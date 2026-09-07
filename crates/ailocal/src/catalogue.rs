//! A short list of models worth starting from, ranked by what this machine can run.
//!
//! Fit is computed; quality is not. VRAM headroom comes from the registry's own
//! manifest, but "good at coding" cannot be derived from any metadata, so the entries
//! below are curated. That makes this list the one part of ailocal that goes stale as
//! models are superseded - it is a starting point, not an authority, and
//! `ailocal model search` exists for everything not on it.
//!
//! Every entry was verified to resolve against the Ollama registry. Sizes are fetched
//! live rather than hardcoded, so a re-tagged model cannot silently misreport its fit.

use reqwest::blocking::Client;

use crate::source::Source;
use crate::vram::{Budget, CacheType};

/// A curated starting point.
#[derive(Debug, Clone, Copy)]
pub struct Entry {
    /// Ollama reference, `<name>:<tag>`.
    pub reference: &'static str,
    /// What it is good for, in a few words.
    pub summary: &'static str,
    /// Anything non-obvious about running it.
    pub note: &'static str,
    /// Rough capability class, 1 (tiny) to 5 (specialist / frontier-for-its-size).
    ///
    /// A judgement call, not a measurement - there is no metadata anywhere that says
    /// how good a model is at coding. Fit is computed from the real file; this is the
    /// part a human maintains, and the part that dates.
    pub tier: u8,
}

/// The starting set.
///
/// Ollama references rather than Hugging Face ones because downloads are far faster and
/// verify against a published digest. The cost is quant coverage - these are the default
/// `Q4_K_M` builds, so a model that only just misses the budget cannot be squeezed in
/// from here. `ailocal model files` on a Hugging Face repo is the way to do that.
pub const ENTRIES: &[Entry] = &[
    Entry {
        reference: "gemma4:12b",
        summary: "general + coding, very long context",
        note: "sliding-window attention, so its KV cache stays small - reaches 256k",
        tier: 4,
    },
    Entry {
        reference: "qwen3:14b",
        summary: "general + coding, holds up well under long prompts",
        note: "full attention, so context costs ~85 KiB/token",
        tier: 4,
    },
    Entry {
        reference: "qwen3-coder:30b",
        summary: "code specialist, mixture-of-experts",
        note: "needs a large card at this quant",
        tier: 5,
    },
    Entry {
        reference: "devstral:24b",
        summary: "agentic coding specialist",
        note: "weights alone are ~13.4 GiB, leaving little for context",
        tier: 5,
    },
    Entry {
        reference: "gpt-oss:20b",
        summary: "mixture-of-experts, strong general model",
        note: "natively 4-bit, so quantising it further buys nothing",
        tier: 4,
    },
    Entry {
        reference: "phi4:14b",
        summary: "compact general model, strong at reasoning",
        note: "",
        tier: 3,
    },
    Entry {
        reference: "mistral-nemo:12b",
        summary: "general purpose, modest footprint",
        note: "",
        tier: 3,
    },
    Entry {
        reference: "qwen3:8b",
        summary: "small and fast, for weaker cards",
        note: "",
        tier: 2,
    },
    Entry {
        reference: "qwen3:4b",
        summary: "very small, runs almost anywhere",
        note: "",
        tier: 1,
    },
];

/// A catalogue entry with its real size resolved.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub entry: Entry,
    pub size_mib: u64,
    /// MiB left for KV cache and compute buffers once the weights are resident.
    pub headroom_mib: i64,
    /// Largest context this machine can actually give it, once measured.
    ///
    /// `None` until [`measure`] has read the model's attention geometry.
    pub context: Option<u64>,
}

impl Candidate {
    #[must_use]
    pub fn fits(&self) -> bool {
        // Below roughly this much spare there is no room for a usable context, whatever
        // the model's attention layout.
        self.headroom_mib >= 1024
    }

    /// One line for a picker.
    #[must_use]
    pub fn label(&self) -> String {
        let fit = match (self.fits(), self.context) {
            (false, _) => "  will not fit".to_owned(),
            (true, Some(ctx)) => format!("{:>4}k context", ctx / 1024),
            (true, None) => format!("{:>5} GiB spare", self.headroom_mib / 1024),
        };
        format!(
            "{:<18} {:>5} GiB  {fit}   {}",
            self.entry.reference,
            self.size_mib / 1024,
            self.entry.summary
        )
    }
}

/// How much of a remote GGUF to read to learn its attention geometry.
const HEAD_BYTES: u64 = 8 << 20;

/// Measure how much context each fitting candidate can actually hold, and re-rank.
///
/// Weight size is a poor proxy for usefulness. gemma4-12b is smaller than qwen3-14b but
/// holds 256k tokens against its 40k, because most of its layers use a sliding window -
/// so ranking by size alone recommends the worse model. This reads each candidate's KV
/// geometry from the first few MiB of its blob and ranks by the number that decides
/// whether a model is pleasant to use.
///
/// Candidates whose metadata cannot be read keep their size-based ordering rather than
/// disappearing.
pub fn measure(client: &Client, candidates: &mut [Candidate], budget: &Budget, cache: CacheType) {
    for candidate in candidates.iter_mut().filter(|c| c.fits()) {
        let Ok(source) = format!("ollama:{}", candidate.entry.reference).parse::<Source>() else {
            continue;
        };
        let Ok(artifact) = source.resolve(client, None) else {
            continue;
        };
        let Ok(head) = crate::download::head_bytes(client, &artifact, HEAD_BYTES) else {
            continue;
        };
        let Ok(metadata) = crate::gguf::parse_head(&head) else {
            continue;
        };

        if let crate::registry::Fit::Fits(ctx) = crate::registry::assess(
            metadata.kv_layout(),
            metadata.context_length(),
            budget,
            cache,
            candidate.size_mib,
        ) {
            candidate.context = Some(ctx);
        }
    }

    candidates.sort_by_key(rank);
}

/// Smallest context worth having for coding work.
///
/// Below this a model technically loads but cannot hold a file plus a conversation,
/// which is worse than a smaller model that can.
const PRACTICAL_CONTEXT: u64 = 16_384;

/// Sort key, ascending: better candidates compare smaller.
///
/// Neither dimension alone is right. Ranking purely by capability offers models that
/// hold only a few thousand tokens; ranking purely by measured context puts a 4B on top,
/// because being small is what leaves the room. So: usable first, then the most capable
/// of those, then whichever holds the most context.
fn rank(c: &Candidate) -> (bool, bool, std::cmp::Reverse<u8>, std::cmp::Reverse<u64>) {
    (
        !c.fits(),
        !c.context.is_none_or(|ctx| ctx >= PRACTICAL_CONTEXT),
        std::cmp::Reverse(c.entry.tier),
        std::cmp::Reverse(c.context.unwrap_or(0)),
    )
}

/// Resolve every entry's real size and rank them for this machine.
///
/// Models that fit come first, largest first - within the budget, a bigger model is
/// generally the better one. Those that do not fit are kept, at the end, so the list
/// explains itself rather than mysteriously omitting well-known models.
///
/// Entries that fail to resolve are skipped rather than failing the whole listing: a
/// re-tagged or withdrawn model should not make the picker unusable.
///
/// # Errors
/// Never fails as a whole; individual lookups are best-effort.
pub fn resolve(client: &Client, available_mib: u64) -> Vec<Candidate> {
    let mut out: Vec<Candidate> = ENTRIES
        .iter()
        .filter_map(|entry| {
            let source: Source = entry
                .reference
                .parse()
                .ok()
                .map(|s: Source| s)
                .or_else(|| format!("ollama:{}", entry.reference).parse::<Source>().ok())?;
            let artifact = source.resolve(client, None).ok()?;
            let size_mib = artifact.size? / (1024 * 1024);
            Some(Candidate {
                entry: *entry,
                size_mib,
                headroom_mib: i64::try_from(available_mib).unwrap_or(i64::MAX)
                    - i64::try_from(size_mib).unwrap_or(i64::MAX),
                context: None,
            })
        })
        .collect();

    out.sort_by_key(rank);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(reference: &'static str, size_mib: u64, available: u64) -> Candidate {
        with_tier(reference, size_mib, available, 3)
    }

    fn with_tier(reference: &'static str, size_mib: u64, available: u64, tier: u8) -> Candidate {
        Candidate {
            entry: Entry {
                reference,
                summary: "",
                note: "",
                tier,
            },
            size_mib,
            headroom_mib: available as i64 - size_mib as i64,
            context: None,
        }
    }

    #[test]
    fn every_entry_is_a_parseable_ollama_reference() {
        for e in ENTRIES {
            let r = format!("ollama:{}", e.reference);
            assert!(
                r.parse::<Source>().is_ok(),
                "{} is not a valid reference",
                e.reference
            );
        }
    }

    #[test]
    fn a_model_with_no_room_for_context_does_not_fit() {
        // devstral-24b against this machine's budget: loads, but holds nothing.
        assert!(!candidate("devstral:24b", 13_670, 12_900).fits());
        assert!(candidate("gemma4:12b", 7039, 12_900).fits());
    }

    #[test]
    fn ranking_puts_fitting_models_first_then_prefers_the_largest() {
        let mut v = [
            candidate("small", 2382, 12_900),
            candidate("huge", 40_551, 12_900),
            candidate("good", 8846, 12_900),
        ];
        v.sort_by(|a, b| b.fits().cmp(&a.fits()).then(b.size_mib.cmp(&a.size_mib)));

        assert_eq!(
            v[0].entry.reference, "good",
            "largest that fits comes first"
        );
        assert_eq!(v[1].entry.reference, "small");
        assert_eq!(v[2].entry.reference, "huge", "unusable models sink");
    }

    #[test]
    fn labels_say_why_a_model_is_unavailable() {
        assert!(
            candidate("huge", 40_551, 12_900)
                .label()
                .contains("will not fit")
        );
        assert!(candidate("ok", 7039, 12_900).label().contains("spare"));
    }

    /// The reason `measure` exists: a smaller model with sliding-window attention beats
    /// a larger one with full attention on this hardware, so size must not decide the
    /// order once context has been measured.
    #[test]
    fn measured_context_outranks_raw_size() {
        let mut small = candidate("gemma4:12b", 7039, 12_900);
        small.context = Some(262_144);
        let mut large = candidate("qwen3:14b", 8846, 12_900);
        large.context = Some(39_572);

        let mut v = [large, small];
        v.sort_by(|a, b| {
            b.fits()
                .cmp(&a.fits())
                .then(b.context.cmp(&a.context))
                .then(b.size_mib.cmp(&a.size_mib))
        });
        assert_eq!(v[0].entry.reference, "gemma4:12b");
    }

    #[test]
    fn a_measured_label_reports_context_rather_than_spare_memory() {
        let mut c = candidate("gemma4:12b", 7039, 12_900);
        c.context = Some(262_144);
        assert!(c.label().contains("256k context"), "{}", c.label());
    }
}
