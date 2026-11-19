//! A short list of models worth starting from, ranked by what this machine can run.
//!
//! Fit is computed; quality is not. VRAM headroom comes from the registry's own
//! manifest, but "good at coding" cannot be derived from any metadata, so the entries
//! below are curated. That makes this list the one part of ailocal that goes stale as
//! models are superseded - it is a starting point, not an authority, and
//! `ailocal model search` exists for everything not on it.
//!
//! Every entry was verified to resolve against the Ollama registry, and to be a text-only
//! GGUF. Ollama ships some multimodal models with the vision tower inside the weights -
//! `devstral-small-2:24b` carries 222 `v.*`/`mm.*` tensors in its one blob - and
//! llama.cpp refuses a file holding tensors its text model does not create. A `-coding`
//! tag is not a different model either: it shares the base tag's blob and changes only
//! Ollama's own parameters, which llama-server never reads. Sizes are fetched live
//! rather than hardcoded, so a re-tagged model cannot silently misreport its fit.

use reqwest::blocking::Client;

use crate::source::Source;
use crate::vram::{Budget, CacheType};

/// A curated starting point.
#[derive(Debug, Clone, Copy, serde::Serialize)]
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
        reference: "qwen3.6:35b-a3b",
        summary: "agentic coding, mixture-of-experts",
        note: "hybrid attention - one block in four keeps a KV cache, so context is cheap",
        tier: 5,
    },
    Entry {
        reference: "qwen3.8:27b",
        summary: "latest Qwen, coding and long agentic tasks",
        note: "dense, so slower per token than the 35B mixture-of-experts",
        tier: 5,
    },
    Entry {
        reference: "gemma4:31b",
        summary: "largest Gemma, general + coding",
        note: "sliding-window attention, so its KV cache stays small",
        tier: 5,
    },
    Entry {
        reference: "gemma4:26b-a4b",
        summary: "mixture-of-experts Gemma, fast for its size",
        note: "sliding-window attention, so its KV cache stays small",
        tier: 4,
    },
    Entry {
        reference: "qwen3-coder:30b",
        summary: "code specialist, mixture-of-experts",
        note: "no thinking mode - its template has no switch to turn one on",
        tier: 4,
    },
    Entry {
        reference: "gemma4:12b",
        summary: "general + coding, very long context",
        note: "sliding-window attention, so its KV cache stays small - reaches 256k",
        tier: 4,
    },
    Entry {
        reference: "gpt-oss:20b",
        summary: "mixture-of-experts, strong general model",
        note: "natively 4-bit, so quantising it further buys nothing",
        tier: 4,
    },
    Entry {
        reference: "qwen3.5:9b",
        summary: "small and fast, for weaker cards",
        note: "",
        tier: 3,
    },
    Entry {
        reference: "gemma4:e4b",
        summary: "small Gemma, long context on little memory",
        note: "",
        tier: 2,
    },
    Entry {
        reference: "qwen3.5:4b",
        summary: "very small, runs almost anywhere",
        note: "",
        tier: 2,
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

/// Spare VRAM below which no context is worth having, whatever the attention layout.
///
/// One constant rather than a number per caller: a catalogue entry and a Hugging Face
/// file are the same question asked about two sources, and a window that showed a
/// quant as usable which the catalogue would have called too large is telling a
/// person something the next step will contradict.
pub const MIN_HEADROOM_MIB: i64 = 1024;

impl Candidate {
    #[must_use]
    pub fn fits(&self) -> bool {
        self.headroom_mib >= MIN_HEADROOM_MIB
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

/// The fields, and whether it fits.
///
/// Written out by hand for that last one: [`Candidate::fits`] is where the line is
/// drawn, and a window that worked it out again from `headroom_mib` could draw it
/// somewhere else.
impl serde::Serialize for Candidate {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct as _;

        let mut out = serializer.serialize_struct("Candidate", 5)?;
        out.serialize_field("entry", &self.entry)?;
        out.serialize_field("size_mib", &self.size_mib)?;
        out.serialize_field("headroom_mib", &self.headroom_mib)?;
        out.serialize_field("context", &self.context)?;
        out.serialize_field("fits", &self.fits())?;
        out.end()
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
///
/// Every candidate is read at once. One at a time, `model pick` took 28 to 31 s on the
/// M4 Max: twenty requests in a row, a manifest 0.6 s and an 8 MiB head 0.85 s each,
/// at 8.6 MB/s for one stream. Four streams together drew 15 MB/s, so reading in
/// parallel uses the link instead of waiting on it, and the whole pick took 9.4 to
/// 10.9 s - most of that the seven heads, 56 MiB the link cannot move faster.
pub fn measure(client: &Client, candidates: &mut [Candidate], budget: &Budget, cache: CacheType) {
    std::thread::scope(|scope| {
        for candidate in candidates.iter_mut().filter(|c| c.fits()) {
            scope.spawn(move || {
                if let Some(ctx) = usable_context(client, candidate, budget, cache) {
                    candidate.context = Some(ctx);
                }
            });
        }
    });

    candidates.sort_by_key(rank);
}

/// The context `candidate` would get here, read from the head of its blob.
fn usable_context(
    client: &Client,
    candidate: &Candidate,
    budget: &Budget,
    cache: CacheType,
) -> Option<u64> {
    let source = format!("ollama:{}", candidate.entry.reference)
        .parse::<Source>()
        .ok()?;
    let artifact = source.resolve(client, None).ok()?;
    let head = crate::download::head_bytes(client, &artifact, HEAD_BYTES).ok()?;
    let metadata = crate::gguf::parse_head(&head).ok()?;

    match crate::registry::assess(
        metadata.kv_layout(),
        metadata.context_length(),
        budget,
        cache,
        candidate.size_mib,
    ) {
        crate::registry::Fit::Fits(ctx) => Some(ctx),
        _ => None,
    }
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
/// Every entry is asked at once, for the reason [`measure`] gives. Each answer goes
/// back into its entry's own slot, so ties in the ranking still fall in catalogue
/// order.
///
/// # Errors
/// Never fails as a whole; individual lookups are best-effort.
pub fn resolve(client: &Client, available_mib: u64) -> Vec<Candidate> {
    let mut slots: Vec<Option<Candidate>> = ENTRIES.iter().map(|_| None).collect();
    std::thread::scope(|scope| {
        for (entry, slot) in ENTRIES.iter().zip(slots.iter_mut()) {
            scope.spawn(move || *slot = candidate(client, entry, available_mib));
        }
    });

    let mut out: Vec<Candidate> = slots.into_iter().flatten().collect();
    out.sort_by_key(rank);
    out
}

/// `entry` with its real size, or `None` if the registry does not answer for it.
fn candidate(client: &Client, entry: &Entry, available_mib: u64) -> Option<Candidate> {
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

    /// The window reads `fits` rather than comparing `headroom_mib` itself, so the JSON
    /// has to put the line at the same 1024 MiB the picker does.
    #[test]
    fn the_wire_says_whether_it_fits_at_the_pickers_line() {
        let json =
            |available| serde_json::to_value(candidate("gemma4:12b", 7039, available)).unwrap();

        let just = json(7039 + 1024);
        assert_eq!(just["headroom_mib"], 1024);
        assert_eq!(just["fits"], true);
        assert_eq!(just["entry"]["reference"], "gemma4:12b");
        assert_eq!(json(7039 + 1023)["fits"], false);
    }
}
