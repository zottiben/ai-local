//! What models are on disk, and whether this machine can actually run them.
//!
//! The registry is derived from the filesystem rather than stored, so it cannot drift
//! from reality: a model deleted by hand simply stops being listed. Each entry carries
//! the verdict from [`crate::vram`], because "is it downloaded" is a far less useful
//! question than "can it run here, and with how much context".

use std::path::{Path, PathBuf};

use crate::gguf::{self, Metadata};
use crate::vram::{Budget, CacheType, KvLayout};

/// How much of a GGUF to read when probing metadata. Enough for the header and tensor
/// info of a large model without pulling gigabytes through the page cache.
const HEAD_BYTES: usize = 16 << 20;

/// Smallest context worth having a model for at all.
pub const MIN_USEFUL_CONTEXT: u64 = 4096;

/// Whether a model can run on this machine.
///
/// The cases are kept distinct on purpose. An earlier version collapsed 'metadata
/// unreadable' and 'weights do not fit' into one `None` and proceeded on both, which
/// downloaded 16 GB of a model that could never load. Not knowing and knowing it will
/// fail are opposite answers, and only one of them is safe to continue from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    /// Metadata could not be read, so no judgement was possible.
    Unknown,
    /// Runs, with this much context.
    Fits(u64),
    /// Loads, but with too little context to be worth it.
    ContextTooSmall(u64),
    /// The weights alone exceed the budget.
    WeightsTooLarge,
}

/// Decide whether a model fits, given what we know about it.
///
/// `kv` is `None` when the GGUF metadata could not be parsed - distinct from the
/// weights being too large, which [`Budget::max_context`] signals with its own `None`.
#[must_use]
pub fn assess(
    kv: Option<KvLayout>,
    trained_context: Option<u64>,
    budget: &Budget,
    cache: CacheType,
    weights_mib: u64,
) -> Fit {
    let Some(kv) = kv else {
        return Fit::Unknown;
    };
    let Some(ctx) = budget.max_context(&kv, cache, weights_mib) else {
        return Fit::WeightsTooLarge;
    };
    let ctx = trained_context.map_or(ctx, |trained| ctx.min(trained));
    if ctx >= MIN_USEFUL_CONTEXT {
        Fit::Fits(ctx)
    } else {
        Fit::ContextTooSmall(ctx)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pub name: String,
    pub path: PathBuf,
    pub size_mib: u64,
    /// `None` when the file could not be parsed as GGUF.
    pub arch: Option<String>,
    pub trained_context: Option<u64>,
    pub kv: Option<KvLayout>,
    /// Sliding-window models cost far less per token than [`KvLayout`] predicts.
    pub sliding_window: bool,
}

/// Find an installed model that is the artifact `stem` would be downloaded as.
///
/// An exact name match is the easy case. The other one matters more in practice: a
/// file fetched earlier, or by hand, often carries the quantisation in its name where
/// the reference would not. This machine holds `gemma4-12b-Q4_K_M.gguf` while
/// `ollama:gemma4:12b` would store `gemma4-12b.gguf`, and the two are byte-for-byte the
/// same blob - 7381382048 bytes either way. Offering to fetch seven gigabytes someone
/// already has is the thing worth avoiding, so a name that extends the stem and a size
/// that agrees to the megabyte counts as the same model.
///
/// The size check is what keeps this honest: different quantisations of one model
/// differ by hundreds of megabytes, so `gemma4-12b-Q5_K_M` will not be mistaken for
/// `gemma4-12b-Q4_K_M`.
#[must_use]
pub fn already_have<'a>(installed: &'a [Model], stem: &str, size_mib: u64) -> Option<&'a Model> {
    installed
        .iter()
        .find(|m| m.name == stem || (m.name.starts_with(stem) && m.size_mib == size_mib))
}

impl Model {
    /// Largest context this model can hold under `budget`, or `None` if the weights
    /// alone do not fit or the metadata was unreadable.
    ///
    /// Capped at the trained context: a larger window is not usable just because the
    /// memory is free.
    #[must_use]
    pub fn max_context(&self, budget: &Budget, cache: CacheType) -> Option<u64> {
        let kv = self.kv?;
        let fits = budget.max_context(&kv, cache, self.size_mib)?;
        Some(match self.trained_context {
            Some(trained) => fits.min(trained),
            None => fits,
        })
    }

    /// Whether the model can run at all, at any context.
    #[must_use]
    pub fn is_runnable(&self, budget: &Budget, cache: CacheType) -> bool {
        self.max_context(budget, cache).is_some_and(|c| c > 0)
    }
}

/// List the models in `dir`, sorted by name.
///
/// A file that fails to parse is reported with `arch: None` rather than dropped, so a
/// corrupt or partial download is visible instead of silently missing.
///
/// # Errors
/// If the directory cannot be read.
pub fn scan(dir: &Path) -> anyhow::Result<Vec<Model>> {
    use anyhow::Context as _;

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(anyhow::Error::new(e).context(format!("reading {}", dir.display()))),
    };

    let mut models = Vec::new();
    for entry in entries {
        let entry = entry.with_context(|| format!("walking {}", dir.display()))?;
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "gguf") {
            continue;
        }

        let size_mib = entry
            .metadata()
            .map(|m| m.len() / (1024 * 1024))
            .unwrap_or(0);
        let name = path.file_stem().map_or_else(
            || path.display().to_string(),
            |s| s.to_string_lossy().into_owned(),
        );

        let md: Option<Metadata> = gguf::read_file(&path, HEAD_BYTES).ok();
        models.push(Model {
            name,
            path,
            size_mib,
            arch: md
                .as_ref()
                .and_then(|m| m.architecture().map(str::to_owned)),
            trained_context: md.as_ref().and_then(Metadata::context_length),
            kv: md.as_ref().and_then(Metadata::kv_layout),
            sliding_window: md.as_ref().is_some_and(Metadata::has_sliding_window),
        });
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named(name: &str, size_mib: u64) -> Model {
        Model {
            name: name.into(),
            path: PathBuf::from(format!("/tmp/{name}.gguf")),
            size_mib,
            arch: None,
            trained_context: None,
            kv: None,
            sliding_window: false,
        }
    }

    #[test]
    fn an_exact_name_is_the_same_model() {
        let installed = vec![named("qwen3-14b", 8846)];
        assert!(already_have(&installed, "qwen3-14b", 8846).is_some());
        assert!(already_have(&installed, "qwen3-8b", 4000).is_none());
    }

    /// The real case on this machine: `ollama:gemma4:12b` stores `gemma4-12b.gguf`,
    /// but the copy already here is `gemma4-12b-Q4_K_M.gguf` and the two are the same
    /// 7381382048 bytes. Re-downloading that is seven gigabytes wasted.
    #[test]
    fn a_quantisation_suffix_does_not_hide_a_model_we_have() {
        let installed = vec![named("gemma4-12b-Q4_K_M", 7038)];
        let found = already_have(&installed, "gemma4-12b", 7038).expect("same blob");
        assert_eq!(found.name, "gemma4-12b-Q4_K_M");
    }

    /// ...but a different quantisation is a different file, and must still be offered.
    #[test]
    fn a_different_quantisation_is_a_different_model() {
        let installed = vec![named("gemma4-12b-Q5_K_M", 8600)];
        assert!(already_have(&installed, "gemma4-12b", 7038).is_none());
    }

    /// A longer name that merely shares a prefix is not the same model, and the size
    /// is what has to prove it.
    #[test]
    fn a_shared_prefix_alone_is_not_enough() {
        let installed = vec![named("qwen3-14b-instruct", 12000)];
        assert!(already_have(&installed, "qwen3-14b", 8846).is_none());
    }

    #[test]
    fn nothing_installed_matches_nothing() {
        assert!(already_have(&[], "gemma4-12b", 7038).is_none());
    }

    fn model(size_mib: u64, kv: KvLayout, trained: u64) -> Model {
        Model {
            name: "t".into(),
            path: PathBuf::from("/tmp/t.gguf"),
            size_mib,
            arch: Some("test".into()),
            trained_context: Some(trained),
            kv: Some(kv),
            sliding_window: false,
        }
    }

    fn qwen3_14b() -> KvLayout {
        KvLayout::dense(40, 8, 128, 128)
    }

    #[test]
    fn context_is_capped_by_training_not_just_memory() {
        // Plenty of spare VRAM, but the model was only trained to 32k.
        let m = model(8836, qwen3_14b(), 32_768);
        assert_eq!(
            m.max_context(&Budget::new(900), CacheType::Q8_0),
            Some(32_768)
        );
    }

    #[test]
    fn a_model_too_large_to_load_is_not_runnable() {
        // devstral-24b's 13.34 GiB, which could not hold even a 4096 context.
        let m = model(13_660, qwen3_14b(), 131_072);
        assert!(!m.is_runnable(&Budget::new(900), CacheType::Q8_0));
    }

    #[test]
    fn unparseable_models_are_listed_but_not_runnable() {
        let m = Model {
            name: "partial".into(),
            path: PathBuf::from("/tmp/partial.gguf"),
            size_mib: 2,
            arch: None,
            trained_context: None,
            kv: None,
            sliding_window: false,
        };
        assert!(!m.is_runnable(&Budget::new(900), CacheType::Q8_0));
    }

    #[test]
    fn scanning_a_missing_directory_is_empty_not_an_error() {
        assert!(
            scan(Path::new("/nonexistent/ailocal/models"))
                .unwrap()
                .is_empty()
        );
    }

    fn qwen36_27b() -> KvLayout {
        KvLayout::dense(65, 4, 256, 256)
    }

    /// The regression: qwen3.6:27b at Q4_K_M is 16057 MiB against ~13000 MiB available.
    /// This must be a refusal, not an "unknown" that installs 16 GB anyway.
    #[test]
    fn oversized_weights_are_a_refusal_not_an_unknown() {
        let fit = assess(
            Some(qwen36_27b()),
            Some(262_144),
            &Budget::new(900),
            CacheType::Q8_0,
            16_057,
        );
        assert_eq!(fit, Fit::WeightsTooLarge);
        assert_ne!(
            fit,
            Fit::Unknown,
            "must not be confused with missing metadata"
        );
    }

    #[test]
    fn unreadable_metadata_is_unknown_not_a_refusal() {
        assert_eq!(
            assess(None, None, &Budget::new(900), CacheType::Q8_0, 100),
            Fit::Unknown
        );
    }

    /// A 27B at IQ3_XXS loads and holds ~16k, which is worth downloading.
    #[test]
    fn a_model_that_loads_with_usable_context_fits() {
        match assess(
            Some(qwen36_27b()),
            Some(262_144),
            &Budget::new(900),
            CacheType::Q8_0,
            10_428,
        ) {
            Fit::Fits(ctx) => assert!((14_000..18_000).contains(&ctx), "got {ctx}"),
            other => panic!("expected Fits, got {other:?}"),
        }
    }

    /// Barely loading is not the same as being useful.
    #[test]
    fn a_model_with_a_sliver_of_context_is_refused() {
        match assess(
            Some(qwen36_27b()),
            Some(262_144),
            &Budget::new(900),
            CacheType::Q8_0,
            12_900,
        ) {
            Fit::ContextTooSmall(ctx) => assert!(ctx < MIN_USEFUL_CONTEXT, "got {ctx}"),
            other => panic!("expected ContextTooSmall, got {other:?}"),
        }
    }
}
