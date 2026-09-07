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
}
