//! The sampling settings a model's publisher recommends, kept beside its weights.
//!
//! llama-server falls back to generic defaults - temperature 0.8, top-k 40, min-p 0.05,
//! no presence penalty - for anything the GGUF does not carry, and most carry nothing.
//! Measured on `qwen3.6-35b-a3b`: its GGUF has no `general.sampling.*` key at all, so it
//! ran on those defaults while its publisher recommends top-k 20, min-p 0 and a presence
//! penalty of 1.5.
//!
//! The Ollama registry ships those recommendations as a separate `params` layer beside
//! the weights. Install keeps that layer as `<name>.params.json`, verbatim, and serve
//! passes it on as server defaults - so a harness that sends its own values still wins,
//! request by request. Kept verbatim rather than translated so that a key this build
//! does not map yet is still on disk for one that does.

use std::path::{Path, PathBuf};

use anyhow::Context as _;

/// What llama-server is told to sample with, when the publisher said.
///
/// Every field is optional because a `params` layer names only what its publisher
/// chose to set, and a field left out must stay llama.cpp's - or the GGUF's own
/// `general.sampling.*`, which llama.cpp reads for anything not given on the command
/// line.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Sampling {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f64>,
    /// How far back the three penalties look. Ollama and llama.cpp both default to 64.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i64>,
}

impl Sampling {
    /// Read a `params` layer. Keys that are not sampling - `stop`, `num_ctx`,
    /// `draft_num_predict` - are ignored: the chat template already stops generation,
    /// and the context is sized from the card rather than taken from a publisher who
    /// never saw it.
    ///
    /// # Errors
    /// If the bytes are not a JSON object of the expected types.
    pub fn parse(bytes: &[u8]) -> anyhow::Result<Self> {
        serde_json::from_slice(bytes).context("parsing sampling parameters")
    }

    /// Whether this sets nothing, so llama.cpp decides everything.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// The llama-server flags for what was set, and nothing for what was not.
    #[must_use]
    pub fn args(&self) -> Vec<String> {
        let floats = [
            ("--temp", self.temperature),
            ("--top-p", self.top_p),
            ("--min-p", self.min_p),
            ("--presence-penalty", self.presence_penalty),
            ("--frequency-penalty", self.frequency_penalty),
            ("--repeat-penalty", self.repeat_penalty),
        ];
        let mut args = Vec::new();
        if let Some(k) = self.top_k {
            args.extend(["--top-k".to_owned(), k.to_string()]);
        }
        for (flag, value) in floats {
            if let Some(v) = value {
                args.extend([flag.to_owned(), v.to_string()]);
            }
        }
        if let Some(n) = self.repeat_last_n {
            args.extend(["--repeat-last-n".to_owned(), n.to_string()]);
        }
        args
    }
}

/// One line for `ps` and the installer, in the order the model cards list them.
impl std::fmt::Display for Sampling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let named = [
            ("temperature", self.temperature.map(|v| v.to_string())),
            ("top-k", self.top_k.map(|v| v.to_string())),
            ("top-p", self.top_p.map(|v| v.to_string())),
            ("min-p", self.min_p.map(|v| v.to_string())),
            ("presence", self.presence_penalty.map(|v| v.to_string())),
            ("frequency", self.frequency_penalty.map(|v| v.to_string())),
            ("repeat", self.repeat_penalty.map(|v| v.to_string())),
            ("repeat-last-n", self.repeat_last_n.map(|v| v.to_string())),
        ];
        let set: Vec<String> = named
            .into_iter()
            .filter_map(|(name, value)| value.map(|v| format!("{name} {v}")))
            .collect();
        f.write_str(&set.join(", "))
    }
}

/// Where the settings for the weights at `weights` live.
#[must_use]
pub fn path_for(weights: &Path) -> PathBuf {
    weights.with_extension("params.json")
}

/// The settings kept for `weights`, or `None` when there are none.
///
/// None is the ordinary case for a model from Hugging Face, which publishes no such
/// layer. A file that is there but unreadable is an error rather than a `None`: it was
/// validated when it was written, so something has damaged it since, and serving on
/// llama.cpp's defaults without saying so is the bug this module exists to fix.
///
/// # Errors
/// If the file exists but cannot be read or parsed.
pub fn load(weights: &Path) -> anyhow::Result<Option<Sampling>> {
    let path = path_for(weights);
    match std::fs::read(&path) {
        Ok(bytes) => Sampling::parse(&bytes).map(Some).with_context(|| {
            format!(
                "reading {}; delete it to serve on llama.cpp's defaults",
                path.display()
            )
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(anyhow::Error::new(e).context(format!("reading {}", path.display()))),
    }
}

/// Keep a publisher's `params` layer beside `weights`, as it came.
///
/// Parsed before anything is written, so a layer this build cannot read never lands
/// where serve would then refuse to start on it.
///
/// # Errors
/// If the bytes do not parse or the file cannot be written.
pub fn save(weights: &Path, bytes: &[u8]) -> anyhow::Result<Sampling> {
    let sampling = Sampling::parse(bytes)?;
    let path = path_for(weights);
    std::fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(sampling)
}

/// Delete the settings kept for `weights`, if there are any.
///
/// # Errors
/// If the file is there and cannot be removed.
pub fn remove(weights: &Path) -> anyhow::Result<()> {
    let path = path_for(weights);
    match std::fs::remove_file(&path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            Err(anyhow::Error::new(e).context(format!("deleting {}", path.display())))
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layer the registry served for `qwen3.6:35b-a3b` on 2026-11-23, byte for byte.
    const QWEN36: &[u8] = br#"{"draft_num_predict":2,"min_p":0,"presence_penalty":1.5,"repeat_penalty":1,"temperature":1,"top_k":20,"top_p":0.95}"#;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ailocal-sampling-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("qwen3.6-35b-a3b.gguf")
    }

    #[test]
    fn a_published_layer_becomes_llama_server_flags() {
        let sampling = Sampling::parse(QWEN36).unwrap();
        assert_eq!(
            sampling.args(),
            [
                "--top-k",
                "20",
                "--temp",
                "1",
                "--top-p",
                "0.95",
                "--min-p",
                "0",
                "--presence-penalty",
                "1.5",
                "--repeat-penalty",
                "1",
            ]
        );
        assert_eq!(
            sampling.to_string(),
            "temperature 1, top-k 20, top-p 0.95, min-p 0, presence 1.5, repeat 1"
        );
    }

    /// A field the publisher left out has to stay out. Passing llama.cpp's own default
    /// for it would override a `general.sampling.*` value in the GGUF.
    #[test]
    fn only_what_was_set_is_passed() {
        let sampling = Sampling::parse(br#"{"temperature":0.6,"stop":["<|im_end|>"]}"#).unwrap();
        assert_eq!(sampling.args(), ["--temp", "0.6"]);
        assert!(!sampling.is_empty());
        assert!(
            Sampling::parse(br#"{"stop":["</s>"],"num_ctx":4096}"#)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn the_layer_is_kept_verbatim_beside_the_weights_and_read_back() {
        let weights = scratch("roundtrip");
        assert_eq!(load(&weights).unwrap(), None);

        let saved = save(&weights, QWEN36).unwrap();
        assert_eq!(
            path_for(&weights).file_name().unwrap(),
            "qwen3.6-35b-a3b.params.json"
        );
        assert_eq!(std::fs::read(path_for(&weights)).unwrap(), QWEN36);
        assert_eq!(load(&weights).unwrap(), Some(saved));

        remove(&weights).unwrap();
        assert_eq!(load(&weights).unwrap(), None);
        remove(&weights).unwrap();
    }

    #[test]
    fn a_layer_that_does_not_parse_is_never_written() {
        let weights = scratch("invalid");
        assert!(save(&weights, br#"{"top_k":"twenty"}"#).is_err());
        assert!(!path_for(&weights).exists());
    }

    #[test]
    fn a_damaged_file_is_an_error_not_a_silent_default() {
        let weights = scratch("damaged");
        std::fs::write(path_for(&weights), b"{").unwrap();
        let err = load(&weights).unwrap_err();
        assert!(format!("{err:#}").contains("delete it"), "{err:#}");
    }
}
