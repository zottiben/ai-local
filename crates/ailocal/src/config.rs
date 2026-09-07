//! Where things live on this machine.
//!
//! Every large artifact - weights, adapters, datasets, the Hugging Face cache - is
//! placed by configuration, never by a library default. The defaults here point at the
//! big disk because the alternatives silently fill a partition: `HF_HOME` defaults to
//! `~/.cache/huggingface`, and a single model is larger than the free space on `/home`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Root for everything too big to live in the repo or a home directory.
const DATA_ROOT: &str = "/mnt/kingston/ailocal";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Directory holding GGUF weights.
    #[serde(default = "default_models_dir")]
    pub models_dir: PathBuf,

    /// Value to export as `HF_HOME` for any child process touching the HF hub.
    #[serde(default = "default_hf_home")]
    pub hf_home: PathBuf,

    /// KV cache quantisation. `q8_0` roughly halves cache size for no measurable
    /// throughput cost, and is what makes long contexts fit.
    #[serde(default = "default_cache_type")]
    pub cache_type: String,

    /// Whether models may emit chain-of-thought: `auto`, `on` or `off`.
    ///
    /// Both current models are reasoning models, and they spend heavily on it - gemma4
    /// burned ~700 tokens of thinking on "say hello in three words" and never reached
    /// an answer. Thinking goes to a separate `reasoning_content` field, so an
    /// exhausted budget looks to a client like an empty reply. `off` trades that
    /// capability for a direct answer.
    #[serde(default = "default_reasoning")]
    pub reasoning: String,

    /// Token ceiling on thinking. `-1` is unrestricted, `0` ends it immediately.
    #[serde(default = "default_reasoning_budget")]
    pub reasoning_budget: i64,

    /// Model the service unit loads at boot. `None` means load nothing and let the
    /// gateway pull one in on the first request.
    #[serde(default)]
    pub default_model: Option<String>,

    /// Address the gateway service binds to.
    ///
    /// Loopback by default. The the tunnel host tunnel reaches this host over the LAN, so
    /// exposing it there means binding `0.0.0.0` - which is safe only because the
    /// gateway requires a bearer key.
    #[serde(default = "default_gateway_host")]
    pub gateway_host: String,

    #[serde(default = "default_gateway_port")]
    pub gateway_port: u16,
}

fn default_models_dir() -> PathBuf {
    Path::new(DATA_ROOT).join("models")
}

fn default_hf_home() -> PathBuf {
    Path::new(DATA_ROOT).join("hf")
}

fn default_cache_type() -> String {
    "q8_0".to_owned()
}

fn default_reasoning() -> String {
    "auto".to_owned()
}

fn default_reasoning_budget() -> i64 {
    -1
}

fn default_gateway_host() -> String {
    "127.0.0.1".to_owned()
}

fn default_gateway_port() -> u16 {
    8081
}

impl Default for Config {
    fn default() -> Self {
        Self {
            models_dir: default_models_dir(),
            hf_home: default_hf_home(),
            cache_type: default_cache_type(),
            reasoning: default_reasoning(),
            reasoning_budget: default_reasoning_budget(),
            default_model: None,
            gateway_host: default_gateway_host(),
            gateway_port: default_gateway_port(),
        }
    }
}

impl Config {
    /// Path to the config file, honouring `XDG_CONFIG_HOME`.
    ///
    /// # Errors
    /// If neither `XDG_CONFIG_HOME` nor `HOME` is set.
    pub fn path() -> anyhow::Result<PathBuf> {
        if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
            return Ok(PathBuf::from(xdg).join("ailocal/config.toml"));
        }
        let home = std::env::var_os("HOME")
            .ok_or_else(|| anyhow::anyhow!("neither XDG_CONFIG_HOME nor HOME is set"))?;
        Ok(PathBuf::from(home).join(".config/ailocal/config.toml"))
    }

    /// Load the config, falling back to defaults when the file does not exist.
    ///
    /// A missing file is not an error - the defaults are correct for this machine and
    /// writing one out is the user's choice, not a precondition for the tool working.
    ///
    /// # Errors
    /// If the file exists but cannot be read or parsed.
    pub fn load() -> anyhow::Result<Self> {
        use anyhow::Context as _;

        let path = Self::path()?;
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(anyhow::Error::new(e).context(format!("reading {}", path.display()))),
        }
    }

    /// Write the config out, creating parent directories.
    ///
    /// # Errors
    /// If the file cannot be serialised or written.
    pub fn save(&self) -> anyhow::Result<PathBuf> {
        use anyhow::Context as _;

        let path = Self::path()?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        std::fs::write(&path, toml::to_string_pretty(self)?)
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_point_at_the_big_disk() {
        let c = Config::default();
        assert!(c.models_dir.starts_with(DATA_ROOT));
        assert!(
            c.hf_home.starts_with(DATA_ROOT),
            "HF_HOME must not default into /home"
        );
    }

    #[test]
    fn partial_config_keeps_defaults_for_the_rest() {
        let c: Config = toml::from_str(r#"models_dir = "/tmp/models""#).unwrap();
        assert_eq!(c.models_dir, Path::new("/tmp/models"));
        assert_eq!(c.hf_home, default_hf_home());
        assert_eq!(c.cache_type, "q8_0");
        assert_eq!(c.reasoning, "auto");
        assert_eq!(c.reasoning_budget, -1);
    }

    #[test]
    fn round_trips() {
        let c = Config::default();
        let back: Config = toml::from_str(&toml::to_string_pretty(&c).unwrap()).unwrap();
        assert_eq!(back.models_dir, c.models_dir);
        assert_eq!(back.cache_type, c.cache_type);
    }

    /// A typo in a key should be reported, not silently ignored into a default.
    #[test]
    fn unknown_keys_are_rejected() {
        assert!(toml::from_str::<Config>(r#"model_dir = "/tmp""#).is_err());
    }
}
