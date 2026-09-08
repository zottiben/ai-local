//! Where things live on this machine.
//!
//! Every large artifact - weights, adapters, datasets, the Hugging Face cache - is
//! placed by configuration, never by a library default. Left to themselves the
//! libraries choose badly: `HF_HOME` defaults to `~/.cache/huggingface`, and a single
//! model is larger than the free space on a lot of home partitions.
//!
//! One knob decides all of it. [`Config::data_dir`] is the root, and `models_dir`,
//! `hf_home` and `eval_dir` hang off it unless they are set individually - so moving
//! everything to a bigger disk is one line, while putting only the weights on a
//! scratch volume is still possible.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Environment override for the data root, taking precedence over the default but not
/// over the config file. Mostly for tests and for one-off runs against another disk.
pub const DATA_DIR_ENV: &str = "AILOCAL_DATA_DIR";

/// The data root when nothing says otherwise.
///
/// `~/.ailocal`, matching where the rest of this ecosystem puts model weights
/// (`~/.ollama`, `~/.cache/huggingface`) rather than an XDG data directory, because
/// these are tens of gigabytes that people relocate - the default only has to be
/// obvious and writable, and being the same string on every platform makes it one
/// thing to document and one thing to move.
///
/// It is a default, not a policy. Anywhere with room is a better answer, which is why
/// `ailocal setup` asks.
#[must_use]
pub fn default_data_dir() -> PathBuf {
    data_root_from(std::env::var_os(DATA_DIR_ENV), std::env::var_os("HOME"))
}

/// The resolution rule, separated from the environment so it can be tested.
///
/// Setting an environment variable is `unsafe` under edition 2024 and this workspace
/// forbids `unsafe`, so a test that poked the real environment could not be written at
/// all - which is reason enough to keep the decision in a pure function.
fn data_root_from(
    explicit: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
) -> PathBuf {
    if let Some(explicit) = explicit {
        return PathBuf::from(explicit);
    }
    // The relative fallback is unreachable in practice - `Config::path` needs HOME too
    // and fails first - and keeps this total rather than panicking inside a getter.
    home.map_or_else(
        || PathBuf::from(".ailocal"),
        |home| PathBuf::from(home).join(".ailocal"),
    )
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "Raw", into = "Raw")]
pub struct Config {
    /// Root for everything too big to live in a repo: weights, the HF cache, adapters,
    /// datasets, eval scratch.
    ///
    /// Set this and the three paths below follow. Set one of those and it wins for that
    /// one thing.
    pub data_dir: PathBuf,

    /// Directory holding GGUF weights.
    pub models_dir: PathBuf,

    /// Value to export as `HF_HOME` for any child process touching the HF hub.
    pub hf_home: PathBuf,

    /// Where the `eval` extra keeps run records and its compile scratch.
    ///
    /// Reports themselves are kilobytes, but the scratch directory holds compiled test
    /// binaries, and the training slices this eval gates will put datasets alongside
    /// them.
    pub eval_dir: PathBuf,

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
    /// Loopback by default. A Cloudflare tunnel reaches this host over the LAN, so
    /// exposing it there means binding `0.0.0.0` - which is safe only because the
    /// gateway requires a bearer key.
    #[serde(default = "default_gateway_host")]
    pub gateway_host: String,

    #[serde(default = "default_gateway_port")]
    pub gateway_port: u16,
}

/// The serialised shape.
///
/// Separate from [`Config`] because serde defaults cannot depend on another field, and
/// the derived paths must. Going through this also means a written config records only
/// what was actually chosen: a `models_dir` that matches `data_dir/models` is left out,
/// so `data_dir` stays the single knob instead of being silently overridden by three
/// stale absolute paths the next time someone moves their disk.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    data_dir: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    models_dir: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hf_home: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    eval_dir: Option<PathBuf>,

    #[serde(default = "default_cache_type")]
    cache_type: String,
    #[serde(default = "default_reasoning")]
    reasoning: String,
    #[serde(default = "default_reasoning_budget")]
    reasoning_budget: i64,
    #[serde(default)]
    default_model: Option<String>,
    #[serde(default = "default_gateway_host")]
    gateway_host: String,
    #[serde(default = "default_gateway_port")]
    gateway_port: u16,
}

/// Point at the likely cause when a config will not parse.
///
/// Unknown keys are rejected so a typo is reported rather than silently ignored, but
/// that same strictness makes a config written by a *newer* ailocal fail every command
/// with nothing but "unknown field". Seen for real: a 0.2.0 service could not read the
/// `data_dir` a 0.3 binary had written, and restart-looped with no hint of why.
fn unknown_field_hint(text: &str) -> Option<String> {
    // Only fields this build does not know about can produce that error, so a key we
    // do not recognise is the thing worth naming.
    const KNOWN: &[&str] = &[
        "data_dir",
        "models_dir",
        "hf_home",
        "eval_dir",
        "cache_type",
        "reasoning",
        "reasoning_budget",
        "default_model",
        "gateway_host",
        "gateway_port",
    ];

    let unknown: Vec<&str> = text
        .lines()
        .filter_map(|l| l.split_once('=').map(|(k, _)| k.trim()))
        .filter(|k| !k.is_empty() && !k.starts_with('#') && !KNOWN.contains(k))
        .collect();

    (!unknown.is_empty()).then(|| {
        format!(
            "this ailocal ({}) does not know: {}\n\
             If the config was written by a newer version, `ailocal update` will \
             catch this binary up.",
            env!("CARGO_PKG_VERSION"),
            unknown.join(", ")
        )
    })
}

/// Work out the data root from paths named the way we would have named them.
///
/// A config written before `data_dir` existed names each path individually and has no
/// root. Falling straight back to the platform default would then quietly relocate
/// whatever that old config did not happen to mention - on upgrade, eval reports would
/// move from the big disk to the home directory and simply stop being found, which
/// looks like losing them.
///
/// A path shaped like `<dir>/models` is evidence that `<dir>` is the root. One shaped
/// like `/scratch/weights` is a deliberate override for that one thing and says nothing
/// about a root, so it is ignored here. Paths that disagree about the parent mean there
/// is no single root to find.
fn infer_root(raw: &Raw) -> Option<PathBuf> {
    let mut root: Option<&Path> = None;

    for (path, name) in [
        (raw.models_dir.as_deref(), "models"),
        (raw.hf_home.as_deref(), "hf"),
        (raw.eval_dir.as_deref(), "eval"),
    ] {
        let Some(path) = path else { continue };
        if path.file_name().is_none_or(|f| f != name) {
            continue;
        }
        let Some(parent) = path.parent() else {
            continue;
        };
        match root {
            None => root = Some(parent),
            Some(seen) if seen != parent => return None,
            Some(_) => {}
        }
    }
    root.map(Path::to_path_buf)
}

impl From<Raw> for Config {
    fn from(raw: Raw) -> Self {
        let data_dir = raw
            .data_dir
            .clone()
            .or_else(|| infer_root(&raw))
            .unwrap_or_else(default_data_dir);
        Self {
            models_dir: raw.models_dir.unwrap_or_else(|| data_dir.join("models")),
            hf_home: raw.hf_home.unwrap_or_else(|| data_dir.join("hf")),
            eval_dir: raw.eval_dir.unwrap_or_else(|| data_dir.join("eval")),
            data_dir,
            cache_type: raw.cache_type,
            reasoning: raw.reasoning,
            reasoning_budget: raw.reasoning_budget,
            default_model: raw.default_model,
            gateway_host: raw.gateway_host,
            gateway_port: raw.gateway_port,
        }
    }
}

impl From<Config> for Raw {
    fn from(c: Config) -> Self {
        // Omit anything that is exactly what `data_dir` already implies.
        let derived = |actual: &Path, name: &str| {
            (actual != c.data_dir.join(name)).then(|| actual.to_path_buf())
        };
        Self {
            models_dir: derived(&c.models_dir, "models"),
            hf_home: derived(&c.hf_home, "hf"),
            eval_dir: derived(&c.eval_dir, "eval"),
            data_dir: Some(c.data_dir),
            cache_type: c.cache_type,
            reasoning: c.reasoning,
            reasoning_budget: c.reasoning_budget,
            default_model: c.default_model,
            gateway_host: c.gateway_host,
            gateway_port: c.gateway_port,
        }
    }
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
        let data_dir = default_data_dir();
        Self {
            models_dir: data_dir.join("models"),
            hf_home: data_dir.join("hf"),
            eval_dir: data_dir.join("eval"),
            data_dir,
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
    /// Move the data root, carrying every path that was following it.
    ///
    /// Paths that were set individually stay where they are: someone who pinned
    /// `models_dir` to a scratch volume meant it, and silently dragging it along with
    /// the root would be the opposite of what they asked for.
    pub fn set_data_dir(&mut self, root: PathBuf) {
        for (path, name) in [
            (&mut self.models_dir, "models"),
            (&mut self.hf_home, "hf"),
            (&mut self.eval_dir, "eval"),
        ] {
            if *path == self.data_dir.join(name) {
                *path = root.join(name);
            }
        }
        self.data_dir = root;
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
    /// A missing file is not an error - the defaults are reasonable and writing one out
    /// is the user's choice, not a precondition for the tool working.
    ///
    /// # Errors
    /// If the file exists but cannot be read or parsed.
    pub fn load() -> anyhow::Result<Self> {
        let path = Self::path()?;
        match std::fs::read_to_string(&path) {
            Ok(text) => toml::from_str(&text).map_err(|e| {
                anyhow::Error::new(e).context(unknown_field_hint(&text).map_or_else(
                    || format!("parsing {}", path.display()),
                    |hint| format!("parsing {}\n{hint}", path.display()),
                ))
            }),
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

    /// Everything large hangs off one root, so a user with a different disk layout has
    /// exactly one thing to change.
    #[test]
    fn every_large_path_derives_from_the_data_root() {
        let c = Config::default();
        for path in [&c.models_dir, &c.hf_home, &c.eval_dir] {
            assert!(
                path.starts_with(&c.data_dir),
                "{} escapes the data root {}",
                path.display(),
                c.data_dir.display()
            );
        }
    }

    /// The Hugging Face libraries default to `~/.cache/huggingface`, which is how a
    /// single download fills a home partition. Ours must never be that.
    #[test]
    fn hf_home_is_ours_rather_than_the_library_default() {
        let c = Config::default();
        assert!(!c.hf_home.ends_with(".cache/huggingface"));
        assert_eq!(c.hf_home, c.data_dir.join("hf"));
    }

    fn os(s: &str) -> std::ffi::OsString {
        std::ffi::OsString::from(s)
    }

    #[test]
    fn the_environment_can_override_the_default_root() {
        assert_eq!(
            data_root_from(Some(os("/scratch/ai")), Some(os("/home/u"))),
            Path::new("/scratch/ai")
        );
    }

    #[test]
    fn the_default_root_sits_in_the_home_directory() {
        assert_eq!(
            data_root_from(None, Some(os("/Users/someone"))),
            Path::new("/Users/someone/.ailocal"),
            "a fresh machine with no config must land somewhere writable"
        );
    }

    #[test]
    fn a_missing_home_does_not_panic() {
        assert_eq!(data_root_from(None, None), Path::new(".ailocal"));
    }

    #[test]
    fn setting_the_root_moves_the_paths_that_followed_it() {
        let mut c = Config::default();
        c.set_data_dir(PathBuf::from("/mnt/big/ailocal"));
        assert_eq!(c.models_dir, Path::new("/mnt/big/ailocal/models"));
        assert_eq!(c.hf_home, Path::new("/mnt/big/ailocal/hf"));
        assert_eq!(c.eval_dir, Path::new("/mnt/big/ailocal/eval"));
    }

    /// Someone who pinned the weights to a scratch volume meant it. Moving the root
    /// must not drag that along.
    #[test]
    fn setting_the_root_leaves_an_individually_chosen_path_alone() {
        let mut c: Config = toml::from_str(r#"models_dir = "/scratch/weights""#).unwrap();
        c.set_data_dir(PathBuf::from("/mnt/big/ailocal"));
        assert_eq!(c.models_dir, Path::new("/scratch/weights"));
        assert_eq!(c.hf_home, Path::new("/mnt/big/ailocal/hf"));
    }

    #[test]
    fn a_data_dir_alone_places_everything_under_it() {
        let c: Config = toml::from_str(r#"data_dir = "/mnt/kingston/ailocal""#).unwrap();
        assert_eq!(c.models_dir, Path::new("/mnt/kingston/ailocal/models"));
        assert_eq!(c.hf_home, Path::new("/mnt/kingston/ailocal/hf"));
        assert_eq!(c.eval_dir, Path::new("/mnt/kingston/ailocal/eval"));
    }

    #[test]
    fn an_individual_path_overrides_the_root() {
        let c: Config = toml::from_str(
            r#"
            data_dir = "/mnt/big/ailocal"
            models_dir = "/scratch/weights"
            "#,
        )
        .unwrap();
        assert_eq!(c.models_dir, Path::new("/scratch/weights"));
        assert_eq!(c.hf_home, Path::new("/mnt/big/ailocal/hf"));
    }

    /// A config written before `data_dir` existed names its paths explicitly and has no
    /// root. Upgrading must not relocate anything - including the paths that config
    /// never mentioned, which is the case that silently loses data rather than
    /// obviously breaking.
    #[test]
    fn a_config_from_before_the_data_root_stays_where_it_is() {
        // Exactly the shape this project's own config had: written before `eval_dir`
        // existed, so nothing in the file says where eval data should go.
        let older = r#"
            models_dir = "/mnt/kingston/ailocal/models"
            hf_home = "/mnt/kingston/ailocal/hf"
            cache_type = "q8_0"
            reasoning = "off"
        "#;
        let c: Config = toml::from_str(older).unwrap();
        assert_eq!(c.models_dir, Path::new("/mnt/kingston/ailocal/models"));
        assert_eq!(c.hf_home, Path::new("/mnt/kingston/ailocal/hf"));
        assert_eq!(
            c.eval_dir,
            Path::new("/mnt/kingston/ailocal/eval"),
            "an upgrade must not move data the old config did not mention"
        );
        assert_eq!(c.reasoning, "off");

        // And rewriting it collapses to the root it always implied.
        let text = toml::to_string_pretty(&c).unwrap();
        assert!(
            text.contains(r#"data_dir = "/mnt/kingston/ailocal""#),
            "got:\n{text}"
        );
        assert!(!text.contains("models_dir"), "got:\n{text}");
    }

    #[test]
    fn partial_config_keeps_defaults_for_the_rest() {
        let c: Config = toml::from_str(r#"models_dir = "/tmp/models""#).unwrap();
        assert_eq!(c.models_dir, Path::new("/tmp/models"));
        assert_eq!(c.cache_type, "q8_0");
        assert_eq!(c.reasoning, "auto");
        assert_eq!(c.reasoning_budget, -1);
    }

    /// `<dir>/models` says `<dir>` is the data root, so the rest follows it there
    /// rather than staying behind in the home directory.
    #[test]
    fn a_canonically_named_path_reveals_the_root() {
        let c: Config = toml::from_str(r#"models_dir = "/mnt/big/ailocal/models""#).unwrap();
        assert_eq!(c.data_dir, Path::new("/mnt/big/ailocal"));
        assert_eq!(c.hf_home, Path::new("/mnt/big/ailocal/hf"));
    }

    /// A path that is not named after what it holds is an override for that one thing,
    /// not a statement about where everything lives.
    #[test]
    fn a_bespoke_path_reveals_nothing_about_the_root() {
        let c: Config = toml::from_str(r#"models_dir = "/scratch/weights""#).unwrap();
        assert_eq!(c.data_dir, default_data_dir());
        assert_eq!(c.hf_home, default_data_dir().join("hf"));
    }

    #[test]
    fn paths_on_different_disks_yield_no_root() {
        let c: Config = toml::from_str(
            r#"
            models_dir = "/mnt/big/models"
            hf_home = "/mnt/other/hf"
            "#,
        )
        .unwrap();
        assert_eq!(c.data_dir, default_data_dir());
    }

    #[test]
    fn round_trips() {
        let c = Config::default();
        let back: Config = toml::from_str(&toml::to_string_pretty(&c).unwrap()).unwrap();
        assert_eq!(back.data_dir, c.data_dir);
        assert_eq!(back.models_dir, c.models_dir);
        assert_eq!(back.cache_type, c.cache_type);
    }

    /// A written config records the root and omits paths that merely restate it, so
    /// changing `data_dir` later actually moves things instead of being overridden by
    /// three absolute paths nobody chose.
    #[test]
    fn a_written_config_names_the_root_and_not_what_it_implies() {
        let text = toml::to_string_pretty(&Config::default()).unwrap();
        assert!(text.contains("data_dir"), "got:\n{text}");
        assert!(!text.contains("models_dir"), "got:\n{text}");
        assert!(!text.contains("hf_home"), "got:\n{text}");
        assert!(!text.contains("eval_dir"), "got:\n{text}");
    }

    #[test]
    fn a_written_config_keeps_a_path_that_was_chosen_separately() {
        let c = Config {
            models_dir: PathBuf::from("/scratch/weights"),
            ..Config::default()
        };
        let text = toml::to_string_pretty(&c).unwrap();
        assert!(text.contains("/scratch/weights"), "got:\n{text}");

        let back: Config = toml::from_str(&text).unwrap();
        assert_eq!(back.models_dir, Path::new("/scratch/weights"));
        assert_eq!(back.hf_home, c.hf_home);
    }

    /// A typo in a key should be reported, not silently ignored into a default.
    #[test]
    fn unknown_keys_are_rejected() {
        assert!(toml::from_str::<Config>(r#"model_dir = "/tmp""#).is_err());
    }

    /// The same strictness makes a config from a newer ailocal unreadable, which is
    /// how a 0.2.0 service ended up restart-looping on a `data_dir` it had never heard
    /// of. The error has to say that rather than just "unknown field".
    #[test]
    fn an_unrecognised_key_suggests_updating() {
        let hint = unknown_field_hint("data_dir = \"/x\"\nfuture_setting = 3\n")
            .expect("should have spotted the unknown key");
        assert!(hint.contains("future_setting"), "got: {hint}");
        assert!(hint.contains("ailocal update"), "got: {hint}");
        assert!(
            !hint.contains("data_dir"),
            "a known key is not the problem: {hint}"
        );
    }

    #[test]
    fn a_config_of_known_keys_produces_no_hint() {
        assert!(unknown_field_hint("data_dir = \"/x\"\ncache_type = \"q8_0\"\n").is_none());
        assert!(unknown_field_hint("# just a comment\n").is_none());
    }
}
