//! Writing our endpoint into a coding harness's own configuration.
//!
//! These files belong to other tools and hold live credentials, so every write here is
//! a merge rather than a rewrite, is backed up first, and is a no-op when the settings
//! already say what we want. `revert` puts the newest backup back.
//!
//! Pi's schema was read from its shipped bundle rather than guessed: the llama.cpp
//! provider stores `{"type":"api_key","key":...,"env":{"LLAMA_BASE_URL":...}}` under
//! the provider id, normalises the URL by stripping any trailing `/v1`, and then calls
//! `<base>/models` for the catalogue and `<base>/v1/chat/completions` for inference.

use std::path::{Path, PathBuf};

use anyhow::Context as _;

/// Pi's provider id for a llama.cpp-compatible server.
pub const PI_PROVIDER_ID: &str = "llama.cpp";

/// What a configure call did.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Settings already matched; nothing was written.
    AlreadyConfigured,
    Configured {
        backup: Option<PathBuf>,
    },
}

/// Path to Pi's credential store.
///
/// # Errors
/// If `HOME` is not set.
pub fn pi_auth_path() -> anyhow::Result<PathBuf> {
    pi_file("auth.json")
}

/// Path to Pi's cached model catalogue.
///
/// # Errors
/// If `HOME` is not set.
pub fn pi_models_store_path() -> anyhow::Result<PathBuf> {
    pi_file("models-store.json")
}

fn pi_file(name: &str) -> anyhow::Result<PathBuf> {
    let home = std::env::var_os("HOME").ok_or_else(|| anyhow::anyhow!("HOME is not set"))?;
    Ok(PathBuf::from(home).join(".pi/agent").join(name))
}

/// A model to advertise to Pi.
#[derive(Debug, Clone)]
pub struct PiModel {
    pub id: String,
    pub context_window: u64,
    pub reasoning: bool,
}

/// Why a particular model was chosen, so the choice is visible rather than mysterious.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chose {
    /// Named on the command line.
    Requested,
    /// `default_model` from the config.
    Configured,
    /// Whatever is loaded right now.
    Loaded,
    /// Nothing said anything, so the list order decided.
    FirstAvailable,
}

impl Chose {
    #[must_use]
    pub const fn why(self) -> &'static str {
        match self {
            Self::Requested => "as requested",
            Self::Configured => "the configured default_model",
            Self::Loaded => "currently loaded",
            Self::FirstAvailable => "first alphabetically, since nothing named one",
        }
    }
}

/// Pick the one model a single-model harness should be pointed at.
///
/// Claude Code takes one `ANTHROPIC_MODEL`, so something has to choose. Sorting the
/// installed models and taking the first was silently wrong: install `qwen3-14b`
/// alongside `gemma4-12b` and it keeps pinning gemma4 because `g` sorts before `q`,
/// ignoring both the model you just installed and the configured default.
///
/// The order is what a person would expect: what you asked for, then what you
/// configured, then what is actually running, and only then the arbitrary one.
///
/// # Errors
/// If `wanted` names a model that is not in `models` - a typo there must not silently
/// fall through to a different model.
pub fn preferred<'a>(
    models: &'a [PiModel],
    wanted: Option<&str>,
    configured: Option<&str>,
    loaded: Option<&str>,
) -> anyhow::Result<(&'a PiModel, Chose)> {
    let find = |name: &str| models.iter().find(|m| m.id == name);

    if let Some(name) = wanted {
        let found = find(name).ok_or_else(|| {
            anyhow::anyhow!(
                "no model named {name:?} can run here; available: {}",
                models
                    .iter()
                    .map(|m| m.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        return Ok((found, Chose::Requested));
    }

    if let Some(found) = configured.and_then(find) {
        return Ok((found, Chose::Configured));
    }
    if let Some(found) = loaded.and_then(find) {
        return Ok((found, Chose::Loaded));
    }

    models
        .first()
        .map(|m| (m, Chose::FirstAvailable))
        .ok_or_else(|| anyhow::anyhow!("no model can run here"))
}

/// Largest completion Pi should request.
///
/// Capped well below the context window: these are reasoning models, and an
/// over-generous cap mostly buys longer thinking rather than a longer answer.
const MAX_OUTPUT_TOKENS: u64 = 8192;

/// The catalogue entry Pi caches for a local model.
///
/// `api` must be `openai-completions` and `provider` must be the llama.cpp id, or Pi
/// filters the entry out when it refreshes.
#[must_use]
pub fn pi_model_entry(model: &PiModel, base_url: &str) -> serde_json::Value {
    serde_json::json!({
        "id": model.id,
        "name": format!("{} (local)", model.id),
        "api": "openai-completions",
        "provider": PI_PROVIDER_ID,
        "baseUrl": format!("{}/v1", normalize_base_url(base_url)),
        "reasoning": model.reasoning,
        "input": ["text"],
        // Local inference is free, and Pi renders these figures directly.
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
        "contextWindow": model.context_window,
        "maxTokens": model.context_window.min(MAX_OUTPUT_TOKENS),
    })
}

/// The credential Pi expects for a llama.cpp server.
#[must_use]
pub fn pi_credential(base_url: &str, api_key: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "api_key",
        "key": api_key,
        "env": { "LLAMA_BASE_URL": normalize_base_url(base_url) },
    })
}

/// Strip a trailing `/v1` and trailing slashes, as Pi does before storing.
///
/// Writing a URL Pi would normalise differently makes the settings look changed on
/// every run, so we store exactly what it would.
#[must_use]
pub fn normalize_base_url(url: &str) -> String {
    let trimmed = url.trim().trim_end_matches('/');
    trimmed.strip_suffix("/v1").unwrap_or(trimmed).to_owned()
}

/// Point Pi at our gateway.
///
/// Writes two files, because the credential alone is not enough: Pi builds its provider
/// registry from the cached model catalogue, so with no catalogue entry the provider
/// does not exist at all and `--provider llama.cpp` fails with "Unknown provider".
///
/// # Errors
/// If either file cannot be read, is not a JSON object, or cannot be written.
pub fn configure_pi(base_url: &str, api_key: &str, models: &[PiModel]) -> anyhow::Result<Outcome> {
    let credential = merge_json(
        &pi_auth_path()?,
        PI_PROVIDER_ID,
        pi_credential(base_url, api_key),
    )?;

    let catalogue = serde_json::json!({
        "models": models
            .iter()
            .map(|m| pi_model_entry(m, base_url))
            .collect::<Vec<_>>(),
        "checkedAt": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0),
    });
    let catalogue = merge_json(&pi_models_store_path()?, PI_PROVIDER_ID, catalogue)?;

    match (credential, catalogue) {
        (None, None) => Ok(Outcome::AlreadyConfigured),
        (a, b) => Ok(Outcome::Configured { backup: a.or(b) }),
    }
}

/// Set `key` to `value` in a JSON object file, backing it up if that changes anything.
///
/// Returns the backup path when a write happened, `None` when the file already said
/// what we wanted. Volatile bookkeeping fields are ignored in the comparison, or the
/// catalogue would look changed on every run and rewrite the file each time.
fn merge_json(path: &Path, key: &str, value: serde_json::Value) -> anyhow::Result<Option<PathBuf>> {
    anyhow::ensure!(
        path.exists(),
        "{} does not exist - run pi at least once first",
        path.display()
    );

    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut root: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    let object = root
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("{} is not a JSON object", path.display()))?;

    if object
        .get(key)
        .is_some_and(|existing| stable_part(existing) == stable_part(&value))
    {
        return Ok(None);
    }

    let backup = back_up(path)?;
    object.insert(key.to_owned(), value);
    write_atomically(path, &format!("{}\n", serde_json::to_string_pretty(&root)?))?;
    Ok(Some(backup))
}

/// A value with volatile bookkeeping fields removed, for comparison.
fn stable_part(value: &serde_json::Value) -> serde_json::Value {
    let mut copy = value.clone();
    if let Some(o) = copy.as_object_mut() {
        o.remove("checkedAt");
        o.remove("lastModified");
        o.remove("etag");
    }
    copy
}

/// Remove our entries from both of Pi's files, leaving everything else alone.
///
/// # Errors
/// If a file cannot be read or written.
pub fn unconfigure_pi() -> anyhow::Result<bool> {
    let mut removed = false;
    for path in [pi_auth_path()?, pi_models_store_path()?] {
        if !path.exists() {
            continue;
        }
        let raw = std::fs::read_to_string(&path)?;
        let mut root: serde_json::Value = serde_json::from_str(&raw)?;
        let Some(object) = root.as_object_mut() else {
            continue;
        };
        if object.remove(PI_PROVIDER_ID).is_none() {
            continue;
        }
        back_up(&path)?;
        write_atomically(
            &path,
            &format!("{}\n", serde_json::to_string_pretty(&root)?),
        )?;
        removed = true;
    }
    Ok(removed)
}

/// Render the environment Claude Code needs to talk to the gateway.
///
/// Deliberately a file to source rather than an edit to `~/.claude/settings.json`:
/// settings apply to *every* Claude Code session on the machine, so writing them there
/// would silently redirect work you wanted to run against the real Anthropic API. An
/// env file is opt-in per shell and reverts by closing it.
#[must_use]
pub fn claude_code_env(base_url: &str, api_key: &str, model: &str, context: u64) -> String {
    let base = normalize_base_url(base_url);
    format!(
        "# Source this to point Claude Code at the local model:\n\
         #   source {}\n\
         # Written by `ailocal harness configure claude-code`.\n\
         export ANTHROPIC_BASE_URL={base}\n\
         export ANTHROPIC_AUTH_TOKEN={api_key}\n\
         export ANTHROPIC_MODEL={model}\n\
         export ANTHROPIC_SMALL_FAST_MODEL={model}\n\
         # Claude Code does not know this model, so it would otherwise assume a 200k\n\
         # window and auto-compact far too early.\n\
         export CLAUDE_CODE_MAX_CONTEXT_TOKENS={context}\n",
        claude_code_env_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "<path>".into()),
    )
}

/// Where the Claude Code env file is written.
///
/// # Errors
/// If neither `XDG_CONFIG_HOME` nor `HOME` is set.
pub fn claude_code_env_path() -> anyhow::Result<PathBuf> {
    Ok(crate::config::Config::path()?.with_file_name("claude-code.env"))
}

/// Write the Claude Code env file.
///
/// # Errors
/// If the file cannot be written.
pub fn configure_claude_code(
    base_url: &str,
    api_key: &str,
    model: &str,
    context: u64,
) -> anyhow::Result<(PathBuf, Outcome)> {
    let path = claude_code_env_path()?;
    let desired = claude_code_env(base_url, api_key, model, context);

    if std::fs::read_to_string(&path).is_ok_and(|existing| existing == desired) {
        return Ok((path, Outcome::AlreadyConfigured));
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&path, &desired).with_context(|| format!("writing {}", path.display()))?;

    // Holds the gateway key.
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).ok();

    Ok((path, Outcome::Configured { backup: None }))
}

/// Copy a file next to itself with a timestamped suffix.
fn back_up(path: &Path) -> anyhow::Result<PathBuf> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let name = format!(
        "{}.ailocal-{stamp}.bak",
        path.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    );
    let backup = path.with_file_name(name);
    std::fs::copy(path, &backup)
        .with_context(|| format!("backing up {} to {}", path.display(), backup.display()))?;
    Ok(backup)
}

/// Write via a temporary file and rename, so an interrupted write cannot truncate
/// somebody else's credential store.
fn write_atomically(path: &Path, contents: &str) -> anyhow::Result<()> {
    let temp = path.with_extension("ailocal-tmp");
    std::fs::write(&temp, contents).with_context(|| format!("writing {}", temp.display()))?;

    // Preserve the original's permissions, since this file holds credentials.
    if let Ok(meta) = std::fs::metadata(path) {
        std::fs::set_permissions(&temp, meta.permissions()).ok();
    }
    std::fs::rename(&temp, path).with_context(|| format!("replacing {}", path.display()))
}

#[cfg(test)]
mod tests {
    /// The bug this ordering exists to fix: `gemma4` sorts before `qwen3`, so taking
    /// the first installed model pinned Claude Code to a model the user never chose.
    #[test]
    fn a_named_model_beats_alphabetical_order() {
        let models = catalogue_of(&["gemma4-12b", "qwen3-14b"]);
        let (chosen, why) = preferred(&models, Some("qwen3-14b"), None, None).unwrap();
        assert_eq!(chosen.id, "qwen3-14b");
        assert_eq!(why, Chose::Requested);
    }

    #[test]
    fn the_configured_default_beats_alphabetical_order() {
        let models = catalogue_of(&["gemma4-12b", "qwen3-14b"]);
        let (chosen, why) = preferred(&models, None, Some("qwen3-14b"), None).unwrap();
        assert_eq!(chosen.id, "qwen3-14b");
        assert_eq!(why, Chose::Configured);
    }

    #[test]
    fn what_is_loaded_beats_alphabetical_order() {
        let models = catalogue_of(&["gemma4-12b", "qwen3-14b"]);
        let (chosen, why) = preferred(&models, None, None, Some("qwen3-14b")).unwrap();
        assert_eq!(chosen.id, "qwen3-14b");
        assert_eq!(why, Chose::Loaded);
    }

    #[test]
    fn an_explicit_request_outranks_everything_else() {
        let models = catalogue_of(&["a", "b", "c"]);
        let (chosen, _) = preferred(&models, Some("c"), Some("b"), Some("a")).unwrap();
        assert_eq!(chosen.id, "c");
    }

    #[test]
    fn the_configured_default_outranks_what_is_loaded() {
        let models = catalogue_of(&["a", "b", "c"]);
        let (chosen, _) = preferred(&models, None, Some("b"), Some("a")).unwrap();
        assert_eq!(chosen.id, "b");
    }

    /// A default naming a model that cannot run here must not win, or the harness gets
    /// pointed at something that will fail to load.
    #[test]
    fn a_hint_naming_an_unavailable_model_is_skipped() {
        let models = catalogue_of(&["gemma4-12b"]);
        let (chosen, why) = preferred(&models, None, Some("deleted-model"), None).unwrap();
        assert_eq!(chosen.id, "gemma4-12b");
        assert_eq!(why, Chose::FirstAvailable);
    }

    /// A typo in `--model` has to stop, not quietly configure a different model.
    #[test]
    fn an_explicit_request_for_an_unavailable_model_is_an_error() {
        let models = catalogue_of(&["gemma4-12b", "qwen3-14b"]);
        let err = preferred(&models, Some("qwen3-14"), None, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("qwen3-14"), "got: {err}");
        assert!(
            err.contains("gemma4-12b, qwen3-14b"),
            "must list what is available: {err}"
        );
    }

    #[test]
    fn nothing_installed_is_an_error_rather_than_a_panic() {
        assert!(preferred(&[], None, None, None).is_err());
    }

    fn catalogue_of(names: &[&str]) -> Vec<PiModel> {
        names
            .iter()
            .map(|id| PiModel {
                id: (*id).to_owned(),
                context_window: 4096,
                reasoning: false,
            })
            .collect()
    }

    use super::*;

    #[test]
    fn base_urls_are_normalised_the_way_pi_stores_them() {
        for (input, want) in [
            ("http://127.0.0.1:8081", "http://127.0.0.1:8081"),
            ("http://127.0.0.1:8081/", "http://127.0.0.1:8081"),
            ("http://127.0.0.1:8081/v1", "http://127.0.0.1:8081"),
            ("http://127.0.0.1:8081/v1/", "http://127.0.0.1:8081"),
            ("  http://host:9/v1  ", "http://host:9"),
        ] {
            assert_eq!(normalize_base_url(input), want, "for {input:?}");
        }
    }

    /// The stored value has to be byte-identical to what Pi would write, or every run
    /// looks like a change and rewrites the file.
    #[test]
    fn credential_matches_pi_login_output() {
        let cred = pi_credential("http://127.0.0.1:8081/v1", "ail_secret");
        assert_eq!(cred["type"], "api_key");
        assert_eq!(cred["key"], "ail_secret");
        assert_eq!(cred["env"]["LLAMA_BASE_URL"], "http://127.0.0.1:8081");
    }

    #[test]
    fn the_same_settings_produce_an_identical_credential() {
        assert_eq!(
            pi_credential("http://h:1", "k"),
            pi_credential("http://h:1/v1/", "k"),
        );
    }

    #[test]
    fn claude_code_env_sets_what_claude_code_reads() {
        let env = claude_code_env("http://127.0.0.1:8081/v1", "ail_k", "gemma4", 262_144);
        // Claude Code appends /v1 itself, so the stored value must not already have it.
        assert!(env.contains("export ANTHROPIC_BASE_URL=http://127.0.0.1:8081\n"));
        assert!(env.contains("export ANTHROPIC_AUTH_TOKEN=ail_k"));
        assert!(env.contains("export ANTHROPIC_MODEL=gemma4"));
        assert!(env.contains("export CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144"));
    }

    #[test]
    fn pi_model_entries_carry_the_fields_pi_filters_on() {
        let entry = pi_model_entry(
            &PiModel {
                id: "m".into(),
                context_window: 1024,
                reasoning: false,
            },
            "http://h:1",
        );
        // Pi drops catalogue entries that do not match both of these exactly.
        assert_eq!(entry["api"], "openai-completions");
        assert_eq!(entry["provider"], PI_PROVIDER_ID);
        assert_eq!(entry["baseUrl"], "http://h:1/v1");
        assert_eq!(entry["contextWindow"], 1024);
        assert_eq!(entry["maxTokens"], 1024);
    }
}
