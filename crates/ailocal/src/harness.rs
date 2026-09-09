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
//!
//! Which is why capability goes in a third file. `models-store.json` is Pi's *cache* of
//! that `<base>/models` call, rebuilt from it on every refresh by a function that
//! hardcodes `reasoning: false` - so anything we write there about what a model can do
//! is erased the next time Pi opens `/model`. `models.json` is user configuration that
//! Pi composes *over* the provider on every read, so that is where a capability has to
//! be stated to survive.

use std::path::{Path, PathBuf};

use anyhow::Context as _;

use crate::gguf::Thinking;

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

/// Path to Pi's custom-model configuration.
///
/// # Errors
/// If `HOME` is not set.
pub fn pi_models_json_path() -> anyhow::Result<PathBuf> {
    pi_file("models.json")
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
    /// What thinking control the model's own chat template exposes.
    pub thinking: Thinking,
}

/// Every harness this knows how to configure.
pub const NAMES: [&str; 2] = ["pi", "claude-code"];

/// Whether `name` has already been pointed at our gateway.
///
/// Read-only, so that "what still needs doing" can be answered without writing
/// anything. Pi is configured when its credential store holds our provider; Claude
/// Code when the env file exists, since that file has no purpose other than this.
#[must_use]
pub fn is_configured(name: &str) -> bool {
    match name {
        "pi" => pi_auth_path().is_ok_and(|path| {
            std::fs::read_to_string(path)
                .ok()
                .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
                .is_some_and(|json| json.get(PI_PROVIDER_ID).is_some())
        }),
        "claude-code" => claude_code_env_path().is_ok_and(|path| path.is_file()),
        _ => false,
    }
}

/// The harnesses already pointed at us.
#[must_use]
pub fn configured() -> Vec<&'static str> {
    NAMES.into_iter().filter(|n| is_configured(n)).collect()
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
/// Has to leave room for the whole ladder to differ. Pi derives a thinking budget from
/// the level - 1024, 2048, 8192, 16384 - then clamps it to `maxTokens` less 1024 kept
/// back for the answer. Cap output at 8192 and `medium` and `high` both clamp to 7168,
/// so two levels the user can select become one thing on the wire. 32768 keeps all
/// four distinct, and is a ceiling rather than a demand.
const MAX_OUTPUT_TOKENS: u64 = 32768;

/// The catalogue entry Pi caches for a local model.
///
/// `api` must be `openai-completions` and `provider` must be the llama.cpp id, or Pi
/// filters the entry out when it refreshes.
///
/// Pi overwrites all of this from `<base>/models` at its next refresh, so it seeds the
/// cache rather than settling anything. What must outlive a refresh goes in
/// [`pi_model_override`].
#[must_use]
pub fn pi_model_entry(model: &PiModel, base_url: &str) -> serde_json::Value {
    serde_json::json!({
        "id": model.id,
        "name": format!("{} (local)", model.id),
        "api": "openai-completions",
        "provider": PI_PROVIDER_ID,
        "baseUrl": format!("{}/v1", normalize_base_url(base_url)),
        "reasoning": model.thinking.is_available(),
        "input": ["text"],
        // Local inference is free, and Pi renders these figures directly.
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
        "contextWindow": model.context_window,
        "maxTokens": model.context_window.min(MAX_OUTPUT_TOKENS),
    })
}

/// The per-request field llama.cpp actually reads to cap thinking.
///
/// Measured, because the obvious name is wrong: `thinking_budget_tokens` is accepted
/// and silently ignored on `/v1/chat/completions` (it is only the internal name the
/// `/v1/messages` converter uses), while `reasoning_budget_tokens` type-checks as a
/// number and demonstrably shortens the thinking - 554 characters of
/// `reasoning_content` unbudgeted against 88 at a budget of 24.
const THINKING_BUDGET_FIELD: &str = "reasoning_budget_tokens";

/// What Pi should believe about this model, in the file a refresh cannot touch.
///
/// `thinkingLevelMap` decides which levels the harness offers, and its values are what
/// get sent. Three shapes, one per capability:
///
/// - no thinking: `reasoning: false`, and Pi offers `off` alone. Nothing here can
///   change that, and pretending otherwise just moves the failure to `/effort`.
/// - a switch: `off` has to be spelled out as `"none"`, because with the key absent Pi
///   sends *no* field when thinking is off - indistinguishable from a harness that
///   never mentioned it, which leaves llama-server on its launch default. The levels
///   between are left at Pi's defaults and differ for real, since each carries a
///   budget. `xhigh` and `max` are refused: Pi folds both onto `high` before looking
///   the budget up, so offering them would be two controls that do nothing.
/// - an effort dial: every level maps to itself and llama.cpp hands it to the template.
#[must_use]
pub fn pi_model_override(model: &PiModel) -> serde_json::Value {
    let mut entry = serde_json::json!({
        "reasoning": model.thinking.is_available(),
        "contextWindow": model.context_window,
        "maxTokens": model.context_window.min(MAX_OUTPUT_TOKENS),
    });

    let levels = match model.thinking {
        Thinking::None => return entry,
        Thinking::Toggle => serde_json::json!({
            "off": "none",
            "xhigh": serde_json::Value::Null,
            "max": serde_json::Value::Null,
        }),
        Thinking::Effort => serde_json::json!({
            "off": "none",
            "minimal": "minimal",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh",
            "max": "max",
        }),
    };

    entry["thinkingLevelMap"] = levels;
    entry["compat"] = serde_json::json!({
        "supportsReasoningEffort": true,
        "thinkingTokenBudgetField": THINKING_BUDGET_FIELD,
    });
    entry
}

/// The whole `modelOverrides` block for a set of models.
#[must_use]
pub fn pi_model_overrides(models: &[PiModel]) -> serde_json::Value {
    serde_json::Value::Object(
        models
            .iter()
            .map(|m| (m.id.clone(), pi_model_override(m)))
            .collect(),
    )
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
/// Writes three files, because no one of them is enough on its own. The credential
/// alone leaves the provider unregistered, since Pi builds its provider registry from
/// the cached model catalogue and `--provider llama.cpp` fails with "Unknown provider"
/// without an entry there. The catalogue alone loses every capability at Pi's next
/// refresh, which rebuilds it from `<base>/models` with `reasoning` hardcoded off.
/// `models.json` is the only one of the three Pi composes rather than replaces.
///
/// # Errors
/// If a file cannot be read, is not a JSON object, or cannot be written.
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

    let overrides = set_pi_overrides(pi_model_overrides(models))?;

    match (credential, catalogue, overrides) {
        (None, None, Outcome::AlreadyConfigured) => Ok(Outcome::AlreadyConfigured),
        (a, b, c) => Ok(Outcome::Configured {
            backup: a.or(b).or(match c {
                Outcome::Configured { backup } => backup,
                Outcome::AlreadyConfigured => None,
            }),
        }),
    }
}

/// Replace our `modelOverrides` block in Pi's `models.json`.
///
/// The block is rewritten whole rather than merged key by key, so a model that is no
/// longer installed stops being described. Everything outside it - other providers,
/// other keys on this one, a hand-written `apiKey` - is left exactly as found.
///
/// Unlike Pi's other two files this one need not already exist: it is optional
/// configuration rather than state Pi maintains, so an absent file is created.
fn set_pi_overrides(wanted: serde_json::Value) -> anyhow::Result<Outcome> {
    let path = pi_models_json_path()?;
    let mut root = read_json_object(&path)?;

    if pi_overrides_in(&root) == Some(&wanted) {
        return Ok(Outcome::AlreadyConfigured);
    }

    // No backup when we are creating the file - there is nothing yet to lose, and a
    // zero-byte `.bak` reads as if there were.
    let backup = path.exists().then(|| back_up(&path)).transpose()?;
    object_at(&mut root, &path, &["providers", PI_PROVIDER_ID])?
        .insert("modelOverrides".to_owned(), wanted);
    write_json(&path, &root)?;
    Ok(Outcome::Configured { backup })
}

/// Stop describing one model, leaving the rest of the block alone.
///
/// Returns whether anything was there to remove. For `ailocal model rm`: an override
/// naming weights that no longer exist has Pi advertising a model it cannot load.
///
/// # Errors
/// If `models.json` cannot be read, is not a JSON object, or cannot be written.
pub fn forget_pi_model(id: &str) -> anyhow::Result<bool> {
    let path = pi_models_json_path()?;
    if !path.exists() {
        return Ok(false);
    }

    let mut root = read_json_object(&path)?;
    let Some(overrides) = pi_overrides_in(&root).and_then(serde_json::Value::as_object) else {
        return Ok(false);
    };
    if !overrides.contains_key(id) {
        return Ok(false);
    }

    let mut kept = overrides.clone();
    kept.remove(id);
    back_up(&path)?;
    object_at(&mut root, &path, &["providers", PI_PROVIDER_ID])?
        .insert("modelOverrides".to_owned(), serde_json::Value::Object(kept));
    write_json(&path, &root)?;
    Ok(true)
}

/// Our overrides block, if Pi's config has one.
fn pi_overrides_in(root: &serde_json::Value) -> Option<&serde_json::Value> {
    root.get("providers")?
        .get(PI_PROVIDER_ID)?
        .get("modelOverrides")
}

/// Read a JSON object file, treating absent or empty as `{}`.
fn read_json_object(path: &Path) -> anyhow::Result<serde_json::Value> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    if raw.trim().is_empty() {
        return Ok(serde_json::json!({}));
    }

    let root: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    anyhow::ensure!(root.is_object(), "{} is not a JSON object", path.display());
    Ok(root)
}

/// Walk to a nested object, creating the levels that are missing.
///
/// Errors rather than panics when a level exists but holds something other than an
/// object: this is somebody else's hand-edited config, and `value["k"] = v` on a
/// string would take the process down with it.
fn object_at<'a>(
    root: &'a mut serde_json::Value,
    path: &Path,
    keys: &[&str],
) -> anyhow::Result<&'a mut serde_json::Map<String, serde_json::Value>> {
    let mut here = root;
    for key in keys {
        here = here
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("{} is not a JSON object", path.display()))?
            .entry((*key).to_owned())
            .or_insert_with(|| serde_json::json!({}));
    }
    here.as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("{:?} in {} is not an object", keys, path.display()))
}

/// Write a JSON document the way Pi writes its own: pretty, newline-terminated.
fn write_json(path: &Path, root: &serde_json::Value) -> anyhow::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    write_atomically(path, &format!("{}\n", serde_json::to_string_pretty(root)?))
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

/// Remove our entries from all three of Pi's files, leaving everything else alone.
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
    Ok(unconfigure_pi_models_json()? || removed)
}

/// Take our capability statement back out of Pi's `models.json`.
///
/// Only ever removes what we put there. The provider key goes too once nothing else is
/// on it, and the file goes once no provider is left - a `models.json` holding an empty
/// `providers` is a file we created and then emptied, and leaving it behind would have
/// `ailocal harness unconfigure pi` still visible in Pi's config directory.
fn unconfigure_pi_models_json() -> anyhow::Result<bool> {
    let path = pi_models_json_path()?;
    if !path.exists() {
        return Ok(false);
    }

    let mut root = read_json_object(&path)?;
    if pi_overrides_in(&root).is_none() {
        return Ok(false);
    }

    back_up(&path)?;
    let providers = object_at(&mut root, &path, &["providers"])?;
    if let Some(provider) = providers
        .get_mut(PI_PROVIDER_ID)
        .and_then(serde_json::Value::as_object_mut)
    {
        provider.remove("modelOverrides");
        if provider.is_empty() {
            providers.remove(PI_PROVIDER_ID);
        }
    }

    // Nothing of anyone else's left: no other provider, and no other top-level key.
    let ours_alone = root["providers"]
        .as_object()
        .is_some_and(serde_json::Map::is_empty)
        && root.as_object().is_some_and(|o| o.len() == 1);

    if ours_alone {
        std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
    } else {
        write_json(&path, &root)?;
    }
    Ok(true)
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
        names.iter().map(|id| model(id, Thinking::None)).collect()
    }

    fn model(id: &str, thinking: Thinking) -> PiModel {
        PiModel {
            id: id.to_owned(),
            context_window: 4096,
            thinking,
        }
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
        let mut m = model("m", Thinking::None);
        m.context_window = 1024;
        let entry = pi_model_entry(&m, "http://h:1");
        // Pi drops catalogue entries that do not match both of these exactly.
        assert_eq!(entry["api"], "openai-completions");
        assert_eq!(entry["provider"], PI_PROVIDER_ID);
        assert_eq!(entry["baseUrl"], "http://h:1/v1");
        assert_eq!(entry["contextWindow"], 1024);
        assert_eq!(entry["maxTokens"], 1024);
    }

    /// The bug the override exists to fix: a model that can think was advertised as
    /// unable to, so `/thinking` in pi offered `off` and nothing else.
    #[test]
    fn a_model_that_can_think_is_advertised_as_able_to() {
        let entry = pi_model_override(&model("gemma4-12b", Thinking::Toggle));
        assert_eq!(entry["reasoning"], true);
        assert_eq!(entry["compat"]["supportsReasoningEffort"], true);
    }

    /// Without a mapping for `off`, pi omits the field entirely when thinking is off -
    /// which the gateway cannot tell from a harness that never mentioned it, leaving
    /// llama-server on whatever it was launched with.
    #[test]
    fn turning_thinking_off_is_something_pi_can_say() {
        for thinking in [Thinking::Toggle, Thinking::Effort] {
            let entry = pi_model_override(&model("m", thinking));
            assert_eq!(entry["thinkingLevelMap"]["off"], "none", "{thinking:?}");
        }
    }

    /// pi folds `xhigh` and `max` onto `high` before choosing a budget, so a switch has
    /// nothing to put behind them. Offering them anyway is three levels that do the
    /// same thing.
    #[test]
    fn a_switch_does_not_pretend_to_have_extended_levels() {
        let entry = pi_model_override(&model("m", Thinking::Toggle));
        assert!(entry["thinkingLevelMap"]["xhigh"].is_null());
        assert!(entry["thinkingLevelMap"]["max"].is_null());
        // The rungs in between are left unmapped on purpose: pi's own defaults apply,
        // and each carries a distinct budget.
        assert!(entry["thinkingLevelMap"].get("medium").is_none());
    }

    /// A template that reads a level gets every level, because llama.cpp hands it
    /// straight through.
    #[test]
    fn an_effort_dial_gets_the_whole_ladder() {
        let entry = pi_model_override(&model("m", Thinking::Effort));
        for level in ["minimal", "low", "medium", "high", "xhigh", "max"] {
            assert_eq!(entry["thinkingLevelMap"][level], level);
        }
    }

    /// A model with no thinking mode must not be given controls for one - that is how
    /// `/effort` ends up refusing a level the user was offered.
    #[test]
    fn a_model_without_thinking_is_given_no_thinking_controls() {
        let entry = pi_model_override(&model("qwen3-coder-30b", Thinking::None));
        assert_eq!(entry["reasoning"], false);
        assert!(entry.get("thinkingLevelMap").is_none());
        assert!(entry.get("compat").is_none());
    }

    /// The budget field is the only thing that makes one level differ from another, and
    /// the plausible spelling is the wrong one - `thinking_budget_tokens` is accepted
    /// and ignored by llama-server.
    #[test]
    fn the_budget_field_is_the_one_llama_cpp_reads() {
        let entry = pi_model_override(&model("m", Thinking::Toggle));
        assert_eq!(
            entry["compat"]["thinkingTokenBudgetField"],
            "reasoning_budget_tokens"
        );
    }

    /// Cap the output too low and pi clamps `medium` and `high` to the same budget,
    /// collapsing two levels the user can select into one thing on the wire. pi's
    /// ladder and the 1024 tokens it keeps back for the answer are both measured.
    #[test]
    fn the_output_cap_leaves_the_levels_distinguishable() {
        let room = MAX_OUTPUT_TOKENS.saturating_sub(1024);
        let clamped: Vec<u64> = [1024_u64, 2048, 8192, 16384]
            .iter()
            .map(|budget| (*budget).min(room))
            .collect();

        let mut distinct = clamped.clone();
        distinct.dedup();
        assert_eq!(
            clamped, distinct,
            "two thinking levels clamp to one budget: {clamped:?}"
        );
    }

    #[test]
    fn overrides_are_keyed_by_model_id() {
        let models = [
            model("gemma4-12b", Thinking::Toggle),
            model("qwen3-coder-30b", Thinking::None),
        ];
        let block = pi_model_overrides(&models);
        assert_eq!(block["gemma4-12b"]["reasoning"], true);
        assert_eq!(block["qwen3-coder-30b"]["reasoning"], false);
        assert_eq!(block.as_object().unwrap().len(), 2);
    }

    /// Pi's config is the user's, and only the one key inside it is ours. Everything
    /// else - another provider, a hand-written key on this one - has to survive.
    #[test]
    fn writing_overrides_leaves_the_rest_of_pi_s_config_alone() {
        let mut root = serde_json::json!({
            "providers": {
                "ollama": { "baseUrl": "http://localhost:11434/v1" },
                PI_PROVIDER_ID: { "headers": { "x-mine": "1" } },
            },
        });
        let path = Path::new("models.json");

        object_at(&mut root, path, &["providers", PI_PROVIDER_ID])
            .unwrap()
            .insert(
                "modelOverrides".to_owned(),
                pi_model_overrides(&[model("m", Thinking::Toggle)]),
            );

        assert_eq!(
            root["providers"]["ollama"]["baseUrl"],
            "http://localhost:11434/v1"
        );
        assert_eq!(root["providers"][PI_PROVIDER_ID]["headers"]["x-mine"], "1");
        assert_eq!(
            root["providers"][PI_PROVIDER_ID]["modelOverrides"]["m"]["reasoning"],
            true
        );
    }

    /// Somebody else's hand-edited config must produce an error naming the file, not a
    /// panic from indexing a string with a key.
    #[test]
    fn a_provider_block_that_is_not_an_object_is_an_error() {
        let mut root = serde_json::json!({ "providers": "oops" });
        let err = object_at(&mut root, Path::new("models.json"), &["providers"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("models.json"), "got: {err}");
    }

    #[test]
    fn missing_levels_of_the_path_are_created() {
        let mut root = serde_json::json!({});
        object_at(&mut root, Path::new("models.json"), &["providers", "p"])
            .unwrap()
            .insert("modelOverrides".to_owned(), serde_json::json!({}));
        assert!(root["providers"]["p"]["modelOverrides"].is_object());
    }
}
