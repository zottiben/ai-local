//! Bringing the machine up: the services, the gateway and the harnesses.
//!
//! The same split as [`crate::models`]: a step that only looks and a step that acts, so
//! the command line can print its plan and ask, and the window can show the same plan
//! in a panel, before anything is written. Each act returns what it changed rather than
//! printing it, because the two callers say it differently.

use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::models::UnitBinary;
use crate::{Refused, auth, gateway, harness, registry, serve, service, status, wire};

/// How long to wait for the gateway to answer after its unit is started.
///
/// "Started" only means the job forked. Reporting success without this is reporting a
/// hope; 20 s is far longer than a gateway that is going to come up at all needs.
const ANSWER_GRACE: std::time::Duration = std::time::Duration::from_secs(20);

// --- services ---

/// One unit, as `service status` lists it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Unit {
    pub name: String,
    pub active: bool,
    pub enabled: bool,
}

/// What the manager says about our units.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Services {
    /// Without lingering the units only start after a login, so a headless reboot
    /// leaves the gateway down (rule 5).
    pub linger: bool,
    pub units: Vec<Unit>,
}

#[must_use]
pub fn services() -> Services {
    Services {
        linger: service::linger_enabled(),
        units: service::service_names()
            .into_iter()
            .map(|name| Unit {
                name: name.to_owned(),
                active: service::is_active(name),
                enabled: service::is_enabled(name),
            })
            .collect(),
    }
}

/// What installing the services did.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Installed {
    pub units: Vec<String>,
    pub unit_dir: PathBuf,
    /// The binary baked into `ExecStart`, which is never the console's own (rule 5).
    pub exe: PathBuf,
    /// Set because nothing named a default and this model was the first that runs here.
    pub chose_default: Option<String>,
    /// No installed model can run here, so only the gateway was written and nothing
    /// will be resident when the first request arrives.
    pub gateway_only: bool,
    /// Whether the gateway answered within the grace. `None` when not started.
    pub answering: Option<bool>,
    pub linger: bool,
}

/// Write the units for the current config, and start them unless asked not to.
///
/// Names a default model when there is not one, because the model unit is only written
/// when a model is named: without it the gateway comes up alone, nothing is resident,
/// and every first prompt pays a cold load on top of the harness's system prompt. That
/// is the difference between a few seconds and a few minutes, and it used to happen in
/// silence.
///
/// # Errors
/// If the config cannot be read or saved, or the units cannot be written or reloaded.
pub fn install_services(exe: &Path, start: bool) -> anyhow::Result<Installed> {
    let mut cfg = Config::load()?;

    let mut chose_default = None;
    let mut gateway_only = false;
    if cfg.default_model.is_none() {
        match status::runnable_models(&cfg)?.first() {
            Some(first) => {
                cfg.default_model = Some(first.id.clone());
                cfg.save()?;
                chose_default = Some(first.id.clone());
            }
            None => gateway_only = true,
        }
    }

    let units = service::install(
        exe,
        &cfg.gateway_host,
        cfg.gateway_port,
        cfg.default_model.as_deref(),
    )?;
    service::reload()?;

    let mut answering = None;
    if start {
        for unit in &units {
            service::enable(unit)?;
        }
        answering = Some(gateway::wait_until_answering(
            &cfg.gateway_host,
            cfg.gateway_port,
            ANSWER_GRACE,
        ));
    }

    Ok(Installed {
        units,
        unit_dir: service::unit_dir()?,
        exe: exe.to_path_buf(),
        chose_default,
        gateway_only,
        answering,
        linger: service::linger_enabled(),
    })
}

/// Stop, disable and delete both units. Returns the ones that were there.
///
/// # Errors
/// If the units cannot be deleted or the manager cannot be reloaded.
pub fn remove_services() -> anyhow::Result<Vec<String>> {
    for unit in service::service_names() {
        service::disable(unit).ok();
    }
    let removed = service::uninstall()?;
    service::reload()?;
    Ok(removed)
}

// --- the gateway ---

/// Why nothing is answering as a gateway, in the terms that name the cause.
///
/// A harness stores the URL and only finds out it is wrong when a prompt fails with a
/// bare "Connection error", which says nothing about the port. Worth answering here,
/// while there is still context.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Trouble {
    /// Nothing holds the port and no service is running: it was never started.
    NotRunning,
    /// Something that is not ailocal holds the port, so the gateway cannot start.
    PortTaken { free: u16 },
    /// The service is up but not on this port, which is what happens when
    /// `gateway_port` is edited by hand without reinstalling the service.
    WrongPort { expected: u16 },
}

/// `None` when the gateway answers.
#[must_use]
pub fn trouble(cfg: &Config) -> Option<Trouble> {
    if gateway::answers_as_gateway(&cfg.gateway_host, cfg.gateway_port) {
        return None;
    }
    Some(
        match gateway::port_status(&cfg.gateway_host, cfg.gateway_port) {
            gateway::PortStatus::Taken => Trouble::PortTaken {
                free: gateway::usable_port(&cfg.gateway_host, cfg.gateway_port)
                    .unwrap_or(cfg.gateway_port + 1),
            },
            _ if service::is_active(service::GATEWAY) => Trouble::WrongPort {
                expected: cfg.gateway_port,
            },
            _ => Trouble::NotRunning,
        },
    )
}

// --- harnesses ---

/// What pointing a harness at the gateway would do, before it is done.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HarnessPlan {
    pub harness: String,
    pub url: String,
    /// The model it would be pointed at, and why that one.
    pub model: String,
    pub context: u64,
    pub why: String,
    /// The other runnable models, which Claude Code would not get.
    pub others: Vec<String>,
    /// Models whose chat template offers no thinking mode, so `/effort` can only ever
    /// answer `off` for them (rule 7b).
    pub without_thinking: Vec<String>,
    /// What is wrong with the gateway it would be pointed at, if anything.
    pub trouble: Option<Trouble>,
    pub already_configured: bool,
}

/// Work out which model a harness would get, and what is wrong with the gateway.
///
/// # Errors
/// If the harness is unknown, no installed model can run here, or `wanted` names one
/// that cannot - a typo there must not fall through to a different model.
pub fn plan_harness(
    cfg: &Config,
    name: &str,
    url: &str,
    wanted: Option<&str>,
) -> anyhow::Result<HarnessPlan> {
    anyhow::ensure!(
        harness::NAMES.contains(&name),
        Refused::Missing(format!(
            "unknown harness {name:?}; expected `pi` or `claude-code`"
        ))
    );

    let models = status::runnable_models(cfg)?;
    anyhow::ensure!(
        !models.is_empty(),
        Refused::Invalid(
            "no installed model can run here; `ailocal model ls` shows why".to_owned()
        )
    );

    let loaded = serve::running()?.map(|i| i.model);
    let (preferred, why) = harness::preferred(
        &models,
        wanted,
        cfg.default_model.as_deref(),
        loaded.as_deref(),
    )?;

    Ok(HarnessPlan {
        harness: name.to_owned(),
        url: url.to_owned(),
        model: preferred.id.clone(),
        context: preferred.context_window,
        why: why.why().to_owned(),
        others: models
            .iter()
            .filter(|m| m.id != preferred.id)
            .map(|m| m.id.clone())
            .collect(),
        without_thinking: models
            .iter()
            .filter(|m| !m.thinking.is_available())
            .map(|m| m.id.clone())
            .collect(),
        trouble: trouble(cfg),
        already_configured: harness::is_configured(name),
    })
}

/// What a harness configure wrote.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HarnessConfigured {
    #[serde(flatten)]
    pub plan: HarnessPlan,
    /// Nothing was written: it already said exactly this.
    pub unchanged: bool,
    /// The file that now points at us. Claude Code's has to be sourced by hand.
    pub path: Option<PathBuf>,
    pub backup: Option<PathBuf>,
}

/// Point a harness at the gateway.
///
/// Written even when the gateway is not answering: that is the right thing when it is
/// merely not started yet, and `setup` configures harnesses as part of a sequence that
/// brings everything up. The trouble travels back in the plan instead.
///
/// # Errors
/// If the plan cannot be made, or the harness's files cannot be read or written.
pub fn configure_harness(
    cfg: &Config,
    name: &str,
    url: &str,
    wanted: Option<&str>,
) -> anyhow::Result<HarnessConfigured> {
    let plan = plan_harness(cfg, name, url, wanted)?;
    let key = auth::load_or_create()?;

    if name == "claude-code" {
        let (path, outcome) = harness::configure_claude_code(url, &key, &plan.model, plan.context)?;
        return Ok(HarnessConfigured {
            unchanged: matches!(outcome, harness::Outcome::AlreadyConfigured),
            path: Some(path),
            backup: None,
            plan,
        });
    }

    // Pi keeps the whole catalogue and chooses per session, so the preference only
    // decides the order it sees them in.
    let mut models = status::runnable_models(cfg)?;
    if let Some(at) = models.iter().position(|m| m.id == plan.model) {
        let chosen = models.remove(at);
        models.insert(0, chosen);
    }

    let outcome = harness::configure_pi(url, &key, &models)?;
    Ok(HarnessConfigured {
        unchanged: matches!(outcome, harness::Outcome::AlreadyConfigured),
        path: Some(harness::pi_models_json_path()?),
        backup: match outcome {
            harness::Outcome::Configured { backup } => backup,
            harness::Outcome::AlreadyConfigured => None,
        },
        plan,
    })
}

/// Stop a harness pointing at us. `false` when it was not configured.
///
/// # Errors
/// If the harness is unknown, or its files cannot be read or written.
pub fn unconfigure_harness(name: &str) -> anyhow::Result<bool> {
    match name {
        "claude-code" => {
            let path = harness::claude_code_env_path()?;
            if !path.exists() {
                return Ok(false);
            }
            std::fs::remove_file(&path)?;
            Ok(true)
        }
        "pi" => harness::unconfigure_pi(),
        _ => Err(Refused::Missing(format!(
            "unknown harness {name:?}; expected `pi` or `claude-code`"
        ))
        .into()),
    }
}

// --- settings ---

/// A change to one setting, with only the fields the caller means to touch.
///
/// Every field is optional and absent means "leave it", so the window can send one
/// field from one input without having to echo the whole config back - which is how a
/// setting someone else changed in a terminal gets quietly reverted.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub cache_type: Option<String>,
    pub reasoning: Option<String>,
    pub reasoning_budget: Option<i64>,
    pub max_context: Option<u64>,
    pub gateway_port: Option<u16>,
    /// `Some(None)` clears it; absent leaves it.
    #[serde(default, deserialize_with = "crate::wire::double_option")]
    pub system_prompt: Option<Option<String>>,
}

/// What changing the settings means for what is already running.
#[derive(Debug, Default, serde::Serialize)]
pub struct Applied {
    /// The settings that are different now, by name.
    pub changed: Vec<String>,
    pub path: Option<PathBuf>,
    /// The gateway unit bakes the port into its `ExecStart`, so the config alone would
    /// leave the service on the old one.
    #[serde(serialize_with = "wire::optional_result")]
    pub gateway_restarted: Option<anyhow::Result<()>>,
    /// A resident model keeps the context it was launched with, so the catalogue would
    /// advertise one number while the server serves another.
    pub reload_model: Option<String>,
    /// Harnesses hold the URL in their own config and do not follow a port change.
    pub repoint_harnesses: bool,
}

/// Validate and write a settings change, saying what it means for what is running.
///
/// # Errors
/// If a value is one the CLI would also refuse - a port something else holds, or a
/// context too small to be worth loading - or the config cannot be saved.
pub fn apply_settings(settings: &Settings, unit_binary: UnitBinary) -> anyhow::Result<Applied> {
    let mut cfg = Config::load()?;
    let mut out = Applied::default();

    if let Some(cache) = &settings.cache_type {
        // Parsed rather than trusted: an unknown cache type is only noticed at launch,
        // where it reads as the model failing to load.
        cache
            .parse::<crate::vram::CacheType>()
            .map_err(|e| Refused::Invalid(format!("{e:#}")))?;
        if *cache != cfg.cache_type {
            cfg.cache_type = cache.clone();
            out.changed.push("cache_type".to_owned());
        }
    }

    if let Some(reasoning) = &settings.reasoning {
        anyhow::ensure!(
            matches!(reasoning.as_str(), "auto" | "on" | "off"),
            Refused::Invalid(format!(
                "reasoning must be `auto`, `on` or `off`, not {reasoning:?}"
            ))
        );
        if *reasoning != cfg.reasoning {
            cfg.reasoning = reasoning.clone();
            out.changed.push("reasoning".to_owned());
        }
    }

    if let Some(budget) = settings
        .reasoning_budget
        .filter(|b| *b != cfg.reasoning_budget)
    {
        cfg.reasoning_budget = budget;
        out.changed.push("reasoning_budget".to_owned());
    }

    if let Some(tokens) = settings.max_context {
        anyhow::ensure!(
            tokens >= registry::MIN_USEFUL_CONTEXT,
            Refused::Invalid(format!(
                "{tokens} tokens is below the {} a model needs to be worth loading",
                registry::MIN_USEFUL_CONTEXT
            ))
        );
        if tokens != cfg.max_context {
            cfg.max_context = tokens;
            out.changed.push("max_context".to_owned());
            out.reload_model = serve::running()?
                .filter(|i| i.context != tokens)
                .map(|i| i.model);
        }
    }

    if let Some(prompt) = &settings.system_prompt {
        // A prompt of only whitespace is someone clearing the box, not a house rule
        // made of spaces, and it would be prepended to every request either way.
        let prompt = prompt.as_ref().filter(|p| !p.trim().is_empty()).cloned();
        if prompt != cfg.system_prompt {
            cfg.system_prompt = prompt;
            out.changed.push("system_prompt".to_owned());
        }
    }

    let mut port_changed = false;
    if let Some(port) = settings.gateway_port.filter(|p| *p != cfg.gateway_port) {
        anyhow::ensure!(
            gateway::port_status(&cfg.gateway_host, port) != gateway::PortStatus::Taken,
            Refused::Invalid(format!(
                "port {port} is in use by something else; pick another"
            ))
        );
        cfg.gateway_port = port;
        port_changed = true;
        out.changed.push("gateway_port".to_owned());
    }

    if out.changed.is_empty() {
        return Ok(out);
    }
    out.path = Some(cfg.save()?);

    if port_changed {
        out.repoint_harnesses = true;
        // The unit bakes the port into its ExecStart, so the config alone would leave
        // a running service on the old one.
        let installed =
            service::is_enabled(service::GATEWAY) || service::is_active(service::GATEWAY);
        if installed {
            out.gateway_restarted = Some(restart_gateway(&cfg, unit_binary));
        }
    }
    Ok(out)
}

/// Rewrite the gateway unit for the current config and restart it.
fn restart_gateway(cfg: &Config, unit_binary: UnitBinary) -> anyhow::Result<()> {
    service::install(
        &unit_binary()?,
        &cfg.gateway_host,
        cfg.gateway_port,
        cfg.default_model.as_deref(),
    )?;
    service::reload()?;
    if service::is_active(service::GATEWAY) {
        service::stop(service::GATEWAY)?;
    }
    service::start(service::GATEWAY)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_settings_patch_leaves_out_what_it_does_not_mention() {
        // The window sends one field from one input. Every other setting has to survive
        // that, including one a terminal changed a second ago.
        let one: Settings = serde_json::from_str(r#"{"max_context": 65536}"#).unwrap();
        assert_eq!(one.max_context, Some(65_536));
        assert!(one.cache_type.is_none());
        assert!(one.system_prompt.is_none());

        // Absent and null mean different things for the prompt: leave it, and clear it.
        let cleared: Settings = serde_json::from_str(r#"{"system_prompt": null}"#).unwrap();
        assert_eq!(cleared.system_prompt, Some(None));
    }

    #[test]
    fn an_unknown_setting_is_refused_rather_than_ignored() {
        // A typo that writes nothing and reports success is worse than an error: the
        // setting stays as it was and the window says it changed.
        let err = serde_json::from_str::<Settings>(r#"{"max_contex": 65536}"#).unwrap_err();
        assert!(err.to_string().contains("unknown field"), "{err}");
    }

    #[test]
    fn trouble_names_the_cause_rather_than_the_symptom() {
        // Each variant carries what the window needs to offer the fix, not just a
        // message: the free port to move to, and the port the unit should have had.
        assert_eq!(
            Trouble::PortTaken { free: 8082 },
            Trouble::PortTaken { free: 8082 }
        );
        let taken = serde_json::to_value(Trouble::PortTaken { free: 8082 }).unwrap();
        assert_eq!(taken, serde_json::json!({ "port_taken": { "free": 8082 } }));
        assert_eq!(
            serde_json::to_value(Trouble::NotRunning).unwrap(),
            serde_json::json!("not_running")
        );
    }

    #[test]
    fn a_service_install_reports_what_it_chose_for_the_window() {
        let installed = Installed {
            units: vec![service::GATEWAY.to_owned()],
            unit_dir: PathBuf::from("/tmp/units"),
            exe: PathBuf::from("/usr/local/bin/ailocal"),
            chose_default: Some("gemma4-12b".to_owned()),
            gateway_only: false,
            answering: Some(true),
            linger: false,
        };
        let json = serde_json::to_value(&installed).unwrap();
        assert_eq!(json["chose_default"], "gemma4-12b");
        assert_eq!(json["answering"], true);
        // Without lingering a headless reboot leaves the gateway down, so the window
        // has to be able to say so (rule 5).
        assert_eq!(json["linger"], false);
    }
}
