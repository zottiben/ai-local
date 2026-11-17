//! The whole picture of this machine, and what is still missing from it.
//!
//! `ailocal status` prints this and the window draws it. Both read the machine rather
//! than anything remembered, because the machine is the state: a model deleted by
//! hand, a service stopped by launchd or a port taken by Metro all have to show up
//! the next time anyone looks.

use crate::config::Config;
use crate::registry::{self, Fit};
use crate::{auth, gateway, gguf, harness, serve, service, vram};

/// Everything `status` reports, gathered in one pass.
#[derive(Debug)]
pub struct Overview {
    /// Installed models that can run here, alphabetically.
    pub models: Vec<harness::PiModel>,
    pub default_model: Option<String>,
    /// What llama-server has resident right now.
    pub loaded: Option<serve::Instance>,
    /// The gateway's URL as a harness on this machine would call it.
    pub gateway_url: String,
    pub gateway: Gateway,
    /// What the gateway offers a harness, asked only while it answers.
    pub advertised: Option<anyhow::Result<Vec<String>>>,
    /// The gateway service, then the model service.
    pub services: Vec<(&'static str, Service)>,
    /// Harnesses already pointed at the gateway.
    pub harnesses: Vec<&'static str>,
    pub next: NextStep,
}

/// Whether the gateway port is doing its job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gateway {
    Answering,
    /// Something that is not our gateway holds the port.
    PortTaken,
    NotAnswering,
}

/// What the service manager says about one of our services.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Service {
    Running,
    InstalledNotRunning,
    NotInstalled,
}

/// Look at every part of the setup.
///
/// Nothing here fails: a part that cannot be read is reported as missing, because the
/// point of asking is to find the broken link, and an error would hide the rest.
#[must_use]
pub fn overview(cfg: &Config) -> Overview {
    let models = runnable_models(cfg).unwrap_or_default();
    let loaded = serve::running().ok().flatten();

    let gateway = if gateway::answers_as_gateway(&cfg.gateway_host, cfg.gateway_port) {
        Gateway::Answering
    } else {
        match gateway::port_status(&cfg.gateway_host, cfg.gateway_port) {
            gateway::PortStatus::Taken => Gateway::PortTaken,
            _ => Gateway::NotAnswering,
        }
    };
    let advertised = (gateway == Gateway::Answering).then(|| advertised_models(cfg));

    let services: Vec<_> = service::service_names()
        .into_iter()
        .map(|name| {
            let state = if service::is_active(name) {
                Service::Running
            } else if service::is_enabled(name) {
                Service::InstalledNotRunning
            } else {
                Service::NotInstalled
            };
            (name, state)
        })
        .collect();
    let harnesses = harness::configured();

    let next = NextStep::from_state(
        !models.is_empty(),
        gateway == Gateway::Answering,
        services
            .iter()
            .any(|&(name, state)| name == service::GATEWAY && state != Service::NotInstalled),
        !harnesses.is_empty(),
    );

    Overview {
        models,
        default_model: cfg.default_model.clone(),
        loaded,
        gateway_url: format!(
            "http://{}:{}",
            gateway::loopback_for(&cfg.gateway_host),
            cfg.gateway_port
        ),
        gateway,
        advertised,
        services,
        harnesses,
        next,
    }
}

/// Every installed model that could actually run here, alphabetically.
///
/// Offering a model a harness cannot load only moves the failure later, so anything
/// that does not fit is left out.
///
/// # Errors
/// If the configured cache type is unknown or the models directory cannot be read.
pub fn runnable_models(cfg: &Config) -> anyhow::Result<Vec<harness::PiModel>> {
    let cache: vram::CacheType = cfg.cache_type.parse()?;
    let budget = serve::budget_or_estimate();

    Ok(registry::scan(&cfg.models_dir)?
        .iter()
        .filter_map(|m| {
            match registry::assess(m.kv, m.trained_context, &budget, cache, m.size_mib) {
                Fit::Fits(ctx) => Some(harness::PiModel {
                    id: m.name.clone(),
                    context_window: ctx,
                    // What the model can do, not how the server happens to be launched
                    // right now. Reporting the launch flag here is what left Pi
                    // offering `off` as the only choice: `reasoning = "off"` in the
                    // config became "this model cannot think" in the catalogue.
                    thinking: gguf::thinking_support(&m.path),
                }),
                _ => None,
            }
        })
        .collect())
}

/// The model names the gateway is offering right now.
///
/// Asked over HTTP rather than computed locally, because a harness gets this list from
/// the gateway and the point is to see what it sees.
///
/// # Errors
/// If the gateway cannot be asked, or does not answer with a model list.
pub fn advertised_models(cfg: &Config) -> anyhow::Result<Vec<String>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let body: serde_json::Value = client
        .get(format!(
            "http://{}:{}/v1/models",
            gateway::loopback_for(&cfg.gateway_host),
            cfg.gateway_port
        ))
        .bearer_auth(auth::load_or_create()?)
        .send()?
        .error_for_status()?
        .json()?;

    Ok(body["data"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| m["id"].as_str().map(str::to_owned))
        .collect())
}

/// What still has to happen before a harness can answer a prompt.
///
/// Worked out from the machine's actual state rather than assumed. The hardcoded
/// version of this pointed from "a model is installed" straight at `harness configure`,
/// which skipped the gateway entirely - so following the tool's own advice produced a
/// configured harness talking to nothing, and the first sign of trouble was Pi
/// answering a prompt with "Connection error".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextStep {
    InstallModel,
    /// Nothing is serving, and no service exists to do it.
    StartGateway,
    /// A service exists but is not answering.
    FixGateway,
    ConfigureHarness,
    /// Everything is in place.
    Ready,
}

impl NextStep {
    /// Decide from the parts rather than by probing, so this is testable.
    #[must_use]
    pub const fn from_state(
        has_model: bool,
        gateway_answering: bool,
        service_installed: bool,
        harness_configured: bool,
    ) -> Self {
        if !has_model {
            return Self::InstallModel;
        }
        if !gateway_answering {
            return if service_installed {
                Self::FixGateway
            } else {
                Self::StartGateway
            };
        }
        if !harness_configured {
            return Self::ConfigureHarness;
        }
        Self::Ready
    }

    /// The step as the command line words it, or `None` when there is nothing to do.
    #[must_use]
    pub fn describe(&self) -> Option<String> {
        Some(match self {
            Self::InstallModel => {
                "Next: ailocal model pick   (choose a model for this machine)".to_owned()
            }
            Self::StartGateway => "Next: ailocal service install   (nothing is serving yet - \
                 this starts the gateway and keeps it running)"
                .to_owned(),
            Self::FixGateway => format!(
                "Next: the gateway service is installed but not answering.\n\
                 \x20     ailocal gateway check      says what is wrong\n\
                 \x20     {}",
                service::log_hint()
            ),
            Self::ConfigureHarness => {
                "Next: ailocal harness configure pi   (or claude-code)".to_owned()
            }
            Self::Ready => return None,
        })
    }
}

/// Inspect this machine and work out what is still missing.
#[must_use]
pub fn next_step(cfg: &Config) -> NextStep {
    NextStep::from_state(
        runnable_models(cfg).is_ok_and(|m| !m.is_empty()),
        gateway::answers_as_gateway(&cfg.gateway_host, cfg.gateway_port),
        service::is_enabled(service::GATEWAY) || service::is_active(service::GATEWAY),
        !harness::configured().is_empty(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact sequence a fresh machine walks, in order.
    #[test]
    fn the_next_step_follows_the_setup_sequence() {
        assert_eq!(
            NextStep::from_state(false, false, false, false),
            NextStep::InstallModel
        );
        assert_eq!(
            NextStep::from_state(true, false, false, false),
            NextStep::StartGateway
        );
        assert_eq!(
            NextStep::from_state(true, true, true, false),
            NextStep::ConfigureHarness
        );
        assert_eq!(
            NextStep::from_state(true, true, true, true),
            NextStep::Ready
        );
    }

    /// The bug this replaced: with a model installed and no gateway, the advice was
    /// "configure a harness", which produced a harness pointed at nothing and a
    /// "Connection error" at the first prompt.
    #[test]
    fn a_model_without_a_gateway_does_not_send_you_to_the_harness() {
        let step = NextStep::from_state(true, false, false, false);
        assert_ne!(step, NextStep::ConfigureHarness);
        let advice = step.describe().unwrap();
        assert!(advice.contains("service install"), "got: {advice}");
    }

    /// A harness configured against a dead gateway is still broken, so the gateway
    /// outranks it however far along the rest is.
    #[test]
    fn a_dead_gateway_outranks_an_already_configured_harness() {
        assert_eq!(
            NextStep::from_state(true, false, true, true),
            NextStep::FixGateway
        );
        assert_eq!(
            NextStep::from_state(true, false, false, true),
            NextStep::StartGateway
        );
    }

    /// "Not installed" and "installed but broken" need opposite advice - one is a
    /// command to run, the other is a log to read.
    #[test]
    fn a_missing_service_and_a_broken_one_are_told_apart() {
        let missing = NextStep::from_state(true, false, false, false)
            .describe()
            .unwrap();
        let broken = NextStep::from_state(true, false, true, false)
            .describe()
            .unwrap();
        assert!(missing.contains("service install"), "got: {missing}");
        assert!(!broken.contains("service install"), "got: {broken}");
        assert!(broken.contains("gateway check"), "got: {broken}");
    }

    #[test]
    fn a_ready_machine_is_told_nothing() {
        assert!(
            NextStep::from_state(true, true, true, true)
                .describe()
                .is_none()
        );
    }
}
