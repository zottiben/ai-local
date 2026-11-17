//! What is installed, what could be, and changing either.
//!
//! Each change is split into a step that only looks and a step that acts, so the
//! command line can print the plan and ask, and the window can show it in a dialog,
//! before anything is deleted or rewritten.

use std::path::PathBuf;

use anyhow::Context as _;
use reqwest::blocking::Client;

use crate::catalogue::{self, Candidate};
use crate::config::Config;
use crate::gguf::{self, Thinking};
use crate::registry::{self, Fit};
use crate::source::Source;
use crate::vram::{Budget, CacheType};
use crate::{harness, serve, service};

/// One installed model, as `model ls` lists it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Installed {
    pub name: String,
    pub path: PathBuf,
    pub size_mib: u64,
    /// `None` when the file could not be parsed as GGUF.
    pub arch: Option<String>,
    pub sliding_window: bool,
    /// KV cache cost of each token of context, at the configured cache type.
    pub kv_bytes_per_token: Option<f64>,
    pub trained_context: Option<u64>,
    /// The largest context a launch would get here, or `None` if it cannot run at all.
    pub max_context: Option<u64>,
    /// What thinking control the model's own chat template offers.
    pub thinking: Thinking,
}

/// Every installed model, sorted by name, with what each can do on this machine.
///
/// Budgeted against what would be free after evicting whatever is loaded. Using live
/// VRAM instead reports the running model as unable to fit, which is absurd on its
/// face and was exactly what `model ls` printed before.
///
/// # Errors
/// If the configured cache type is unknown or the models directory cannot be read.
pub fn inventory(cfg: &Config) -> anyhow::Result<Vec<Installed>> {
    let cache: CacheType = cfg.cache_type.parse()?;
    let budget = serve::budget_or_estimate();

    Ok(registry::scan(&cfg.models_dir)?
        .into_iter()
        .map(|m| Installed {
            kv_bytes_per_token: m.kv.map(|kv| kv.bytes_per_token(cache)),
            max_context: m.max_context(&budget, cache).filter(|&c| c > 0),
            thinking: gguf::thinking_support(&m.path),
            name: m.name,
            path: m.path,
            size_mib: m.size_mib,
            arch: m.arch,
            sliding_window: m.sliding_window,
            trained_context: m.trained_context,
        })
        .collect())
}

/// An installed model the picker offers. Choosing it costs nothing but a config write.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Downloaded {
    pub name: String,
    pub size_mib: u64,
    /// The context a launch would get here.
    pub context: u64,
    /// The catalogue's description, when this is a model the catalogue knows.
    pub summary: Option<&'static str>,
}

/// What `model pick` offers: what is installed and runs here, then what could be.
#[derive(Debug, serde::Serialize)]
pub struct Offer {
    pub available_mib: u64,
    pub downloaded: Vec<Downloaded>,
    /// Catalogue entries not yet on disk, best first.
    pub remote: Vec<Candidate>,
    /// Installed models that cannot run here at all.
    pub unusable: usize,
    #[serde(skip)]
    budget: Budget,
    #[serde(skip)]
    cache: CacheType,
}

impl Offer {
    /// Ask the catalogue what fits in `budget`, and fold in what is already installed.
    ///
    /// A catalogue model already on disk is offered as itself rather than as a
    /// download, so nobody is invited to fetch gigabytes they already have - and so
    /// picking it is how the default is switched.
    ///
    /// # Errors
    /// If the models directory cannot be read, or the registry cannot be reached and
    /// nothing is installed. With a model installed there is still a perfectly good
    /// choice to make offline.
    pub fn gather(cfg: &Config, client: &Client, budget: Budget) -> anyhow::Result<Self> {
        Self::from_candidates(
            cfg,
            budget,
            catalogue::resolve(client, budget.available_mib()),
        )
    }

    /// [`Offer::gather`], from a catalogue already resolved against `budget`.
    ///
    /// The window measures the catalogue once and keeps it, because the registry
    /// takes ten seconds to ask and changes far less often than the models directory
    /// does. What is on disk is still read here, every time.
    ///
    /// # Errors
    /// As [`Offer::gather`].
    pub fn from_candidates(
        cfg: &Config,
        budget: Budget,
        mut remote: Vec<Candidate>,
    ) -> anyhow::Result<Self> {
        let cache: CacheType = cfg.cache_type.parse()?;
        let available_mib = budget.available_mib();
        let on_disk = registry::scan(&cfg.models_dir)?;

        // Keep what the catalogue called a model it found on disk, to describe it.
        let mut described: Vec<(String, &'static str)> = Vec::new();
        remote.retain(|c| {
            let Some(stem) = catalogue_stem(c) else {
                return true;
            };
            match registry::already_have(&on_disk, &stem, c.size_mib) {
                Some(have) => {
                    described.push((have.name.clone(), c.entry.summary));
                    false
                }
                None => true,
            }
        });

        let mut downloaded = Vec::new();
        let mut unusable = 0;
        for m in &on_disk {
            match registry::assess(m.kv, m.trained_context, &budget, cache, m.size_mib) {
                Fit::Fits(context) => downloaded.push(Downloaded {
                    name: m.name.clone(),
                    size_mib: m.size_mib,
                    context,
                    summary: described
                        .iter()
                        .find(|(name, _)| *name == m.name)
                        .map(|(_, summary)| *summary),
                }),
                _ => unusable += 1,
            }
        }

        anyhow::ensure!(
            !remote.is_empty() || !downloaded.is_empty(),
            "could not reach the model registry, and nothing is installed yet"
        );

        Ok(Self {
            available_mib,
            downloaded,
            remote,
            unusable,
            budget,
            cache,
        })
    }

    /// Read each fitting download's real attention geometry for its usable context.
    ///
    /// A few MiB of each, because weight size does not predict context: a 12B with
    /// sliding-window attention holds six times what a 14B with full attention does.
    pub fn measure(&mut self, client: &Client) {
        catalogue::measure(client, &mut self.remote, &self.budget, self.cache);
    }
}

/// The filename stem a catalogue entry would be stored under, for spotting one that is
/// already downloaded.
fn catalogue_stem(candidate: &Candidate) -> Option<String> {
    format!("ollama:{}", candidate.entry.reference)
        .parse::<Source>()
        .ok()
        .map(|s| s.file_name().trim_end_matches(".gguf").to_owned())
}

/// Finds the binary a service definition should run.
///
/// Passed in rather than decided here, because it depends on who is asking: the
/// command line points units at itself, and the window has to point them at the
/// `ailocal` beside it, never at its own executable.
pub type UnitBinary<'a> = &'a dyn Fn() -> anyhow::Result<PathBuf>;

/// What became of the model service after the default changed.
#[derive(Debug)]
pub enum Repoint {
    /// No model service is installed, so there was nothing to change.
    NoService,
    /// It was running, and now serves the new default.
    Restarted,
    /// No default is left, so its definition was removed and it is stopped.
    Cleared,
    /// It was installed but not running; its definition names the new default.
    Updated,
    /// It still names the old model.
    Failed(anyhow::Error),
}

/// Point an installed model service at the configured default.
///
/// The unit embeds the model name in its `ExecStart`, so changing `default_model` in
/// the config does nothing on its own - the service would keep serving the old model
/// and would load it again at the next boot. Reported rather than returned as an
/// error: a service that cannot be updated must not turn a successful download into a
/// failed command.
pub fn repoint_model_service(cfg: &Config, unit_binary: UnitBinary) -> Repoint {
    if !(service::is_enabled(service::MODEL) || service::is_active(service::MODEL)) {
        return Repoint::NoService;
    }

    let outcome = (|| -> anyhow::Result<bool> {
        let exe = unit_binary()?;
        service::install(
            &exe,
            &cfg.gateway_host,
            cfg.gateway_port,
            cfg.default_model.as_deref(),
        )?;
        service::reload()?;
        let running = service::is_active(service::MODEL);
        let wanted = cfg.default_model.is_some();
        if running {
            service::stop(service::MODEL)?;
            // Only start it again if there is something to serve. With no default
            // model `install` deletes the definition, so starting would ask the
            // service manager for a unit that no longer exists.
            if wanted {
                service::start(service::MODEL)?;
            }
        }
        Ok(running && wanted)
    })();

    match outcome {
        Ok(true) => Repoint::Restarted,
        Ok(false) if cfg.default_model.is_none() => Repoint::Cleared,
        Ok(false) => Repoint::Updated,
        Err(e) => Repoint::Failed(e),
    }
}

/// What setting the default did.
#[derive(Debug)]
pub enum DefaultChange {
    /// It already was the default; nothing was written.
    Unchanged,
    Changed(Repoint),
}

/// Record `name` as the model this machine serves, and make that true.
///
/// # Errors
/// If the config cannot be read or written.
pub fn set_default(name: &str, unit_binary: UnitBinary) -> anyhow::Result<DefaultChange> {
    let mut cfg = Config::load()?;
    if cfg.default_model.as_deref() == Some(name) {
        return Ok(DefaultChange::Unchanged);
    }

    cfg.default_model = Some(name.to_owned());
    cfg.save()?;
    Ok(DefaultChange::Changed(repoint_model_service(
        &cfg,
        unit_binary,
    )))
}

/// A model about to be deleted, and what deleting it will also change.
#[derive(Debug, Clone)]
pub struct Removal {
    pub name: String,
    pub path: PathBuf,
    pub size_mib: u64,
    /// llama-server has it open, so it will be stopped first.
    pub loaded: bool,
    /// It is `default_model`, which will be cleared.
    pub is_default: bool,
}

/// Work out what deleting `name` involves, without deleting anything.
///
/// # Errors
/// If no such model is installed, or the running server's record cannot be read.
pub fn plan_removal(cfg: &Config, name: &str) -> anyhow::Result<Removal> {
    let models = registry::scan(&cfg.models_dir)?;
    let model = models.into_iter().find(|m| m.name == name).ok_or_else(|| {
        anyhow::anyhow!(
            "no model named {name:?} in {}; `ailocal model ls` shows what is installed",
            cfg.models_dir.display()
        )
    })?;

    Ok(Removal {
        loaded: serve::running()?.is_some_and(|i| i.model == name),
        is_default: cfg.default_model.as_deref() == Some(name),
        name: model.name,
        path: model.path,
        size_mib: model.size_mib,
    })
}

/// What deleting a model did, beyond the delete.
#[derive(Debug)]
pub struct Removed {
    /// It was loaded, and was stopped before the file went.
    pub stopped: bool,
    /// What became of `default_model`, if it named the model.
    pub default: Option<anyhow::Result<Repoint>>,
    /// Whether Pi had an entry for it, which is now gone.
    pub pi: anyhow::Result<bool>,
    /// Claude Code's env file still names it. That file is sourced by hand rather
    /// than read through us, so it cannot be quietly repointed.
    pub claude_code_pinned: bool,
}

impl Removal {
    /// Delete the weights, and everything that pointed at them.
    ///
    /// Deleting the file is the easy half. The rest of the machine goes on believing
    /// the model is there: `default_model` still names it, so the model service loads
    /// it at the next boot; Pi still has a capability entry for it, so it stays in
    /// `/model` and fails on selection; llama-server may have it open right now. Each
    /// of those is a confusing failure minutes or a reboot away from the delete that
    /// caused it, so they are settled here.
    ///
    /// Whether it is loaded is asked again rather than taken from the plan: the
    /// gateway loads models on demand, and a confirmation can sit open long enough
    /// for it to load this one.
    ///
    /// # Errors
    /// If it is loaded and cannot be stopped, or the file cannot be deleted. Once the
    /// file is gone nothing else fails the call: the weights are already deleted, so
    /// an error would leave no way to finish what it started.
    pub fn remove(&self, cfg: &Config, unit_binary: UnitBinary) -> anyhow::Result<Removed> {
        // Stop before unlinking. llama-server holds the file open and keeps serving
        // from a deleted inode, so the model appears to work until something restarts
        // it.
        let stopped = serve::running()?.is_some_and(|i| i.model == self.name);
        if stopped {
            serve::stop()?;
        }

        std::fs::remove_file(&self.path).with_context(|| {
            format!(
                "deleting {}{}",
                self.path.display(),
                if stopped {
                    " (it was stopped first)"
                } else {
                    ""
                }
            )
        })?;

        let default = (cfg.default_model.as_deref() == Some(self.name.as_str())).then(|| {
            let mut cfg = cfg.clone();
            cfg.default_model = None;
            cfg.save().map(|_| repoint_model_service(&cfg, unit_binary))
        });
        let pi = harness::forget_pi_model(&self.name);
        let claude_code_pinned = harness::claude_code_env_path()
            .ok()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .is_some_and(|env| env.contains(&format!("ANTHROPIC_MODEL={}\n", self.name)));

        Ok(Removed {
            stopped,
            default,
            pi,
            claude_code_pinned,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sandbox(name: &str) -> Config {
        let dir =
            std::env::temp_dir().join(format!("ailocal-models-{name}-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("models")).unwrap();
        Config {
            models_dir: dir.join("models"),
            default_model: Some("gemma4-12b".to_owned()),
            ..Config::default()
        }
    }

    /// The plan names the file and its size, so a 258 MiB model reads as that at the
    /// prompt, and it deletes nothing.
    #[test]
    fn a_removal_plan_only_looks() {
        let cfg = sandbox("plan");
        let path = cfg.models_dir.join("tiny.gguf");
        std::fs::write(&path, vec![0u8; 3 << 20]).unwrap();

        let plan = plan_removal(&cfg, "tiny").unwrap();
        assert_eq!(plan.path, path);
        assert_eq!(plan.size_mib, 3);
        assert!(!plan.is_default);
        assert!(path.exists());
        std::fs::remove_dir_all(cfg.models_dir.parent().unwrap()).ok();
    }

    /// The same error the command has always given for a name that is not installed,
    /// so the window and the terminal say the same thing.
    #[test]
    fn a_removal_plan_for_a_missing_model_says_where_it_looked() {
        let cfg = sandbox("missing");
        let err = plan_removal(&cfg, "nope").unwrap_err().to_string();
        assert!(err.starts_with("no model named \"nope\" in "), "got: {err}");
        assert!(err.contains("ailocal model ls"), "got: {err}");
        std::fs::remove_dir_all(cfg.models_dir.parent().unwrap()).ok();
    }
}
