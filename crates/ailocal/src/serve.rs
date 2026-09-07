//! Running llama-server, with the VRAM ceiling actually enforced.
//!
//! This is where [`crate::vram`] stops being advisory. Two guards, because the failure
//! mode is losing the desktop rather than getting an error:
//!
//! 1. Before spawning, the context is computed from the model's own KV geometry and
//!    clamped to what fits. A load that cannot hold a usable context is refused.
//! 2. While loading, VRAM is sampled and the process killed the moment it crosses the
//!    ceiling - because a mis-estimate must not become a dead session.
//!
//! Only one model runs at a time. With ~13 GB of usable budget a second model is not
//! coexistence, it is eviction, so that is what it does.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use crate::registry::{Fit, Model};
use crate::vram::{self, Budget, CacheType};

/// How long to wait for a model to load before giving up.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);

/// How often to sample VRAM while the model loads.
const WATCHDOG_INTERVAL: Duration = Duration::from_millis(200);

pub const DEFAULT_PORT: u16 = 8080;
pub const DEFAULT_HOST: &str = "127.0.0.1";

/// A running llama-server, as recorded on disk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Instance {
    pub pid: u32,
    pub model: String,
    pub path: PathBuf,
    pub host: String,
    pub port: u16,
    pub context: u64,
    pub cache_type: String,
    /// VRAM held by everything else at the moment this model was loaded.
    ///
    /// Recorded so a later launch can budget against what would be free *after*
    /// evicting this one. Reading current usage instead makes the running model look
    /// like a permanent cost and refuses every replacement.
    #[serde(default)]
    pub desktop_mib: u64,
}

impl Instance {
    #[must_use]
    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }
}

/// The budget a launch should be planned against.
///
/// With a model already running, that is the baseline recorded when it started, since
/// starting another evicts it. Planning against current usage would count the outgoing
/// model's VRAM as unavailable and refuse every swap.
///
/// # Errors
/// If VRAM usage cannot be read and no instance is running to supply a baseline.
pub fn budget_for_next_launch() -> anyhow::Result<Budget> {
    match running()? {
        Some(i) if i.desktop_mib > 0 => Ok(Budget::new(i.desktop_mib)),
        _ => Ok(Budget::new(crate::vram_used_mib()?)),
    }
}

/// Where the running-instance record lives.
///
/// `XDG_RUNTIME_DIR` when available, since the state is meaningless across a reboot
/// and that directory is cleared for us.
///
/// # Errors
/// If no suitable directory can be determined.
pub fn state_path() -> anyhow::Result<PathBuf> {
    let dir = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    Ok(dir.join("ailocal/server.json"))
}

/// Whether a process is still alive.
fn is_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

/// The currently running instance, if there is one.
///
/// A record whose process has died is cleaned up rather than reported, so a crashed
/// server does not leave `ps` lying.
///
/// # Errors
/// If the state file exists but cannot be read or parsed.
pub fn running() -> anyhow::Result<Option<Instance>> {
    let path = state_path()?;
    let raw = match std::fs::read_to_string(&path) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(anyhow::Error::new(e).context(format!("reading {}", path.display()))),
    };

    let instance: Instance =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    if is_alive(instance.pid) {
        Ok(Some(instance))
    } else {
        std::fs::remove_file(&path).ok();
        Ok(None)
    }
}

fn record(instance: &Instance) -> anyhow::Result<()> {
    let path = state_path()?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&path, serde_json::to_string_pretty(instance)?)
        .with_context(|| format!("writing {}", path.display()))
}

/// Signal a process and wait for it to actually exit.
fn terminate(pid: u32) -> anyhow::Result<()> {
    let Ok(raw) = i32::try_from(pid) else {
        anyhow::bail!("implausible pid {pid}")
    };
    let Some(target) = rustix::process::Pid::from_raw(raw) else {
        anyhow::bail!("implausible pid {pid}")
    };

    // SIGTERM first so llama-server can release the GPU cleanly.
    rustix::process::kill_process(target, rustix::process::Signal::TERM).ok();
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if !is_alive(pid) {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    rustix::process::kill_process(target, rustix::process::Signal::KILL).ok();
    Ok(())
}

/// Stop the running instance, returning what was stopped.
///
/// # Errors
/// If the state file cannot be read or the process cannot be signalled.
pub fn stop() -> anyhow::Result<Option<Instance>> {
    let Some(instance) = running()? else {
        return Ok(None);
    };
    terminate(instance.pid)?;
    std::fs::remove_file(state_path()?).ok();
    Ok(Some(instance))
}

/// Knobs for a launch.
#[derive(Debug, Clone)]
pub struct Options {
    /// Requested context. Clamped down to what fits; `None` means "as much as fits".
    pub context: Option<u64>,
    pub host: String,
    pub port: u16,
    pub cache: CacheType,
    /// Bearer token llama-server requires on requests, if any.
    pub api_key: Option<String>,
    /// `auto`, `on` or `off` - see [`crate::config::Config::reasoning`].
    pub reasoning: String,
    /// Token ceiling on thinking; `-1` leaves it unrestricted.
    pub reasoning_budget: i64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            context: None,
            host: DEFAULT_HOST.to_owned(),
            port: DEFAULT_PORT,
            cache: CacheType::Q8_0,
            api_key: None,
            reasoning: "auto".to_owned(),
            reasoning_budget: -1,
        }
    }
}

impl Options {
    /// Options taking their defaults from the config file.
    ///
    /// # Errors
    /// If the configured cache type is not one llama.cpp accepts.
    pub fn from_config(config: &crate::config::Config) -> anyhow::Result<Self> {
        Ok(Self {
            cache: config.cache_type.parse()?,
            reasoning: config.reasoning.clone(),
            reasoning_budget: config.reasoning_budget,
            ..Self::default()
        })
    }
}

/// Decide what context to launch with, refusing loads that cannot work.
///
/// # Errors
/// If the model cannot hold a usable context, or a requested context does not fit.
pub fn plan_context(model: &Model, budget: &Budget, opts: &Options) -> anyhow::Result<u64> {
    let fit = crate::registry::assess(
        model.kv,
        model.trained_context,
        budget,
        opts.cache,
        model.size_mib,
    );

    let ceiling = match fit {
        Fit::Fits(ctx) => ctx,
        Fit::ContextTooSmall(ctx) => anyhow::bail!(
            "{} only holds {ctx} tokens of context here, which is not usable",
            model.name
        ),
        Fit::WeightsTooLarge => anyhow::bail!(
            "{} needs {} MiB of weights but only {} MiB is available",
            model.name,
            model.size_mib,
            budget.available_mib()
        ),
        Fit::Unknown => anyhow::bail!(
            "cannot read {}'s metadata, so cannot prove it is safe to load",
            model.name
        ),
    };

    match opts.context {
        None => Ok(ceiling),
        Some(want) if want <= ceiling => Ok(want),
        Some(want) => anyhow::bail!(
            "requested {want} tokens of context but only {ceiling} fits within the \
             {} MiB ceiling; over-committing VRAM kills the desktop session",
            vram::CEILING_MIB
        ),
    }
}

/// Launch llama-server for `model`, evicting anything already running.
///
/// # Errors
/// If the model cannot be served safely, the binary is missing, or it fails to become
/// healthy before the timeout.
pub fn start(model: &Model, budget: &Budget, opts: &Options) -> anyhow::Result<Instance> {
    let context = plan_context(model, budget, opts)?;

    // Single-tenant: the budget cannot hold two models, so this is eviction. Done only
    // after plan_context has accepted the new load, so a refusal never costs the
    // model that was already serving.
    if let Some(old) = running()? {
        terminate(old.pid)?;
        std::fs::remove_file(state_path()?).ok();
        wait_for_vram_release(old.desktop_mib);
    }

    let log = state_path()?.with_file_name("server.log");
    if let Some(dir) = log.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let out = std::fs::File::create(&log).with_context(|| format!("creating {}", log.display()))?;
    let err = out.try_clone()?;

    // Sampled after eviction, so it reflects the desktop alone.
    let desktop_mib = crate::vram_used_mib().unwrap_or(budget.desktop_mib);

    let mut cmd = std::process::Command::new("llama-server");
    cmd.arg("-m")
        .arg(&model.path)
        .args(["-ngl", "99"])
        .args(["-c", &context.to_string()])
        .args(["-ctk", opts.cache.as_llama_arg()])
        .args(["-ctv", opts.cache.as_llama_arg()])
        .args(["--host", &opts.host])
        .args(["--port", &opts.port.to_string()])
        .args(["-a", &model.name])
        .args(["--reasoning", &opts.reasoning])
        .arg("--no-webui")
        .stdout(out)
        .stderr(err);
    if opts.reasoning_budget >= 0 {
        cmd.args(["--reasoning-budget", &opts.reasoning_budget.to_string()]);
    }
    if let Some(key) = &opts.api_key {
        cmd.args(["--api-key", key]);
    }

    let child = cmd
        .spawn()
        .context("starting llama-server (is llama-cpp installed?)")?;
    let pid = child.id();

    let instance = Instance {
        pid,
        model: model.name.clone(),
        path: model.path.clone(),
        host: opts.host.clone(),
        port: opts.port,
        context,
        cache_type: opts.cache.as_llama_arg().to_owned(),
        desktop_mib,
    };

    match await_healthy(&instance) {
        Ok(()) => {
            record(&instance)?;
            Ok(instance)
        }
        Err(e) => {
            terminate(pid).ok();
            let tail = std::fs::read_to_string(&log)
                .map(|s| s.lines().rev().take(8).collect::<Vec<_>>().join("\n"))
                .unwrap_or_default();
            Err(e.context(format!("llama-server did not come up:\n{tail}")))
        }
    }
}

/// Give the driver a moment to hand back the evicted model's VRAM.
///
/// The process exiting does not mean the allocation is gone yet, and starting the next
/// model against stale usage is how the watchdog trips on a load that would have fit.
fn wait_for_vram_release(baseline_mib: u64) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        match crate::vram_used_mib() {
            Ok(used) if used <= baseline_mib.saturating_add(512) => return,
            Err(_) => return,
            Ok(_) => std::thread::sleep(WATCHDOG_INTERVAL),
        }
    }
}

/// Wait for the server to answer, killing it if VRAM crosses the ceiling first.
///
/// The watchdog is the reason this is not just a health poll. Estimating capacity
/// wrongly must cost a failed start, never the graphical session.
fn await_healthy(instance: &Instance) -> anyhow::Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()?;
    let health = format!("{}/health", instance.base_url());
    let deadline = Instant::now() + STARTUP_TIMEOUT;

    while Instant::now() < deadline {
        if !is_alive(instance.pid) {
            anyhow::bail!("llama-server exited during startup");
        }

        if let Ok(used) = crate::vram_used_mib()
            && used >= vram::CEILING_MIB
        {
            terminate(instance.pid).ok();
            anyhow::bail!(
                "aborted at {used} MiB of VRAM, over the {} MiB ceiling - \
                 killed it rather than let it starve the compositor",
                vram::CEILING_MIB
            );
        }

        if client
            .get(&health)
            .send()
            .is_ok_and(|r| r.status().is_success())
        {
            return Ok(());
        }
        std::thread::sleep(WATCHDOG_INTERVAL);
    }

    anyhow::bail!("timed out after {STARTUP_TIMEOUT:?}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vram::KvLayout;

    fn model(name: &str, size_mib: u64, kv: Option<KvLayout>, trained: Option<u64>) -> Model {
        Model {
            name: name.to_owned(),
            path: PathBuf::from("/tmp/x.gguf"),
            size_mib,
            arch: Some("test".into()),
            trained_context: trained,
            kv,
            sliding_window: false,
        }
    }

    fn qwen3_14b() -> KvLayout {
        KvLayout::dense(40, 8, 128, 128)
    }

    #[test]
    fn defaults_to_the_largest_context_that_fits() {
        let m = model("qwen3-14b", 8836, Some(qwen3_14b()), Some(40_960));
        let ctx = plan_context(&m, &Budget::new(900), &Options::default()).unwrap();
        assert!((32_000..=40_960).contains(&ctx), "got {ctx}");
    }

    #[test]
    fn honours_a_smaller_requested_context() {
        let m = model("qwen3-14b", 8836, Some(qwen3_14b()), Some(40_960));
        let opts = Options {
            context: Some(8192),
            ..Options::default()
        };
        assert_eq!(plan_context(&m, &Budget::new(900), &opts).unwrap(), 8192);
    }

    /// The whole point: asking for the context that killed the desktop must be an
    /// error, not a clamp and definitely not an attempt.
    #[test]
    fn refuses_a_context_that_would_not_fit() {
        let m = model("qwen3-14b", 8836, Some(qwen3_14b()), Some(131_072));
        let opts = Options {
            context: Some(65_536),
            ..Options::default()
        };
        let err = plan_context(&m, &Budget::new(900), &opts).unwrap_err();
        assert!(err.to_string().contains("only"), "unexpected error: {err}");
    }

    #[test]
    fn refuses_a_model_whose_weights_do_not_fit() {
        let m = model("devstral-24b", 13_660, Some(qwen3_14b()), Some(131_072));
        let err = plan_context(&m, &Budget::new(900), &Options::default()).unwrap_err();
        assert!(err.to_string().contains("weights"), "unexpected: {err}");
    }

    /// Unreadable metadata means we cannot prove the load is safe, so we do not try.
    #[test]
    fn refuses_a_model_it_cannot_measure() {
        let m = model("mystery", 4000, None, None);
        let err = plan_context(&m, &Budget::new(900), &Options::default()).unwrap_err();
        assert!(err.to_string().contains("metadata"), "unexpected: {err}");
    }

    fn instance(desktop_mib: u64) -> Instance {
        Instance {
            pid: 1,
            model: "m".into(),
            path: PathBuf::from("/tmp/m.gguf"),
            host: "127.0.0.1".into(),
            port: 8080,
            context: 4096,
            cache_type: "q8_0".into(),
            desktop_mib,
        }
    }

    #[test]
    fn base_url_is_built_from_host_and_port() {
        assert_eq!(instance(900).base_url(), "http://127.0.0.1:8080");
    }

    /// A swap must be planned against the baseline recorded when the outgoing model
    /// loaded. Budgeting against live usage counts the model being evicted as an
    /// immovable cost and refuses every replacement.
    #[test]
    fn a_swap_is_budgeted_against_the_post_eviction_baseline() {
        let running = instance(900);
        let budget = Budget::new(running.desktop_mib);
        let m = model("qwen3-14b", 8836, Some(qwen3_14b()), Some(40_960));
        assert!(plan_context(&m, &budget, &Options::default()).is_ok());

        // The same model against live usage while gemma4 holds 11317 MiB.
        let naive = Budget::new(11_317);
        assert!(plan_context(&m, &naive, &Options::default()).is_err());
    }

    /// State written before `desktop_mib` existed must still load.
    #[test]
    fn older_state_files_without_a_baseline_still_parse() {
        let json = r#"{"pid":1,"model":"m","path":"/tmp/m.gguf","host":"127.0.0.1",
                       "port":8080,"context":4096,"cache_type":"q8_0"}"#;
        let i: Instance = serde_json::from_str(json).unwrap();
        assert_eq!(i.desktop_mib, 0);
    }
}
