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

use std::path::PathBuf;
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
    /// `None` when it could not be measured, which is not the same as zero.
    ///
    /// It has to be an option. As a bare `u64`, zero meant both "never recorded" and
    /// "measured nothing else resident" - and a device that reports all of its memory
    /// free when idle records exactly zero, so the guard against the first case threw
    /// away the second. The budget then fell back to live usage, counting the running
    /// model as a permanent cost and concluding it no longer fits.
    #[serde(default)]
    pub desktop_mib: Option<u64>,
    /// `--reasoning` this server was launched with.
    ///
    /// A launch-time flag with no per-request equivalent, so the only way to know
    /// whether the resident model is thinking is to have written it down. The eval
    /// harness compares reasoning arms and would otherwise have to restart the model
    /// for every task to be sure which mode it was in.
    #[serde(default = "unknown_reasoning")]
    pub reasoning: String,
}

/// What to assume for a state file written before `reasoning` was recorded.
fn unknown_reasoning() -> String {
    "?".to_owned()
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
/// The configured context cap is applied here rather than at the spawn, because every
/// figure a user or a harness is shown has to be the one a launch would actually take.
/// `model ls`, the gateway's catalogue and the window Pi compacts against all come
/// through this function; capping only the spawn would advertise a window the server
/// does not have.
///
/// # Errors
/// If VRAM usage cannot be read and no instance is running to supply a baseline, or
/// the config cannot be parsed.
pub fn budget_for_next_launch() -> anyhow::Result<Budget> {
    // The ceiling comes from the device: a 16 GB card and a 64 GB Mac cannot share a
    // hardcoded constant.
    let mut budget = Budget::for_device(&crate::primary_device()?)
        .capped_at(Some(crate::config::Config::load()?.max_context));

    // With a model already resident, that memory is not a permanent cost - starting
    // another evicts it - so budget against the baseline recorded when it loaded.
    if let Some(instance) = running()?
        && let Some(baseline) = instance.desktop_mib
    {
        budget.desktop_mib = baseline;
    }
    Ok(budget)
}

/// Where the running-instance record lives.
///
/// `XDG_RUNTIME_DIR` when available, since the state is meaningless across a reboot and
/// that directory is cleared for us. macOS sets no such variable, and its per-user
/// `TMPDIR` has the same property, so the fallback is right there rather than a
/// degradation.
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
///
/// `kill(pid, 0)` rather than `/proc/<pid>`: it is portable, and there is no `/proc` on
/// macOS - where the directory check would silently report every process as dead, so
/// `ps` would show nothing running and a second model would load on top of the first.
fn is_alive(pid: u32) -> bool {
    let Ok(raw) = i32::try_from(pid) else {
        return false;
    };
    let Some(target) = rustix::process::Pid::from_raw(raw) else {
        return false;
    };
    // Err(ESRCH) means no such process; Err(EPERM) means it exists but is not ours,
    // which still counts as alive.
    !matches!(
        rustix::process::test_kill_process(target),
        Err(rustix::io::Errno::SRCH)
    )
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

    let capped_by_config = model.kv.is_some_and(|kv| {
        budget.context_cap.is_some_and(|cap| {
            cap == ceiling
                && budget
                    .capped_at(None)
                    .max_context(&kv, opts.cache, model.size_mib)
                    .map(|hardware| model.trained_context.map_or(hardware, |t| hardware.min(t)))
                    .is_some_and(|without_cap| without_cap > cap)
        })
    });

    match opts.context {
        None => Ok(ceiling),
        Some(want) if want <= ceiling => Ok(want),
        // A request the configuration refuses, not one the hardware does. Raising the
        // cap rather than overriding it here keeps one answer to "how big is the
        // window": the harnesses were told this number too.
        Some(want) if capped_by_config => anyhow::bail!(
            "requested {want} tokens of context but max_context is {ceiling}; raise it \
             with `ailocal config max-context {want}` if the latency is worth it"
        ),
        Some(want) if model.trained_context == Some(ceiling) => anyhow::bail!(
            "requested {want} tokens of context but {} was trained for {ceiling}",
            model.name
        ),
        Some(want) => anyhow::bail!(
            "requested {want} tokens of context but only {ceiling} fits within the \
             {} MiB ceiling; over-committing GPU memory can take down the desktop",
            budget.ceiling_mib
        ),
    }
}

/// Launch llama-server for `model`, evicting anything already running.
///
/// # Errors
/// If the model cannot be served safely, the binary is missing, or it fails to become
/// healthy before the timeout.
pub fn start(model: &Model, budget: &Budget, opts: &Options) -> anyhow::Result<Instance> {
    let (instance, child) = spawn(model, budget, opts, Stdio::LogFile)?;
    // Deliberately not waited on: dropping a std::process::Child does not kill it, so
    // the server outlives this command. It does still die on SIGHUP when the terminal
    // closes, which is what `run_foreground` under systemd is for.
    drop(child);
    Ok(instance)
}

/// Run llama-server in the foreground, returning when it exits.
///
/// This is the form a service manager wants: one process to supervise, restart and
/// collect logs from, rather than a command that forks and returns. Terminating this
/// process terminates the model.
///
/// # Errors
/// If the model cannot be served safely or fails to become healthy.
pub fn run_foreground(model: &Model, budget: &Budget, opts: &Options) -> anyhow::Result<()> {
    let (instance, mut child) = spawn(model, budget, opts, Stdio::Inherit)?;
    println!(
        "{} up at {} with {} context",
        instance.model,
        instance.base_url(),
        instance.context
    );

    let status = child.wait().context("waiting for llama-server")?;
    std::fs::remove_file(state_path()?).ok();
    anyhow::ensure!(status.success(), "llama-server exited with {status}");
    Ok(())
}

/// Where llama-server's output should go.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stdio {
    /// A log file, so a detached server's output is still recoverable.
    LogFile,
    /// Our own stdio, so a service manager collects it into the journal.
    Inherit,
}

/// Plan, evict, spawn and wait for health, returning the live child.
fn spawn(
    model: &Model,
    budget: &Budget,
    opts: &Options,
    stdio: Stdio,
) -> anyhow::Result<(Instance, std::process::Child)> {
    let context = plan_context(model, budget, opts)?;

    // Single-tenant: the budget cannot hold two models, so this is eviction. Done only
    // after plan_context has accepted the new load, so a refusal never costs the
    // model that was already serving.
    if let Some(old) = running()? {
        terminate(old.pid)?;
        std::fs::remove_file(state_path()?).ok();
        wait_for_vram_release(old.desktop_mib.unwrap_or(budget.desktop_mib));
    }

    let log = state_path()?.with_file_name("server.log");
    if let Some(dir) = log.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let out = std::fs::File::create(&log).with_context(|| format!("creating {}", log.display()))?;

    // Sampled after eviction, so it reflects the desktop alone.
    // `ok()` rather than a fallback: a reading that failed is unknown, and recording a
    // guess as though it were measured is what makes the next launch budget wrongly.
    let desktop_mib = crate::vram_used_mib().ok();

    let exe = crate::llama_server()?;
    let mut cmd = std::process::Command::new(&exe);
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
        .arg("--no-webui");
    if should_lock_weights() {
        cmd.args(["--load-mode", "mlock"]);
    }
    if stdio == Stdio::LogFile {
        cmd.stdout(out.try_clone()?).stderr(out);
    }
    if opts.reasoning_budget >= 0 {
        cmd.args(["--reasoning-budget", &opts.reasoning_budget.to_string()]);
    }
    if let Some(key) = &opts.api_key {
        cmd.args(["--api-key", key]);
    }

    let child = cmd
        .spawn()
        .with_context(|| format!("starting {}", exe.display()))?;
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
        reasoning: opts.reasoning.clone(),
    };

    match await_healthy(&instance) {
        Ok(()) => {
            record(&instance)?;
            Ok((instance, child))
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

/// Whether to ask the OS to keep the weights resident.
///
/// On unified memory this is a correction rather than an optimisation. The weights
/// are the same pages the GPU reads, so an idle server is a large, untouched
/// allocation and macOS compresses it like any other - and then the next prompt pays
/// to fault all of it back before it can answer. Measured on an M4 Max on 2026-09-11:
/// 13.4 tok/s on the first request after an idle spell against 97.3 tok/s once
/// resident, with 234 MB decompressed during a single three-second generation.
///
/// On a discrete card the host mapping is scratch the driver has already copied into
/// VRAM. Locking it there would pin the weights a second time, in RAM the machine
/// needs for everything else - fatal on the 8.5 GB box this also runs on.
///
/// A probe that fails answers no, because that is the choice that cannot make things
/// worse.
fn should_lock_weights() -> bool {
    crate::primary_device().is_ok_and(|d| d.backend.is_unified_memory())
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

    // Only meaningful where VRAM can be sampled cheaply; see `can_poll_vram`.
    let watchdog = crate::can_poll_vram();
    let ceiling = crate::primary_device()
        .map(|d| vram::Budget::for_device(&d).ceiling_mib)
        .unwrap_or(vram::CEILING_MIB);

    while Instant::now() < deadline {
        if !is_alive(instance.pid) {
            anyhow::bail!("llama-server exited during startup");
        }

        if watchdog
            && let Ok(used) = crate::vram_used_mib()
            && used >= ceiling
        {
            terminate(instance.pid).ok();
            anyhow::bail!(
                "aborted at {used} MiB of VRAM, over the {ceiling} MiB ceiling - \
                 killed it rather than let it starve the compositor"
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
    fn names_the_config_when_it_is_the_only_reason_a_request_is_refused() {
        let m = model("qwen3-14b", 8836, Some(qwen3_14b()), Some(40_960));
        let budget = Budget::new(900).capped_at(Some(8192));
        let opts = Options {
            context: Some(16_384),
            ..Options::default()
        };
        let err = plan_context(&m, &budget, &opts).unwrap_err();
        assert!(err.to_string().contains("max_context"), "unexpected: {err}");
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
            desktop_mib: Some(desktop_mib),
            reasoning: "auto".into(),
        }
    }

    /// State files written by an older build have no `reasoning` field, and must still
    /// deserialise - otherwise an upgrade makes `ailocal ps` report nothing running
    /// while llama-server is very much still holding the VRAM.
    #[test]
    fn a_state_file_without_reasoning_still_loads() {
        let older = r#"{"pid":1,"model":"m","path":"/tmp/m.gguf","host":"127.0.0.1",
                        "port":8080,"context":4096,"cache_type":"q8_0","desktop_mib":900}"#;
        let parsed: Instance = serde_json::from_str(older).unwrap();
        assert_eq!(parsed.reasoning, "?");
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
        let budget = Budget::new(running.desktop_mib.unwrap());
        let m = model("qwen3-14b", 8836, Some(qwen3_14b()), Some(40_960));
        assert!(plan_context(&m, &budget, &Options::default()).is_ok());

        // The same model against live usage while gemma4 holds 11317 MiB.
        let naive = Budget::new(11_317);
        assert!(plan_context(&m, &naive, &Options::default()).is_err());
    }

    /// State written before `desktop_mib` existed must still load, and must be
    /// distinguishable from a measurement that came back zero.
    #[test]
    fn older_state_files_without_a_baseline_still_parse() {
        let json = r#"{"pid":1,"model":"m","path":"/tmp/m.gguf","host":"127.0.0.1",
                       "port":8080,"context":4096,"cache_type":"q8_0"}"#;
        let i: Instance = serde_json::from_str(json).unwrap();
        assert_eq!(i.desktop_mib, None, "unrecorded must not read as zero");
    }

    /// A device with nothing else resident records a baseline of zero, and that is a
    /// measurement like any other.
    ///
    /// Treating it as "unknown" is what broke unified memory: the recorded baseline
    /// was discarded, the budget re-measured live with the model already loaded, and
    /// the model that was serving perfectly well was judged not to fit - so it vanished
    /// from the gateway's list and from every harness catalogue built off it.
    #[test]
    fn a_measured_baseline_of_zero_is_still_a_measurement() {
        let json = r#"{"pid":1,"model":"m","path":"/tmp/m.gguf","host":"127.0.0.1",
                       "port":8080,"context":4096,"cache_type":"q8_0","desktop_mib":0}"#;
        let i: Instance = serde_json::from_str(json).unwrap();
        assert_eq!(i.desktop_mib, Some(0));

        // The whole device is available to the next launch, not none of it.
        let budget = Budget {
            ceiling_mib: 20_000,
            desktop_mib: i.desktop_mib.unwrap(),
            context_cap: None,
        };
        let m = model("qwen3-14b", 8836, Some(qwen3_14b()), Some(40_960));
        assert!(
            plan_context(&m, &budget, &Options::default()).is_ok(),
            "a model that is already serving must still be judged to fit"
        );
    }
}
