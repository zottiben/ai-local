//! `ailocal eval agent` - a coding agent in a repository, scored by what it pushed.
//!
//! The single-answer suite cannot see the habits that decide whether a local model is
//! usable in a harness: formatting what it changed, running the tests before it pushes,
//! reaching for `jq` rather than Python. Those only exist across a whole session, so
//! this runs one - Pi, in a fresh copy of a small crate with a bare origin - and scores
//! two things afterwards. What CI would see on origin, from the repository; and whether
//! the agent checked its work on the way there, from Pi's event stream.
//!
//! Every request goes through a gateway hosted in this process, because the gateway is
//! part of what a harness talks to: it translates Pi's thinking fields, and anything
//! it adds to a prompt is part of the measurement. Pi gets an agent directory of its
//! own holding only our provider, so nothing from the user's own setup - credentials,
//! extensions, instructions, skills - leaks into a run.
//!
//! The repositories live under the system temp directory rather than the data dir.
//! Pi loads `AGENTS.md` from every parent of its working directory, and the data dir
//! defaults to one under `$HOME`, where a personal `AGENTS.md` would become part of
//! every prompt. They are a few kilobytes each and removed after scoring; the build
//! output, which is not small, goes to the eval directory through `CARGO_TARGET_DIR`.

use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::time::{Duration, Instant};

use ailocal::config::Config;
use ailocal::{gateway, harness};
use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use crate::check::Outcome;
use crate::report::CheckResult;
use crate::transcript::{self, Transcript};

/// Bumped when a field changes meaning.
pub const SCHEMA: u32 = 1;

/// The built-in agent suite.
pub const TASKS: &str = include_str!("../tasks/agent.toml");

/// The repository every task starts from, compiled in so a run needs no checkout.
const FIXTURE: [(&str, &str); 8] = [
    (".gitignore", include_str!("../fixtures/slugkit/.gitignore")),
    ("AGENTS.md", include_str!("../fixtures/slugkit/AGENTS.md")),
    ("Cargo.lock", include_str!("../fixtures/slugkit/Cargo.lock")),
    ("Cargo.toml", include_str!("../fixtures/slugkit/Cargo.toml")),
    (
        "releases.json",
        include_str!("../fixtures/slugkit/releases.json"),
    ),
    (
        "rustfmt.toml",
        include_str!("../fixtures/slugkit/rustfmt.toml"),
    ),
    ("src/lib.rs", include_str!("../fixtures/slugkit/src/lib.rs")),
    (
        "tests/slugify.rs",
        include_str!("../fixtures/slugkit/tests/slugify.rs"),
    ),
];

/// Where a task's acceptance test is put in the clone that is scored.
const ACCEPT_TEST: &str = "eval_acceptance";

/// The levels Pi's `--thinking` takes. Pi clamps each to what the model can do.
pub const THINKING: [&str; 7] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

/// The files Pi reads instructions from in its working directory and every parent.
const CONTEXT_FILES: [&str; 5] = [
    "AGENTS.override.md",
    "AGENTS.md",
    "AGENTS.MD",
    "CLAUDE.md",
    "CLAUDE.MD",
];

/// Ceiling on each of the scoring builds. The crate has no dependencies, so a minute
/// is already generous; five covers a cold cache on a slow disk.
const CHECK_TIMEOUT: Duration = Duration::from_secs(300);

// Check labels, in the order a report shows them.
const DONE: &str = "done";
const PUSHED: &str = "pushed";
const TESTS: &str = "tests";
const FMT: &str = "fmt";
const CLIPPY: &str = "clippy";
const TESTED_FIRST: &str = "tested-first";
const NO_PYTHON_JSON: &str = "no-python-json";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskFile {
    #[serde(rename = "task")]
    tasks: Vec<Task>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Task {
    pub id: String,
    pub prompt: String,
    /// An integration test that passes once the task is done.
    pub accept: String,
    /// Commits made on top of the fixture before the agent starts, in order.
    #[serde(default)]
    pub before: Vec<Commit>,
}

/// A commit a task starts from.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Commit {
    pub message: String,
    /// Made in another clone and pushed, as a teammate's would be, so the agent's copy
    /// does not have it until it fetches.
    #[serde(default)]
    pub pushed: bool,
    /// Each file's path and its whole new contents.
    pub files: std::collections::BTreeMap<String, String>,
}

/// The built-in tasks.
///
/// # Errors
/// If the task file does not parse or two tasks share an id - both mistakes in this
/// crate, caught by its tests.
pub fn tasks() -> anyhow::Result<Vec<Task>> {
    let file: TaskFile = toml::from_str(TASKS).context("parsing the agent task file")?;
    anyhow::ensure!(
        !file.tasks.is_empty(),
        "the agent task file defines no tasks"
    );
    let mut seen = std::collections::HashSet::new();
    for t in &file.tasks {
        anyhow::ensure!(
            seen.insert(t.id.as_str()),
            "two tasks share the id {:?}",
            t.id
        );
    }
    Ok(file.tasks)
}

/// Short hex digest of the tasks and the fixture together.
///
/// Either one changing changes what a score means, so both are in it.
#[must_use]
pub fn digest() -> String {
    use sha2::Digest as _;

    let mut hash = sha2::Sha256::new();
    hash.update(TASKS.as_bytes());
    for (path, text) in FIXTURE {
        hash.update(path.as_bytes());
        hash.update([0]);
        hash.update(text.as_bytes());
        hash.update([0]);
    }
    short_hex(&hash.finalize())
}

/// Short hex digest of the gateway's built-in agent rules.
///
/// The text ships with the binary and will be tuned, so a report records which version
/// it scored rather than only that there was one.
#[must_use]
pub fn rules_digest() -> String {
    use sha2::Digest as _;

    short_hex(&sha2::Sha256::digest(gateway::AGENT_RULES.as_bytes()))
}

/// Short hex digest of the Pi extension, recorded for the same reason.
#[must_use]
pub fn extension_digest() -> String {
    use sha2::Digest as _;

    short_hex(&sha2::Sha256::digest(harness::PI_EXTENSION.as_bytes()))
}

fn short_hex(hash: &[u8]) -> String {
    hash.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

// --- reports -------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct Report {
    pub schema: u32,
    pub id: String,
    pub started_at: String,
    pub harness_version: String,
    pub pi_version: String,
    pub arm: Arm,
    /// [`digest`] of the suite this was scored against.
    pub suite: String,
    pub runs: Vec<Run>,
}

/// The conditions an agent was run under.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Arm {
    pub model: String,
    /// The level passed to Pi's `--thinking`.
    pub thinking: String,
    /// The window Pi was told the model has.
    pub context: u64,
    /// The server's default sampling, when it had the publisher's; `None` is
    /// llama.cpp's own defaults. Pi sends none of its own.
    pub sampling: Option<String>,
    /// [`rules_digest`] of the agent rules the gateway added, or `None` when it added
    /// none - as for every report written before there were any.
    #[serde(default)]
    pub rules: Option<String>,
    /// [`extension_digest`] of the Pi extension Pi loaded, or `None` when it loaded
    /// none - as for every report written before there was one.
    #[serde(default)]
    pub extension: Option<String>,
}

impl Arm {
    #[must_use]
    pub fn label(&self) -> String {
        format!(
            "{} (thinking {}, rules {}, extension {})",
            self.model,
            self.thinking,
            on_off(self.rules.is_some()),
            on_off(self.extension.is_some())
        )
    }
}

fn on_off(on: bool) -> &'static str {
    if on { "on" } else { "off" }
}

/// One task, attempted once.
#[derive(Debug, Serialize, Deserialize)]
pub struct Run {
    pub task: String,
    pub rep: u32,
    pub seconds: f64,
    pub checks: Vec<CheckResult>,
    pub transcript: Transcript,
    /// Set when the harness, not the agent, cut the run short.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Run {
    fn outcome(&self, label: &str) -> Option<&Outcome> {
        self.checks
            .iter()
            .find(|c| c.label == label)
            .map(|c| &c.outcome)
    }

    /// Whether CI on origin would be green: something was pushed and all three of the
    /// repository's own checks pass on it.
    #[must_use]
    pub fn ci_green(&self) -> bool {
        [PUSHED, TESTS, FMT, CLIPPY]
            .iter()
            .all(|label| self.outcome(label).is_some_and(Outcome::passed))
    }

    /// What went wrong first, for a one-line summary.
    #[must_use]
    pub fn why(&self) -> String {
        if let Some(e) = self.error.as_ref().or(self.transcript.error.as_ref()) {
            return e.clone();
        }
        self.checks
            .iter()
            .find_map(|c| match &c.outcome {
                Outcome::Fail { why } => Some(format!("{}: {why}", c.label)),
                _ => None,
            })
            .unwrap_or_default()
    }
}

/// Where agent reports and transcripts are kept.
#[must_use]
pub fn runs_dir(eval_dir: &Path) -> PathBuf {
    eval_dir.join("agent")
}

impl Report {
    /// Write the report out beside its transcripts, returning where it went.
    ///
    /// # Errors
    /// If the directory cannot be created or the file cannot be written.
    pub fn save(&self, eval_dir: &Path) -> anyhow::Result<PathBuf> {
        let dir = runs_dir(eval_dir);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = dir.join(format!("{}.json", self.id));
        std::fs::write(&path, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(path)
    }

    /// Fraction of the runs where `label` passed, over the runs it could be judged on.
    fn rate(&self, label: &str) -> Option<f64> {
        let judged: Vec<&Outcome> = self
            .runs
            .iter()
            .filter_map(|r| r.outcome(label))
            .filter(|o| !o.skipped())
            .collect();
        (!judged.is_empty())
            .then(|| judged.iter().filter(|o| o.passed()).count() as f64 / judged.len() as f64)
    }
}

/// One line per arm.
#[must_use]
pub fn summary_table(reports: &[&Report]) -> String {
    let width = reports
        .iter()
        .map(|r| r.arm.label().len())
        .max()
        .unwrap_or(20)
        .max(3);
    let pct = |rate: Option<f64>| {
        rate.map_or_else(
            || "-".to_owned(),
            |r| format!("{:.0}%", crate::report::unsigned_zero(r) * 100.0),
        )
    };

    let mut out = format!(
        "{:<width$}  {:>4} {:>5} {:>5} {:>6} {:>5} {:>6} {:>5} {:>6} {:>4} {:>7}\n",
        "ARM", "RUNS", "DONE", "CI", "TESTED", "FMT", "CLIPPY", "TESTS", "NO-PY", "ERR", "MIN/RUN"
    );
    for r in reports {
        let runs = r.runs.len();
        let green = r.runs.iter().filter(|run| run.ci_green()).count();
        let errors = r
            .runs
            .iter()
            .filter(|run| run.error.is_some() || run.transcript.error.is_some())
            .count();
        let minutes = r.runs.iter().map(|run| run.seconds).sum::<f64>() / 60.0;
        out.push_str(&format!(
            "{:<width$}  {:>4} {:>5} {:>5} {:>6} {:>5} {:>6} {:>5} {:>6} {:>4} {:>7.1}\n",
            r.arm.label(),
            runs,
            pct(r.rate(DONE)),
            pct((runs > 0).then(|| green as f64 / runs as f64)),
            pct(r.rate(TESTED_FIRST)),
            pct(r.rate(FMT)),
            pct(r.rate(CLIPPY)),
            pct(r.rate(TESTS)),
            pct(r.rate(NO_PYTHON_JSON)),
            errors,
            if runs > 0 { minutes / runs as f64 } else { 0.0 },
        ));
    }
    out.push_str(
        "\nDONE and CI are over every run; a run that pushed nothing is neither. TESTED, \
         FMT, CLIPPY\nand TESTS are over the runs that pushed. TESTED is a test run after \
         the last edit and\nbefore the push.\n",
    );
    out
}

/// One line per run, for a single arm.
#[must_use]
pub fn detail_table(report: &Report) -> String {
    let width = report
        .runs
        .iter()
        .map(|r| r.task.len() + 3)
        .max()
        .unwrap_or(16)
        .max(4);
    let mark = |run: &Run, label: &str| match run.outcome(label) {
        Some(Outcome::Pass) => "ok",
        Some(Outcome::Fail { .. }) => "x",
        Some(Outcome::Skip { .. }) | None => "-",
    };

    let mut out = format!(
        "{:<width$}  {:>4} {:>3} {:>6} {:>3} {:>6} {:>5} {:>5} {:>6} {:>5}  WHY\n",
        "RUN", "DONE", "CI", "TESTED", "FMT", "CLIPPY", "TESTS", "NO-PY", "CALLS", "SECS"
    );
    for run in &report.runs {
        out.push_str(&format!(
            "{:<width$}  {:>4} {:>3} {:>6} {:>3} {:>6} {:>5} {:>5} {:>6} {:>5.0}  {}\n",
            format!("{} #{}", run.task, run.rep),
            mark(run, DONE),
            if run.ci_green() { "ok" } else { "x" },
            mark(run, TESTED_FIRST),
            mark(run, FMT),
            mark(run, CLIPPY),
            mark(run, TESTS),
            mark(run, NO_PYTHON_JSON),
            run.transcript.calls.len(),
            run.seconds,
            run.why(),
        ));
    }
    out
}

// --- running -------------------------------------------------------------------------

/// What one invocation measures.
pub struct Plan {
    pub models: Vec<String>,
    /// Whether the gateway adds its agent rules, per arm.
    pub rules: Vec<bool>,
    /// Whether Pi loads our extension, per arm.
    pub extension: Vec<bool>,
    pub thinking: Vec<String>,
    pub tasks: Vec<Task>,
    pub reps: u32,
    pub timeout: Duration,
}

/// The tools a run needs, found once up front.
struct Tools {
    pi: PathBuf,
    pi_version: String,
    /// Without `jq`, Python is the reasonable way to read JSON and is not held against
    /// the agent.
    jq: bool,
}

impl Tools {
    fn find() -> anyhow::Result<Self> {
        let pi = find_pi()?;
        let pi_version = version(Command::new(&pi).arg("--version"))
            .with_context(|| format!("running {} --version", pi.display()))?;

        version(Command::new("git").arg("--version"))
            .context("git is needed to give the agent a repository and an origin")?;
        for component in ["fmt", "clippy"] {
            version(Command::new("cargo").args([component, "--version"])).with_context(|| {
                format!(
                    "`cargo {component}` is needed: the fixture is a Rust crate and its CI \
                     runs it. Install it with `rustup component add {}`",
                    if component == "fmt" {
                        "rustfmt"
                    } else {
                        "clippy"
                    }
                )
            })?;
        }

        Ok(Self {
            pi,
            pi_version,
            jq: version(Command::new("jq").arg("--version")).is_ok(),
        })
    }
}

/// The first line a `--version` prints.
fn version(command: &mut Command) -> anyhow::Result<String> {
    let out = command.stdin(Stdio::null()).output()?;
    anyhow::ensure!(out.status.success(), "exited with {}", out.status);
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .trim()
        .to_owned())
}

/// Pi, on `PATH` or where its installer puts it.
fn find_pi() -> anyhow::Result<PathBuf> {
    let on_path = std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
        .unwrap_or_default();
    let managed = std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".pi/agent/bin"));

    on_path
        .into_iter()
        .chain(managed)
        .map(|dir| dir.join("pi"))
        .find(|p| p.is_file())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "pi is not on PATH or in ~/.pi/agent/bin; install it from https://pi.dev"
            )
        })
}

/// Instruction files Pi would load into every run from above `dir`.
fn inherited_context(dir: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = dir
        .ancestors()
        .flat_map(|d| CONTEXT_FILES.map(|name| d.join(name)))
        .filter(|p| p.is_file())
        .filter_map(|p| p.canonicalize().ok())
        .collect();
    // Case-insensitive filesystems answer to both spellings of one file.
    found.sort();
    found.dedup();
    found
}

/// A gateway serving this run alone, on a port of its own.
///
/// It lives as long as the runtime it was started on.
struct Gateway {
    base_url: String,
    key: String,
}

impl Gateway {
    fn start(runtime: &tokio::runtime::Runtime, config: Config) -> anyhow::Result<Self> {
        let listener = runtime.block_on(tokio::net::TcpListener::bind((
            std::net::Ipv4Addr::LOCALHOST,
            0,
        )))?;
        let address = listener.local_addr()?;
        let key = ailocal::auth::generate()?;

        let state = Arc::new(gateway::AppState {
            key: key.clone(),
            config: ailocal::config::Live::fixed(config),
            client: reqwest::Client::builder().build()?,
            request_log: None,
        });
        runtime.spawn(async move { axum::serve(listener, gateway::router(state)).await });

        Ok(Self {
            base_url: format!("http://{address}"),
            key,
        })
    }
}

/// The three files Pi needs to offer our provider, as `harness configure pi` writes
/// them but holding nothing else.
fn pi_files(
    base_url: &str,
    key: &str,
    model: &harness::PiModel,
) -> [(&'static str, serde_json::Value); 3] {
    let models = std::slice::from_ref(model);
    let ours = |value: serde_json::Value| {
        serde_json::Value::Object(
            [(harness::PI_PROVIDER_ID.to_owned(), value)]
                .into_iter()
                .collect(),
        )
    };
    [
        ("auth.json", ours(harness::pi_credential(base_url, key))),
        (
            "models-store.json",
            ours(harness::pi_catalogue(models, base_url)),
        ),
        (
            "models.json",
            serde_json::json!({
                "providers": ours(serde_json::json!({
                    "modelOverrides": harness::pi_model_overrides(models),
                })),
            }),
        ),
    ]
}

/// Give Pi an agent directory holding only our provider, and our extension if asked.
fn write_pi_dir(
    dir: &Path,
    gateway: &Gateway,
    model: &harness::PiModel,
    extension: bool,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    for (name, value) in pi_files(&gateway.base_url, &gateway.key, model) {
        std::fs::write(dir.join(name), serde_json::to_string_pretty(&value)?)?;
    }
    if extension {
        let extensions = dir.join("extensions");
        std::fs::create_dir_all(&extensions)?;
        std::fs::write(extensions.join("ailocal.ts"), harness::PI_EXTENSION)?;
    }
    Ok(())
}

/// A throwaway directory, removed when dropped.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

/// Run every arm in `plan`, printing progress, and return a report per arm.
///
/// # Errors
/// If a tool is missing, the model cannot be loaded, or a repository cannot be set up.
/// A run the agent fails is a result, not an error.
pub fn run(config: &Config, plan: &Plan) -> anyhow::Result<Vec<Report>> {
    let tools = Tools::find()?;

    let scratch = Scratch(
        std::env::temp_dir()
            .canonicalize()?
            .join(format!("ailocal-eval-agent-{}", std::process::id())),
    );
    let inherited = inherited_context(&scratch.0);
    anyhow::ensure!(
        inherited.is_empty(),
        "Pi would load these into every run, because they sit above {}:\n  {}\n\
         Move them, or point TMPDIR somewhere without them.",
        scratch.0.display(),
        inherited
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join("\n  ")
    );
    std::fs::create_dir_all(&scratch.0)?;

    let target_dir = runs_dir(&config.eval_dir).join("target");
    std::fs::create_dir_all(&target_dir)?;

    eprintln!(
        "This runs pi {} with your privileges in throwaway repositories under {}.\n\
         Its shell commands are not sandboxed.\n",
        tools.pi_version,
        scratch.0.display()
    );
    eprintln!(
        "{} task(s) x {} rep(s), {} arm(s), suite {}{}\n",
        plan.tasks.len(),
        plan.reps,
        plan.models.len() * plan.rules.len() * plan.extension.len() * plan.thinking.len(),
        digest(),
        if tools.jq {
            ""
        } else {
            "; jq is not installed, so Python on JSON is not scored"
        }
    );

    // Taken only when a model has to be loaded. Pi sends its thinking level with every
    // request, so the launch flag that forces `eval run` to reload is irrelevant here,
    // and a model that is already resident and answering is used as it is - with no
    // service stopped and started around it.
    let mut guard = None;

    // Serves the gateways, and stops a running agent on Ctrl-C. One handler only: a
    // second would find the first one's interrupt and exit without cleaning up.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.spawn(stop_on_interrupt());

    // One gateway per rules setting, since the gateway is what adds them. The user's
    // own house rule is theirs, not the model's, and a score that moved when they
    // edited it would be measuring the rule.
    let gateways = plan
        .rules
        .iter()
        .map(|&rules| {
            let gateway = Gateway::start(
                &runtime,
                Config {
                    system_prompt: None,
                    agent_rules: rules,
                    ..config.clone()
                },
            )?;
            Ok((rules, gateway))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let runnable = ailocal::status::runnable_models(config)?;
    let mut reports = Vec::new();
    for model in &plan.models {
        let offered = runnable.iter().find(|m| &m.id == model).ok_or_else(|| {
            anyhow::anyhow!(
                "{model:?} is not a model that can run here; `ailocal model ls` shows \
                 what is installed and what fits"
            )
        })?;
        let resident =
            ailocal::serve::running()?.filter(|i| &i.model == model && crate::run::is_answering(i));
        let instance = match resident {
            Some(instance) => instance,
            None => {
                if guard.is_none() {
                    guard = Some(crate::run::ModelServiceGuard::take()?);
                }
                crate::run::ensure_loaded(config, model, &config.reasoning, None)?
            }
        };

        for (rules, gateway) in &gateways {
            for &extension in &plan.extension {
                let pi_dir = scratch.0.join(format!(
                    "pi-{}-rules-{}-extension-{}",
                    slug(model),
                    on_off(*rules),
                    on_off(extension)
                ));
                write_pi_dir(&pi_dir, gateway, offered, extension)?;

                for thinking in &plan.thinking {
                    let arm = Arm {
                        model: model.clone(),
                        thinking: thinking.clone(),
                        context: offered.context_window,
                        sampling: instance.sampling.as_ref().map(ToString::to_string),
                        rules: rules.then(rules_digest),
                        extension: extension.then(extension_digest),
                    };
                    eprintln!("== {}", arm.label());
                    let report =
                        run_arm(config, plan, &tools, &scratch.0, &pi_dir, &target_dir, arm)?;
                    let path = report.save(&config.eval_dir)?;
                    eprintln!("   saved {}\n", path.display());
                    reports.push(report);
                }
            }
        }
    }
    Ok(reports)
}

/// Directories and tools shared by every run in an arm.
struct Bench<'a> {
    tools: &'a Tools,
    pi_dir: &'a Path,
    target_dir: &'a Path,
    transcripts: PathBuf,
    timeout: Duration,
}

fn run_arm(
    config: &Config,
    plan: &Plan,
    tools: &Tools,
    scratch: &Path,
    pi_dir: &Path,
    target_dir: &Path,
    arm: Arm,
) -> anyhow::Result<Report> {
    let started = crate::report::now_epoch();
    let id = format!(
        "{}-{}-thinking-{}-rules-{}-extension-{}",
        crate::report::utc_compact(started),
        slug(&arm.model),
        arm.thinking,
        on_off(arm.rules.is_some()),
        on_off(arm.extension.is_some())
    );
    let bench = Bench {
        tools,
        pi_dir,
        target_dir,
        transcripts: runs_dir(&config.eval_dir).join(&id),
        timeout: plan.timeout,
    };

    let total = plan.tasks.len() * plan.reps as usize;
    let mut runs = Vec::with_capacity(total);
    // Rep-major, so a run cut short still has every task once before any twice.
    for rep in 1..=plan.reps {
        for task in &plan.tasks {
            eprint!(
                "  [{:>2}/{total}] {:<20} ",
                runs.len() + 1,
                format!("{} #{rep}", task.id)
            );
            let dir = scratch.join(format!("{}-{rep}", task.id));
            let run = one(&bench, &arm, task, rep, &dir)?;
            std::fs::remove_dir_all(&dir).ok();
            let why = run.why();
            eprintln!(
                "{:>4.0}s  {}",
                run.seconds,
                if why.is_empty() {
                    "every check passed"
                } else {
                    &why
                }
            );
            runs.push(run);
        }
    }

    Ok(Report {
        schema: SCHEMA,
        id,
        started_at: crate::report::utc(started),
        harness_version: env!("CARGO_PKG_VERSION").to_owned(),
        pi_version: tools.pi_version.clone(),
        arm,
        suite: digest(),
        runs,
    })
}

/// A working copy and the bare origin it pushes to.
struct Repo {
    work: PathBuf,
    origin: PathBuf,
    /// Origin's `main` as the agent found it.
    initial: String,
}

impl Repo {
    fn create(dir: &Path, before: &[Commit]) -> anyhow::Result<Self> {
        let work = dir.join("work");
        let origin = dir.join("origin.git");
        // Hooks from the user's global config would run on the agent's commits.
        let no_hooks = dir.join("no-hooks");
        std::fs::create_dir_all(&no_hooks)?;

        write_files(&work, FIXTURE)?;
        git(&work, &["init", "-q", "-b", "main"])?;
        configure(&work, "slugkit", &no_hooks)?;
        git(&work, &["add", "-A"])?;
        git(&work, &["commit", "-q", "-m", "Initial commit"])?;

        git(dir, &["init", "-q", "--bare", "-b", "main", "origin.git"])?;
        git(
            &work,
            &["remote", "add", "origin", &origin.to_string_lossy()],
        )?;
        git(&work, &["push", "-q", "-u", "origin", "main"])?;

        let teammate = dir.join("teammate");
        for commit in before {
            let at = if commit.pushed {
                if !teammate.exists() {
                    git(dir, &["clone", "-q", &origin.to_string_lossy(), "teammate"])?;
                    configure(&teammate, "teammate", &no_hooks)?;
                }
                &teammate
            } else {
                &work
            };
            write_files(
                at,
                commit.files.iter().map(|(p, t)| (p.as_str(), t.as_str())),
            )?;
            git(at, &["add", "-A"])?;
            git(at, &["commit", "-q", "-m", &commit.message])?;
            if commit.pushed {
                git(at, &["push", "-q", "origin", "main"])?;
            }
        }
        let initial = git(&origin, &["rev-parse", "main"])?;

        Ok(Self {
            work,
            origin,
            initial,
        })
    }
}

fn write_files<'a>(
    dir: &Path,
    files: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> anyhow::Result<()> {
    for (path, text) in files {
        let file = dir.join(path);
        if let Some(parent) = file.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&file, text)?;
    }
    Ok(())
}

/// Who commits in `dir`, and nothing from the user's global config that would get in
/// the way.
fn configure(dir: &Path, name: &str, no_hooks: &Path) -> anyhow::Result<()> {
    for (key, value) in [
        ("user.name", name),
        ("user.email", &format!("{name}@slugkit.invalid")),
        // A signing prompt would hang the agent's commit until the timeout.
        ("commit.gpgsign", "false"),
        ("tag.gpgsign", "false"),
        ("core.hooksPath", &no_hooks.to_string_lossy()),
    ] {
        git(dir, &["config", key, value])?;
    }
    Ok(())
}

fn git(dir: &Path, args: &[&str]) -> anyhow::Result<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .context("running git")?;
    anyhow::ensure!(
        out.status.success(),
        "git {} in {}: {}",
        args.join(" "),
        dir.display(),
        String::from_utf8_lossy(&out.stderr).trim()
    );
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

/// Give the agent one task in a fresh repository, then score what it left.
fn one(bench: &Bench<'_>, arm: &Arm, task: &Task, rep: u32, dir: &Path) -> anyhow::Result<Run> {
    let repo = Repo::create(dir, &task.before)?;
    let events_path = dir.join("events.jsonl");

    let mut pi = Command::new(&bench.tools.pi);
    pi.current_dir(&repo.work)
        .args(["--mode", "json", "--no-session"])
        // `~/.agents/skills` is read whatever the agent directory, and differs from
        // one machine to the next.
        .arg("--no-skills")
        .args(["--provider", harness::PI_PROVIDER_ID, "--model", &arm.model])
        .args(["--thinking", &arm.thinking])
        .arg(task.prompt.trim())
        .env("PI_CODING_AGENT_DIR", bench.pi_dir)
        .env("PI_OFFLINE", "1")
        .env("PI_SKIP_VERSION_CHECK", "1")
        .env("CARGO_TARGET_DIR", bench.target_dir)
        .stdin(Stdio::null())
        .stdout(std::fs::File::create(&events_path)?)
        .stderr(std::fs::File::create(dir.join("pi-stderr.txt"))?);

    let began = Instant::now();
    let ended = run_bounded(&mut pi, bench.timeout)?;
    let seconds = began.elapsed().as_secs_f64();

    let events = std::fs::read_to_string(&events_path).unwrap_or_default();
    save_transcript(
        &events,
        &bench.transcripts.join(format!("{}-{rep}.jsonl", task.id)),
    )?;
    let transcript = transcript::parse(&events);

    let error = match ended {
        None => Some(format!(
            "stopped after {}s, the timeout",
            bench.timeout.as_secs()
        )),
        Some(status) if !status.success() && transcript.calls.is_empty() => Some(format!(
            "pi exited with {status}: {}",
            last_line(&std::fs::read_to_string(dir.join("pi-stderr.txt")).unwrap_or_default())
        )),
        Some(_) => None,
    };

    let checks = score(bench, &repo, task, &transcript, dir)?;
    Ok(Run {
        task: task.id.clone(),
        rep,
        seconds,
        checks,
        transcript,
        error,
    })
}

/// Keep the events a reader needs. The streaming deltas are a line per token, most of
/// the file, and `message_end` carries the whole message they add up to.
fn save_transcript(events: &str, path: &Path) -> anyhow::Result<()> {
    let kept: String = events
        .lines()
        .filter(|line| {
            serde_json::from_str::<serde_json::Value>(line).is_ok_and(|e| {
                !matches!(
                    e["type"].as_str(),
                    Some("message_update" | "tool_execution_update")
                )
            })
        })
        .map(|line| format!("{line}\n"))
        .collect();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    std::fs::write(path, kept).with_context(|| format!("writing {}", path.display()))
}

/// Score a finished run from origin and from what the agent did.
fn score(
    bench: &Bench<'_>,
    repo: &Repo,
    task: &Task,
    transcript: &Transcript,
    dir: &Path,
) -> anyhow::Result<Vec<CheckResult>> {
    let head = git(&repo.origin, &["rev-parse", "main"])?;
    let pushed = head != repo.initial;

    let (done, tests, fmt, clippy) = if pushed {
        let clone = dir.join("check");
        git(
            dir,
            &["clone", "-q", &repo.origin.to_string_lossy(), "check"],
        )?;
        let cargo = |args: &[&str]| cargo(&clone, bench.target_dir, args);

        let tests = cargo(&["test"])?;
        let fmt = cargo(&["fmt", "--check"])?;
        let clippy = cargo(&["clippy", "--all-targets", "--", "-D", "warnings"])?;

        let accept = clone.join("tests").join(format!("{ACCEPT_TEST}.rs"));
        std::fs::create_dir_all(clone.join("tests"))?;
        std::fs::write(&accept, &task.accept)?;
        let done = cargo(&["test", "--test", ACCEPT_TEST])?;
        (done, tests, fmt, clippy)
    } else {
        let not = || Outcome::Skip {
            why: "nothing was pushed".to_owned(),
        };
        (
            Outcome::Fail {
                why: "nothing was pushed".to_owned(),
            },
            not(),
            not(),
            not(),
        )
    };

    let tested_first = match transcript::tested_before_push(&transcript.calls) {
        None => Outcome::Skip {
            why: "no push succeeded".to_owned(),
        },
        Some(true) => Outcome::Pass,
        Some(false) => Outcome::Fail {
            why: "pushed with no test run since its last edit".to_owned(),
        },
    };

    let no_python_json = if bench.tools.jq {
        transcript
            .calls
            .iter()
            .find(|c| !c.held && c.python_on_json())
            .map_or(Outcome::Pass, |c| Outcome::Fail {
                why: format!(
                    "ran `{}`",
                    one_line(&c.target.replace(&repo.work.display().to_string(), "."))
                ),
            })
    } else {
        Outcome::Skip {
            why: "jq is not installed".to_owned(),
        }
    };

    Ok([
        (DONE, done),
        (
            PUSHED,
            if pushed {
                Outcome::Pass
            } else {
                Outcome::Fail {
                    why: "origin/main never moved".to_owned(),
                }
            },
        ),
        (TESTS, tests),
        (FMT, fmt),
        (CLIPPY, clippy),
        (TESTED_FIRST, tested_first),
        (NO_PYTHON_JSON, no_python_json),
    ]
    .into_iter()
    .map(|(label, outcome)| CheckResult {
        label: label.to_owned(),
        weight: 1.0,
        outcome,
    })
    .collect())
}

/// Run a cargo command in `dir` and judge it by its exit status.
fn cargo(dir: &Path, target_dir: &Path, args: &[&str]) -> anyhow::Result<Outcome> {
    let log = dir.join(".cargo-output");
    let file = std::fs::File::create(&log)?;
    let mut command = Command::new("cargo");
    command
        .current_dir(dir)
        .args(args)
        .env("CARGO_TARGET_DIR", target_dir)
        .env("CARGO_TERM_COLOR", "never")
        .stdin(Stdio::null())
        .stdout(file.try_clone()?)
        .stderr(file);

    let ended = run_bounded(&mut command, CHECK_TIMEOUT)?;
    let output = std::fs::read_to_string(&log).unwrap_or_default();
    std::fs::remove_file(&log).ok();

    Ok(match ended {
        Some(status) if status.success() => Outcome::Pass,
        Some(_) => Outcome::Fail {
            why: first_problem(&output, dir),
        },
        None => Outcome::Fail {
            why: format!(
                "`cargo {}` ran past {}s",
                args.join(" "),
                CHECK_TIMEOUT.as_secs()
            ),
        },
    })
}

/// The line of a failed cargo run that says what failed, with the clone's path taken
/// out so the report reads the same on every machine.
fn first_problem(output: &str, dir: &Path) -> String {
    let prefix = format!("{}/", dir.display());
    output
        .lines()
        .find(|l| l.starts_with("error") || l.starts_with("Diff in") || l.ends_with("FAILED"))
        .map_or_else(|| last_line(output), str::to_owned)
        .replace(&prefix, "")
        .trim()
        .to_owned()
}

fn last_line(text: &str) -> String {
    text.lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .unwrap_or_default()
        .trim()
        .to_owned()
}

/// A command as one short line: a `python3 -c` script is mostly newlines.
fn one_line(text: &str) -> String {
    let line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match line.char_indices().nth(100) {
        Some((cut, _)) => format!("{}...", &line[..cut]),
        None => line,
    }
}

/// Run `command` in a process group of its own, stopping the whole group when it
/// exits or when `limit` passes. `None` means it was stopped.
///
/// The group matters for the agent: whatever its shell commands left running - a
/// watcher, a server it started to test against - goes with it rather than outliving
/// the run.
fn run_bounded(command: &mut Command, limit: Duration) -> anyhow::Result<Option<ExitStatus>> {
    let mut child = command.process_group(0).spawn()?;
    let group = rustix::process::Pid::from_child(&child);
    RUNNING_GROUP.store(group.as_raw_nonzero().get(), Ordering::SeqCst);
    let deadline = Instant::now() + limit;

    let status = loop {
        if let Some(status) = child.try_wait()? {
            break Some(status);
        }
        if Instant::now() >= deadline {
            break None;
        }
        std::thread::sleep(Duration::from_millis(100));
    };

    stop_group(group);
    RUNNING_GROUP.store(0, Ordering::SeqCst);
    if status.is_none() {
        child.wait().ok();
    }
    // A run stopped by Ctrl-C ends like a finished one from here, and scoring it would
    // record "nothing was pushed" against the model.
    anyhow::ensure!(!INTERRUPTED.load(Ordering::SeqCst), "interrupted");
    Ok(status)
}

/// The process group [`run_bounded`] is waiting on, or 0.
///
/// A group of its own is what lets a timeout stop everything the agent started, and
/// also what keeps Ctrl-C from reaching it: the terminal signals its foreground group,
/// which this no longer belongs to. So the interrupt handler stops it from here.
static RUNNING_GROUP: AtomicI32 = AtomicI32::new(0);

/// Set by the first Ctrl-C.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

fn stop_group(group: rustix::process::Pid) {
    if rustix::process::kill_process_group(group, rustix::process::Signal::TERM).is_ok() {
        std::thread::sleep(Duration::from_millis(500));
        rustix::process::kill_process_group(group, rustix::process::Signal::KILL).ok();
    }
}

/// Stop whatever is running on Ctrl-C.
///
/// Stopping the group is enough: [`run_bounded`] then fails, and the error unwinds
/// through everything that cleans up - the scratch directory, a paused model service.
/// With nothing running there is nothing to unwind from, and a second Ctrl-C is
/// someone who does not want to wait for it.
async fn stop_on_interrupt() {
    while tokio::signal::ctrl_c().await.is_ok() {
        let group = rustix::process::Pid::from_raw(RUNNING_GROUP.load(Ordering::SeqCst));
        match group {
            Some(group) if !INTERRUPTED.swap(true, Ordering::SeqCst) => {
                eprintln!("\ninterrupted; stopping the agent and cleaning up");
                stop_group(group);
            }
            _ => std::process::exit(130),
        }
    }
}

/// A model name made safe for a filename.
fn slug(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The suite ships inside the binary, so a mistake in it is a broken build rather
    /// than a broken run an hour in.
    #[test]
    fn the_builtin_tasks_parse() {
        let all = tasks().unwrap();
        assert!(all.len() >= 4);
        assert!(all.iter().all(|t| !t.prompt.trim().is_empty()));
        assert!(all.iter().all(|t| t.accept.contains("#[test]")));
    }

    /// Every prompt has to ask for the push, or "nothing was pushed" scores a task
    /// that never asked for one.
    #[test]
    fn every_task_ends_in_a_push() {
        for t in tasks().unwrap() {
            assert!(t.prompt.contains("push"), "{} never asks for a push", t.id);
        }
    }

    #[test]
    fn the_digest_covers_the_fixture() {
        assert_eq!(digest().len(), 16);
        assert_eq!(digest(), digest());
    }

    /// The release task's answer is only meaningful if the trap is still there: the
    /// newest release overall must be yanked, and the expected version the newest one
    /// that is not.
    #[test]
    fn the_release_fixture_still_sets_its_trap() {
        let (_, text) = FIXTURE.iter().find(|(p, _)| *p == "releases.json").unwrap();
        let releases: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        let newest = |list: &mut dyn Iterator<Item = &serde_json::Value>| {
            list.max_by_key(|r| r["published"].as_str().unwrap().to_owned())
                .map(|r| r["version"].as_str().unwrap().to_owned())
        };

        let overall = releases
            .iter()
            .max_by_key(|r| r["published"].as_str().unwrap().to_owned())
            .unwrap();
        assert_eq!(overall["yanked"], true);
        let standing = newest(&mut releases.iter().filter(|r| r["yanked"] == false)).unwrap();
        let task = tasks()
            .unwrap()
            .into_iter()
            .find(|t| t.id == "release-version")
            .unwrap();
        assert!(
            task.accept.contains(&format!("\"{standing}\"")),
            "{standing} is not what the task accepts"
        );

        // Longer than one read returns, or the agent never needs a query.
        assert!(text.lines().count() > 2000);
    }

    /// Every task has to start green, both in the agent's copy and on origin, or every
    /// run inherits a failure it did not cause - and no acceptance test may pass before
    /// its task is done, or `done` scores nothing. Runs cargo, so it takes a while.
    #[test]
    fn every_task_starts_green_and_none_starts_done() {
        let root =
            std::env::temp_dir().join(format!("ailocal-eval-fixture-{}", std::process::id()));
        let target = root.join("target");

        let mut fixture_checked = false;
        for task in tasks().unwrap() {
            let dir = root.join(&task.id);
            let repo = Repo::create(&dir, &task.before).unwrap();
            let mut copies = Vec::new();
            // Every task with no commits of its own starts from the same fixture.
            if !task.before.is_empty() || !std::mem::replace(&mut fixture_checked, true) {
                copies.push(repo.work.clone());
            }
            if task.before.iter().any(|c| c.pushed) {
                git(
                    &dir,
                    &["clone", "-q", &repo.origin.to_string_lossy(), "origin"],
                )
                .unwrap();
                copies.push(dir.join("origin"));
            }
            for copy in &copies {
                for args in [
                    &["test"][..],
                    &["fmt", "--check"],
                    &["clippy", "--all-targets", "--", "-D", "warnings"],
                ] {
                    assert_eq!(
                        cargo(copy, &target, args).unwrap(),
                        Outcome::Pass,
                        "{}: cargo {args:?} in {}",
                        task.id,
                        copy.display()
                    );
                }
            }

            let accept = repo.work.join("tests").join(format!("{ACCEPT_TEST}.rs"));
            std::fs::write(accept, &task.accept).unwrap();
            let before = cargo(&repo.work, &target, &["test", "--test", ACCEPT_TEST]).unwrap();
            assert!(
                !before.passed(),
                "{} is accepted before it is done",
                task.id
            );
        }
        std::fs::remove_dir_all(&root).ok();
    }

    /// The rebase task measures what happens at `git rebase --continue`, which is only
    /// reached if the two commits really conflict.
    #[test]
    fn the_rebase_task_starts_with_a_conflict() {
        let root = std::env::temp_dir().join(format!("ailocal-eval-rebase-{}", std::process::id()));
        let task = tasks()
            .unwrap()
            .into_iter()
            .find(|t| t.id == "rebase")
            .unwrap();
        let repo = Repo::create(&root, &task.before).unwrap();

        assert_ne!(
            git(&repo.work, &["rev-parse", "origin/main"]).unwrap(),
            repo.initial,
            "the agent's copy already has the teammate's commit"
        );
        git(&repo.work, &["fetch", "-q"]).unwrap();
        let rebase = git(&repo.work, &["rebase", "origin/main"]);
        assert!(rebase.is_err_and(|e| e.to_string().contains("could not apply")));
        std::fs::remove_dir_all(&root).ok();
    }

    /// The one thing the test above cannot see: that the apostrophe task still
    /// contradicts an existing test the agent has to find.
    #[test]
    fn the_apostrophe_trap_is_in_the_existing_tests() {
        let (_, tests) = FIXTURE
            .iter()
            .find(|(p, _)| *p == "tests/slugify.rs")
            .unwrap();
        assert!(tests.contains("rock-n-roll"));
        let (_, lib) = FIXTURE.iter().find(|(p, _)| *p == "src/lib.rs").unwrap();
        assert!(
            !lib.contains("#[test]"),
            "tests in lib.rs would show the trap on first read"
        );
    }

    #[test]
    fn pi_gets_exactly_our_provider_and_nothing_else() {
        let model = harness::PiModel {
            id: "m".to_owned(),
            context_window: 65536,
            thinking: ailocal::gguf::Thinking::Toggle,
        };
        let files = pi_files("http://127.0.0.1:9/v1", "k", &model);
        let names: Vec<&str> = files.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, ["auth.json", "models-store.json", "models.json"]);

        let (_, auth) = &files[0];
        assert_eq!(auth.as_object().unwrap().len(), 1);
        assert_eq!(auth["llama.cpp"]["key"], "k");
        assert_eq!(
            auth["llama.cpp"]["env"]["LLAMA_BASE_URL"],
            "http://127.0.0.1:9"
        );

        let (_, store) = &files[1];
        assert_eq!(store["llama.cpp"]["models"][0]["id"], "m");

        let (_, overrides) = &files[2];
        assert_eq!(
            overrides["providers"]["llama.cpp"]["modelOverrides"]["m"]["thinkingLevelMap"]["off"],
            "none"
        );
    }

    #[test]
    fn the_extension_is_in_the_agent_directory_only_when_asked_for() {
        let root = std::env::temp_dir().join(format!("ailocal-eval-pi-dir-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        let gateway = Gateway {
            base_url: "http://127.0.0.1:9/v1".to_owned(),
            key: "k".to_owned(),
        };
        let model = harness::PiModel {
            id: "m".to_owned(),
            context_window: 65536,
            thinking: ailocal::gguf::Thinking::Toggle,
        };

        write_pi_dir(&root.join("on"), &gateway, &model, true).unwrap();
        let installed = std::fs::read_to_string(root.join("on/extensions/ailocal.ts")).unwrap();
        assert_eq!(installed, harness::PI_EXTENSION);

        write_pi_dir(&root.join("off"), &gateway, &model, false).unwrap();
        assert!(root.join("off/auth.json").is_file());
        assert!(!root.join("off/extensions").exists());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn instruction_files_above_a_directory_are_found() {
        let root = std::env::temp_dir()
            .canonicalize()
            .unwrap()
            .join(format!("ailocal-eval-context-{}", std::process::id()));
        let deep = root.join("a/b");
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::write(root.join("AGENTS.md"), "rules").unwrap();

        let found = inherited_context(&deep);
        assert!(found.contains(&root.join("AGENTS.md")), "{found:?}");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_failing_check_names_the_line_that_failed() {
        let dir = Path::new("/tmp/run/check");
        let output = "   Compiling slugkit\nrunning 5 tests\n\
                      test apostrophes_split_words ... FAILED\nfailures:\n";
        assert_eq!(
            first_problem(output, dir),
            "test apostrophes_split_words ... FAILED"
        );
        assert_eq!(
            first_problem("Diff in /tmp/run/check/src/lib.rs:12:\n", dir),
            "Diff in src/lib.rs:12:"
        );
        assert_eq!(first_problem("something odd\n\n", dir), "something odd");
    }

    #[test]
    fn a_bounded_run_is_stopped_at_its_limit() {
        let mut sleep = Command::new("sleep");
        sleep.arg("30");
        let began = Instant::now();
        assert!(
            run_bounded(&mut sleep, Duration::from_millis(200))
                .unwrap()
                .is_none()
        );
        assert!(began.elapsed() < Duration::from_secs(5));

        let mut ok = Command::new("true");
        assert!(
            run_bounded(&mut ok, Duration::from_secs(5))
                .unwrap()
                .is_some_and(|s| s.success())
        );
    }

    fn run_with(outcomes: &[(&str, Outcome)]) -> Run {
        Run {
            task: "t".to_owned(),
            rep: 1,
            seconds: 60.0,
            checks: outcomes
                .iter()
                .map(|(label, outcome)| CheckResult {
                    label: (*label).to_owned(),
                    weight: 1.0,
                    outcome: outcome.clone(),
                })
                .collect(),
            transcript: Transcript::default(),
            error: None,
        }
    }

    fn fail() -> Outcome {
        Outcome::Fail {
            why: "no".to_owned(),
        }
    }

    fn skip() -> Outcome {
        Outcome::Skip {
            why: "n/a".to_owned(),
        }
    }

    #[test]
    fn ci_is_green_only_when_all_three_checks_pass_on_a_push() {
        let green = run_with(&[
            (PUSHED, Outcome::Pass),
            (TESTS, Outcome::Pass),
            (FMT, Outcome::Pass),
            (CLIPPY, Outcome::Pass),
        ]);
        assert!(green.ci_green());

        let unformatted = run_with(&[
            (PUSHED, Outcome::Pass),
            (TESTS, Outcome::Pass),
            (FMT, fail()),
            (CLIPPY, Outcome::Pass),
        ]);
        assert!(!unformatted.ci_green());

        let nothing = run_with(&[
            (PUSHED, fail()),
            (TESTS, skip()),
            (FMT, skip()),
            (CLIPPY, skip()),
        ]);
        assert!(
            !nothing.ci_green(),
            "skipped checks must not make an empty push green"
        );
    }

    #[test]
    fn rates_leave_out_the_runs_a_check_could_not_judge() {
        let report = Report {
            schema: SCHEMA,
            id: "r".to_owned(),
            started_at: crate::report::utc(0),
            harness_version: "test".to_owned(),
            pi_version: "pi 0".to_owned(),
            arm: Arm {
                model: "m".to_owned(),
                thinking: "off".to_owned(),
                context: 65536,
                sampling: None,
                rules: None,
                extension: None,
            },
            suite: digest(),
            runs: vec![
                run_with(&[(FMT, Outcome::Pass)]),
                run_with(&[(FMT, fail())]),
                run_with(&[(FMT, skip())]),
            ],
        };
        assert_eq!(report.rate(FMT), Some(0.5));
        assert_eq!(report.rate(CLIPPY), None);

        let table = summary_table(&[&report]);
        assert!(
            table.contains("m (thinking off, rules off, extension off)"),
            "{table}"
        );
        assert!(table.contains("50%"), "{table}");
    }

    #[test]
    fn a_command_is_reported_on_one_short_line() {
        assert_eq!(
            one_line("python3 -c \"\nimport json\nprint(1)\n\""),
            "python3 -c \" import json print(1) \""
        );
        assert!(one_line(&"x".repeat(300)).ends_with("..."));
    }

    #[test]
    fn model_names_become_safe_filenames() {
        assert_eq!(slug("owner/model:tag"), "owner_model_tag");
        assert_eq!(slug("qwen3.6-35b-a3b"), "qwen3.6-35b-a3b");
    }
}
