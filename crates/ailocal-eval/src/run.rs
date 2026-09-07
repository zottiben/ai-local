//! Driving one arm: load a model in a known state, ask every task, score the answers.
//!
//! Requests go straight to llama-server rather than through the gateway. The gateway is
//! a thin authenticated proxy that `ailocal gateway check` already verifies, and putting
//! it in the path would add its behaviour - model swapping in particular - to a
//! measurement that is about the model.
//!
//! No system prompt is sent. Each task states its own requirements, so the score
//! reflects the model rather than a prompt someone tuned, and a later run cannot be
//! made to look better by editing a preamble.

use std::time::Instant;

use ailocal::config::Config;
use ailocal::{registry, serve, service};

use crate::check;
use crate::report::{Arm, CheckResult, CorpusRef, Report, TaskResult};
use crate::task::{Loaded, Task};

/// Sampling temperature. Zero because a benchmark that moves between runs cannot
/// settle anything.
const TEMPERATURE: f64 = 0.0;

/// Fixed seed, recorded in the report. llama.cpp still samples at temperature zero, so
/// pinning the seed is what makes two runs of the same arm land in the same place.
pub const DEFAULT_SEED: i64 = 1_337;

/// Ceiling on one request. Long, because a cold model plus a reasoning arm plus a
/// thousand tokens is minutes, and cutting that off would score a slow answer as no
/// answer.
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(900);

/// What to measure, and how.
pub struct Plan {
    pub model: String,
    pub reasoning: String,
    pub seed: i64,
    pub allow_exec: bool,
    /// Token budget for every task, overriding the corpus.
    ///
    /// Exists so that "reasoning is slower" can be told apart from "reasoning ran out
    /// of room". Those are different findings and the default budget cannot
    /// distinguish them.
    pub max_tokens: Option<u32>,
}

/// Temporary ownership of the model server.
///
/// `reasoning` is a llama-server launch flag, so measuring two modes means restarting
/// the model - and if a service manager is supervising it, killing llama-server makes
/// the supervisor's `serve --foreground` exit and get restarted, which evicts the model
/// the eval just loaded. So the unit is paused for the duration and started again
/// afterwards, including when a run fails part way.
///
/// Taken **before** deciding whether anything needs reloading, not after. Deciding
/// first and stopping second is a race, and it is not theoretical: `systemctl start`
/// returns as soon as a `Type=simple` unit has forked, long before its model has
/// loaded, so a second eval run started straight after a first one read a state file
/// describing a server that was seconds from being replaced, judged that nothing needed
/// reloading, and then failed every task with a connection error.
///
/// The cost is one model load per run even when the resident model was already right.
/// Against a run measured in minutes that is noise, and it buys the invariant the whole
/// measurement rests on: for the duration, nothing else decides what is loaded.
///
/// The unit is deliberately not left down - coming back from an eval to a machine with
/// no model would be a nasty surprise, and the gateway keeps serving throughout.
pub struct ModelServiceGuard {
    resume: bool,
}

impl ModelServiceGuard {
    /// Pause the model unit if it is running, otherwise do nothing.
    ///
    /// # Errors
    /// If the unit is active but cannot be stopped, since proceeding would then race
    /// the supervisor for the GPU.
    pub fn take() -> anyhow::Result<Self> {
        if !service::is_active(service::MODEL_UNIT) {
            return Ok(Self { resume: false });
        }
        eprintln!(
            "pausing {} for the run; it will be started again afterwards",
            service::MODEL_UNIT
        );
        service::stop(service::MODEL_UNIT)?;
        Ok(Self { resume: true })
    }
}

impl Drop for ModelServiceGuard {
    fn drop(&mut self) {
        if !self.resume {
            return;
        }
        // Starting the unit relaunches the configured model, which also evicts whatever
        // the last arm left resident - that is the point, it restores the machine to
        // the state the eval found it in.
        eprintln!("\nrestarting {}", service::MODEL_UNIT);
        if let Err(e) = service::start(service::MODEL_UNIT) {
            eprintln!(
                "WARNING: could not restart {}: {e}\n\
                 Start it yourself with `systemctl --user start {}`",
                service::MODEL_UNIT,
                service::MODEL_UNIT
            );
        }
    }
}

/// Whether a recorded server is actually answering.
///
/// [`serve::running`] proves only that the process exists. That is not the same as
/// serving: a supervisor restarting the model replaces the process within a second or
/// two, and a record can be live while the port behind it is already dead. Without this
/// the run reuses a corpse and every single task fails with a connection error, which
/// reads as a broken harness rather than as one server that needed reloading.
fn is_answering(instance: &serve::Instance) -> bool {
    reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .and_then(|c| c.get(format!("{}/health", instance.base_url())).send())
        .is_ok_and(|r| r.status().is_success())
}

/// Make `model` the resident server, in `reasoning` mode.
///
/// Reuses what is already loaded when it is already right *and* still answering. A
/// reload costs tens of seconds and evicts the model the gateway is serving, so it is
/// worth not doing twice - but reasoning is a launch-time flag, so an arm that wants a
/// different mode has no choice.
///
/// # Errors
/// If the model is not installed, or cannot be loaded within the VRAM budget.
pub fn ensure_loaded(config: &Config, plan: &Plan) -> anyhow::Result<serve::Instance> {
    if let Some(current) = serve::running()?
        && current.model == plan.model
        && current.reasoning == plan.reasoning
        && is_answering(&current)
    {
        return Ok(current);
    }

    let models = registry::scan(&config.models_dir)?;
    let model = models
        .iter()
        .find(|m| m.name == plan.model)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no model named {:?}; `ailocal model ls` shows what is installed",
                plan.model
            )
        })?;

    let options = serve::Options {
        reasoning: plan.reasoning.clone(),
        ..serve::Options::from_config(config)?
    };
    let budget = serve::budget_for_next_launch()?;

    eprintln!(
        "loading {} with reasoning {} ...",
        plan.model, plan.reasoning
    );
    serve::start(model, &budget, &options)
}

/// Run every task in `corpus` against a loaded model, printing progress.
///
/// # Errors
/// If the model cannot be loaded. A failing individual request is recorded on the task
/// rather than aborting the arm - one bad response should not cost the other thirteen.
pub fn arm(
    config: &Config,
    plan: &Plan,
    corpus: &Loaded,
    tasks: &[&Task],
) -> anyhow::Result<Report> {
    let instance = ensure_loaded(config, plan)?;

    let client = reqwest::blocking::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()?;
    let endpoint = format!("{}/v1/chat/completions", instance.base_url());

    let scratch = crate::scratch_dir(config);
    std::fs::create_dir_all(&scratch)
        .map_err(|e| anyhow::anyhow!("creating {}: {e}", scratch.display()))?;
    let ctx = check::Context {
        scratch,
        allow_exec: plan.allow_exec,
    };

    let started = crate::report::now_epoch();
    let mut results = Vec::with_capacity(tasks.len());

    for (n, task) in tasks.iter().enumerate() {
        eprint!("  [{:>2}/{}] {:<28} ", n + 1, tasks.len(), task.id);
        let result = one(&client, &endpoint, plan, task, &ctx);
        eprintln!("{}", verdict_line(&result));
        results.push(result);
    }

    Ok(Report {
        schema: crate::report::SCHEMA,
        id: run_id(started, &plan.model, &plan.reasoning),
        started_at: crate::report::utc(started),
        harness_version: env!("CARGO_PKG_VERSION").to_owned(),
        arm: Arm {
            model: instance.model.clone(),
            reasoning: plan.reasoning.clone(),
            context: instance.context,
            cache_type: instance.cache_type.clone(),
            temperature: TEMPERATURE,
            seed: plan.seed,
            exec: plan.allow_exec,
            max_tokens: plan.max_tokens,
        },
        corpus: CorpusRef {
            origin: corpus.origin.clone(),
            digest: corpus.digest.clone(),
            splits: {
                let mut s: Vec<String> = tasks.iter().map(|t| t.split.to_string()).collect();
                s.sort();
                s.dedup();
                s
            },
        },
        tasks: results,
    })
}

/// Ask one task and score the answer.
fn one(
    client: &reqwest::blocking::Client,
    endpoint: &str,
    plan: &Plan,
    task: &Task,
    ctx: &check::Context,
) -> TaskResult {
    let body = serde_json::json!({
        "model": plan.model,
        "messages": [{ "role": "user", "content": task.prompt }],
        "temperature": TEMPERATURE,
        "seed": plan.seed,
        "max_tokens": plan.max_tokens.unwrap_or(task.max_tokens),
        "stream": false,
    });

    let began = Instant::now();
    let answered = client
        .post(endpoint)
        .json(&body)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .and_then(reqwest::blocking::Response::json::<serde_json::Value>);
    let seconds = began.elapsed().as_secs_f64();

    let blank = |error: Option<String>| TaskResult {
        id: task.id.clone(),
        category: task.category.clone(),
        split: task.split,
        score: None,
        passed: None,
        checks: Vec::new(),
        answer: String::new(),
        reasoning: String::new(),
        finish_reason: String::new(),
        prompt_tokens: 0,
        completion_tokens: 0,
        seconds,
        error,
    };

    let value = match answered {
        Ok(v) => v,
        // A transport or HTTP failure is the harness's problem, not the model's, so it
        // is recorded as unjudgeable rather than scored zero.
        Err(e) => return blank(Some(format!("request failed: {e}"))),
    };

    let choice = &value["choices"][0];
    let answer = choice["message"]["content"].as_str().unwrap_or_default();
    // llama.cpp puts chain-of-thought here and leaves `content` empty until it is done.
    // Scoring it would credit a model for thinking rather than answering.
    let thinking = choice["message"]["reasoning_content"]
        .as_str()
        .unwrap_or_default();

    let checks: Vec<CheckResult> = task
        .checks
        .iter()
        .map(|c| CheckResult {
            label: c.label().to_owned(),
            weight: c.weight,
            outcome: c.evaluate(answer, ctx),
        })
        .collect();

    let (score, passed) = tally(&checks);

    TaskResult {
        id: task.id.clone(),
        category: task.category.clone(),
        split: task.split,
        score,
        passed,
        checks,
        answer: answer.to_owned(),
        reasoning: thinking.to_owned(),
        finish_reason: choice["finish_reason"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        prompt_tokens: value["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
        completion_tokens: value["usage"]["completion_tokens"].as_u64().unwrap_or(0),
        seconds,
        error: None,
    }
}

/// Weighted score over the checks that could actually be judged.
///
/// Skipped checks leave the denominator rather than counting as failures - a machine
/// without `rustc` should report that it could not judge, not that the model is bad.
fn tally(checks: &[CheckResult]) -> (Option<f64>, Option<bool>) {
    let judged: Vec<&CheckResult> = checks.iter().filter(|c| !c.outcome.skipped()).collect();
    if judged.is_empty() {
        return (None, None);
    }

    let total: f64 = judged.iter().map(|c| c.weight).sum();
    if total <= 0.0 {
        return (None, None);
    }
    let earned: f64 = judged
        .iter()
        .filter(|c| c.outcome.passed())
        .map(|c| c.weight)
        .sum();

    (
        Some(crate::report::unsigned_zero(earned / total)),
        Some(judged.iter().all(|c| c.outcome.passed())),
    )
}

/// The one-line progress note printed as each task finishes.
fn verdict_line(result: &TaskResult) -> String {
    if let Some(error) = &result.error {
        return format!("ERROR  {error}");
    }
    if result.thought_itself_out_of_an_answer() {
        return format!(
            "EMPTY  spent {} chars thinking and ran out of budget ({:.0}s)",
            result.reasoning.chars().count(),
            result.seconds
        );
    }
    let score = result
        .score
        .map_or_else(|| " n/a".to_owned(), |s| format!("{:>3.0}%", s * 100.0));
    format!("{score}  {:>5.1}s", result.seconds)
}

/// A sortable, filesystem-safe name for a run.
///
/// The timestamp leads and is fixed width, because `eval ls` sorts by filename and a
/// variable-width prefix would put the runs in the wrong order.
fn run_id(epoch: u64, model: &str, reasoning: &str) -> String {
    let safe: String = model
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{}-{safe}-{reasoning}", crate::report::utc_compact(epoch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::{Check, Outcome};

    fn result(outcomes: Vec<(f64, Outcome)>) -> Vec<CheckResult> {
        outcomes
            .into_iter()
            .map(|(weight, outcome)| CheckResult {
                label: "x".to_owned(),
                weight,
                outcome,
            })
            .collect()
    }

    fn fail() -> Outcome {
        Outcome::Fail {
            why: "no".to_owned(),
        }
    }

    fn skip() -> Outcome {
        Outcome::Skip {
            why: "no rustc".to_owned(),
        }
    }

    #[test]
    fn all_checks_passing_is_a_full_score() {
        let (score, passed) = tally(&result(vec![(1.0, Outcome::Pass), (1.0, Outcome::Pass)]));
        assert_eq!(score, Some(1.0));
        assert_eq!(passed, Some(true));
    }

    #[test]
    fn partial_credit_is_weighted() {
        // Compiling is worth three times as much as mentioning a keyword, so a model
        // that compiles but misses the keyword should score well above half.
        let (score, passed) = tally(&result(vec![(3.0, Outcome::Pass), (1.0, fail())]));
        assert_eq!(score, Some(0.75));
        assert_eq!(passed, Some(false));
    }

    #[test]
    fn skipped_checks_leave_the_denominator() {
        let (score, passed) = tally(&result(vec![(1.0, Outcome::Pass), (1.0, skip())]));
        assert_eq!(score, Some(1.0), "a skip must not dilute a perfect score");
        assert_eq!(passed, Some(true));
    }

    /// std sums f64 from -0.0, so a task where nothing passed must not come back as
    /// negative zero and render as "-0%".
    #[test]
    fn a_task_where_nothing_passed_scores_positive_zero() {
        let (score, passed) = tally(&result(vec![(1.0, fail()), (2.0, fail())]));
        assert_eq!(score, Some(0.0));
        assert!(
            score.unwrap().is_sign_positive(),
            "got negative zero, which renders as -0%"
        );
        assert_eq!(passed, Some(false));
    }

    #[test]
    fn a_task_that_could_not_be_judged_at_all_has_no_score() {
        let (score, passed) = tally(&result(vec![(1.0, skip())]));
        assert_eq!(score, None);
        assert_eq!(passed, None);
        assert_eq!(tally(&[]), (None, None));
    }

    #[test]
    fn zero_weights_cannot_divide_by_zero() {
        assert_eq!(tally(&result(vec![(0.0, Outcome::Pass)])), (None, None));
    }

    /// `eval ls` sorts by filename, so the timestamp has to lead and be fixed width.
    #[test]
    fn run_ids_sort_chronologically() {
        let early = run_id(0, "gemma4-12b-Q4_K_M", "off");
        let late = run_id(1_000_000_000, "gemma4-12b-Q4_K_M", "off");
        assert!(early < late, "{early} should sort before {late}");
    }

    /// Underscores are left alone - `Q4_K_M` is part of how a model is named, and
    /// mangling it would make a run id unreadable.
    #[test]
    fn a_real_model_name_survives_intact() {
        assert_eq!(
            run_id(0, "gemma4-12b-Q4_K_M", "off"),
            "19700101T000000Z-gemma4-12b-Q4_K_M-off"
        );
    }

    #[test]
    fn model_names_with_awkward_characters_are_flattened() {
        assert_eq!(
            run_id(0, "owner/model:tag", "on"),
            "19700101T000000Z-owner_model_tag-on"
        );
    }

    /// A check list is scored against `content` only. If a model puts its answer in
    /// `reasoning_content`, a harness never sees it and neither does the score.
    #[test]
    fn only_content_is_scored() {
        let check: Check = toml::from_str("kind = \"contains\"\nvalue = \"answer\"").unwrap();
        let ctx = check::Context {
            scratch: std::env::temp_dir().join("ailocal-eval-tests"),
            allow_exec: false,
        };
        assert!(!check.evaluate("", &ctx).passed());
    }
}
