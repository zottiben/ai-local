//! Run records: what was measured, under exactly which conditions, and how it scored.
//!
//! A number is only useful next to the conditions that produced it, so a report carries
//! the model, the reasoning mode, the context, the KV quantisation, the sampling
//! settings and a hash of the task file. Two reports that disagree on any of those are
//! not a comparison, and `compare` says so rather than printing a difference that means
//! nothing.
//!
//! Reports are JSON on disk because they are meant to be re-read - by `compare`, by a
//! later session, and eventually by the training slices deciding whether an adapter
//! earned its place.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::check::Outcome;
use crate::task::Split;

/// Bumped when a field changes meaning, so an old report is refused rather than
/// silently misread into a comparison.
pub const SCHEMA: u32 = 2;

/// Turn `-0.0` into `0.0`.
///
/// std sums `f64` starting from `-0.0`, which is the right additive identity for
/// floats but means a task where nothing passed scores negative zero - and renders as
/// `-0%`, which reads as a bug in the harness rather than as a zero.
#[must_use]
pub fn unsigned_zero(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub schema: u32,
    /// Filename stem, and how a run is named on the command line.
    pub id: String,
    pub started_at: String,
    /// Version of the harness that produced this, since scoring could change.
    pub harness_version: String,
    pub arm: Arm,
    pub corpus: CorpusRef,
    pub tasks: Vec<TaskResult>,
}

/// The conditions a run was measured under.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Arm {
    pub model: String,
    /// `auto`, `on` or `off`, as passed to llama-server.
    pub reasoning: String,
    pub context: u64,
    pub cache_type: String,
    pub temperature: f64,
    pub seed: i64,
    /// Whether compiled test binaries were allowed to run.
    pub exec: bool,
    /// Token budget override, when the per-task one was not used.
    ///
    /// Recorded because it is the single setting most able to manufacture a result: a
    /// reasoning model given too small a budget returns nothing at all, which reads as
    /// incompetence rather than as a budget that was too small.
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

impl Arm {
    /// The part of an arm that must match for two runs to be comparable.
    ///
    /// Context and KV type are excluded: they are chosen by the VRAM budget rather than
    /// by the operator, and refusing to compare a 256k model with a 50k one would rule
    /// out the comparison the budget exists to inform.
    fn comparable_to(&self, other: &Self) -> Vec<String> {
        let mut differences = Vec::new();
        if self.temperature != other.temperature {
            differences.push(format!(
                "temperature {} vs {}",
                self.temperature, other.temperature
            ));
        }
        if self.seed != other.seed {
            differences.push(format!("seed {} vs {}", self.seed, other.seed));
        }
        if self.exec != other.exec {
            differences.push(format!("exec {} vs {}", self.exec, other.exec));
        }
        if self.max_tokens != other.max_tokens {
            let show = |m: Option<u32>| m.map_or_else(|| "per-task".to_owned(), |n| n.to_string());
            differences.push(format!(
                "max_tokens {} vs {}",
                show(self.max_tokens),
                show(other.max_tokens)
            ));
        }
        differences
    }

    /// How this arm reads in a table.
    #[must_use]
    pub fn label(&self) -> String {
        match self.max_tokens {
            None => format!("{} (reasoning {})", self.model, self.reasoning),
            Some(n) => format!("{} (reasoning {}, {n} tok)", self.model, self.reasoning),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusRef {
    pub origin: String,
    pub digest: String,
    pub splits: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub id: String,
    pub category: String,
    pub split: Split,

    /// Weighted fraction of the checks that could be judged. `None` when none could.
    pub score: Option<f64>,
    /// Whether every judgeable check passed. `None` when none could.
    pub passed: Option<bool>,

    pub checks: Vec<CheckResult>,

    /// What the model actually said.
    ///
    /// Kept in full. A report that says a task failed without showing the answer
    /// cannot be acted on - there is no way to tell a model that got it wrong from a
    /// task whose wording was ambiguous, and those need opposite responses.
    pub answer: String,
    /// What it thought first, which the client never sees.
    pub reasoning: String,
    /// `stop`, `length`, or whatever llama-server reported.
    pub finish_reason: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub seconds: f64,

    /// Set when the request itself failed, as distinct from the model answering badly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl TaskResult {
    /// An answer that never arrived because the budget went on thinking.
    ///
    /// The signature of the failure described in the project's own notes: content is
    /// empty, `finish_reason` is `length`, and the tokens went to `reasoning_content`.
    /// It looks to a harness like a broken model, so it is counted separately rather
    /// than folded into a low score.
    #[must_use]
    pub fn thought_itself_out_of_an_answer(&self) -> bool {
        self.answer.is_empty() && self.finish_reason == "length" && !self.reasoning.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub label: String,
    pub weight: f64,
    #[serde(flatten)]
    pub outcome: Outcome,
}

/// Aggregate numbers, derived rather than stored so an old report cannot disagree with
/// the way scores are computed today.
pub struct Summary {
    pub scored: usize,
    pub unjudgeable: usize,
    pub errors: usize,
    /// Mean task score over judgeable tasks, 0.0 to 1.0.
    pub score: f64,
    /// Fraction of judgeable tasks where every check passed.
    pub pass_rate: f64,
    pub empty_answers: usize,
    pub truncated: usize,
    pub thought_out: usize,
    pub seconds: f64,
    pub completion_tokens: u64,
}

impl Summary {
    #[must_use]
    pub fn of(report: &Report) -> Self {
        let judgeable: Vec<&TaskResult> =
            report.tasks.iter().filter(|t| t.score.is_some()).collect();
        let n = judgeable.len();

        let mean = |total: f64| {
            if n == 0 {
                0.0
            } else {
                unsigned_zero(total / n as f64)
            }
        };

        Self {
            scored: n,
            unjudgeable: report.tasks.len() - n,
            errors: report.tasks.iter().filter(|t| t.error.is_some()).count(),
            score: mean(judgeable.iter().filter_map(|t| t.score).sum()),
            pass_rate: mean(judgeable.iter().filter(|t| t.passed == Some(true)).count() as f64),
            empty_answers: report
                .tasks
                .iter()
                .filter(|t| t.answer.trim().is_empty())
                .count(),
            truncated: report
                .tasks
                .iter()
                .filter(|t| t.finish_reason == "length")
                .count(),
            thought_out: report
                .tasks
                .iter()
                .filter(|t| t.thought_itself_out_of_an_answer())
                .count(),
            seconds: report.tasks.iter().map(|t| t.seconds).sum(),
            completion_tokens: report.tasks.iter().map(|t| t.completion_tokens).sum(),
        }
    }

    /// Generated tokens per second across the whole run.
    #[must_use]
    pub fn tokens_per_second(&self) -> f64 {
        if self.seconds <= 0.0 {
            return 0.0;
        }
        self.completion_tokens as f64 / self.seconds
    }
}

// --- persistence ---------------------------------------------------------------------

/// Where runs are kept.
#[must_use]
pub fn runs_dir(eval_dir: &Path) -> PathBuf {
    eval_dir.join("runs")
}

impl Report {
    /// Write the report out, returning where it went.
    ///
    /// # Errors
    /// If the directory cannot be created or the file cannot be written.
    pub fn save(&self, eval_dir: &Path) -> anyhow::Result<PathBuf> {
        use anyhow::Context as _;

        let dir = runs_dir(eval_dir);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = dir.join(format!("{}.json", self.id));
        std::fs::write(&path, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(path)
    }

    /// Read a report from a path.
    ///
    /// # Errors
    /// If the file cannot be read, parsed, or was written by a different schema.
    pub fn read(path: &Path) -> anyhow::Result<Self> {
        use anyhow::Context as _;

        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let report: Self =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        anyhow::ensure!(
            report.schema == SCHEMA,
            "{} was written by schema {} but this build reads {SCHEMA}; re-run it",
            path.display(),
            report.schema
        );
        Ok(report)
    }
}

/// Every saved run, oldest first.
///
/// # Errors
/// If the directory exists but cannot be listed.
pub fn list(eval_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let dir = runs_dir(eval_dir);
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| anyhow::anyhow!("listing {}: {e}", dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    // Run ids begin with a sortable UTC timestamp, so this is chronological.
    found.sort();
    Ok(found)
}

/// Find a run by id, by filename, or by a substring that matches exactly one.
///
/// # Errors
/// If nothing matches, or more than one does.
pub fn resolve(eval_dir: &Path, needle: &str) -> anyhow::Result<Report> {
    // An explicit path wins, so a report copied somewhere else is still readable.
    let as_path = Path::new(needle);
    if as_path.is_file() {
        return Report::read(as_path);
    }

    let all = list(eval_dir)?;
    let stem = |p: &Path| {
        p.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    };

    if let Some(exact) = all.iter().find(|p| stem(p) == needle) {
        return Report::read(exact);
    }

    let matches: Vec<&PathBuf> = all.iter().filter(|p| stem(p).contains(needle)).collect();
    match matches.as_slice() {
        [one] => Report::read(one),
        [] => anyhow::bail!(
            "no run matching {needle:?} in {}; `ailocal eval ls` lists them",
            runs_dir(eval_dir).display()
        ),
        many => anyhow::bail!(
            "{needle:?} matches {} runs: {}",
            many.len(),
            many.iter().map(|p| stem(p)).collect::<Vec<_>>().join(", ")
        ),
    }
}

// --- rendering -------------------------------------------------------------------

/// One line per arm, which is the answer to "which of these is better".
#[must_use]
pub fn summary_table(reports: &[&Report]) -> String {
    let mut out = String::new();
    let width = reports
        .iter()
        .map(|r| r.arm.label().len())
        .max()
        .unwrap_or(20)
        .max(4);

    out.push_str(&format!(
        "{:<width$}  {:>6} {:>6} {:>7} {:>6} {:>7}\n",
        "ARM", "SCORE", "PASS", "EMPTY", "CUT", "TOK/S"
    ));
    for r in reports {
        let s = Summary::of(r);
        out.push_str(&format!(
            "{:<width$}  {:>5.0}% {:>5.0}% {:>7} {:>6} {:>7.1}\n",
            r.arm.label(),
            s.score * 100.0,
            s.pass_rate * 100.0,
            s.empty_answers,
            s.truncated,
            s.tokens_per_second(),
        ));
    }

    // Only worth explaining when it happened, and when it did it explains everything.
    if reports.iter().any(|r| Summary::of(r).thought_out > 0) {
        out.push_str(
            "\nEMPTY answers with CUT set are the reasoning failure: the model spent its\n\
             whole token budget in reasoning_content and never wrote an answer. To a\n\
             harness that is an empty reply, not a slow one.\n",
        );
    }

    // A request that never completed is a hole in the measurement, not a low score,
    // and a score computed over the rest should not be read without knowing about it.
    let failed: usize = reports.iter().map(|r| Summary::of(r).errors).sum();
    if failed > 0 {
        out.push_str(&format!(
            "\n{failed} request(s) failed outright and are excluded from these scores.\n\
             `ailocal eval show <run>` names them.\n"
        ));
    }
    out
}

/// Per-task detail for a single run.
#[must_use]
pub fn detail_table(report: &Report) -> String {
    let mut out = String::new();
    let width = report
        .tasks
        .iter()
        .map(|t| t.id.len())
        .max()
        .unwrap_or(16)
        .max(4);

    out.push_str(&format!(
        "{:<width$}  {:>6} {:>7} {:>6}  WHY\n",
        "TASK", "SCORE", "SECONDS", "TOKENS"
    ));
    for t in &report.tasks {
        let score = t
            .score
            .map_or_else(|| "  n/a".to_owned(), |s| format!("{:>5.0}%", s * 100.0));

        // The first thing that went wrong is what a person needs; the rest is noise.
        let why = t.error.clone().unwrap_or_else(|| {
            t.checks
                .iter()
                .find_map(|c| match &c.outcome {
                    Outcome::Fail { why } => Some(format!("{}: {why}", c.label)),
                    Outcome::Skip { why } => Some(format!("{} skipped: {why}", c.label)),
                    Outcome::Pass => None,
                })
                .unwrap_or_default()
        });

        out.push_str(&format!(
            "{:<width$}  {score} {:>7.1} {:>6}  {why}\n",
            t.id, t.seconds, t.completion_tokens,
        ));
    }
    out
}

/// Two runs side by side, joined on task id.
///
/// Reports the tasks that changed rather than every task: a diff of forty identical
/// rows hides the four that moved.
#[must_use]
pub fn comparison(a: &Report, b: &Report) -> String {
    let mut out = String::new();

    let differences = a.arm.comparable_to(&b.arm);
    if !differences.is_empty() {
        out.push_str(&format!(
            "WARNING: these runs were not measured the same way ({}).\n\
             The difference below is not attributable to the model.\n\n",
            differences.join(", ")
        ));
    }
    if a.corpus.digest != b.corpus.digest {
        out.push_str(&format!(
            "WARNING: different task sets ({} vs {}). Scores are not comparable.\n\n",
            a.corpus.digest, b.corpus.digest
        ));
    }

    out.push_str(&summary_table(&[a, b]));

    let (sa, sb) = (Summary::of(a), Summary::of(b));
    out.push_str(&format!(
        "\nB scores {:+.0} points, {:+.0} points of pass rate, and takes {:.1}x the wall clock.\n\n",
        (sb.score - sa.score) * 100.0,
        (sb.pass_rate - sa.pass_rate) * 100.0,
        if sa.seconds > 0.0 {
            sb.seconds / sa.seconds
        } else {
            0.0
        }
    ));

    let by_id: std::collections::HashMap<&str, &TaskResult> =
        b.tasks.iter().map(|t| (t.id.as_str(), t)).collect();

    let mut moved = Vec::new();
    for left in &a.tasks {
        let Some(right) = by_id.get(left.id.as_str()) else {
            continue;
        };
        if left.score != right.score {
            moved.push((left, *right));
        }
    }

    if moved.is_empty() {
        out.push_str("No task changed score.\n");
        return out;
    }

    let width = moved.iter().map(|(l, _)| l.id.len()).max().unwrap_or(16);
    out.push_str(&format!("{:<width$}  {:>7} {:>7}\n", "TASK", "A", "B"));
    for (left, right) in moved {
        let pct = |s: Option<f64>| {
            s.map_or_else(
                || "n/a".to_owned(),
                |v| format!("{:.0}%", unsigned_zero(v) * 100.0),
            )
        };
        out.push_str(&format!(
            "{:<width$}  {:>7} {:>7}\n",
            left.id,
            pct(left.score),
            pct(right.score)
        ));
    }
    out
}

// --- time ---------------------------------------------------------------------------

/// Seconds since the Unix epoch, right now.
#[must_use]
pub fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// `YYYY-MM-DDTHH:MM:SSZ` for epoch seconds.
///
/// Hand-rolled rather than pulling in a date library for one format string. UTC only,
/// which is what a machine-readable record wants anyway - a local timestamp is
/// ambiguous twice a year.
#[must_use]
pub fn utc(epoch: u64) -> String {
    let (y, m, d) = civil_from_days((epoch / 86_400) as i64);
    let rem = epoch % 86_400;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// `YYYYmmddTHHMMSSZ`, for use inside a filename.
#[must_use]
pub fn utc_compact(epoch: u64) -> String {
    utc(epoch).replace(['-', ':'], "")
}

/// Howard Hinnant's `civil_from_days`, the standard closed form. Correct for any date
/// in the proleptic Gregorian calendar, which is more than a run timestamp needs, but
/// it is no longer than an approximation would be.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(id: &str, score: Option<f64>) -> TaskResult {
        TaskResult {
            id: id.to_owned(),
            category: "c".to_owned(),
            split: Split::HeldOut,
            score,
            passed: score.map(|s| s >= 1.0),
            checks: Vec::new(),
            answer: "an answer".to_owned(),
            reasoning: String::new(),
            finish_reason: "stop".to_owned(),
            prompt_tokens: 5,
            completion_tokens: 20,
            seconds: 2.0,
            error: None,
        }
    }

    fn report(id: &str, scores: &[Option<f64>]) -> Report {
        Report {
            schema: SCHEMA,
            id: id.to_owned(),
            started_at: utc(0),
            harness_version: "test".to_owned(),
            arm: Arm {
                model: "m".to_owned(),
                reasoning: "off".to_owned(),
                context: 4096,
                cache_type: "q8_0".to_owned(),
                temperature: 0.0,
                seed: 1,
                exec: true,
                max_tokens: None,
            },
            corpus: CorpusRef {
                origin: "built-in".to_owned(),
                digest: "abcd".to_owned(),
                splits: vec!["held-out".to_owned()],
            },
            tasks: scores
                .iter()
                .enumerate()
                .map(|(i, s)| task(&format!("t{i}"), *s))
                .collect(),
        }
    }

    #[test]
    fn the_mean_ignores_tasks_that_could_not_be_judged() {
        let r = report("r", &[Some(1.0), Some(0.0), None]);
        let s = Summary::of(&r);
        assert_eq!(s.scored, 2);
        assert_eq!(s.unjudgeable, 1);
        assert!(
            (s.score - 0.5).abs() < 1e-9,
            "a skipped task must not count as zero, got {}",
            s.score
        );
    }

    #[test]
    fn an_empty_run_does_not_divide_by_zero() {
        let s = Summary::of(&report("r", &[]));
        assert_eq!(s.score, 0.0);
        assert_eq!(s.tokens_per_second(), 0.0);
    }

    #[test]
    fn pass_rate_counts_only_perfect_tasks() {
        let r = report("r", &[Some(1.0), Some(0.9), Some(1.0)]);
        let s = Summary::of(&r);
        assert!(
            (s.pass_rate - 2.0 / 3.0).abs() < 1e-9,
            "got {}",
            s.pass_rate
        );
    }

    /// The specific failure the project has already hit once, and the reason the
    /// summary reports EMPTY and CUT rather than only a score.
    #[test]
    fn an_answer_lost_to_thinking_is_identified() {
        let mut t = task("t", Some(0.0));
        t.answer = String::new();
        t.reasoning = "thinking about it at length".repeat(80);
        t.finish_reason = "length".to_owned();
        assert!(t.thought_itself_out_of_an_answer());

        // A merely truncated answer is a different thing and must not be counted here.
        t.answer = "pub fn f() {".to_owned();
        assert!(!t.thought_itself_out_of_an_answer());
    }

    #[test]
    fn a_comparison_warns_when_the_arms_were_measured_differently() {
        let a = report("a", &[Some(1.0)]);
        let mut b = report("b", &[Some(0.0)]);
        b.arm.seed = 99;
        let text = comparison(&a, &b);
        assert!(text.contains("not measured the same way"), "got:\n{text}");
        assert!(text.contains("seed 1 vs 99"), "got:\n{text}");
    }

    /// The budget is the setting most able to manufacture a result, so a comparison
    /// across two different ones has to say so.
    #[test]
    fn a_comparison_warns_when_the_token_budgets_differ() {
        let a = report("a", &[Some(0.0)]);
        let mut b = report("b", &[Some(1.0)]);
        b.arm.max_tokens = Some(4096);
        let text = comparison(&a, &b);
        assert!(text.contains("max_tokens per-task vs 4096"), "got:\n{text}");
    }

    #[test]
    fn an_arm_with_an_overridden_budget_says_so_in_its_label() {
        let mut a = report("a", &[Some(1.0)]);
        assert!(!a.arm.label().contains("tok"));
        a.arm.max_tokens = Some(4096);
        assert!(a.arm.label().contains("4096 tok"), "got {}", a.arm.label());
    }

    #[test]
    fn a_comparison_warns_when_the_task_sets_differ() {
        let a = report("a", &[Some(1.0)]);
        let mut b = report("b", &[Some(1.0)]);
        b.corpus.digest = "9999".to_owned();
        assert!(comparison(&a, &b).contains("different task sets"));
    }

    #[test]
    fn a_comparison_lists_only_the_tasks_that_moved() {
        let a = report("a", &[Some(1.0), Some(0.0)]);
        let b = report("b", &[Some(1.0), Some(1.0)]);
        let text = comparison(&a, &b);
        assert!(text.contains("t1"), "the changed task must appear:\n{text}");
        // t0 scored the same in both, so it is noise.
        let table = text.split("TASK").nth(1).unwrap_or("");
        assert!(
            !table.contains("t0"),
            "unchanged tasks must be omitted:\n{text}"
        );
    }

    #[test]
    fn identical_runs_compare_to_no_difference() {
        let a = report("a", &[Some(1.0)]);
        let b = report("b", &[Some(1.0)]);
        assert!(comparison(&a, &b).contains("No task changed score"));
    }

    /// A task where nothing passed is `0%`, not `-0%`.
    #[test]
    fn a_zero_score_renders_without_a_sign() {
        let a = report("a", &[Some(1.0)]);
        let mut b = report("b", &[Some(0.0)]);
        b.tasks[0].score = Some(-0.0);
        let text = comparison(&a, &b);
        assert!(!text.contains("-0%"), "negative zero leaked out:\n{text}");
    }

    #[test]
    fn timestamps_render_as_utc() {
        assert_eq!(utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(utc(1_000_000_000), "2001-09-09T01:46:40Z");
        // A leap day, which a naive 365-day calculation gets wrong.
        assert_eq!(utc(1_709_164_800), "2024-02-29T00:00:00Z");
        assert_eq!(utc_compact(0), "19700101T000000Z");
    }

    #[test]
    fn a_report_round_trips_through_disk() {
        let dir = std::env::temp_dir().join(format!("ailocal-eval-report-{}", std::process::id()));
        let r = report("20260101T000000Z-m-off", &[Some(1.0)]);
        let path = r.save(&dir).unwrap();

        let back = resolve(&dir, "20260101T000000Z-m-off").unwrap();
        assert_eq!(back.id, r.id);
        assert_eq!(back.arm, r.arm);

        // A substring is enough while it is unambiguous.
        assert!(resolve(&dir, "-m-off").is_ok());
        assert!(resolve(&dir, "nothing-like-this").is_err());

        assert_eq!(list(&dir).unwrap(), vec![path]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_report_from_another_schema_is_refused() {
        let dir = std::env::temp_dir().join(format!("ailocal-eval-schema-{}", std::process::id()));
        let mut r = report("20260101T000000Z-m-off", &[Some(1.0)]);
        r.schema = SCHEMA + 1;
        let path = r.save(&dir).unwrap();

        let err = Report::read(&path).unwrap_err().to_string();
        assert!(err.contains("schema"), "got: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn listing_an_absent_directory_is_empty_rather_than_an_error() {
        let dir = std::env::temp_dir().join("ailocal-eval-definitely-absent-xyz");
        assert!(list(&dir).unwrap().is_empty());
    }
}
