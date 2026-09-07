//! The task corpus: prompts paired with checks that can be scored without a judge.
//!
//! Every check here is a program, never another model. A model-graded eval cannot
//! settle the questions this exists to settle - "is reasoning worth its latency", "is a
//! quantised 27B better than a clean 12B" - because the grader's own quality becomes a
//! confounder, and because two runs a week apart would not be comparable. Deterministic
//! checks give a number that means the same thing every time it is produced.
//!
//! The corpus is compiled into the binary rather than downloaded, so a run needs no
//! network and a report can name the exact task set by hash.

use serde::{Deserialize, Serialize};

/// The built-in coding suite.
pub const BUILTIN: &str = include_str!("../tasks/coding.toml");

/// A parsed task file.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Corpus {
    #[serde(rename = "task")]
    pub tasks: Vec<Task>,
}

/// How much room a task's answer is given when it does not say.
///
/// Generous on purpose. A reasoning model spends its budget thinking before it writes
/// anything, so a tight ceiling returns an empty answer and makes the model look
/// incapable when it was only cut off - which is precisely the confound this harness
/// exists to measure rather than create.
const DEFAULT_MAX_TOKENS: u32 = 1024;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Task {
    /// Stable identifier. Reports are joined on this, so renaming one breaks history.
    pub id: String,

    /// What the task is measuring, for grouping in the report.
    pub category: String,

    /// `dev` tasks may be looked at while iterating; `held-out` ones may not.
    ///
    /// The distinction has no teeth today - both run by default - but it will the
    /// moment an adapter is trained, because a score on tasks you tuned against is not
    /// evidence of anything. Marking them now costs nothing and cannot be done
    /// honestly later.
    #[serde(default = "default_split")]
    pub split: Split,

    pub prompt: String,

    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    #[serde(default, rename = "check")]
    pub checks: Vec<crate::check::Check>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Split {
    Dev,
    HeldOut,
}

impl std::fmt::Display for Split {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Dev => "dev",
            Self::HeldOut => "held-out",
        })
    }
}

impl std::str::FromStr for Split {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "dev" => Ok(Self::Dev),
            "held-out" | "heldout" => Ok(Self::HeldOut),
            other => anyhow::bail!("unknown split {other:?}; expected `dev` or `held-out`"),
        }
    }
}

fn default_split() -> Split {
    Split::HeldOut
}

const fn default_max_tokens() -> u32 {
    DEFAULT_MAX_TOKENS
}

/// A task file plus the identity of the file it came from.
pub struct Loaded {
    pub tasks: Vec<Task>,
    /// Where it came from, for the report.
    pub origin: String,
    /// Hash of the file's bytes.
    ///
    /// Two runs are only comparable if they were scored against the same tasks, and a
    /// corpus edited between them is otherwise invisible in the numbers.
    pub digest: String,
}

/// Parse a corpus, either the built-in one or a file.
///
/// # Errors
/// If the file cannot be read or parsed, or if it is internally inconsistent.
pub fn load(path: Option<&std::path::Path>) -> anyhow::Result<Loaded> {
    use anyhow::Context as _;

    let (text, origin) = match path {
        None => (BUILTIN.to_owned(), "built-in".to_owned()),
        Some(p) => (
            std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?,
            p.display().to_string(),
        ),
    };

    let corpus: Corpus =
        toml::from_str(&text).with_context(|| format!("parsing the task file at {origin}"))?;
    validate(&corpus.tasks).with_context(|| format!("in the task file at {origin}"))?;

    Ok(Loaded {
        tasks: corpus.tasks,
        origin,
        digest: digest(&text),
    })
}

/// Refuse a corpus that cannot produce a meaningful score.
fn validate(tasks: &[Task]) -> anyhow::Result<()> {
    anyhow::ensure!(!tasks.is_empty(), "the task file defines no tasks");

    let mut seen = std::collections::HashSet::new();
    for t in tasks {
        anyhow::ensure!(
            seen.insert(t.id.as_str()),
            "two tasks share the id {:?}; reports are joined on it",
            t.id
        );
        anyhow::ensure!(!t.prompt.trim().is_empty(), "task {:?} has no prompt", t.id);
        // A task with no checks would silently contribute a perfect score to every
        // model, which is worse than not having the task at all.
        anyhow::ensure!(
            !t.checks.is_empty(),
            "task {:?} has no checks, so it cannot be scored",
            t.id
        );
    }
    Ok(())
}

/// Short hex digest of a corpus, enough to tell two task sets apart.
fn digest(text: &str) -> String {
    use sha2::Digest as _;

    let hash = sha2::Sha256::digest(text.as_bytes());
    hash.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The corpus ships inside the binary, so a mistake in it is a broken build rather
    /// than a broken run an hour into an eval.
    #[test]
    fn the_builtin_corpus_is_valid() {
        let loaded = load(None).unwrap();
        assert!(
            loaded.tasks.len() >= 10,
            "a suite this small cannot separate two models"
        );
        assert_eq!(loaded.digest.len(), 16);
    }

    #[test]
    fn the_builtin_corpus_holds_most_tasks_back() {
        let loaded = load(None).unwrap();
        let held = loaded
            .tasks
            .iter()
            .filter(|t| t.split == Split::HeldOut)
            .count();
        assert!(
            held * 2 > loaded.tasks.len(),
            "most tasks must be held out, or there is nothing honest left to measure \
             a trained adapter against"
        );
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let text = r#"
            [[task]]
            id = "a"
            category = "x"
            prompt = "p"
            [[task.check]]
            kind = "contains"
            value = "y"

            [[task]]
            id = "a"
            category = "x"
            prompt = "p"
            [[task.check]]
            kind = "contains"
            value = "y"
        "#;
        let corpus: Corpus = toml::from_str(text).unwrap();
        let err = validate(&corpus.tasks).unwrap_err().to_string();
        assert!(err.contains("share the id"), "got: {err}");
    }

    #[test]
    fn a_task_without_checks_is_rejected() {
        let text = r#"
            [[task]]
            id = "a"
            category = "x"
            prompt = "p"
        "#;
        let corpus: Corpus = toml::from_str(text).unwrap();
        let err = validate(&corpus.tasks).unwrap_err().to_string();
        assert!(err.contains("cannot be scored"), "got: {err}");
    }

    #[test]
    fn a_different_corpus_hashes_differently() {
        assert_ne!(digest("a"), digest("b"));
        assert_eq!(digest("a"), digest("a"));
    }

    /// A correct answer to every task, used only by the test below.
    #[derive(Deserialize)]
    struct References {
        answers: std::collections::HashMap<String, String>,
    }

    /// Every task must be solvable, and its checks must say so.
    ///
    /// This is the test that keeps the harness honest. A `rust_test` that does not
    /// compile against a right answer scores every model zero on that task, and there
    /// is nothing in a run's output to distinguish that from a genuinely hard task -
    /// it would read as a finding about the model rather than a bug in the corpus.
    #[test]
    fn every_task_is_solved_by_its_reference_answer() {
        let references: References =
            toml::from_str(include_str!("../tasks/reference.toml")).unwrap();
        let corpus = load(None).unwrap();
        let ctx = crate::check::Context {
            scratch: std::env::temp_dir().join("ailocal-eval-reference"),
            allow_exec: true,
        };

        for task in &corpus.tasks {
            let answer = references.answers.get(&task.id).unwrap_or_else(|| {
                panic!(
                    "task {:?} has no reference answer in tasks/reference.toml, so \
                     nothing proves it can be solved",
                    task.id
                )
            });

            for check in &task.checks {
                let outcome = check.evaluate(answer, &ctx);
                assert!(
                    outcome.passed(),
                    "task {:?}: the reference answer fails its own {} check: {outcome:?}",
                    task.id,
                    check.label()
                );
            }
        }
    }

    /// The converse: a reference for a task that no longer exists is dead weight, and
    /// usually means a task was renamed and its answer left behind.
    #[test]
    fn no_reference_answer_is_orphaned() {
        let references: References =
            toml::from_str(include_str!("../tasks/reference.toml")).unwrap();
        let corpus = load(None).unwrap();

        for id in references.answers.keys() {
            assert!(
                corpus.tasks.iter().any(|t| &t.id == id),
                "tasks/reference.toml answers {id:?}, which is not a task"
            );
        }
    }

    #[test]
    fn splits_round_trip_through_their_names() {
        for s in [Split::Dev, Split::HeldOut] {
            assert_eq!(s.to_string().parse::<Split>().unwrap(), s);
        }
        assert!("nonsense".parse::<Split>().is_err());
    }
}
