//! Deterministic checks over a model's answer.
//!
//! Two families. Text checks look at what was said; `rust_compiles` and `rust_test`
//! hand the answer to `rustc` and find out whether it is true. The second family is the
//! reason this harness is worth building - "looks like Rust" and "is Rust that passes
//! its tests" are very different claims, and only one of them predicts whether a model
//! is useful in a coding harness.
//!
//! `rustc` rather than `cargo`: a single file with no manifest, no dependency
//! resolution and no lockfile compiles in well under a second, which is what makes it
//! affordable to run on every task of every arm.
//!
//! # Running model output
//!
//! `rust_test` executes code a language model wrote. That is inherent to measuring
//! whether code works, and it is what every coding benchmark does, but it is worth
//! saying out loud rather than burying: an eval run executes untrusted code as you,
//! with your privileges. It is bounded by a timeout, not by a sandbox. `--no-exec`
//! degrades those checks to compile-only.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Ceiling on a single `rustc` invocation. Generous: this is a guard against a wedged
/// compiler, not a performance budget.
const COMPILE_TIMEOUT: Duration = Duration::from_secs(90);

/// Ceiling on a compiled test binary.
///
/// Short on purpose. A wrong loop bound is one of the commonest coding-model mistakes,
/// and the binary-search task in the built-in corpus hangs forever if the model does
/// not fix it - so "did not terminate" has to be a scoreable outcome rather than a
/// stalled eval.
const TEST_TIMEOUT: Duration = Duration::from_secs(10);

/// Edition the model's code is compiled under. Matches the workspace, so a model that
/// writes current Rust is not marked down for it.
const EDITION: &str = "2024";

/// One scoreable assertion about an answer.
#[derive(Debug, Clone, Deserialize)]
pub struct Check {
    /// Relative contribution to the task's score.
    ///
    /// Partial credit is deliberate: "compiles but one case is wrong" and "is not Rust
    /// at all" are different results, and collapsing them to zero throws away most of
    /// what distinguishes two models.
    #[serde(default = "default_weight")]
    pub weight: f64,

    #[serde(flatten)]
    pub kind: Kind,
}

fn default_weight() -> f64 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Kind {
    /// The answer contains this substring.
    Contains {
        value: String,
        #[serde(default)]
        ignore_case: bool,
    },
    /// The answer does not contain this substring.
    NotContains {
        value: String,
        #[serde(default)]
        ignore_case: bool,
    },
    /// The answer contains at least one of these.
    ContainsAny {
        values: Vec<String>,
        #[serde(default)]
        ignore_case: bool,
    },
    /// The answer contains every one of these.
    ContainsAll {
        values: Vec<String>,
        #[serde(default)]
        ignore_case: bool,
    },
    /// The whole answer, trimmed, is exactly this.
    Equals {
        value: String,
        #[serde(default)]
        ignore_case: bool,
    },
    /// The first word of the answer is this, compared without case or punctuation.
    ///
    /// For one-word questions, where a model that answers "No." or "**no**" has still
    /// answered correctly and should not be marked down for typography.
    FirstWord { value: String },
    /// The answer is no longer than this many whitespace-separated words.
    ///
    /// The scoreable form of "reply with only the answer". Preamble is not a style
    /// complaint in a coding harness - it ends up pasted into files.
    MaxWords { value: usize },
    /// The answer is a JSON object carrying these keys, ignoring any code fence.
    Json {
        #[serde(default)]
        keys: Vec<String>,
    },
    /// The Rust in the answer compiles as a library.
    RustCompiles,
    /// The Rust in the answer compiles with this test appended, and the test passes.
    RustTest { test: String },
}

/// What a check decided.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome {
    Pass,
    Fail {
        why: String,
    },
    /// Could not be judged here - no `rustc`, or execution was declined.
    ///
    /// Kept separate from a failure and excluded from the denominator, because a score
    /// that silently counts "could not check" as "wrong" is a lie about the model.
    Skip {
        why: String,
    },
}

impl Outcome {
    fn fail(why: impl Into<String>) -> Self {
        Self::Fail { why: why.into() }
    }

    fn skip(why: impl Into<String>) -> Self {
        Self::Skip { why: why.into() }
    }

    #[must_use]
    pub const fn passed(&self) -> bool {
        matches!(self, Self::Pass)
    }

    #[must_use]
    pub const fn skipped(&self) -> bool {
        matches!(self, Self::Skip { .. })
    }
}

/// Everything a check needs that is not the answer itself.
pub struct Context {
    /// Directory to compile in. On the big disk, per hard rule 1.
    pub scratch: PathBuf,
    /// Whether compiled test binaries may be run.
    pub allow_exec: bool,
}

impl Check {
    /// A short stable name, for the per-task detail in a report.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self.kind {
            Kind::Contains { .. } => "contains",
            Kind::NotContains { .. } => "not_contains",
            Kind::ContainsAny { .. } => "contains_any",
            Kind::ContainsAll { .. } => "contains_all",
            Kind::Equals { .. } => "equals",
            Kind::FirstWord { .. } => "first_word",
            Kind::MaxWords { .. } => "max_words",
            Kind::Json { .. } => "json",
            Kind::RustCompiles => "rust_compiles",
            Kind::RustTest { .. } => "rust_test",
        }
    }

    /// Judge an answer.
    #[must_use]
    pub fn evaluate(&self, answer: &str, ctx: &Context) -> Outcome {
        match &self.kind {
            Kind::Contains { value, ignore_case } => {
                if holds(answer, value, *ignore_case) {
                    Outcome::Pass
                } else {
                    Outcome::fail(format!("does not contain {value:?}"))
                }
            }
            Kind::NotContains { value, ignore_case } => {
                if holds(answer, value, *ignore_case) {
                    Outcome::fail(format!("contains {value:?}"))
                } else {
                    Outcome::Pass
                }
            }
            Kind::ContainsAny {
                values,
                ignore_case,
            } => {
                if values.iter().any(|v| holds(answer, v, *ignore_case)) {
                    Outcome::Pass
                } else {
                    Outcome::fail(format!("contains none of {values:?}"))
                }
            }
            Kind::ContainsAll {
                values,
                ignore_case,
            } => match values.iter().find(|v| !holds(answer, v, *ignore_case)) {
                None => Outcome::Pass,
                Some(missing) => Outcome::fail(format!("does not contain {missing:?}")),
            },
            Kind::Equals { value, ignore_case } => {
                let got = answer.trim();
                let same = if *ignore_case {
                    got.eq_ignore_ascii_case(value.trim())
                } else {
                    got == value.trim()
                };
                if same {
                    Outcome::Pass
                } else {
                    Outcome::fail(format!("expected {value:?}, got {:?}", clip(got, 80)))
                }
            }
            Kind::FirstWord { value } => match first_word(answer) {
                Some(word) if word.eq_ignore_ascii_case(value) => Outcome::Pass,
                Some(word) => Outcome::fail(format!("first word was {word:?}, wanted {value:?}")),
                None => Outcome::fail("the answer is empty"),
            },
            Kind::MaxWords { value } => {
                let words = answer.split_whitespace().count();
                if words <= *value {
                    Outcome::Pass
                } else {
                    Outcome::fail(format!("{words} words, at most {value} allowed"))
                }
            }
            Kind::Json { keys } => json_check(answer, keys),
            Kind::RustCompiles => compile_check(answer, ctx),
            Kind::RustTest { test } => test_check(answer, test, ctx),
        }
    }
}

fn holds(haystack: &str, needle: &str, ignore_case: bool) -> bool {
    if ignore_case {
        haystack.to_lowercase().contains(&needle.to_lowercase())
    } else {
        haystack.contains(needle)
    }
}

/// The first run of alphanumerics in an answer, lowercased.
///
/// Skips whatever a model puts in front of a one-word answer - a fence, `**`, a bullet.
fn first_word(answer: &str) -> Option<String> {
    let start = answer.find(char::is_alphanumeric)?;
    let word: String = answer[start..]
        .chars()
        .take_while(|c| c.is_alphanumeric())
        .collect();
    Some(word.to_lowercase())
}

fn clip(s: &str, max: usize) -> String {
    let flat = s.replace('\n', " ");
    if flat.chars().count() <= max {
        return flat;
    }
    let kept: String = flat.chars().take(max).collect();
    format!("{kept}...")
}

fn json_check(answer: &str, keys: &[String]) -> Outcome {
    // A fence around JSON is a formatting slip, not a wrong answer, so strip it before
    // parsing and let a `max_words` check be the thing that polices formatting.
    let all = blocks(answer);
    let text = pick(&all, &["json"]).map_or_else(|| answer.trim().to_owned(), |b| b.body.clone());

    let value: serde_json::Value = match serde_json::from_str(text.trim()) {
        Ok(v) => v,
        Err(e) => return Outcome::fail(format!("not JSON: {e}")),
    };
    let Some(object) = value.as_object() else {
        return Outcome::fail("JSON, but not an object");
    };
    match keys.iter().find(|k| !object.contains_key(*k)) {
        None => Outcome::Pass,
        Some(missing) => Outcome::fail(format!("no {missing:?} key")),
    }
}

// --- Rust checks ------------------------------------------------------------------

fn compile_check(answer: &str, ctx: &Context) -> Outcome {
    if !rustc_available() {
        return Outcome::skip("rustc is not on PATH");
    }
    let Some(code) = rust_from(answer) else {
        return Outcome::fail("the answer contains no code");
    };

    let scratch = match Scratch::new(&ctx.scratch) {
        Ok(s) => s,
        Err(e) => return Outcome::skip(format!("no scratch directory: {e}")),
    };

    match compile(&code, &scratch, Emit::Metadata) {
        Ok(_) => Outcome::Pass,
        Err(e) => Outcome::fail(e),
    }
}

fn test_check(answer: &str, test: &str, ctx: &Context) -> Outcome {
    if !rustc_available() {
        return Outcome::skip("rustc is not on PATH");
    }
    if !ctx.allow_exec {
        return Outcome::skip("--no-exec: compiled but not run");
    }
    let Some(code) = rust_from(answer) else {
        return Outcome::fail("the answer contains no code");
    };

    let scratch = match Scratch::new(&ctx.scratch) {
        Ok(s) => s,
        Err(e) => return Outcome::skip(format!("no scratch directory: {e}")),
    };

    // The task states the exact signature to write, so the test is appended verbatim
    // and a mismatch is a real failure to follow the instruction, not a harness quirk.
    let source =
        format!("{code}\n\n#[cfg(test)]\nmod ailocal_eval_tests {{\n use super::*;\n{test}\n}}\n");

    let binary = match compile(&source, &scratch, Emit::TestBinary) {
        Ok(path) => path,
        Err(e) => return Outcome::fail(e),
    };

    match run_bounded(&binary, TEST_TIMEOUT) {
        Ok(Verdict::Passed) => Outcome::Pass,
        Ok(Verdict::Failed(out)) => Outcome::fail(format!("tests failed: {}", clip(&out, 200))),
        Ok(Verdict::TimedOut) => Outcome::fail(format!(
            "did not finish in {}s - most likely a loop that never ends",
            TEST_TIMEOUT.as_secs()
        )),
        Err(e) => Outcome::skip(format!("could not run the test binary: {e}")),
    }
}

/// Whether a Rust compiler is present, asked once.
fn rustc_available() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| ailocal::update::which("rustc").is_some())
}

#[derive(Clone, Copy)]
enum Emit {
    /// Type-check only. No codegen, so it is several times faster.
    Metadata,
    /// A runnable test binary.
    TestBinary,
}

/// A temporary directory that removes itself, including when a check returns early.
struct Scratch(PathBuf);

impl Scratch {
    fn new(root: &Path) -> std::io::Result<Self> {
        // Process id plus a counter: checks run one at a time today, but a directory
        // name that collides would make two tasks overwrite each other's source.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = root.join(format!("scratch-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        Ok(Self(dir))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

/// Compile `code`, returning the output path or rustc's complaint.
fn compile(code: &str, scratch: &Scratch, emit: Emit) -> Result<PathBuf, String> {
    let source = scratch.0.join("answer.rs");
    std::fs::write(&source, code).map_err(|e| format!("writing the answer out: {e}"))?;

    let out = scratch.0.join(match emit {
        Emit::Metadata => "answer.rmeta",
        Emit::TestBinary => "answer-test",
    });

    let mut cmd = Command::new("rustc");
    cmd.args(["--edition", EDITION]);
    match emit {
        Emit::Metadata => {
            cmd.args(["--crate-type", "lib", "--emit", "metadata"]);
        }
        Emit::TestBinary => {
            cmd.arg("--test");
        }
    }
    // A fixed crate name because the model's file name should not leak into error
    // messages, and `-A warnings` because unused imports are not what is being scored.
    cmd.args(["--crate-name", "answer"])
        .args(["-A", "warnings"])
        .arg("-o")
        .arg(&out)
        .arg(&source)
        .current_dir(&scratch.0)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| format!("starting rustc: {e}"))?;
    if !wait_bounded(&mut child, COMPILE_TIMEOUT)? {
        return Err(format!(
            "rustc did not finish within {}s",
            COMPILE_TIMEOUT.as_secs()
        ));
    }
    let output = child
        .wait_with_output()
        .map_err(|e| format!("collecting rustc output: {e}"))?;

    if output.status.success() {
        return Ok(out);
    }
    Err(format!(
        "does not compile: {}",
        clip(
            &first_rustc_error(&String::from_utf8_lossy(&output.stderr)),
            200
        )
    ))
}

/// The first real diagnostic from a rustc run.
///
/// The whole stderr is several hundred lines of spans and notes; the report needs the
/// one line that says what is wrong.
fn first_rustc_error(stderr: &str) -> String {
    stderr
        .lines()
        .find(|l| l.starts_with("error"))
        .unwrap_or_else(|| stderr.lines().next().unwrap_or("unknown error"))
        .to_owned()
}

enum Verdict {
    Passed,
    Failed(String),
    TimedOut,
}

/// Run a compiled test binary under a deadline.
fn run_bounded(binary: &Path, limit: Duration) -> Result<Verdict, String> {
    let mut child = Command::new(binary)
        // One thread so a failure names the test that failed rather than interleaving.
        .args(["--test-threads", "1"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{e}"))?;

    if !wait_bounded(&mut child, limit)? {
        return Ok(Verdict::TimedOut);
    }

    let output = child.wait_with_output().map_err(|e| format!("{e}"))?;
    if output.status.success() {
        return Ok(Verdict::Passed);
    }

    // The assertion message is on stdout for a normal failure and on stderr for a
    // panic outside a test, so report whichever has something to say.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let detail = stdout
        .lines()
        .find(|l| l.contains("assertion") || l.contains("panicked"))
        .map(str::to_owned)
        .unwrap_or_else(|| String::from_utf8_lossy(&output.stderr).trim().to_owned());
    Ok(Verdict::Failed(detail))
}

/// Wait for `child`, killing it and returning `false` if `limit` passes first.
fn wait_bounded(child: &mut Child, limit: Duration) -> Result<bool, String> {
    let deadline = Instant::now() + limit;
    loop {
        match child.try_wait().map_err(|e| format!("{e}"))? {
            Some(_) => return Ok(true),
            None if Instant::now() >= deadline => {
                child.kill().ok();
                // Reap it, or the process lingers as a zombie for the rest of the run.
                child.wait().ok();
                return Ok(false);
            }
            None => std::thread::sleep(Duration::from_millis(25)),
        }
    }
}

// --- pulling code out of prose ------------------------------------------------------

/// One fenced block in a response.
struct Block {
    lang: String,
    body: String,
    /// Whether a closing fence was found.
    ///
    /// False means the response was cut off mid-block, which models truncated by
    /// `max_tokens` do constantly - and an unfinished block is a much worse guess at
    /// what the model meant than a finished one earlier in the same answer.
    terminated: bool,
}

/// Every fenced block in a response, in order.
fn blocks(text: &str) -> Vec<Block> {
    let mut found = Vec::new();
    let mut rest = text;

    while let Some(open) = rest.find("```") {
        let after = &rest[open + 3..];
        let (lang, body_start) = match after.find('\n') {
            Some(nl) => (after[..nl].trim().to_lowercase(), nl + 1),
            // A fence with no newline after it opens nothing.
            None => break,
        };
        let body = &after[body_start..];
        match body.find("```") {
            Some(close) => {
                found.push(Block {
                    lang,
                    body: body[..close].to_owned(),
                    terminated: true,
                });
                rest = &body[close + 3..];
            }
            None => {
                found.push(Block {
                    lang,
                    body: body.to_owned(),
                    terminated: false,
                });
                break;
            }
        }
    }
    found
}

/// The block a reader would take as the answer.
///
/// The **last complete** block, not the first. Models routinely write an attempt, spot
/// their own mistake and write a corrected version - gemma4 does exactly this on the
/// Display task - and scoring the first block marks a model down for catching its own
/// bug, which is the opposite of what the number should reward. Completeness beats
/// recency because a truncated final block is a fragment rather than an answer.
///
/// Blocks tagged with one of `langs` win outright; failing that, any block will do,
/// since a model told to reply with only code sometimes omits the tag.
fn pick<'a>(all: &'a [Block], langs: &[&str]) -> Option<&'a Block> {
    let tagged: Vec<&Block> = all
        .iter()
        .filter(|b| langs.contains(&b.lang.as_str()))
        .collect();
    let pool: Vec<&Block> = if tagged.is_empty() {
        all.iter().collect()
    } else {
        tagged
    };

    pool.iter()
        .rev()
        .find(|b| b.terminated)
        .or_else(|| pool.last())
        .copied()
}

/// The Rust in a response, or the bare text when there is no fence at all - refusing to
/// compile a correct answer for want of a fence would measure formatting rather than
/// Rust.
fn rust_from(answer: &str) -> Option<String> {
    let all = blocks(answer);
    let chosen = pick(&all, &["rust", "rs"])
        .map(|b| b.body.clone())
        .unwrap_or_else(|| answer.to_owned());

    (!chosen.trim().is_empty()).then_some(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> Context {
        Context {
            scratch: std::env::temp_dir().join("ailocal-eval-tests"),
            allow_exec: true,
        }
    }

    fn check(toml_src: &str) -> Check {
        toml::from_str(toml_src).unwrap()
    }

    #[test]
    fn contains_respects_case_by_default() {
        let c = check(
            r#"kind = "contains"
value = "Vec""#,
        );
        assert!(c.evaluate("uses a Vec", &ctx()).passed());
        assert!(!c.evaluate("uses a vec", &ctx()).passed());
    }

    #[test]
    fn ignore_case_is_opt_in() {
        let c = check(
            r#"kind = "contains"
value = "Vec"
ignore_case = true"#,
        );
        assert!(c.evaluate("uses a vec", &ctx()).passed());
    }

    #[test]
    fn weight_defaults_to_one_and_can_be_set() {
        let c = check(
            r#"kind = "contains"
value = "x""#,
        );
        assert!((c.weight - 1.0).abs() < f64::EPSILON);
        let heavy = check(
            r#"kind = "contains"
value = "x"
weight = 3.0"#,
        );
        assert!((heavy.weight - 3.0).abs() < f64::EPSILON);
    }

    /// A one-word answer wrapped in typography is still that word.
    #[test]
    fn first_word_ignores_decoration() {
        let c = check(
            r#"kind = "first_word"
value = "no""#,
        );
        assert!(c.evaluate("**No.**", &ctx()).passed());
        assert!(c.evaluate("  no, it does not", &ctx()).passed());
        assert!(!c.evaluate("Yes", &ctx()).passed());
        assert!(!c.evaluate("", &ctx()).passed());
    }

    #[test]
    fn max_words_catches_preamble() {
        let c = check(
            r#"kind = "max_words"
value = 3"#,
        );
        assert!(c.evaluate("O(log n)", &ctx()).passed());
        assert!(!c.evaluate("Sure! The answer is O(log n).", &ctx()).passed());
    }

    #[test]
    fn json_accepts_a_fenced_object_and_requires_its_keys() {
        let c = check(
            r#"kind = "json"
keys = ["name", "params"]"#,
        );
        assert!(
            c.evaluate("```json\n{\"name\":\"f\",\"params\":2}\n```", &ctx())
                .passed()
        );
        // A corrected second object wins here too, for the same reason.
        assert!(
            c.evaluate(
                "```json\n{\"name\":\"f\"}\n```\nsorry:\n```json\n{\"name\":\"f\",\"params\":2}\n```",
                &ctx()
            )
            .passed()
        );
        assert!(c.evaluate("{\"name\":\"f\",\"params\":2}", &ctx()).passed());
        assert!(!c.evaluate("{\"name\":\"f\"}", &ctx()).passed());
        assert!(!c.evaluate("not json at all", &ctx()).passed());
    }

    #[test]
    fn contains_all_names_the_missing_one() {
        let c = check(
            r#"kind = "contains_all"
values = ["alpha", "beta"]"#,
        );
        match c.evaluate("only alpha here", &ctx()) {
            Outcome::Fail { why } => assert!(why.contains("beta"), "got: {why}"),
            other => panic!("expected a failure naming beta, got {other:?}"),
        }
    }

    #[test]
    fn a_rust_block_is_preferred_over_a_shell_block() {
        let answer = "First install it:\n```sh\ncargo add x\n```\nThen:\n```rust\nfn f() {}\n```";
        assert_eq!(rust_from(answer).unwrap().trim(), "fn f() {}");
    }

    /// The behaviour gemma4 forced: it writes a buggy impl, says "wait, that is a
    /// typo", and writes a corrected one. Scoring the first block marks it down for
    /// noticing.
    #[test]
    fn the_last_complete_block_is_the_answer() {
        let answer = "```rust\nfn f() { wrong() }\n```\n\
                      Wait, that is a typo. Corrected:\n\
                      ```rust\nfn f() { right() }\n```";
        assert_eq!(rust_from(answer).unwrap().trim(), "fn f() { right() }");
    }

    /// ...but a final block cut off by `max_tokens` is a fragment, not a correction,
    /// so the last *complete* one is the better guess at what was meant.
    #[test]
    fn a_truncated_final_block_loses_to_the_last_complete_one() {
        let answer = "```rust\nfn f() { complete() }\n```\n\
                      Actually:\n\
                      ```rust\nfn f() { half_writ";
        assert_eq!(rust_from(answer).unwrap().trim(), "fn f() { complete() }");
    }

    #[test]
    fn bare_code_with_no_fence_is_still_code() {
        assert_eq!(rust_from("fn f() {}").unwrap().trim(), "fn f() {}");
    }

    /// With nothing complete to fall back to, the fragment still reaches rustc, which
    /// rejects it - a real failure rather than a skip.
    #[test]
    fn an_unterminated_fence_still_yields_code() {
        let answer = "```rust\nfn f() {\n";
        assert!(rust_from(answer).unwrap().contains("fn f()"));
    }

    #[test]
    fn an_empty_answer_yields_no_code() {
        assert!(rust_from("   \n  ").is_none());
    }

    #[test]
    fn skipped_checks_are_not_failures() {
        let skip = Outcome::skip("no rustc");
        assert!(skip.skipped());
        assert!(!skip.passed());
        assert!(!Outcome::fail("wrong").skipped());
    }

    #[test]
    fn the_first_rustc_error_is_the_one_reported() {
        let stderr =
            "warning: unused\n   --> a.rs:1\nerror[E0425]: cannot find value `x`\n   --> a.rs:2";
        assert!(first_rustc_error(stderr).contains("E0425"));
    }

    // The remaining tests shell out to rustc. It is present wherever this crate can be
    // built, including CI, so they are real assertions rather than conditional ones.

    #[test]
    fn correct_code_compiles_and_passes_its_test() {
        let c = check(
            r#"kind = "rust_test"
test = """
#[test]
fn t() { assert_eq!(double(2), 4); }
""""#,
        );
        let answer = "```rust\npub fn double(x: i32) -> i32 { x * 2 }\n```";
        assert_eq!(c.evaluate(answer, &ctx()), Outcome::Pass);
    }

    #[test]
    fn wrong_code_compiles_but_fails_its_test() {
        let c = check(
            r#"kind = "rust_test"
test = """
#[test]
fn t() { assert_eq!(double(2), 4); }
""""#,
        );
        let answer = "```rust\npub fn double(x: i32) -> i32 { x * 3 }\n```";
        match c.evaluate(answer, &ctx()) {
            Outcome::Fail { why } => assert!(why.contains("tests failed"), "got: {why}"),
            other => panic!("expected a test failure, got {other:?}"),
        }
    }

    #[test]
    fn code_that_does_not_compile_says_so() {
        let c = check(r#"kind = "rust_compiles""#);
        match c.evaluate("```rust\npub fn f() -> i32 { \"nope\" }\n```", &ctx()) {
            Outcome::Fail { why } => assert!(why.contains("does not compile"), "got: {why}"),
            other => panic!("expected a compile failure, got {other:?}"),
        }
    }

    /// The failure mode that makes a timeout non-negotiable: without one, a single
    /// wrong loop bound stalls the whole eval rather than scoring zero.
    #[test]
    fn a_loop_that_never_ends_is_scored_not_waited_on() {
        let c = check(
            r#"kind = "rust_test"
test = """
#[test]
fn t() { spin(); }
""""#,
        );
        let answer = "```rust\npub fn spin() { loop { std::hint::spin_loop(); } }\n```";
        let started = Instant::now();
        match c.evaluate(answer, &ctx()) {
            Outcome::Fail { why } => assert!(why.contains("did not finish"), "got: {why}"),
            other => panic!("expected a timeout failure, got {other:?}"),
        }
        assert!(
            started.elapsed() < TEST_TIMEOUT + COMPILE_TIMEOUT,
            "the timeout did not bound the run"
        );
    }

    /// `--no-exec` must not silently score model code as wrong.
    #[test]
    fn no_exec_skips_rather_than_fails() {
        let c = check(
            r#"kind = "rust_test"
test = """
#[test]
fn t() { assert_eq!(double(2), 4); }
""""#,
        );
        let ctx = Context {
            scratch: std::env::temp_dir().join("ailocal-eval-tests"),
            allow_exec: false,
        };
        assert!(
            c.evaluate("```rust\npub fn double(x: i32) -> i32 { x * 2 }\n```", &ctx)
                .skipped()
        );
    }

    #[test]
    fn the_scratch_directory_is_removed_afterwards() {
        let root = std::env::temp_dir().join("ailocal-eval-tests");
        let path = {
            let s = Scratch::new(&root).unwrap();
            assert!(s.0.exists());
            s.0.clone()
        };
        assert!(!path.exists(), "scratch must not outlive the check");
    }
}
