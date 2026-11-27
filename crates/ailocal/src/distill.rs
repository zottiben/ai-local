//! Turning a correction into a lesson, or into nothing.
//!
//! The distiller is a chat model told what the agent did and what the person said
//! about it. By default it is the model ailocal serves, asked directly with a JSON
//! schema the answer must follow, so capture needs no other provider. A frontier
//! model is opt-in and asked through `pi -p`, so its credential stays Pi's.
//!
//! It is asked two things, one at a time: whether the message teaches a lesson, and
//! only then whether that lesson is one already kept. Asked both at once with the kept
//! lessons in view, qwen3.6-35b-a3b called seven different corrections in a row
//! repeats of the first lesson it had written.
//!
//! Nothing it learns applies before the person accepts it, so it is tuned to find
//! lessons rather than to be sure of each: a third question, whether a draft was a
//! standing rule, took precision on 136 labelled corrections from 68% to 95% and
//! recall from 67% to 43%. A false lesson costs one rejection; a lesson never found
//! is steering the person goes on doing.

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use anyhow::Context as _;
use serde::Deserialize;

use crate::capture::Candidate;
use crate::lessons::{Lesson, State};

/// How long one answer may take. A local model answers in seconds; a minute is a
/// server that is busy with someone's session or has gone away.
const TIMEOUT: Duration = Duration::from_secs(300);
/// Room for the answer's few short fields, and a stop for a model that rambles.
const MAX_TOKENS: u32 = 600;
/// How much of what the person had asked the distiller is shown.
const MAX_ASKED: usize = 1500;
/// A reason is one sentence. Uncapped, a model reasoned inside it until the answer ran
/// out of tokens before its verdict.
const MAX_REASON: usize = 400;

const INSTRUCTIONS: &str = "\
You read a person's message to a coding agent, with what the agent had just done, and \
decide whether the message teaches a lesson the agent should keep for next time.

A lesson is a rule about how to work that would still apply to a different task \
tomorrow: which template a pull request body follows, never adding eslint-disable, \
running the formatter before committing, leaving review comments on GitHub unanswered. \
A message that asks for work can teach one in passing, as a standing rule about how \
to do it: \"Do not reply to the comments on the PR\". \
The message teaches none when it:
- asks for new work, changes or adds a requirement of the product, or answers the \
agent's question;
- is about this task's code alone: a name, a value, a design choice that does not \
carry over to other work;
- reports a bug in the product, or how a run turned out;
- is about the person's own mistake, or a change of mind.

Write a lesson as one imperative sentence of at most 25 words that makes sense with no \
other context, naming the exact file, command or text it is about. Its scope is \"repository\" \
when it is about this codebase's conventions and \"everywhere\" when it is about how to \
work in any codebase.

Triggers stop the agent at the moment it is about to break the lesson. Give one only \
when that exact action breaks it, written literally:
- commands: the first words of a command the lesson is about, such as \"gh pr create\". \
Never a program alone, such as \"git\" or \"npm\".
- adds: text whose addition by an edit breaks the lesson, such as \"eslint-disable\".
- files: a file or directory the lesson says not to edit, such as \"CHANGELOG.md\".
Most lessons have none: leave each list empty unless the lesson names the thing itself.";

const SAME_INSTRUCTIONS: &str = "\
You decide whether a new lesson for a coding agent is one it has already: the same rule, \
perhaps in other words, so that keeping both would tell the agent one thing twice. A \
lesson about the same subject that asks for something else is new. Answer with the id \
of the lesson it repeats, or \"new\".";

/// A model to distil with.
#[derive(Debug, Clone)]
pub enum Distiller {
    /// The model ailocal serves, at its llama-server's base URL.
    Local { base: String, model: String },
    /// Any model Pi reaches, by `provider/id`.
    Pi { pi: PathBuf, model: String },
}

impl Distiller {
    /// What wrote a lesson, as its `source` says.
    #[must_use]
    pub fn name(&self) -> String {
        match self {
            Self::Local { model, .. } => format!("llama.cpp/{model}"),
            Self::Pi { model, .. } => model.clone(),
        }
    }
}

/// What the distiller made of a correction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// A new lesson.
    Lesson(Draft),
    /// The lesson with this id, taught again.
    Repeat(String),
    /// No lesson.
    None,
}

/// A lesson as the distiller wrote it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Draft {
    pub text: String,
    /// Whether it is about this repository's conventions rather than work anywhere.
    pub repository: bool,
    pub commands: Vec<String>,
    pub adds: Vec<String>,
    pub files: Vec<String>,
}

/// The answer to whether a message teaches a lesson, as the schema asks for it.
#[derive(Debug, Deserialize)]
struct Answer {
    reason: String,
    verdict: String,
    #[serde(default)]
    lesson: String,
    #[serde(default)]
    scope: String,
    #[serde(default)]
    commands: Vec<String>,
    #[serde(default)]
    adds: Vec<String>,
    #[serde(default)]
    files: Vec<String>,
}

/// The answer to whether a lesson is one already kept.
#[derive(Debug, Deserialize)]
struct Same {
    same: String,
}

/// Decide what `candidate` teaches, given the lessons already kept, and why.
///
/// # Errors
/// If the model cannot be asked, or answers with something other than the JSON asked
/// for, or names a lesson that does not exist.
pub fn distill(
    distiller: &Distiller,
    candidate: &Candidate,
    repo: Option<&str>,
    kept: &[Lesson],
) -> anyhow::Result<(Verdict, String)> {
    let answer: Answer = ask(
        distiller,
        INSTRUCTIONS,
        &prompt(candidate, repo),
        &schema(),
        SHAPE,
    )?;
    let Some(draft) = draft(&answer)? else {
        return Ok((Verdict::None, answer.reason));
    };
    if kept.is_empty() {
        return Ok((Verdict::Lesson(draft), answer.reason));
    }
    let same: Same = ask(
        distiller,
        SAME_INSTRUCTIONS,
        &same_prompt(&draft, kept),
        &same_schema(kept),
        SAME_SHAPE,
    )?;
    Ok((
        repeat(&same, kept)?.map_or(Verdict::Lesson(draft), Verdict::Repeat),
        answer.reason,
    ))
}

/// The lesson an answer drafts, if it found one.
fn draft(answer: &Answer) -> anyhow::Result<Option<Draft>> {
    let literal = |list: &[String]| -> Vec<String> {
        list.iter()
            .map(|t| t.trim().to_owned())
            .filter(|t| !t.is_empty())
            .collect()
    };
    match answer.verdict.as_str() {
        "none" => Ok(None),
        "lesson" => {
            anyhow::ensure!(
                !answer.lesson.trim().is_empty(),
                "the distiller found a lesson and wrote nothing"
            );
            Ok(Some(Draft {
                text: answer.lesson.trim().to_owned(),
                repository: answer.scope == "repository",
                commands: literal(&answer.commands)
                    .into_iter()
                    .filter(|c| c.split_whitespace().count() > 1)
                    .collect(),
                adds: literal(&answer.adds),
                files: literal(&answer.files),
            }))
        }
        other => anyhow::bail!("the distiller answered a verdict of {other:?}"),
    }
}

/// The kept lesson an answer says a draft repeats, or `None` for a new one.
fn repeat(same: &Same, kept: &[Lesson]) -> anyhow::Result<Option<String>> {
    let id = same.same.trim();
    if id == "new" {
        return Ok(None);
    }
    anyhow::ensure!(
        kept.iter().any(|l| l.id == id),
        "the distiller called it a repeat of {id:?}, which is no lesson"
    );
    Ok(Some(id.to_owned()))
}

/// What the distiller is shown of a candidate: where, what was asked and done, and
/// what the person said about it.
fn prompt(candidate: &Candidate, repo: Option<&str>) -> String {
    let mut text = format!("Repository: {}\n", repo.unwrap_or("none, or not known"));
    if let Some(asked) = &candidate.asked {
        text.push_str(&format!("\nThe person had asked:\n{}\n", clip(asked)));
    }
    text.push_str("\nThe agent then:\n");
    for step in &candidate.did {
        text.push_str(&format!("{step}\n"));
    }
    text.push_str(&format!("\nThe person said:\n{}\n", candidate.said));
    text
}

fn clip(text: &str) -> &str {
    match text.char_indices().nth(MAX_ASKED) {
        Some((cut, _)) => &text[..cut],
        None => text,
    }
}

/// The fields in the order the model writes them, which is the order it decides in:
/// a reason first, then what it concludes. Sorted, it drafted the lesson and its
/// triggers before deciding there was one.
fn schema() -> serde_json::Value {
    let list = serde_json::json!({"type": "array", "items": {"type": "string"}, "maxItems": 3});
    serde_json::json!({
        "type": "object",
        "properties": {
            "reason": {"type": "string", "maxLength": MAX_REASON},
            "verdict": {"enum": ["lesson", "none"]},
            "lesson": {"type": "string"},
            "scope": {"enum": ["repository", "everywhere"]},
            "commands": list,
            "adds": list,
            "files": list,
        },
        "required": ["reason", "verdict", "lesson", "scope", "commands", "adds", "files"],
        "additionalProperties": false,
    })
}

/// The schema's shape in words, for a model that is not held to it.
const SHAPE: &str = r#"Answer with only this JSON object:
{"reason": "<one short sentence>", "verdict": "lesson" | "none", "lesson": "<the lesson, or empty>", "scope": "repository" | "everywhere", "commands": [], "adds": [], "files": []}"#;

fn same_prompt(draft: &Draft, kept: &[Lesson]) -> String {
    let mut text = format!("New lesson:\n{}\n\nLessons it has:\n", draft.text);
    for lesson in kept {
        text.push_str(&format!("- {}: {}\n", lesson.id, lesson.text));
    }
    text
}

/// The id must be one of those kept, so a local model cannot name another.
fn same_schema(kept: &[Lesson]) -> serde_json::Value {
    let mut ids: Vec<&str> = kept.iter().map(|l| l.id.as_str()).collect();
    ids.push("new");
    serde_json::json!({
        "type": "object",
        "properties": {
            "reason": {"type": "string", "maxLength": MAX_REASON},
            "same": {"enum": ids},
        },
        "required": ["reason", "same"],
        "additionalProperties": false,
    })
}

const SAME_SHAPE: &str = r#"Answer with only this JSON object:
{"reason": "<one short sentence>", "same": "<id of the lesson it repeats, or new>"}"#;

/// Ask `distiller` one question and read the JSON it answers.
fn ask<T: serde::de::DeserializeOwned>(
    distiller: &Distiller,
    instructions: &str,
    prompt: &str,
    schema: &serde_json::Value,
    shape: &str,
) -> anyhow::Result<T> {
    let raw = match distiller {
        Distiller::Local { base, model } => ask_local(base, model, instructions, prompt, schema)?,
        Distiller::Pi { pi, model } => {
            ask_pi(pi, model, &format!("{instructions}\n\n{prompt}\n{shape}"))?
        }
    };
    serde_json::from_str(json_object(&raw))
        .with_context(|| format!("the distiller answered {raw:?}"))
}

/// One of llama-server's slots, as `/slots` reports it.
#[derive(Debug, Deserialize)]
pub struct Slot {
    pub id: u32,
    pub is_processing: bool,
    /// The tokens of the last prompt it served, which its cache still holds.
    #[serde(default)]
    pub n_prompt_tokens: u64,
}

/// The slots of the llama-server at `base`.
///
/// # Errors
/// If it cannot be asked, or answers something else.
pub fn slots(base: &str) -> anyhow::Result<Vec<Slot>> {
    client()?
        .get(format!("{base}/slots"))
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .with_context(|| format!("asking {base}/slots"))?
        .json()
        .with_context(|| format!("reading {base}/slots"))
}

/// The slot to distil in: an idle one holding the fewest tokens. Another slot may hold
/// a session the person will come back to, and a request it served would evict that
/// session's cache.
fn quietest(slots: &[Slot]) -> Option<u32> {
    slots
        .iter()
        .filter(|s| !s.is_processing)
        .min_by_key(|s| s.n_prompt_tokens)
        .map(|s| s.id)
}

fn client() -> reqwest::Result<reqwest::blocking::Client> {
    // llama-server closes an idle connection after about five seconds; see
    // `gateway::upstream_client`.
    reqwest::blocking::Client::builder()
        .timeout(TIMEOUT)
        .pool_max_idle_per_host(0)
        .build()
}

fn ask_local(
    base: &str,
    model: &str,
    instructions: &str,
    prompt: &str,
    schema: &serde_json::Value,
) -> anyhow::Result<String> {
    let endpoint = format!("{base}/v1/chat/completions");
    let mut body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": false},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema},
        },
    });
    if let Some(id) = slots(base).ok().as_deref().and_then(quietest) {
        body["id_slot"] = id.into();
    }
    let response: serde_json::Value = client()?
        .post(&endpoint)
        .json(&body)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .with_context(|| format!("asking {endpoint}"))?
        .json()?;
    response["choices"][0]["message"]["content"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| anyhow::anyhow!("{endpoint} answered no message: {response}"))
}

/// Ask through Pi, with the person's extensions loaded, because a provider such as
/// `claude-subscription` is one. The instructions go in the message: that provider puts
/// its own system prompt first, and one passed as `--system-prompt` was not followed.
fn ask_pi(pi: &std::path::Path, model: &str, message: &str) -> anyhow::Result<String> {
    let output = Command::new(pi)
        .current_dir(std::env::temp_dir())
        .args([
            "-p",
            "--no-session",
            "--no-tools",
            "--no-mcp",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
            "--no-approve",
            "--model",
            model,
        ])
        .arg(message)
        .output()
        .with_context(|| format!("running {}", pi.display()))?;
    anyhow::ensure!(
        output.status.success(),
        "{} -p --model {model} failed: {}",
        pi.display(),
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The JSON object in a model's answer, which a model not held to a schema may wrap
/// in a fence or a sentence.
fn json_object(text: &str) -> &str {
    match (text.find('{'), text.rfind('}')) {
        (Some(start), Some(end)) if start < end => &text[start..=end],
        _ => text,
    }
}

/// A new pending lesson from a draft, learned from `candidate`.
#[must_use]
pub fn pending(draft: Draft, repo: Option<&str>, source: String, created: u64) -> Lesson {
    Lesson {
        id: String::new(),
        text: draft.text,
        repo: repo.filter(|_| draft.repository).map(str::to_owned),
        commands: draft.commands,
        adds: draft.adds,
        files: draft.files,
        state: State::Pending,
        source,
        created,
        corrections: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kept() -> Vec<Lesson> {
        vec![Lesson {
            id: "pr-template".to_owned(),
            text: "Use the template in .github/ for a pull request body.".to_owned(),
            repo: None,
            commands: vec!["gh pr create".to_owned()],
            adds: Vec::new(),
            files: Vec::new(),
            state: State::Active,
            source: String::new(),
            created: 0,
            corrections: Vec::new(),
        }]
    }

    fn answer(json: &str) -> anyhow::Result<Option<Draft>> {
        draft(&serde_json::from_str(json_object(json)).unwrap())
    }

    fn same(json: &str) -> anyhow::Result<Option<String>> {
        repeat(&serde_json::from_str(json).unwrap(), &kept())
    }

    #[test]
    fn an_answer_becomes_a_draft() {
        let lesson = answer(
            r#"Here it is: ```json
            {"reason": "r", "verdict": "lesson", "scope": "repository",
             "lesson": " Fix a lint error at its cause; never add eslint-disable. ",
             "commands": ["npm", "npm run lint", " "], "adds": ["eslint-disable"], "files": []}
            ```"#,
        )
        .unwrap();
        assert_eq!(
            lesson,
            Some(Draft {
                text: "Fix a lint error at its cause; never add eslint-disable.".to_owned(),
                repository: true,
                commands: vec!["npm run lint".to_owned()],
                adds: vec!["eslint-disable".to_owned()],
                files: Vec::new(),
            })
        );
        assert_eq!(
            answer(r#"{"reason": "r", "verdict": "none"}"#).unwrap(),
            None
        );
        assert_eq!(
            same(r#"{"reason": "r", "same": "pr-template"}"#).unwrap(),
            Some("pr-template".to_owned())
        );
        assert_eq!(same(r#"{"reason": "r", "same": "new"}"#).unwrap(), None);
    }

    #[test]
    fn an_answer_that_names_nothing_real_is_refused() {
        for json in [
            r#"{"reason": "r", "verdict": "lesson", "lesson": "  "}"#,
            r#"{"reason": "r", "verdict": "maybe"}"#,
            r#"{"reason": "r", "verdict": "repeat"}"#,
        ] {
            assert!(answer(json).is_err(), "{json}");
        }
        for json in [
            r#"{"reason": "r", "same": "no-such-lesson"}"#,
            r#"{"reason": "r", "same": ""}"#,
        ] {
            assert!(same(json).is_err(), "{json}");
        }
    }

    #[test]
    fn a_distiller_takes_the_idle_slot_that_holds_least() {
        let slot = |id, is_processing, n_prompt_tokens| Slot {
            id,
            is_processing,
            n_prompt_tokens,
        };
        let busy_small = [
            slot(0, false, 80_000),
            slot(1, true, 10),
            slot(2, false, 900),
        ];
        assert_eq!(quietest(&busy_small), Some(2));
        assert_eq!(quietest(&[slot(0, true, 1)]), None);
    }

    /// A model held to a schema writes its fields in the schema's order, so the reason
    /// has to come before what it decides.
    #[test]
    fn a_schema_asks_for_the_reason_first() {
        let sent = serde_json::to_string(&schema()).unwrap();
        let at = |key: &str| sent.find(&format!("\"{key}\"")).unwrap();
        assert!(at("reason") < at("verdict"), "{sent}");
        assert!(at("verdict") < at("lesson"), "{sent}");
        assert!(at("lesson") < at("commands"), "{sent}");
    }

    #[test]
    fn the_schema_allows_only_kept_ids() {
        assert_eq!(
            same_schema(&kept())["properties"]["same"]["enum"],
            serde_json::json!(["pr-template", "new"])
        );
    }

    #[test]
    fn the_prompts_show_the_exchange_then_the_kept_lessons() {
        let candidate = Candidate {
            session: PathBuf::from("s.jsonl"),
            entry: "u2".to_owned(),
            timestamp: String::new(),
            cwd: PathBuf::from("/w"),
            provider: None,
            model: None,
            said: "Why didn't you use the PR template?".to_owned(),
            asked: Some("Open a PR.".to_owned()),
            did: vec!["$ gh pr create".to_owned()],
        };
        assert_eq!(
            prompt(&candidate, Some("github.com/o/r")),
            "Repository: github.com/o/r\n\
             \nThe person had asked:\nOpen a PR.\n\
             \nThe agent then:\n$ gh pr create\n\
             \nThe person said:\nWhy didn't you use the PR template?\n"
        );
        let draft = Draft {
            text: "Follow .github/pull_request_template.md.".to_owned(),
            repository: false,
            commands: Vec::new(),
            adds: Vec::new(),
            files: Vec::new(),
        };
        assert_eq!(
            same_prompt(&draft, &kept()),
            "New lesson:\nFollow .github/pull_request_template.md.\n\
             \nLessons it has:\n\
             - pr-template: Use the template in .github/ for a pull request body.\n"
        );
    }
}
