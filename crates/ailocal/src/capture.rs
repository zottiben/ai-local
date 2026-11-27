//! Lessons learned from the person's corrections in Pi's sessions.
//!
//! A pass reads each session file from where the last pass stopped and takes every
//! message of the person's that reads like a correction - a literal filter, so the
//! cost of a pass is reading the new bytes. Each one goes, with what the agent had
//! just done, to a distiller model, which drops a request or a one-off and turns the
//! rest into lessons that wait for the person's review.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use crate::distill::{Distiller, Verdict};
use crate::lessons::{Correction, Store};

/// How much of the person's message a candidate keeps. A longer one is a new task or
/// a pasted log, not a correction.
const MAX_SAID: usize = 1500;
/// How much of each thing the agent did is kept to show the distiller.
const MAX_STEP: usize = 300;
/// How many of the agent's last steps are kept.
const MAX_STEPS: usize = 12;
/// How much of a correction a lesson keeps, to show at review.
const MAX_KEPT: usize = 300;
/// Answers in a row the distiller may fail before a pass stops: one is a model that
/// misread a message, several are a model that has gone away.
const MAX_FAILURES: usize = 3;
/// How often the gateway looks for new corrections.
const EVERY: std::time::Duration = std::time::Duration::from_secs(10 * 60);

/// Learn from the person's corrections for as long as the gateway runs, with the model
/// it serves, while that model has nothing else to do. Never returns.
pub fn keep_capturing(enabled: impl Fn() -> bool) {
    loop {
        std::thread::sleep(EVERY);
        if enabled()
            && let Err(e) = capture_while_idle()
        {
            eprintln!("lessons: {e:#}");
        }
    }
}

/// One pass, which stops before the next candidate once any slot is busy: the model is
/// the person's first, and a pass can always go on later.
fn capture_while_idle() -> anyhow::Result<()> {
    let Some(instance) = crate::serve::running()? else {
        return Ok(());
    };
    let base = instance.base_url();
    let idle = || crate::distill::slots(&base).is_ok_and(|s| s.iter().all(|s| !s.is_processing));
    if !idle() {
        return Ok(());
    }
    let distiller = Distiller::Local {
        base: base.clone(),
        model: instance.model,
    };
    pass(
        &distiller,
        &session_files()?,
        &crate::lessons::path()?,
        &Progress::path()?,
        idle,
        |_, outcome| match outcome {
            Outcome::Learned(id) => println!("lessons: learned {id}, for review"),
            Outcome::Repeated(id) => println!("lessons: {id} taught again"),
            Outcome::Nothing => {}
            Outcome::Failed(e) => eprintln!("lessons: {e:#}"),
        },
    )
}

/// Where a pass got to in each session file.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Progress {
    #[serde(default)]
    pub sessions: BTreeMap<PathBuf, u64>,
}

impl Progress {
    /// Beside the lessons, so the eval's own store is never read as far as the
    /// person's.
    ///
    /// # Errors
    /// If the lessons have no path.
    pub fn path() -> anyhow::Result<PathBuf> {
        Ok(crate::lessons::path()?.with_file_name("lessons-read.json"))
    }

    /// # Errors
    /// If the file exists and cannot be read or parsed.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }

    /// # Errors
    /// If the file cannot be written.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        crate::write_atomically(path, &serde_json::to_string_pretty(self)?)
    }
}

/// Pi's session files, oldest first. Pi keeps them in a directory per working
/// directory under `sessions`, or all in one when `PI_CODING_AGENT_SESSION_DIR` says.
///
/// # Errors
/// If `HOME` is not set, or the directory cannot be listed.
pub fn session_files() -> anyhow::Result<Vec<PathBuf>> {
    let dir = match std::env::var_os("PI_CODING_AGENT_SESSION_DIR").filter(|d| !d.is_empty()) {
        Some(dir) => PathBuf::from(dir),
        None => match std::env::var_os("PI_CODING_AGENT_DIR").filter(|d| !d.is_empty()) {
            Some(agent) => PathBuf::from(agent),
            None => PathBuf::from(
                std::env::var_os("HOME").ok_or_else(|| anyhow::anyhow!("HOME is not set"))?,
            )
            .join(".pi/agent"),
        }
        .join("sessions"),
    };
    let mut files = Vec::new();
    let list = |dir: &Path| -> anyhow::Result<Vec<PathBuf>> {
        match std::fs::read_dir(dir) {
            Ok(entries) => Ok(entries.flatten().map(|e| e.path()).collect()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(e).with_context(|| format!("listing {}", dir.display())),
        }
    };
    for path in list(&dir)? {
        if path.is_dir() {
            files.extend(list(&path)?);
        } else {
            files.push(path);
        }
    }
    files.retain(|p| p.extension().is_some_and(|e| e == "jsonl"));
    // Pi names a session by the time it started.
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    Ok(files)
}

/// What a pass did with one candidate.
#[derive(Debug)]
pub enum Outcome {
    /// A new pending lesson, by id.
    Learned(String),
    /// Another correction for the lesson with this id.
    Repeated(String),
    /// No lesson.
    Nothing,
    /// The distiller could not be asked, or answered nonsense.
    Failed(anyhow::Error),
}

/// Read every session file from where the last pass stopped, and distil each
/// candidate in it into the store at `lessons`. The store is read again before each
/// change and saved right after it, so a review in the meantime is not undone.
///
/// A file's progress is saved once every candidate in it has an answer; a pass that
/// stopped part of the way through one, or failed on one, reads it again next time
/// and skips what was learned from. `go_on` is asked before each candidate, so the
/// gateway's pass can give way to the person.
///
/// # Errors
/// If another pass is running, a file cannot be read, the store cannot be read or
/// written, or the distiller fails `MAX_FAILURES` times in a row.
pub fn pass(
    distiller: &Distiller,
    files: &[PathBuf],
    lessons: &Path,
    progress: &Path,
    mut go_on: impl FnMut() -> bool,
    mut each: impl FnMut(&Candidate, &Outcome),
) -> anyhow::Result<()> {
    let _lock = lock(progress)?;
    let mut read = Progress::load(progress)?;
    let mut failures = 0;
    let mut repos: HashMap<PathBuf, Option<String>> = HashMap::new();

    for file in files {
        let from = read.sessions.get(file).copied().unwrap_or(0);
        let size = std::fs::metadata(file).map_or(0, |m| m.len());
        if size <= from {
            continue;
        }
        let (found, end) = read_session(file, from)?;
        let mut complete = true;
        for candidate in &found {
            if !go_on() {
                return Ok(());
            }
            let kept = Store::load(lessons)?;
            if kept.has_learned_from(&candidate.session, &candidate.entry) {
                continue;
            }
            let repo = repos
                .entry(candidate.cwd.clone())
                .or_insert_with(|| repo_of(&candidate.cwd))
                .clone();
            let outcome =
                match crate::distill::distill(distiller, candidate, repo.as_deref(), &kept.lessons)
                {
                    Ok((verdict, _)) => learn(lessons, distiller, candidate, repo, verdict)?,
                    Err(e) => Outcome::Failed(e),
                };
            each(candidate, &outcome);
            if let Outcome::Failed(e) = outcome {
                complete = false;
                failures += 1;
                if failures == MAX_FAILURES {
                    return Err(e.context(format!(
                        "the distiller failed {MAX_FAILURES} times in a row"
                    )));
                }
            } else {
                failures = 0;
            }
        }
        if complete {
            read.sessions.insert(file.clone(), end);
            read.save(progress)?;
        }
    }
    Ok(())
}

/// Hold the lock beside `progress` for as long as the file it returns is open. Two
/// passes at once would each learn the same lesson.
fn lock(progress: &Path) -> anyhow::Result<std::fs::File> {
    let path = progress.with_extension("lock");
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let file =
        std::fs::File::create(&path).with_context(|| format!("opening {}", path.display()))?;
    rustix::fs::flock(&file, rustix::fs::FlockOperation::NonBlockingLockExclusive).map_err(
        |_| anyhow::anyhow!("lessons are being captured already, by the gateway or another `ailocal lessons capture`"),
    )?;
    Ok(file)
}

fn learn(
    lessons: &Path,
    distiller: &Distiller,
    candidate: &Candidate,
    repo: Option<String>,
    verdict: Verdict,
) -> anyhow::Result<Outcome> {
    let correction = Correction {
        session: candidate.session.clone(),
        entry: candidate.entry.clone(),
        at: candidate.timestamp.clone(),
        said: clip(&candidate.said, MAX_KEPT),
    };
    let mut store = Store::load(lessons)?;
    let outcome = match verdict {
        Verdict::None => return Ok(Outcome::Nothing),
        Verdict::Lesson(draft) => {
            let created = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            let mut lesson =
                crate::distill::pending(draft, repo.as_deref(), distiller.name(), created);
            lesson.corrections.push(correction);
            Outcome::Learned(store.add(lesson)?)
        }
        Verdict::Repeat(id) => {
            // A review in the meantime may have removed it.
            match store.lesson_mut(&id) {
                Ok(lesson) => lesson.corrections.push(correction),
                Err(_) => return Ok(Outcome::Nothing),
            }
            Outcome::Repeated(id)
        }
    };
    store.save(lessons)?;
    Ok(outcome)
}

/// A message of the person's that reads like a correction, with what it answers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// The session file it is in.
    pub session: PathBuf,
    /// Its entry in that file.
    pub entry: String,
    pub timestamp: String,
    /// Where the session ran.
    pub cwd: PathBuf,
    /// The model the agent turn it answers came from.
    pub provider: Option<String>,
    pub model: Option<String>,
    /// What the person said.
    pub said: String,
    /// What the person had asked before it, if anything.
    pub asked: Option<String>,
    /// What the agent did in between, oldest first: its calls, then its last words.
    pub did: Vec<String>,
}

/// One session file, read from `from`: the candidates after that point, and the
/// offset a later pass starts from. A last line still being written is left for it.
///
/// # Errors
/// If the file cannot be read.
pub fn read_session(path: &Path, from: u64) -> anyhow::Result<(Vec<Candidate>, u64)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mut cwd = PathBuf::new();
    let mut nodes: HashMap<String, Node> = HashMap::new();
    let mut found = Vec::new();
    let mut at = 0usize;

    while let Some(end) = bytes[at..].iter().position(|&b| b == b'\n').map(|n| at + n) {
        let line = &bytes[at..end];
        let start = at as u64;
        at = end + 1;
        let Ok(entry) = serde_json::from_slice::<serde_json::Value>(line) else {
            continue;
        };
        if entry["type"] == "session" {
            cwd = PathBuf::from(entry["cwd"].as_str().unwrap_or_default());
            continue;
        }
        let Some(id) = entry["id"].as_str() else {
            continue;
        };
        let node = Node::from(&entry);
        if start >= from
            && let Kind::Person(said) = &node.kind
            && corrects(said)
        {
            found.extend(candidate(path, &cwd, &entry, said, &node, &nodes));
        }
        nodes.insert(id.to_owned(), node);
    }
    Ok((found, at as u64))
}

/// The candidate a message of the person's makes, or none when it answers no agent
/// turn: a correction is of something the agent did, and a first prompt that says
/// "don't" is an instruction. Measured here, none of the 164 such was a correction.
fn candidate(
    session: &Path,
    cwd: &Path,
    entry: &serde_json::Value,
    said: &str,
    node: &Node,
    nodes: &HashMap<String, Node>,
) -> Option<Candidate> {
    let mut asked = None;
    let mut turns = Vec::new();
    let mut parent = node.parent.as_deref();
    while let Some(node) = parent.and_then(|id| nodes.get(id)) {
        match &node.kind {
            Kind::Person(text) => {
                asked = Some(text.clone());
                break;
            }
            Kind::Agent(turn) => turns.push(turn),
            Kind::Other => {}
        }
        parent = node.parent.as_deref();
    }

    let last = turns.first()?;
    let mut did: Vec<String> = turns.iter().rev().flat_map(|t| t.calls.clone()).collect();
    did.drain(..did.len().saturating_sub(MAX_STEPS));
    if let Some(words) = turns.iter().find_map(|t| t.words.as_ref()) {
        did.push(format!("said: {words}"));
    }
    Some(Candidate {
        session: session.to_owned(),
        entry: entry["id"].as_str().unwrap_or_default().to_owned(),
        timestamp: entry["timestamp"].as_str().unwrap_or_default().to_owned(),
        cwd: cwd.to_owned(),
        provider: last.provider.clone(),
        model: last.model.clone(),
        said: said.to_owned(),
        asked,
        did,
    })
}

/// The repository a session ran in, as lessons name it, from its origin as configured.
/// `None` when the directory is gone, as a removed worktree is, or has no origin.
#[must_use]
pub fn repo_of(cwd: &Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(["config", "--get", "remote.origin.url"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| crate::lessons::repo_id(&String::from_utf8_lossy(&output.stdout)))
        .flatten()
}

/// What a session entry is, as much of it as a candidate needs.
struct Node {
    parent: Option<String>,
    kind: Kind,
}

enum Kind {
    /// A message the person typed.
    Person(String),
    Agent(Turn),
    /// Tool results, compactions, ailocal's own messages and the rest.
    Other,
}

struct Turn {
    provider: Option<String>,
    model: Option<String>,
    calls: Vec<String>,
    words: Option<String>,
}

impl Node {
    fn from(entry: &serde_json::Value) -> Self {
        let parent = entry["parentId"].as_str().map(str::to_owned);
        let message = &entry["message"];
        let kind = match (entry["type"].as_str(), message["role"].as_str()) {
            (Some("message"), Some("user")) => {
                let text = texts(&message["content"]);
                if text.is_empty() {
                    Kind::Other
                } else {
                    Kind::Person(text)
                }
            }
            (Some("message"), Some("assistant")) => {
                let parts = message["content"].as_array().map_or(&[][..], Vec::as_slice);
                let words = texts(&message["content"]);
                Kind::Agent(Turn {
                    provider: message["provider"].as_str().map(str::to_owned),
                    model: message["model"].as_str().map(str::to_owned),
                    calls: parts
                        .iter()
                        .filter(|p| p["type"] == "toolCall")
                        .map(step)
                        .collect(),
                    words: (!words.is_empty()).then(|| clip(&words, MAX_STEP)),
                })
            }
            _ => Kind::Other,
        };
        Self { parent, kind }
    }
}

/// The text parts of a message's content, which is a string or a list of parts.
fn texts(content: &serde_json::Value) -> String {
    if let Some(text) = content.as_str() {
        return text.trim().to_owned();
    }
    let parts: Vec<&str> = content
        .as_array()
        .into_iter()
        .flatten()
        .filter(|p| p["type"] == "text")
        .filter_map(|p| p["text"].as_str())
        .collect();
    parts.join("\n").trim().to_owned()
}

/// One tool call, on one line: a command as typed, an edit by its file and the text
/// it adds, any other tool by its first argument.
fn step(call: &serde_json::Value) -> String {
    let name = call["name"].as_str().unwrap_or("tool");
    let args = &call["arguments"];
    let text = match name {
        "bash" => format!("$ {}", args["command"].as_str().unwrap_or_default()),
        "edit" => {
            let added: Vec<&str> = args["edits"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|e| e["newText"].as_str())
                .collect();
            format!(
                "edit {}: {}",
                args["path"].as_str().unwrap_or_default(),
                added.join(" / ")
            )
        }
        _ => {
            let first = args
                .as_object()
                .and_then(|a| a.values().next())
                .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_owned))
                .unwrap_or_default();
            format!("{name} {first}")
        }
    };
    clip(&text, MAX_STEP)
}

/// `text` on one line, cut to about `max` characters.
fn clip(text: &str, max: usize) -> String {
    let line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match line.char_indices().nth(max) {
        Some((cut, _)) => format!("{}...", &line[..cut]),
        None => line,
    }
}

/// What a correction says, as literals rather than patterns: told to the agent about
/// what it did, asked of it about why, or said about next time.
const CORRECTING: [&str; 43] = [
    "you didn't",
    "you did not",
    "you don't",
    "you do not",
    "you forgot",
    "you ignored",
    "you missed",
    "you keep",
    "you kept",
    "you failed",
    "you have failed",
    "you've failed",
    "you should have",
    "you shouldn't have",
    "you were supposed to",
    "why did you",
    "why didn't you",
    "why are you",
    "why aren't you",
    "why do you",
    "why don't you",
    "why would you",
    "why is there",
    "why was",
    "i said",
    "i asked",
    "i told you",
    "i specifically",
    "again, you",
    "again you",
    "not what i",
    "that's not",
    "that is not",
    "that's wrong",
    "is not a solution",
    "isn't a solution",
    "incorrect",
    "instead of",
    "don't",
    "do not",
    "never",
    "in future",
    "in the future",
];

/// Whether the person's message reads like a correction: short enough to be one, and
/// saying something of the kind. The distiller decides whether it is one.
#[must_use]
pub fn corrects(said: &str) -> bool {
    if said.chars().count() > MAX_SAID {
        return false;
    }
    let said = said.to_lowercase().replace(['\u{2019}', '`'], "'");
    CORRECTING.iter().any(|phrase| said.contains(phrase))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(name: &str, lines: &[serde_json::Value]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ailocal-capture-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
        std::fs::write(&path, text).unwrap();
        path
    }

    fn user(id: &str, parent: &str, text: &str) -> serde_json::Value {
        serde_json::json!({"type": "message", "id": id, "parentId": parent,
            "timestamp": "2026-11-25T23:30:00.000Z",
            "message": {"role": "user", "content": [{"type": "text", "text": text}]}})
    }

    fn agent(id: &str, parent: &str, content: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"type": "message", "id": id, "parentId": parent,
            "message": {"role": "assistant", "provider": "llama.cpp",
                        "model": "qwen3.6-35b-a3b", "content": content}})
    }

    #[test]
    fn a_correction_is_read_with_what_it_answers() {
        let path = write(
            "answers.jsonl",
            &[
                serde_json::json!({"type": "session", "version": 3, "id": "s", "cwd": "/w/repo"}),
                user("u1", "", "Open a PR for this branch."),
                agent(
                    "a1",
                    "u1",
                    serde_json::json!([
                        {"type": "thinking", "thinking": "..."},
                        {"type": "toolCall", "name": "bash",
                         "arguments": {"command": "gh pr create --title T\n  --body '## Summary'"}},
                    ]),
                ),
                serde_json::json!({"type": "message", "id": "r1", "parentId": "a1",
                    "message": {"role": "toolResult", "content": [{"type": "text", "text": "url"}]}}),
                serde_json::json!({"type": "custom_message", "id": "c1", "parentId": "r1",
                    "customType": "ailocal-checks", "content": "ailocal: why did you not test?"}),
                agent(
                    "a2",
                    "c1",
                    serde_json::json!([{"type": "text", "text": "Opened the PR."}]),
                ),
                user("u2", "a2", "Why didn't you use the PR template in .github?"),
            ],
        );

        let (found, end) = read_session(&path, 0).unwrap();
        assert_eq!(end, std::fs::metadata(&path).unwrap().len());
        assert_eq!(
            found,
            [Candidate {
                session: path.clone(),
                entry: "u2".to_owned(),
                timestamp: "2026-11-25T23:30:00.000Z".to_owned(),
                cwd: PathBuf::from("/w/repo"),
                provider: Some("llama.cpp".to_owned()),
                model: Some("qwen3.6-35b-a3b".to_owned()),
                said: "Why didn't you use the PR template in .github?".to_owned(),
                asked: Some("Open a PR for this branch.".to_owned()),
                did: vec![
                    "$ gh pr create --title T --body '## Summary'".to_owned(),
                    "said: Opened the PR.".to_owned(),
                ],
            }]
        );
    }

    /// A pass takes up where the last one stopped, and leaves a line still being
    /// written for the next.
    #[test]
    fn a_pass_reads_only_what_is_new() {
        let lines = [
            serde_json::json!({"type": "session", "version": 3, "id": "s", "cwd": "/w"}),
            user("u0", "", "Lint it."),
            agent(
                "a0",
                "u0",
                serde_json::json!([{"type": "text", "text": "Done."}]),
            ),
            user("u1", "a0", "Don't add eslint-disable, fix the rule."),
            agent(
                "a1",
                "u1",
                serde_json::json!([{"type": "text", "text": "Fixed."}]),
            ),
            user("u2", "a1", "You did not run prettier again."),
        ];
        let path = write("passes.jsonl", &lines[..4]);
        let (first, end) = read_session(&path, 0).unwrap();
        assert_eq!(first.len(), 1);

        let mut text = std::fs::read_to_string(&path).unwrap();
        for line in &lines[4..] {
            text.push_str(&format!("{line}\n"));
        }
        text.push_str(r#"{"type":"message","id":"u3""#);
        std::fs::write(&path, &text).unwrap();
        let (second, next) = read_session(&path, end).unwrap();
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].entry, "u2");
        assert_eq!(second[0].asked.as_deref(), Some(first[0].said.as_str()));
        assert_eq!(next as usize, text.rfind('\n').unwrap() + 1);
    }

    /// A first prompt says "do not" as an instruction; nothing before it was the
    /// agent's to correct.
    #[test]
    fn a_message_that_answers_no_agent_turn_is_no_correction() {
        let path = write(
            "first.jsonl",
            &[
                serde_json::json!({"type": "session", "version": 3, "id": "s", "cwd": "/w"}),
                user(
                    "u1",
                    "",
                    "Reply with exactly the word READY. Do nothing else.",
                ),
            ],
        );
        assert!(corrects(
            "Reply with exactly the word READY. Do nothing else."
        ));
        assert_eq!(read_session(&path, 0).unwrap().0, []);
    }

    #[test]
    fn what_reads_like_a_correction() {
        for said in [
            "You did not use the default PR description template like i asked you to.",
            "Why didn\u{2019}t you run Prettier?",
            "Adding `eslint-disable` is not a solution to fixing linting errors.",
            "Again, you have failed to use the default PR description template.",
            "I don't think this warrants a new component.",
            "Tell me why, so that I can prevent you from doing this in future.",
        ] {
            assert!(corrects(said), "{said}");
        }
        for said in [
            "hello?",
            "Great. Can you address the review comments and push?",
            "Run aip handoff and continue the TOOL10 plan.",
            &format!("Why did you {}", "x".repeat(MAX_SAID)),
        ] {
            assert!(!corrects(said), "{said}");
        }
    }

    /// A llama-server that gives `answers` in turn and keeps what it was asked.
    fn distiller(
        answers: Vec<serde_json::Value>,
    ) -> (Distiller, std::sync::mpsc::Receiver<String>) {
        use std::io::{BufRead as _, BufReader, Read as _, Write as _};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let (asked, questions) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut answers = answers.into_iter();
            for stream in listener.incoming() {
                let mut stream = BufReader::new(stream.unwrap());
                let (mut length, mut first) = (0, String::new());
                stream.read_line(&mut first).unwrap();
                loop {
                    let mut line = String::new();
                    stream.read_line(&mut line).unwrap();
                    if let Some(n) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        length = n.trim().parse().unwrap();
                    }
                    if line == "\r\n" {
                        break;
                    }
                }
                if first.starts_with("GET /slots") {
                    let slots = r#"[{"id":0,"is_processing":false,"n_prompt_tokens":0}]"#;
                    write!(
                        stream.get_mut(),
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{slots}",
                        slots.len()
                    )
                    .unwrap();
                    continue;
                }
                let Some(answer) = answers.next() else {
                    return;
                };
                let mut body = vec![0; length];
                stream.read_exact(&mut body).unwrap();
                asked.send(String::from_utf8(body).unwrap()).unwrap();
                let reply =
                    serde_json::json!({"choices": [{"message": {"content": answer.to_string()}}]})
                        .to_string();
                write!(
                    stream.get_mut(),
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{reply}",
                    reply.len()
                )
                .unwrap();
            }
        });
        let distiller = Distiller::Local {
            base,
            model: "m".to_owned(),
        };
        (distiller, questions)
    }

    fn verdict(verdict: &str, lesson: &str) -> serde_json::Value {
        serde_json::json!({"reason": "r", "verdict": verdict, "lesson": lesson,
            "scope": "everywhere", "commands": ["gh pr create"], "adds": [], "files": []})
    }

    fn same(id: &str) -> serde_json::Value {
        serde_json::json!({"reason": "r", "same": id})
    }

    /// A lesson is learned once, a repeat adds to it, and a later pass reads only what
    /// was written since.
    #[test]
    fn a_pass_learns_a_lesson_once_and_counts_each_repeat() {
        let lines = [
            serde_json::json!({"type": "session", "version": 3, "id": "s", "cwd": "/nowhere"}),
            user("u0", "", "Open a PR."),
            agent(
                "a0",
                "u0",
                serde_json::json!([{"type": "text", "text": "Opened."}]),
            ),
            user("u1", "a0", "You did not use the PR template."),
            agent(
                "a1",
                "u1",
                serde_json::json!([{"type": "text", "text": "Fixed."}]),
            ),
            user("u2", "a1", "Thanks. Don't forget it is due today."),
            agent(
                "a2",
                "u2",
                serde_json::json!([{"type": "text", "text": "Opened."}]),
            ),
            user("u3", "a2", "Again, you did not use the PR template."),
        ];
        let session = write("pass.jsonl", &lines[..6]);
        let dir = session.with_file_name("pass");
        std::fs::create_dir_all(&dir).unwrap();
        let (store, progress) = (dir.join("lessons.json"), dir.join("lessons-read.json"));
        let text = "Use the template in .github/ for a pull request body.";

        // With nothing kept yet, whether it is one already kept is never asked, nor is
        // it for a message that teaches nothing.
        let (first, asked) = distiller(vec![verdict("lesson", text), verdict("none", "")]);
        let mut seen = Vec::new();
        pass(
            &first,
            std::slice::from_ref(&session),
            &store,
            &progress,
            || true,
            |c, o| {
                seen.push(format!("{} {o:?}", c.entry));
            },
        )
        .unwrap();
        assert_eq!(
            seen,
            ["u1 Learned(\"use-template-github-pull\")", "u2 Nothing"]
        );
        let asked: Vec<String> = asked.try_iter().collect();
        assert_eq!(asked.len(), 2);
        assert!(asked[0].contains("You did not use the PR template."));
        assert!(asked[1].contains("Don't forget it is due today."));

        let mut more = std::fs::read_to_string(&session).unwrap();
        for line in &lines[6..] {
            more.push_str(&format!("{line}\n"));
        }
        std::fs::write(&session, more).unwrap();
        let (second, asked) = distiller(vec![
            verdict("lesson", "Follow the pull request template."),
            same("use-template-github-pull"),
        ]);
        pass(
            &second,
            std::slice::from_ref(&session),
            &store,
            &progress,
            || true,
            |_, _| {},
        )
        .unwrap();
        let asked: Vec<String> = asked.try_iter().collect();
        assert!(asked[0].contains("Again, you did not use the PR template."));
        assert!(asked[1].contains("- use-template-github-pull: Use the template"));

        let lessons = Store::load(&store).unwrap().lessons;
        assert_eq!(lessons.len(), 1);
        assert_eq!(lessons[0].text, text);
        assert_eq!(lessons[0].state, crate::lessons::State::Pending);
        assert_eq!(lessons[0].commands, ["gh pr create"]);
        assert_eq!(lessons[0].source, "llama.cpp/m");
        let from: Vec<_> = lessons[0]
            .corrections
            .iter()
            .map(|c| c.entry.as_str())
            .collect();
        assert_eq!(from, ["u1", "u3"]);

        // Nothing new: a distiller that answers nothing is never asked.
        let (idle, _asked) = distiller(Vec::new());
        pass(
            &idle,
            &[session],
            &store,
            &progress,
            || true,
            |_, _| panic!("asked"),
        )
        .unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A candidate the distiller failed on is asked again by the next pass.
    #[test]
    fn a_pass_that_fails_on_a_candidate_reads_it_again() {
        let session = write(
            "failed.jsonl",
            &[
                serde_json::json!({"type": "session", "version": 3, "id": "s", "cwd": "/nowhere"}),
                user("u0", "", "Lint it."),
                agent(
                    "a0",
                    "u0",
                    serde_json::json!([{"type": "text", "text": "Done."}]),
                ),
                user("u1", "a0", "Don't add eslint-disable."),
            ],
        );
        let dir = session.with_file_name("failed");
        std::fs::create_dir_all(&dir).unwrap();
        let (store, progress) = (dir.join("lessons.json"), dir.join("lessons-read.json"));

        let (confused, _asked) = distiller(vec![serde_json::json!("no idea")]);
        let mut outcomes = Vec::new();
        pass(
            &confused,
            std::slice::from_ref(&session),
            &store,
            &progress,
            || true,
            |_, o| {
                outcomes.push(matches!(o, Outcome::Failed(_)));
            },
        )
        .unwrap();
        assert_eq!(outcomes, [true]);
        assert!(Progress::load(&progress).unwrap().sessions.is_empty());

        let (fine, _asked) = distiller(vec![verdict("lesson", "Fix a lint error at its cause.")]);
        pass(&fine, &[session], &store, &progress, || true, |_, _| {}).unwrap();
        assert_eq!(Store::load(&store).unwrap().lessons.len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_step_is_one_short_line() {
        let edit = serde_json::json!({"type": "toolCall", "name": "edit", "arguments": {
            "path": "src/a.ts", "edits": [{"oldText": "x", "newText": "// eslint-disable-next-line\nx"}]}});
        assert_eq!(step(&edit), "edit src/a.ts: // eslint-disable-next-line x");
        let read = serde_json::json!({"type": "toolCall", "name": "read",
            "arguments": {"path": "AGENTS.md"}});
        assert_eq!(step(&read), "read AGENTS.md");
        assert_eq!(clip(&"a ".repeat(400), 6), "a a a ...");
    }
}
