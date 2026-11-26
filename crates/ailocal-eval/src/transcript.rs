//! What a coding agent did, read back from Pi's `--mode json` event stream.
//!
//! The repository shows what an agent left behind; only the event stream shows how it
//! got there - whether the tests ran before the push or only CI found out. So both are
//! scored, and this is the half that reads the stream.
//!
//! Commands are classified by their words rather than by a pattern over the raw text,
//! because a model chains them: `cargo test && git commit -am x && git push` is a test
//! run and a push in one call, and only the order of the words says which came first.

use serde::{Deserialize, Serialize};

/// How the result of a call that ailocal's Pi extension held begins - its `HELD`.
///
/// Pi reports a held call as an ordinary failed one, and only the text says the
/// command never ran.
const HELD: &str = "ailocal: ";

/// How the result of a call held for a lesson begins, after [`HELD`].
const LESSON_HELD: &str = "this call was held once, because ";

/// The custom message the extension gives lessons with, alongside a prompt.
const LESSONS_MESSAGE: &str = "ailocal-lessons";

/// One tool call, as Pi reported it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Call {
    pub tool: String,
    /// The shell command, for `bash`; the path, for a file tool.
    pub target: String,
    pub is_error: bool,
    /// Held by ailocal's Pi extension, so it never ran.
    #[serde(default)]
    pub held: bool,
    /// What the extension said when it held the call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Everything a run's score needs from its event stream.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Transcript {
    pub calls: Vec<Call>,
    /// The last thing the agent said, which is usually its account of what it did.
    pub answer: String,
    pub turns: u32,
    pub generated_tokens: u64,
    /// The most context any one request used, against the window it was given.
    pub peak_context: u64,
    /// Set when Pi stopped on an error rather than because the agent finished.
    pub error: Option<String>,
    /// The lessons the extension gave alongside a prompt, by id.
    #[serde(default)]
    pub lessons_given: Vec<String>,
}

/// Read a `--mode json` stream. Lines that are not JSON are skipped, since Pi's own
/// diagnostics share the process.
#[must_use]
pub fn parse(events: &str) -> Transcript {
    let mut t = Transcript::default();
    let mut pending: Vec<(String, usize)> = Vec::new();

    for line in events.lines() {
        let Ok(event) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        match event["type"].as_str().unwrap_or_default() {
            "tool_execution_start" => {
                let args = &event["args"];
                let target = args["command"]
                    .as_str()
                    .or_else(|| args["path"].as_str())
                    .unwrap_or_default();
                pending.push((
                    event["toolCallId"].as_str().unwrap_or_default().to_owned(),
                    t.calls.len(),
                ));
                t.calls.push(Call {
                    tool: event["toolName"].as_str().unwrap_or_default().to_owned(),
                    target: target.to_owned(),
                    is_error: false,
                    held: false,
                    reason: None,
                });
            }
            "tool_execution_end" => {
                let id = event["toolCallId"].as_str().unwrap_or_default();
                if let Some(&(_, index)) = pending.iter().find(|(p, _)| p == id) {
                    let is_error = event["isError"].as_bool().unwrap_or(false);
                    let text = event["result"]["content"][0]["text"].as_str();
                    let call = &mut t.calls[index];
                    call.is_error = is_error;
                    call.held = is_error && text.is_some_and(|t| t.starts_with(HELD));
                    if call.held {
                        call.reason = text.map(str::to_owned);
                    }
                }
            }
            "message_end" if event["message"]["customType"] == LESSONS_MESSAGE => {
                let ids = event["message"]["details"]["lessons"].as_array();
                t.lessons_given.extend(
                    ids.into_iter()
                        .flatten()
                        .filter_map(|id| id.as_str().map(str::to_owned)),
                );
            }
            "turn_start" => t.turns += 1,
            "message_end" if event["message"]["role"] == "assistant" => {
                let message = &event["message"];
                let usage = &message["usage"];
                let count = |key: &str| usage[key].as_u64().unwrap_or(0);
                t.generated_tokens += count("output");
                t.peak_context = t
                    .peak_context
                    .max(count("input") + count("cacheRead") + count("output"));

                let text: String = message["content"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|part| part["type"] == "text")
                    .filter_map(|part| part["text"].as_str())
                    .collect();
                if !text.trim().is_empty() {
                    t.answer = text;
                }
                if message["stopReason"] == "error" {
                    t.error = Some(
                        message["errorMessage"]
                            .as_str()
                            .unwrap_or("the provider returned an error")
                            .to_owned(),
                    );
                }
            }
            _ => {}
        }
    }
    t
}

impl Call {
    fn commands(&self) -> Vec<Vec<String>> {
        if self.tool == "bash" {
            simple_commands(&self.target)
        } else {
            Vec::new()
        }
    }

    /// Whether the extension held this call for a lesson, rather than for a check.
    #[must_use]
    pub fn held_for_lesson(&self) -> bool {
        self.reason
            .as_deref()
            .and_then(|r| r.strip_prefix(HELD))
            .is_some_and(|r| r.starts_with(LESSON_HELD))
    }

    /// Whether this call changed a file through a file tool.
    ///
    /// A `sed -i` in the shell is not counted, which is why [`tested_before_push`]
    /// falls back to "any test before the push" when it finds no edit.
    #[must_use]
    pub fn changes_files(&self) -> bool {
        matches!(self.tool.as_str(), "edit" | "write")
    }

    fn first(&self, wanted: fn(&[String]) -> bool) -> Option<usize> {
        self.commands().iter().position(|c| wanted(c))
    }

    #[must_use]
    pub fn runs_tests(&self) -> bool {
        self.first(is_test).is_some()
    }

    #[must_use]
    pub fn pushes(&self) -> bool {
        self.first(is_push).is_some()
    }

    /// `cargo test && git push` in one call: the push only happens after the tests.
    fn tests_then_pushes(&self) -> bool {
        matches!((self.first(is_test), self.first(is_push)), (Some(t), Some(p)) if t < p)
    }

    /// Whether this shell command runs Python over JSON.
    ///
    /// Any mention of `json` alongside a Python interpreter counts - `python3 -c
    /// 'import json...'`, `python3 -m json.tool`, a script fed `data.json`, or a
    /// heredoc that imports it.
    #[must_use]
    pub fn python_on_json(&self) -> bool {
        let invokes_python = self.commands().iter().flatten().any(|word| is_python(word));
        invokes_python && self.target.to_ascii_lowercase().contains("json")
    }
}

/// Whether a test run came after the last edit and before the first successful push.
///
/// `None` when nothing was pushed, since there was nothing for a test to come before.
#[must_use]
pub fn tested_before_push(calls: &[Call]) -> Option<bool> {
    let push = calls.iter().position(|c| c.pushes() && !c.is_error)?;
    let from = calls[..push]
        .iter()
        .rposition(Call::changes_files)
        .map_or(0, |edit| edit + 1);
    Some(calls[from..push].iter().any(Call::runs_tests) || calls[push].tests_then_pushes())
}

/// `cargo test`, `cargo +stable test`, `cargo -q nextest run`.
fn is_test(words: &[String]) -> bool {
    subcommand(words, "cargo", &[]).is_some_and(|s| s == "test" || s == "nextest")
}

/// `git push`, `git -C dir push`.
fn is_push(words: &[String]) -> bool {
    subcommand(words, "git", &["-C", "-c"]).is_some_and(|s| s == "push")
}

/// The first word after `program` that is not an option, skipping the argument of any
/// option in `takes_value`.
fn subcommand<'a>(words: &'a [String], program: &str, takes_value: &[&str]) -> Option<&'a str> {
    let start = words.iter().position(|w| w == program)? + 1;
    let mut rest = words[start..].iter();
    while let Some(word) = rest.next() {
        if takes_value.contains(&word.as_str()) {
            rest.next();
        } else if !word.starts_with('-') && !word.starts_with('+') {
            return Some(word);
        }
    }
    None
}

/// `python`, `python3`, `python3.12`, or any of those as a path.
fn is_python(word: &str) -> bool {
    let name = word.rsplit('/').next().unwrap_or(word);
    name.strip_prefix("python")
        .is_some_and(|version| version.chars().all(|c| c.is_ascii_digit() || c == '.'))
}

/// Split a shell command into its simple commands, each a list of words.
///
/// Enough of the shell to read what a model typed: quotes group words and hide the
/// separators inside them, and `;`, `&`, `|`, newlines and parentheses end a command.
/// It does not expand anything, which is right for classifying intent.
fn simple_commands(command: &str) -> Vec<Vec<String>> {
    let mut commands = Vec::new();
    let mut words: Vec<String> = Vec::new();
    let mut word = String::new();
    let mut quote: Option<char> = None;
    let mut chars = command.chars();

    let end_word = |word: &mut String, words: &mut Vec<String>| {
        if !word.is_empty() {
            words.push(std::mem::take(word));
        }
    };

    while let Some(c) = chars.next() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some('"'), '\\') => word.extend(chars.next()),
            (Some(_), c) => word.push(c),
            (None, '\'' | '"') => quote = Some(c),
            (None, '\\') => word.extend(chars.next()),
            (None, ';' | '&' | '|' | '\n' | '(' | ')') => {
                end_word(&mut word, &mut words);
                if !words.is_empty() {
                    commands.push(std::mem::take(&mut words));
                }
            }
            (None, c) if c.is_whitespace() => end_word(&mut word, &mut words),
            (None, c) => word.push(c),
        }
    }
    end_word(&mut word, &mut words);
    if !words.is_empty() {
        commands.push(words);
    }
    commands
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bash(command: &str) -> Call {
        Call {
            tool: "bash".to_owned(),
            target: command.to_owned(),
            is_error: false,
            held: false,
            reason: None,
        }
    }

    fn edit() -> Call {
        Call {
            tool: "edit".to_owned(),
            target: "src/lib.rs".to_owned(),
            is_error: false,
            held: false,
            reason: None,
        }
    }

    #[test]
    fn test_runs_are_recognised_with_options_and_wrappers() {
        for command in [
            "cargo test",
            "cd /tmp/x && cargo test 2>&1 | tail -20",
            "cargo +stable test --all",
            "cargo -q nextest run",
        ] {
            assert!(bash(command).runs_tests(), "{command}");
        }
        for command in [
            "cargo build",
            "cargo fmt",
            "echo 'cargo test'",
            "grep test src",
        ] {
            assert!(!bash(command).runs_tests(), "{command}");
        }
    }

    #[test]
    fn pushes_are_recognised_through_git_options() {
        assert!(bash("git push").pushes());
        assert!(bash("git -C /tmp/work push origin main").pushes());
        assert!(bash("git add -A && git commit -m 'fix' && git push").pushes());
        assert!(!bash("git commit -m 'push it'").pushes());
        assert!(!bash("git log --oneline").pushes());
    }

    /// A file tool is not a shell command, whatever its path says.
    #[test]
    fn only_bash_calls_are_classified() {
        let read = Call {
            tool: "read".to_owned(),
            target: "cargo test && git push".to_owned(),
            is_error: false,
            held: false,
            reason: None,
        };
        assert!(!read.runs_tests() && !read.pushes());
    }

    #[test]
    fn a_test_after_the_last_edit_counts() {
        let calls = [edit(), bash("cargo test"), bash("git push")];
        assert_eq!(tested_before_push(&calls), Some(true));
    }

    /// The case the whole check exists for: the tests ran, then the agent changed
    /// something and pushed what it never tested.
    #[test]
    fn a_test_before_the_last_edit_does_not() {
        let calls = [edit(), bash("cargo test"), edit(), bash("git push")];
        assert_eq!(tested_before_push(&calls), Some(false));
    }

    #[test]
    fn a_chained_test_counts_only_when_it_comes_first() {
        let tested = [edit(), bash("cargo test && git commit -am x && git push")];
        assert_eq!(tested_before_push(&tested), Some(true));

        let after = [edit(), bash("git commit -am x && git push && cargo test")];
        assert_eq!(tested_before_push(&after), Some(false));
    }

    /// An edit made with `sed -i` is invisible here, so with no file-tool edit the
    /// check still wants some test before the push rather than passing vacuously.
    #[test]
    fn with_no_edit_any_earlier_test_counts_and_none_fails() {
        assert_eq!(
            tested_before_push(&[bash("sed -i s/a/b/ Cargo.toml"), bash("git push")]),
            Some(false)
        );
        assert_eq!(
            tested_before_push(&[bash("cargo test"), bash("git push")]),
            Some(true)
        );
    }

    #[test]
    fn a_failed_push_is_not_the_push() {
        let mut rejected = bash("git push");
        rejected.is_error = true;
        let calls = [edit(), rejected, bash("cargo test"), bash("git push")];
        assert_eq!(tested_before_push(&calls), Some(true));
    }

    #[test]
    fn nothing_pushed_is_not_judged() {
        assert_eq!(tested_before_push(&[edit(), bash("cargo test")]), None);
    }

    #[test]
    fn python_over_json_is_recognised_in_its_usual_forms() {
        for command in [
            "python3 -c 'import json; print(json.load(open(\"releases.json\")))'",
            "cat releases.json | python3 -m json.tool",
            "/usr/bin/python3 script.py data.json",
            "python3 - <<'EOF'\nimport json\nEOF",
            "cd /x && python -c \"import json,sys\"",
        ] {
            assert!(bash(command).python_on_json(), "{command}");
        }
        for command in [
            "jq '.[] | select(.yanked | not)' releases.json",
            "python3 build.py",
            "echo pythonic json",
        ] {
            assert!(!bash(command).python_on_json(), "{command}");
        }
    }

    #[test]
    fn quotes_hide_separators_and_group_words() {
        assert_eq!(
            simple_commands("git commit -m 'a; b' && git push"),
            vec![
                vec![
                    "git".to_owned(),
                    "commit".to_owned(),
                    "-m".to_owned(),
                    "a; b".to_owned()
                ],
                vec!["git".to_owned(), "push".to_owned()],
            ]
        );
    }

    #[test]
    fn a_stream_is_read_into_calls_tokens_and_the_last_answer() {
        let events = [
            r#"{"type":"session","version":3}"#,
            "not json at all",
            r#"{"type":"turn_start"}"#,
            r#"{"type":"tool_execution_start","toolCallId":"a","toolName":"bash","args":{"command":"cargo test"}}"#,
            r#"{"type":"tool_execution_end","toolCallId":"a","toolName":"bash","isError":true}"#,
            r#"{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"Done."}],"usage":{"input":100,"output":20,"cacheRead":900},"stopReason":"stop"}}"#,
            r#"{"type":"turn_start"}"#,
            r#"{"type":"message_end","message":{"role":"assistant","content":[{"type":"toolCall"}],"usage":{"input":5,"output":7,"cacheRead":10},"stopReason":"error","errorMessage":"Context size has been exceeded."}}"#,
        ]
        .join("\n");

        let t = parse(&events);
        assert_eq!(
            t.calls,
            vec![Call {
                is_error: true,
                ..bash("cargo test")
            }]
        );
        assert_eq!(t.turns, 2);
        assert_eq!(t.generated_tokens, 27);
        assert_eq!(t.peak_context, 1020);
        assert_eq!(
            t.answer, "Done.",
            "a message with no text must not erase it"
        );
        assert_eq!(t.error.as_deref(), Some("Context size has been exceeded."));
    }

    /// A held call never ran, so it must not count as Python on JSON that did - while a
    /// command that ran and failed still does.
    #[test]
    fn a_call_the_extension_held_is_told_from_one_that_failed() {
        let end = |id: &str, text: &str| {
            serde_json::json!({
                "type": "tool_execution_end", "toolCallId": id, "toolName": "bash",
                "isError": true, "result": {"content": [{"type": "text", "text": text}]},
            })
            .to_string()
        };
        let events = [
            r#"{"type":"tool_execution_start","toolCallId":"a","toolName":"bash","args":{"command":"python3 -m json.tool a.json"}}"#.to_owned(),
            end("a", &format!("{HELD}jq is installed, so read and query JSON with jq")),
            r#"{"type":"tool_execution_start","toolCallId":"b","toolName":"bash","args":{"command":"python3 -m json.tool a.json"}}"#.to_owned(),
            end("b", "Expecting value: line 1 column 1 (char 0)"),
        ]
        .join("\n");

        let t = parse(&events);
        assert!(t.calls[0].is_error && t.calls[0].held);
        assert!(t.calls[1].is_error && !t.calls[1].held);
    }

    #[test]
    fn the_prefix_is_the_one_the_extension_holds_with() {
        let declared = format!("export const HELD = \"{HELD}\";");
        assert!(
            ailocal::harness::PI_EXTENSION.contains(&declared),
            "the extension no longer declares {declared}"
        );
        for said in [
            format!("`{LESSON_HELD}${{one ? \"a lesson\" : \"lessons\"}}"),
            format!("customType: \"{LESSONS_MESSAGE}\""),
            "details: { lessons: relevant.map((lesson) => lesson.id) }".to_owned(),
        ] {
            assert!(
                ailocal::harness::PI_EXTENSION.contains(&said),
                "the extension no longer says {said}"
            );
        }
    }

    /// What the lesson arms are scored on: which lessons went along with the prompt,
    /// and which calls were held for one rather than for a check.
    #[test]
    fn lessons_given_and_held_are_read_from_the_stream() {
        let events = [
            serde_json::json!({
                "type": "message_end",
                "message": {"role": "custom", "customType": "ailocal-lessons",
                            "content": "...", "details": {"lessons": ["pr-template", "x"]}},
            }),
            serde_json::json!({"type": "tool_execution_start", "toolCallId": "a",
                               "toolName": "bash", "args": {"command": "gh pr create"}}),
            serde_json::json!({"type": "tool_execution_end", "toolCallId": "a", "isError": true,
                               "result": {"content": [{"type": "text",
                                          "text": format!("{HELD}{LESSON_HELD}a lesson ...")}]}}),
            serde_json::json!({"type": "tool_execution_start", "toolCallId": "b",
                               "toolName": "bash", "args": {"command": "git push"}}),
            serde_json::json!({"type": "tool_execution_end", "toolCallId": "b", "isError": true,
                               "result": {"content": [{"type": "text",
                                          "text": format!("{HELD}this push was held once")}]}}),
        ]
        .map(|e| e.to_string())
        .join("\n");

        let t = parse(&events);
        assert_eq!(t.lessons_given, ["pr-template", "x"]);
        assert!(t.calls[0].held_for_lesson());
        assert!(t.calls[1].held && !t.calls[1].held_for_lesson());
    }
}
