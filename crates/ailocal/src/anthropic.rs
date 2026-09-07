//! Translation between the Anthropic Messages API and OpenAI chat completions.
//!
//! This is what lets `ANTHROPIC_BASE_URL` point at us: Claude Code speaks Anthropic and
//! llama-server speaks OpenAI, so something has to sit between them. It is the same
//! shape as pointing Claude Code at any other Anthropic-compatible endpoint.
//!
//! Text is the easy half. The work is in tool calls, where the two schemas disagree on
//! structure rather than naming:
//!
//! - Anthropic puts tool results in a *user* message as `tool_result` blocks; OpenAI
//!   wants a separate message per result with `role: "tool"`.
//! - Anthropic streams tool arguments as `input_json_delta` fragments against a block
//!   index; OpenAI streams them as `arguments` fragments against a tool-call index.
//! - Anthropic sends a JSON object for `input`; OpenAI sends `arguments` as a *string*
//!   of JSON.

use serde_json::{Map, Value, json};

/// Convert an Anthropic Messages request into an OpenAI chat completion request.
///
/// # Errors
/// If `messages` is missing or not an array.
pub fn request_to_openai(request: &Value) -> anyhow::Result<Value> {
    let mut out = Map::new();

    out.insert("model".into(), request["model"].clone());
    if let Some(v) = request.get("max_tokens") {
        out.insert("max_tokens".into(), v.clone());
    }
    for key in ["temperature", "top_p", "stream"] {
        if let Some(v) = request.get(key) {
            out.insert(key.into(), v.clone());
        }
    }
    if let Some(stop) = request.get("stop_sequences") {
        out.insert("stop".into(), stop.clone());
    }

    let mut messages = Vec::new();

    // Anthropic carries the system prompt beside the conversation; OpenAI wants it as
    // the first message.
    if let Some(text) = system_text(request.get("system")) {
        messages.push(json!({ "role": "system", "content": text }));
    }

    let source = request
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("request has no \"messages\" array"))?;

    for message in source {
        translate_message(message, &mut messages);
    }
    out.insert("messages".into(), Value::Array(messages));

    if let Some(tools) = request.get("tools").and_then(Value::as_array) {
        let converted: Vec<Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description").cloned().unwrap_or(Value::Null),
                        "parameters": t.get("input_schema").cloned().unwrap_or(json!({
                            "type": "object", "properties": {}
                        })),
                    }
                })
            })
            .collect();
        if !converted.is_empty() {
            out.insert("tools".into(), Value::Array(converted));
        }
    }

    if let Some(choice) = tool_choice_to_openai(request.get("tool_choice")) {
        out.insert("tool_choice".into(), choice);
    }

    Ok(Value::Object(out))
}

/// Flatten Anthropic's `system`, which may be a string or an array of text blocks.
fn system_text(system: Option<&Value>) -> Option<String> {
    match system? {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::Array(blocks) => {
            let joined = blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n");
            (!joined.is_empty()).then_some(joined)
        }
        _ => None,
    }
}

/// Translate one Anthropic message, appending one or more OpenAI messages.
///
/// One in does not mean one out: a user turn carrying several `tool_result` blocks
/// becomes one OpenAI `tool` message per result, plus a user message for any text.
fn translate_message(message: &Value, out: &mut Vec<Value>) {
    let role = message["role"].as_str().unwrap_or("user");

    let blocks = match &message["content"] {
        Value::String(text) => {
            out.push(json!({ "role": role, "content": text }));
            return;
        }
        Value::Array(blocks) => blocks,
        _ => return,
    };

    let mut text = String::new();
    let mut tool_calls = Vec::new();

    for block in blocks {
        match block["type"].as_str() {
            Some("text") => {
                if let Some(t) = block["text"].as_str() {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(t);
                }
            }
            Some("tool_use") => {
                // OpenAI wants arguments as a JSON *string*, not an object.
                let arguments =
                    serde_json::to_string(&block["input"]).unwrap_or_else(|_| "{}".into());
                tool_calls.push(json!({
                    "id": block["id"],
                    "type": "function",
                    "function": { "name": block["name"], "arguments": arguments },
                }));
            }
            Some("tool_result") => {
                // Must be its own message, and must be emitted before any text this
                // turn also carries, so it directly follows the assistant's call.
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"],
                    "content": tool_result_text(block),
                }));
            }
            // Images and thinking blocks have no place in a text-only OpenAI request.
            _ => {}
        }
    }

    if !tool_calls.is_empty() {
        let mut msg = Map::new();
        msg.insert("role".into(), json!("assistant"));
        msg.insert(
            "content".into(),
            if text.is_empty() {
                Value::Null
            } else {
                json!(text)
            },
        );
        msg.insert("tool_calls".into(), Value::Array(tool_calls));
        out.push(Value::Object(msg));
    } else if !text.is_empty() {
        out.push(json!({ "role": role, "content": text }));
    }
}

/// Flatten a `tool_result` block's content, which may be a string or blocks.
fn tool_result_text(block: &Value) -> String {
    match &block["content"] {
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(|i| i.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        other if !other.is_null() => other.to_string(),
        _ => String::new(),
    }
}

fn tool_choice_to_openai(choice: Option<&Value>) -> Option<Value> {
    match choice?["type"].as_str()? {
        "auto" => Some(json!("auto")),
        "any" => Some(json!("required")),
        "none" => Some(json!("none")),
        "tool" => Some(json!({
            "type": "function",
            "function": { "name": choice?["name"] }
        })),
        _ => None,
    }
}

/// Map an OpenAI finish reason onto Anthropic's vocabulary.
#[must_use]
pub fn stop_reason(finish: Option<&str>) -> Value {
    match finish {
        Some("stop") => json!("end_turn"),
        Some("length") => json!("max_tokens"),
        Some("tool_calls") | Some("function_call") => json!("tool_use"),
        Some("content_filter") => json!("stop_sequence"),
        _ => Value::Null,
    }
}

/// Convert a complete OpenAI response into an Anthropic message.
#[must_use]
pub fn response_to_anthropic(response: &Value, fallback_model: &str) -> Value {
    let choice = &response["choices"][0];
    let message = &choice["message"];

    let mut content = Vec::new();
    if let Some(text) = message["content"].as_str()
        && !text.is_empty()
    {
        content.push(json!({ "type": "text", "text": text }));
    }

    if let Some(calls) = message["tool_calls"].as_array() {
        for call in calls {
            let raw = call["function"]["arguments"].as_str().unwrap_or("{}");
            content.push(json!({
                "type": "tool_use",
                "id": call["id"],
                "name": call["function"]["name"],
                // Anthropic wants a parsed object. A model can emit malformed JSON, so
                // fall back to an empty object rather than failing the whole turn.
                "input": serde_json::from_str::<Value>(raw).unwrap_or_else(|_| json!({})),
            }));
        }
    }

    let usage = &response["usage"];
    json!({
        "id": response["id"].as_str().unwrap_or("msg_local"),
        "type": "message",
        "role": "assistant",
        "model": response["model"].as_str().unwrap_or(fallback_model),
        "content": content,
        "stop_reason": stop_reason(choice["finish_reason"].as_str()),
        "stop_sequence": Value::Null,
        "usage": {
            "input_tokens": usage["prompt_tokens"].as_u64().unwrap_or(0),
            "output_tokens": usage["completion_tokens"].as_u64().unwrap_or(0),
        },
    })
}

/// A single server-sent event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub name: &'static str,
    pub data: String,
}

impl Event {
    fn new(name: &'static str, data: &Value) -> Self {
        Self {
            name,
            data: data.to_string(),
        }
    }

    /// Render in SSE wire format.
    #[must_use]
    pub fn encode(&self) -> String {
        format!("event: {}\ndata: {}\n\n", self.name, self.data)
    }
}

/// Turns a stream of OpenAI chunks into Anthropic's event sequence.
///
/// Anthropic's protocol is block-structured where OpenAI's is flat, so this has to
/// track which content block is open and close it before opening another. Tool calls
/// arrive as fragments against an index, and their first fragment carries the id and
/// name while later ones carry only argument text.
pub struct StreamTranslator {
    model: String,
    started: bool,
    /// Index of the currently open Anthropic content block.
    open_block: Option<usize>,
    /// Next content block index to hand out.
    next_index: usize,
    /// OpenAI tool-call index to the Anthropic block index we opened for it.
    tool_blocks: std::collections::BTreeMap<u64, usize>,
    stop_reason: Value,
    output_tokens: u64,
}

impl StreamTranslator {
    #[must_use]
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_owned(),
            started: false,
            open_block: None,
            next_index: 0,
            tool_blocks: std::collections::BTreeMap::new(),
            stop_reason: Value::Null,
            output_tokens: 0,
        }
    }

    /// Feed one parsed OpenAI chunk, producing the Anthropic events it implies.
    pub fn push(&mut self, chunk: &Value) -> Vec<Event> {
        let mut events = Vec::new();

        if !self.started {
            self.started = true;
            events.push(Event::new(
                "message_start",
                &json!({
                    "type": "message_start",
                    "message": {
                        "id": chunk["id"].as_str().unwrap_or("msg_local"),
                        "type": "message",
                        "role": "assistant",
                        "model": chunk["model"].as_str().unwrap_or(&self.model),
                        "content": [],
                        "stop_reason": Value::Null,
                        "stop_sequence": Value::Null,
                        "usage": { "input_tokens": 0, "output_tokens": 0 },
                    }
                }),
            ));
        }

        if let Some(n) = chunk["usage"]["completion_tokens"].as_u64() {
            self.output_tokens = n;
        }

        let choice = &chunk["choices"][0];
        let delta = &choice["delta"];

        if let Some(text) = delta["content"].as_str()
            && !text.is_empty()
        {
            let index = self.ensure_text_block(&mut events);
            events.push(Event::new(
                "content_block_delta",
                &json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": { "type": "text_delta", "text": text },
                }),
            ));
            self.output_tokens += 1;
        }

        if let Some(calls) = delta["tool_calls"].as_array() {
            for call in calls {
                self.push_tool_fragment(call, &mut events);
            }
        }

        if let Some(finish) = choice["finish_reason"].as_str() {
            self.stop_reason = stop_reason(Some(finish));
        }

        events
    }

    /// Close any open block and emit the terminal events.
    pub fn finish(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        self.close_block(&mut events);

        // A stream that ended without a finish_reason still has to say something, and
        // a normal end of turn is the honest default.
        let reason = if self.stop_reason.is_null() {
            json!("end_turn")
        } else {
            self.stop_reason.clone()
        };

        events.push(Event::new(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": { "stop_reason": reason, "stop_sequence": Value::Null },
                "usage": { "output_tokens": self.output_tokens },
            }),
        ));
        events.push(Event::new(
            "message_stop",
            &json!({ "type": "message_stop" }),
        ));
        events
    }

    fn ensure_text_block(&mut self, events: &mut Vec<Event>) -> usize {
        if let Some(index) = self.open_block
            && self.tool_blocks.values().all(|v| *v != index)
        {
            return index;
        }
        self.close_block(events);

        let index = self.next_index;
        self.next_index += 1;
        self.open_block = Some(index);
        events.push(Event::new(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": index,
                "content_block": { "type": "text", "text": "" },
            }),
        ));
        index
    }

    fn push_tool_fragment(&mut self, call: &Value, events: &mut Vec<Event>) {
        let slot = call["index"].as_u64().unwrap_or(0);

        let index = if let Some(existing) = self.tool_blocks.get(&slot) {
            *existing
        } else {
            self.close_block(events);
            let index = self.next_index;
            self.next_index += 1;
            self.tool_blocks.insert(slot, index);
            self.open_block = Some(index);
            events.push(Event::new(
                "content_block_start",
                &json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": call["id"].as_str().unwrap_or("toolu_local"),
                        "name": call["function"]["name"].as_str().unwrap_or(""),
                        "input": {},
                    },
                }),
            ));
            index
        };

        if let Some(fragment) = call["function"]["arguments"].as_str()
            && !fragment.is_empty()
        {
            events.push(Event::new(
                "content_block_delta",
                &json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": { "type": "input_json_delta", "partial_json": fragment },
                }),
            ));
        }
    }

    fn close_block(&mut self, events: &mut Vec<Event>) {
        if let Some(index) = self.open_block.take() {
            events.push(Event::new(
                "content_block_stop",
                &json!({ "type": "content_block_stop", "index": index }),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompt_becomes_the_first_message() {
        let req = json!({
            "model": "m", "max_tokens": 10,
            "system": "be terse",
            "messages": [{ "role": "user", "content": "hi" }],
        });
        let out = request_to_openai(&req).unwrap();
        assert_eq!(out["messages"][0]["role"], "system");
        assert_eq!(out["messages"][0]["content"], "be terse");
        assert_eq!(out["messages"][1]["content"], "hi");
    }

    #[test]
    fn a_system_block_array_is_flattened() {
        let req = json!({
            "model": "m",
            "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "messages": [{ "role": "user", "content": "hi" }],
        });
        let out = request_to_openai(&req).unwrap();
        assert_eq!(out["messages"][0]["content"], "a\nb");
    }

    #[test]
    fn tools_are_rewritten_into_openai_functions() {
        let req = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            }],
        });
        let out = request_to_openai(&req).unwrap();
        let f = &out["tools"][0];
        assert_eq!(f["type"], "function");
        assert_eq!(f["function"]["name"], "read_file");
        assert_eq!(
            f["function"]["parameters"]["properties"]["path"]["type"],
            "string"
        );
    }

    /// The structural difference that matters: Anthropic nests tool results inside a
    /// user turn, OpenAI wants a standalone message per result.
    #[test]
    fn tool_results_become_their_own_messages() {
        let req = json!({
            "model": "m",
            "messages": [
                { "role": "user", "content": "read it" },
                { "role": "assistant", "content": [
                    { "type": "tool_use", "id": "toolu_1", "name": "read_file",
                      "input": { "path": "/tmp/x" } }
                ]},
                { "role": "user", "content": [
                    { "type": "tool_result", "tool_use_id": "toolu_1", "content": "file contents" }
                ]},
            ],
        });
        let out = request_to_openai(&req).unwrap();
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);

        // Arguments must be a JSON string, not an object.
        let call = &msgs[1]["tool_calls"][0];
        assert_eq!(call["id"], "toolu_1");
        assert_eq!(call["function"]["arguments"], "{\"path\":\"/tmp/x\"}");

        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "toolu_1");
        assert_eq!(msgs[2]["content"], "file contents");
    }

    #[test]
    fn tool_result_blocks_are_flattened_to_text() {
        let req = json!({
            "model": "m",
            "messages": [{ "role": "user", "content": [
                { "type": "tool_result", "tool_use_id": "t1",
                  "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}] }
            ]}],
        });
        let out = request_to_openai(&req).unwrap();
        assert_eq!(out["messages"][0]["content"], "one\ntwo");
    }

    #[test]
    fn tool_choice_is_mapped() {
        let mk = |c: Value| {
            request_to_openai(&json!({
                "model": "m", "messages": [{"role": "user", "content": "x"}], "tool_choice": c
            }))
            .unwrap()["tool_choice"]
                .clone()
        };
        assert_eq!(mk(json!({"type": "auto"})), json!("auto"));
        assert_eq!(mk(json!({"type": "any"})), json!("required"));
        assert_eq!(
            mk(json!({"type": "tool", "name": "f"})),
            json!({"type": "function", "function": {"name": "f"}})
        );
    }

    #[test]
    fn stop_reasons_map_to_anthropic_vocabulary() {
        assert_eq!(stop_reason(Some("stop")), json!("end_turn"));
        assert_eq!(stop_reason(Some("length")), json!("max_tokens"));
        assert_eq!(stop_reason(Some("tool_calls")), json!("tool_use"));
        assert_eq!(stop_reason(None), Value::Null);
    }

    #[test]
    fn a_response_with_tool_calls_becomes_tool_use_blocks() {
        let openai = json!({
            "id": "chatcmpl-1", "model": "gemma4",
            "choices": [{ "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "",
                "tool_calls": [{ "id": "call_1", "type": "function",
                    "function": { "name": "read_file", "arguments": "{\"path\":\"/tmp/x\"}" } }],
            }}],
            "usage": { "prompt_tokens": 7, "completion_tokens": 3 },
        });
        let out = response_to_anthropic(&openai, "fallback");
        assert_eq!(out["stop_reason"], "tool_use");
        assert_eq!(out["content"][0]["type"], "tool_use");
        assert_eq!(out["content"][0]["name"], "read_file");
        // Parsed back into an object, which is what Anthropic clients expect.
        assert_eq!(out["content"][0]["input"]["path"], "/tmp/x");
        assert_eq!(out["usage"]["input_tokens"], 7);
    }

    /// A model can emit malformed arguments; that must not fail the whole turn.
    #[test]
    fn malformed_tool_arguments_degrade_to_an_empty_object() {
        let openai = json!({
            "choices": [{ "finish_reason": "tool_calls", "message": {
                "tool_calls": [{ "id": "c1", "function": { "name": "f", "arguments": "{not json" } }],
            }}],
        });
        let out = response_to_anthropic(&openai, "m");
        assert_eq!(out["content"][0]["input"], json!({}));
    }

    fn names(events: &[Event]) -> Vec<&str> {
        events.iter().map(|e| e.name).collect()
    }

    #[test]
    fn a_text_stream_produces_the_expected_event_sequence() {
        let mut t = StreamTranslator::new("m");
        let mut all = t.push(&json!({
            "id": "c1", "model": "m",
            "choices": [{ "delta": { "content": "Hel" } }]
        }));
        all.extend(t.push(&json!({ "choices": [{ "delta": { "content": "lo" } }] })));
        all.extend(t.push(&json!({ "choices": [{ "delta": {}, "finish_reason": "stop" }] })));
        all.extend(t.finish());

        assert_eq!(
            names(&all),
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let last_delta: Value = serde_json::from_str(&all[5].data).unwrap();
        assert_eq!(last_delta["delta"]["stop_reason"], "end_turn");
    }

    /// The hard case: fragmented tool arguments have to be reassembled against the
    /// right block, and only the first fragment carries the id and name.
    #[test]
    fn a_tool_call_stream_reassembles_fragmented_arguments() {
        let mut t = StreamTranslator::new("m");
        let mut all = t.push(&json!({
            "id": "c1",
            "choices": [{ "delta": { "tool_calls": [
                { "index": 0, "id": "call_1", "function": { "name": "read_file", "arguments": "" } }
            ]}}]
        }));
        all.extend(t.push(&json!({
            "choices": [{ "delta": { "tool_calls": [
                { "index": 0, "function": { "arguments": "{\"path\":" } }
            ]}}]
        })));
        all.extend(t.push(&json!({
            "choices": [{ "delta": { "tool_calls": [
                { "index": 0, "function": { "arguments": "\"/tmp/x\"}" } }
            ]}}], "finish_reason": "tool_calls"
        })));
        all.extend(t.finish());

        let start: Value = serde_json::from_str(&all[1].data).unwrap();
        assert_eq!(start["content_block"]["type"], "tool_use");
        assert_eq!(start["content_block"]["id"], "call_1");
        assert_eq!(start["content_block"]["name"], "read_file");

        let fragments: Vec<String> = all
            .iter()
            .filter(|e| e.name == "content_block_delta")
            .map(|e| {
                serde_json::from_str::<Value>(&e.data).unwrap()["delta"]["partial_json"]
                    .as_str()
                    .unwrap_or("")
                    .to_owned()
            })
            .collect();
        assert_eq!(fragments.concat(), "{\"path\":\"/tmp/x\"}");
        assert!(names(&all).contains(&"content_block_stop"));
    }

    /// Text followed by a tool call must close the text block before opening the tool
    /// block, or the client sees two blocks open at the same index.
    #[test]
    fn switching_from_text_to_a_tool_closes_the_text_block() {
        let mut t = StreamTranslator::new("m");
        let mut all = t.push(&json!({ "choices": [{ "delta": { "content": "thinking" } }] }));
        all.extend(t.push(&json!({
            "choices": [{ "delta": { "tool_calls": [
                { "index": 0, "id": "c1", "function": { "name": "f", "arguments": "{}" } }
            ]}}]
        })));
        all.extend(t.finish());

        assert_eq!(
            names(&all),
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
    }

    #[test]
    fn two_parallel_tool_calls_get_separate_blocks() {
        let mut t = StreamTranslator::new("m");
        let mut all = t.push(&json!({
            "choices": [{ "delta": { "tool_calls": [
                { "index": 0, "id": "a", "function": { "name": "f", "arguments": "{}" } }
            ]}}]
        }));
        all.extend(t.push(&json!({
            "choices": [{ "delta": { "tool_calls": [
                { "index": 1, "id": "b", "function": { "name": "g", "arguments": "{}" } }
            ]}}]
        })));
        all.extend(t.finish());

        let starts: Vec<Value> = all
            .iter()
            .filter(|e| e.name == "content_block_start")
            .map(|e| serde_json::from_str(&e.data).unwrap())
            .collect();
        assert_eq!(starts.len(), 2);
        assert_eq!(starts[0]["index"], 0);
        assert_eq!(starts[1]["index"], 1);
        assert_eq!(starts[1]["content_block"]["id"], "b");
    }

    #[test]
    fn events_encode_in_sse_wire_format() {
        let e = Event::new("message_stop", &json!({ "type": "message_stop" }));
        assert_eq!(
            e.encode(),
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        );
    }

    #[test]
    fn a_request_without_messages_is_rejected() {
        assert!(request_to_openai(&json!({ "model": "m" })).is_err());
    }
}
