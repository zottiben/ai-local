//! An authenticated OpenAI-compatible front end for whichever model is loaded.
//!
//! llama-server already speaks OpenAI, so this exists for the three things it does not
//! do: require a credential, present the whole installed catalogue rather than the one
//! model that happens to be resident, and swap models on demand. Harnesses list models
//! once and expect to pick any of them; the card only fits one at a time.
//!
//! Streaming is proxied through untouched. A coding harness is unusable without
//! token-by-token output, and Cloudflare will drop a long-idle non-streaming request.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

use crate::anthropic;
use crate::config::Config;
use crate::registry::{self, Fit};
use crate::serve;
use crate::vram;

pub struct AppState {
    pub key: String,
    pub config: Config,
    pub client: reqwest::Client,
    /// Where to record incoming conversations, if asked to.
    pub request_log: Option<std::path::PathBuf>,
}

/// Append a conversation to the request log, exactly as the harness sent it.
///
/// Recorded *before* this gateway changes anything, so the log answers "what is my
/// harness actually sending?" rather than showing our own edits back to us. That is
/// otherwise unanswerable from either end - the harness shows you its own view, the
/// model shows you its answer, and the wire between them is the only place the truth
/// lives. It settled whether Pi puts AGENTS.md in the prompt (it does, in a `developer`
/// message) which no amount of reading either side's documentation could.
///
/// Off unless asked for, because a prompt log is a transcript of your work.
fn record_request(path: &std::path::Path, endpoint: &str, body: &serde_json::Value) {
    let messages = body["messages"].as_array().map_or_else(Vec::new, |all| {
        all.iter()
            .map(|m| {
                serde_json::json!({
                    "role": m["role"],
                    // Content is a string for text and an array for anything richer,
                    // so measure the serialised form rather than report zero for every
                    // multimodal message.
                    "chars": m["content"]
                        .as_str()
                        .map_or_else(|| m["content"].to_string().len(), str::len),
                    "content": m["content"],
                })
            })
            .collect()
    });

    let line = serde_json::json!({
        "at": crate::gateway::now_rfc3339(),
        "endpoint": endpoint,
        "model": body["model"],
        "tools": body["tools"].as_array().map_or(0, Vec::len),
        "messages": messages,
    });

    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        && let Ok(text) = serde_json::to_string(&line)
    {
        use std::io::Write as _;
        writeln!(file, "{text}").ok();
    }
}

/// Seconds since the epoch, which is all a log line needs to be orderable.
fn now_rfc3339() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// What is holding a port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortStatus {
    /// Nothing is listening; the gateway can have it.
    Free,
    /// A gateway of ours is already there.
    Ours,
    /// Something else is listening.
    ///
    /// 8081 is a popular port - React Native's Metro bundler defaults to it - so on a
    /// development machine this is not unusual, and it has to be named rather than
    /// left as a bind error in a log file nobody reads.
    Taken,
}

/// The address a local client should use to reach a gateway bound to `host`.
///
/// A wildcard bind is not an address you can connect to, so probing `0.0.0.0` would
/// report the port free while the gateway is sitting on it.
#[must_use]
pub fn loopback_for(host: &str) -> &str {
    match host {
        "0.0.0.0" | "::" | "[::]" | "" => "127.0.0.1",
        other => other,
    }
}

/// Find out whether the gateway can bind `host:port`, and who has it if not.
///
/// Binding is the honest test - it is the same syscall the gateway will make, so it
/// cannot disagree with what happens a moment later the way a connect probe can.
#[must_use]
pub fn port_status(host: &str, port: u16) -> PortStatus {
    if std::net::TcpListener::bind((host, port)).is_ok() {
        return PortStatus::Free;
    }
    if answers_as_gateway(host, port) {
        PortStatus::Ours
    } else {
        PortStatus::Taken
    }
}

/// Whether whatever is on this port behaves like one of our gateways.
///
/// Two probes rather than one: plenty of things answer `/health` with 200, but a
/// service that also rejects an unauthenticated `/v1/models` with 401 is ours to a
/// degree worth acting on.
#[must_use]
pub fn answers_as_gateway(host: &str, port: u16) -> bool {
    let base = format!("http://{}:{port}", loopback_for(host));
    let Ok(client) = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
    else {
        return false;
    };

    let healthy = client
        .get(format!("{base}/health"))
        .send()
        .is_ok_and(|r| r.status().is_success());
    if !healthy {
        return false;
    }
    client
        .get(format!("{base}/v1/models"))
        .send()
        .is_ok_and(|r| r.status() == StatusCode::UNAUTHORIZED)
}

/// How many ports past the preferred one to try before giving up.
const PORT_SEARCH_RANGE: u16 = 20;

/// A port the gateway can actually have, starting from `preferred`.
///
/// Returns `preferred` when it is free or already ours, so an existing setup is never
/// moved out from under its harness configuration.
#[must_use]
pub fn usable_port(host: &str, preferred: u16) -> Option<u16> {
    (preferred..preferred.saturating_add(PORT_SEARCH_RANGE))
        .find(|&port| matches!(port_status(host, port), PortStatus::Free | PortStatus::Ours))
}

/// Wait for a gateway to start answering, up to `limit`.
///
/// A service manager reports success as soon as it has forked, which is well before
/// the listener exists - so anything that checks immediately after starting the
/// service races it and reports a healthy gateway as broken.
#[must_use]
pub fn wait_until_answering(host: &str, port: u16, limit: std::time::Duration) -> bool {
    let deadline = std::time::Instant::now() + limit;
    loop {
        if answers_as_gateway(host, port) {
            return true;
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
}

/// An error rendered in the shape OpenAI clients expect.
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        // Clients surface `error.message`, so anything the user needs to act on has to
        // be in there rather than only in our logs.
        let body = serde_json::json!({
            "error": {
                "message": self.message,
                "type": "invalid_request_error",
                "code": self.status.as_u16(),
            }
        });
        (self.status, Json(body)).into_response()
    }
}

type ApiResult<T> = Result<T, ApiError>;

/// Build the router.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        // Unauthenticated on purpose: the tunnel and any monitoring need to see
        // liveness without holding a credential, and it discloses nothing.
        .route("/health", get(|| async { "ok" }))
        .merge(
            Router::new()
                .route("/v1/models", get(list_models))
                .route("/v1/chat/completions", post(proxy))
                .route("/v1/completions", post(proxy))
                .route("/v1/embeddings", post(proxy))
                // llama.cpp router API. Pi speaks this natively via `/login llama.cpp`,
                // and serving it ourselves means Pi's model picker drives our loader -
                // with the VRAM ceiling enforced - rather than a bare llama-server
                // router that would happily load past it.
                // Anthropic Messages API, which is what makes ANTHROPIC_BASE_URL work
                // for Claude Code.
                .route("/v1/messages", post(messages))
                .route("/models", get(router_list))
                .route("/models/load", post(router_load))
                .route("/models/unload", post(router_unload))
                .route("/props", get(props))
                .layer(axum::middleware::from_fn_with_state(
                    state.clone(),
                    require_bearer,
                )),
        )
        .with_state(state)
}

/// Reject anything without the right bearer token.
async fn require_bearer(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Response {
    // Anthropic clients authenticate with `x-api-key`; OpenAI ones with a bearer
    // token. Accept either, since we serve both protocols on the same port.
    let presented = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(crate::auth::bearer)
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()));

    match presented {
        Some(token) if crate::auth::constant_time_eq(token, &state.key) => next.run(request).await,
        _ => ApiError::new(
            StatusCode::UNAUTHORIZED,
            "missing or invalid API key; send `Authorization: Bearer <key>` or \
             `x-api-key: <key>` (see `ailocal gateway key`)",
        )
        .into_response(),
    }
}

/// Advertise every installed model that could actually run here.
///
/// Only one is resident at a time, but a harness lists models once at startup and
/// expects to choose later, so listing only the loaded one would make the others
/// unreachable. Models that cannot fit are omitted rather than offered and then failed.
async fn list_models(State(state): State<Arc<AppState>>) -> ApiResult<Json<serde_json::Value>> {
    let cache: vram::CacheType = state
        .config
        .cache_type
        .parse()
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?;

    let models = registry::scan(&state.config.models_dir).map_err(|e| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("reading model directory: {e}"),
        )
    })?;
    let budget = serve::budget_for_next_launch()
        .unwrap_or_else(|_| vram::Budget::new(crate::vram_used_mib().unwrap_or(900)));

    let data: Vec<serde_json::Value> = models
        .iter()
        .filter(|m| {
            matches!(
                registry::assess(m.kv, m.trained_context, &budget, cache, m.size_mib),
                Fit::Fits(_)
            )
        })
        .map(|m| {
            serde_json::json!({
                "id": m.name,
                "object": "model",
                "owned_by": "ailocal",
                "created": 0,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({ "object": "list", "data": data })))
}

/// Catalogue in llama.cpp router form.
///
/// Pi requires only `id` and `status.value`, and treats anything other than
/// `unloaded` as available. Models that cannot fit are still listed, so they are
/// visible rather than mysteriously absent - loading one fails with the reason.
///
/// `meta` is not optional in practice even though Pi tolerates its absence. Pi reads
/// `meta.n_ctx` for the context window and falls back to a flat 128000 without it, so
/// omitting it had a harness auto-compacting a 256k session at half its window. The
/// figure reported is the one a launch would actually get, not the trained maximum:
/// promising a context this machine cannot hold only moves the failure later.
async fn router_list(State(state): State<Arc<AppState>>) -> ApiResult<Json<serde_json::Value>> {
    let cache: vram::CacheType = state
        .config
        .cache_type
        .parse()
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?;

    let models = registry::scan(&state.config.models_dir).map_err(|e| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("reading model directory: {e}"),
        )
    })?;
    let running = serve::running().ok().flatten();
    let budget = serve::budget_for_next_launch()
        .unwrap_or_else(|_| vram::Budget::new(crate::vram_used_mib().unwrap_or(900)));

    let data: Vec<serde_json::Value> = models
        .iter()
        .map(|m| {
            let live = running
                .as_ref()
                .filter(|i| i.model == m.name)
                .map(|i| i.context);
            let fits = match registry::assess(m.kv, m.trained_context, &budget, cache, m.size_mib) {
                Fit::Fits(ctx) => Some(ctx),
                _ => None,
            };

            serde_json::json!({
                "id": m.name,
                "object": "model",
                "owned_by": "ailocal",
                "created": 0,
                "status": { "value": if live.is_some() { "loaded" } else { "unloaded" } },
                "meta": {
                    "n_ctx": live.or(fits),
                    "n_ctx_train": m.trained_context,
                },
            })
        })
        .collect();

    Ok(Json(serde_json::json!({ "object": "list", "data": data })))
}

#[derive(serde::Deserialize)]
struct ModelRef {
    model: String,
}

async fn router_load(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ModelRef>,
) -> ApiResult<Json<serde_json::Value>> {
    let instance = ensure_loaded(&state, &body.model).await?;
    Ok(Json(serde_json::json!({
        "success": true,
        "model": instance.model,
        "context": instance.context,
    })))
}

async fn router_unload(Json(body): Json<ModelRef>) -> ApiResult<Json<serde_json::Value>> {
    let stopped = tokio::task::spawn_blocking(serve::stop)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;

    Ok(Json(serde_json::json!({
        "success": true,
        "unloaded": stopped.map(|i| i.model).unwrap_or(body.model),
    })))
}

/// Server properties.
///
/// `models_autoload: false` tells Pi that loading is explicit, which is true here: the
/// budget only holds one model, so loading is a decision rather than a side effect.
async fn props() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "models_autoload": false }))
}

/// Make sure `model` is the one that is loaded, swapping if it is not.
///
/// Blocking work goes on a blocking thread: a model load takes tens of seconds and
/// would otherwise stall the whole runtime.
async fn ensure_loaded(state: &Arc<AppState>, model: &str) -> ApiResult<serve::Instance> {
    if let Ok(Some(current)) = serve::running()
        && current.model == model
    {
        return Ok(current);
    }

    let config = state.config.clone();
    let wanted = model.to_owned();
    tokio::task::spawn_blocking(move || -> anyhow::Result<serve::Instance> {
        let models = registry::scan(&config.models_dir)?;
        let found = models
            .iter()
            .find(|m| m.name == wanted)
            .ok_or_else(|| anyhow::anyhow!("no model named {wanted:?}"))?;

        let budget = serve::budget_for_next_launch()?;
        serve::start(found, &budget, &serve::Options::from_config(&config)?)
    })
    .await
    .map_err(|e| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("load panicked: {e}"),
        )
    })?
    .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, format!("{e:#}")))
}

/// Anthropic Messages endpoint.
///
/// Translates in, forwards to llama-server's OpenAI surface, and translates back -
/// including the streaming case, where the two protocols disagree on structure rather
/// than only on naming.
async fn messages(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> ApiResult<Response> {
    let request: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")))?;

    let model = request["model"]
        .as_str()
        .ok_or_else(|| ApiError::new(StatusCode::BAD_REQUEST, "request has no \"model\" field"))?
        .to_owned();

    let wants_stream = request["stream"].as_bool().unwrap_or(false);
    let instance = ensure_loaded(&state, &model).await?;

    let mut translated = anthropic::request_to_openai(&request)
        .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, format!("{e:#}")))?;

    // Anthropic clients express thinking with a `thinking` block, which survives the
    // translation above as-is; both paths need the same treatment to reach llama.cpp.
    apply_thinking_request(&mut translated);
    if let Some(extra) = &state.config.system_prompt {
        inject_system_prompt(&mut translated, extra);
    }

    if wants_stream {
        // Ask for usage on the final chunk so the reported output token count is the
        // server's rather than our estimate.
        translated["stream_options"] = serde_json::json!({ "include_usage": true });
    }

    let upstream = state
        .client
        .post(format!("{}/v1/chat/completions", instance.base_url()))
        .json(&translated)
        .send()
        .await
        .map_err(|e| {
            ApiError::new(
                StatusCode::BAD_GATEWAY,
                format!("upstream llama-server did not answer: {e}"),
            )
        })?;

    if !wants_stream {
        let openai: serde_json::Value = upstream.json().await.map_err(|e| {
            ApiError::new(StatusCode::BAD_GATEWAY, format!("upstream returned: {e}"))
        })?;
        return Ok(Json(anthropic::response_to_anthropic(&openai, &model)).into_response());
    }

    let stream = anthropic_event_stream(upstream, model);
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response()))
}

/// Re-frame an OpenAI SSE stream as Anthropic events.
///
/// Buffers only up to an event boundary: SSE frames are separated by a blank line and
/// can be split across TCP reads, so a naive per-chunk parse drops deltas.
fn anthropic_event_stream(
    upstream: reqwest::Response,
    model: String,
) -> impl futures_util::Stream<Item = Result<String, std::io::Error>> {
    use futures_util::StreamExt as _;

    async_stream::stream! {
        let mut translator = anthropic::StreamTranslator::new(&model);
        let mut bytes = upstream.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = bytes.next().await {
            let Ok(chunk) = chunk else { break };
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(split) = buffer.find("\n\n") {
                let frame = buffer[..split].to_owned();
                buffer.drain(..split + 2);

                for line in frame.lines() {
                    let Some(payload) = line.strip_prefix("data: ") else { continue };
                    if payload.trim() == "[DONE]" {
                        continue;
                    }
                    let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
                        continue;
                    };
                    for event in translator.push(&value) {
                        yield Ok(event.encode());
                    }
                }
            }
        }

        for event in translator.finish() {
            yield Ok(event.encode());
        }
    }
}

/// Translate a harness's "think about this" into the one knob llama.cpp acts on.
///
/// Measured against llama-server directly: `reasoning_effort` and Anthropic's
/// `thinking` block are both accepted and both silently ignored, while
/// `chat_template_kwargs: {"enable_thinking": true}` turns thinking on *even when the
/// server was launched with `--reasoning off`*. So the launch flag is a default, not a
/// lock, and a harness can drive this per request - it just has to be told in the right
/// dialect.
///
/// An explicit `chat_template_kwargs` from the caller wins; this only fills in what the
/// caller expressed some other way.
///
/// `minimal` means think briefly, not don't. It used to be read as off, which was a
/// fair guess while the level was all we had - but a harness that asks for `minimal`
/// now sends a `reasoning_budget_tokens` cap with it, and llama.cpp honours that, so
/// the level and the budget contradicted each other. Only the two spellings that
/// actually mean nothing turn thinking off.
fn apply_thinking_request(body: &mut serde_json::Value) {
    // Anthropic: {"thinking": {"type": "enabled"}}. OpenAI: reasoning_effort.
    let wanted = match (&body["thinking"]["type"], &body["reasoning_effort"]) {
        (serde_json::Value::String(t), _) => Some(t == "enabled"),
        (_, serde_json::Value::String(effort)) => Some(!matches!(effort.as_str(), "none" | "off")),
        _ => None,
    };

    let Some(wanted) = wanted else { return };
    if !body["chat_template_kwargs"]["enable_thinking"].is_null() {
        return;
    }
    if let Some(object) = body.as_object_mut() {
        object
            .entry("chat_template_kwargs")
            .or_insert_with(|| serde_json::json!({}))["enable_thinking"] =
            serde_json::Value::Bool(wanted);
    }
}

/// Prepend a configured instruction to the conversation's system message.
///
/// Harnesses build their own system prompt and have no notion of a per-machine one, so
/// this is the only place a local instruction can be added once and apply to all of
/// them. Appended to the existing system message rather than replacing it - the
/// harness's prompt is what makes its tools work - and inserted as a new first message
/// only when there is none.
fn inject_system_prompt(body: &mut serde_json::Value, extra: &str) {
    if extra.trim().is_empty() {
        return;
    }
    let Some(messages) = body["messages"].as_array_mut() else {
        return;
    };

    // `developer` as well as `system`: OpenAI renamed the role for newer models and Pi
    // uses the new name, so matching only `system` appended nothing and inserted a
    // second instruction block instead - two sets of instructions arguing with each
    // other, which is the one outcome this is meant to avoid.
    if let Some(existing) = messages
        .iter_mut()
        .find(|m| matches!(m["role"].as_str(), Some("system" | "developer")))
        && let Some(text) = existing["content"].as_str()
    {
        existing["content"] = serde_json::Value::String(format!("{text}\n\n{extra}"));
        return;
    }
    messages.insert(0, serde_json::json!({ "role": "system", "content": extra }));
}

/// Forward a request to llama-server, preserving streaming.
async fn proxy(State(state): State<Arc<AppState>>, request: Request) -> ApiResult<Response> {
    let path = request.uri().path().to_owned();

    let bytes = axum::body::to_bytes(request.into_body(), usize::MAX)
        .await
        .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, format!("reading body: {e}")))?;

    let mut payload: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")))?;

    let model = payload["model"]
        .as_str()
        .ok_or_else(|| ApiError::new(StatusCode::BAD_REQUEST, "request has no \"model\" field"))?
        .to_owned();

    let instance = ensure_loaded(&state, &model).await?;

    // Only conversations get rewritten. `/v1/completions` and `/v1/embeddings` have no
    // messages and no chat template, so there is nothing here that applies to them.
    let body = if path.ends_with("/chat/completions") {
        if let Some(log) = &state.request_log {
            record_request(log, &path, &payload);
        }
        apply_thinking_request(&mut payload);
        if let Some(extra) = &state.config.system_prompt {
            inject_system_prompt(&mut payload, extra);
        }
        serde_json::to_vec(&payload)
            .map(axum::body::Bytes::from)
            .unwrap_or(bytes)
    } else {
        bytes
    };

    let upstream = state
        .client
        .post(format!("{}{path}", instance.base_url()))
        .header(header::CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .map_err(|e| {
            ApiError::new(
                StatusCode::BAD_GATEWAY,
                format!("upstream llama-server did not answer: {e}"),
            )
        })?;

    let status = StatusCode::from_u16(upstream.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let content_type = upstream
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_owned();

    // Stream rather than buffer: tokens have to reach the client as they are produced,
    // and a buffered proxy would also hold a whole response in memory.
    let body = Body::from_stream(upstream.bytes_stream());

    Ok(Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, content_type)
        // Proxies and browsers will otherwise batch an SSE stream into chunks.
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat(extra: serde_json::Value) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        });
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        body
    }

    /// Measured against llama-server: it ignores `reasoning_effort` outright, and acts
    /// only on `chat_template_kwargs`. Without this translation a harness's thinking
    /// control does nothing at all, which is indistinguishable from the model refusing
    /// to think.
    #[test]
    fn openai_reasoning_effort_becomes_the_knob_llama_cpp_reads() {
        let mut body = chat(serde_json::json!({"reasoning_effort": "high"}));
        apply_thinking_request(&mut body);
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], true);
    }

    #[test]
    fn effort_levels_that_mean_do_not_think_turn_it_off() {
        for effort in ["none", "off"] {
            let mut body = chat(serde_json::json!({ "reasoning_effort": effort }));
            apply_thinking_request(&mut body);
            assert_eq!(
                body["chat_template_kwargs"]["enable_thinking"], false,
                "for {effort}"
            );
        }
    }

    /// `minimal` arrives with a budget of its own, so it is the smallest amount of
    /// thinking rather than none of it. Reading it as off silently discarded the
    /// lowest rung of the harness's ladder.
    #[test]
    fn a_minimal_effort_still_thinks() {
        for effort in ["minimal", "low", "medium", "high", "xhigh", "max"] {
            let mut body = chat(serde_json::json!({ "reasoning_effort": effort }));
            apply_thinking_request(&mut body);
            assert_eq!(
                body["chat_template_kwargs"]["enable_thinking"], true,
                "for {effort}"
            );
        }
    }

    /// llama.cpp reads this one, so it has to survive the proxy untouched - it is the
    /// only thing that makes one level differ from another.
    #[test]
    fn a_thinking_budget_is_passed_through() {
        let mut body = chat(serde_json::json!({
            "reasoning_effort": "low",
            "reasoning_budget_tokens": 2048,
        }));
        apply_thinking_request(&mut body);
        assert_eq!(body["reasoning_budget_tokens"], 2048);
    }

    #[test]
    fn the_anthropic_thinking_block_is_understood_too() {
        let mut on = chat(serde_json::json!({"thinking": {"type": "enabled"}}));
        apply_thinking_request(&mut on);
        assert_eq!(on["chat_template_kwargs"]["enable_thinking"], true);

        let mut off = chat(serde_json::json!({"thinking": {"type": "disabled"}}));
        apply_thinking_request(&mut off);
        assert_eq!(off["chat_template_kwargs"]["enable_thinking"], false);
    }

    /// A caller who already speaks llama.cpp's dialect has said exactly what it wants.
    #[test]
    fn an_explicit_template_kwarg_is_not_overridden() {
        let mut body = chat(serde_json::json!({
            "reasoning_effort": "high",
            "chat_template_kwargs": {"enable_thinking": false},
        }));
        apply_thinking_request(&mut body);
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
    }

    /// Saying nothing must stay saying nothing, so the server's launch flag decides.
    #[test]
    fn a_request_that_asks_for_nothing_is_left_alone() {
        let mut body = chat(serde_json::json!({}));
        apply_thinking_request(&mut body);
        assert!(body["chat_template_kwargs"].is_null());
    }

    /// The harness's own system prompt is what makes its tools work, so ours is added
    /// to it rather than put in its place.
    #[test]
    fn an_injected_instruction_is_appended_to_the_harness_prompt() {
        let mut body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "hi"},
            ]
        });
        inject_system_prompt(&mut body, "Read AGENTS.md first.");
        let system = body["messages"][0]["content"].as_str().unwrap();
        assert!(
            system.starts_with("You are a coding agent."),
            "got: {system}"
        );
        assert!(system.contains("Read AGENTS.md first."), "got: {system}");
        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
    }

    /// Pi sends `developer`, OpenAI's newer name for the system role. Missing it meant
    /// our instruction arrived as a second, competing block instead of being appended -
    /// found by logging what Pi actually sends rather than assuming.
    #[test]
    fn a_developer_role_counts_as_the_system_prompt() {
        let mut body = serde_json::json!({
            "messages": [
                {"role": "developer", "content": "<project_instructions>x</project_instructions>"},
                {"role": "user", "content": "hi"},
            ]
        });
        inject_system_prompt(&mut body, "Read AGENTS.md first.");

        assert_eq!(
            body["messages"].as_array().unwrap().len(),
            2,
            "must append, not add a second instruction message"
        );
        let text = body["messages"][0]["content"].as_str().unwrap();
        assert!(text.starts_with("<project_instructions>"), "got: {text}");
        assert!(text.ends_with("Read AGENTS.md first."), "got: {text}");
    }

    #[test]
    fn an_injected_instruction_becomes_the_system_prompt_when_there_is_none() {
        let mut body = serde_json::json!({"messages": [{"role": "user", "content": "hi"}]});
        inject_system_prompt(&mut body, "Read AGENTS.md first.");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "Read AGENTS.md first.");
        assert_eq!(body["messages"][1]["role"], "user");
    }

    #[test]
    fn an_empty_instruction_changes_nothing() {
        let mut body = serde_json::json!({"messages": [{"role": "user", "content": "hi"}]});
        inject_system_prompt(&mut body, "   ");
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn api_errors_render_in_the_openai_shape() {
        let resp = ApiError::new(StatusCode::UNAUTHORIZED, "nope").into_response();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    /// A wildcard bind is not something a client can connect to. Probing it directly
    /// would report the port free while the gateway is sitting on it.
    #[test]
    fn a_wildcard_bind_is_probed_over_loopback() {
        assert_eq!(loopback_for("0.0.0.0"), "127.0.0.1");
        assert_eq!(loopback_for("::"), "127.0.0.1");
        assert_eq!(loopback_for(""), "127.0.0.1");
        assert_eq!(loopback_for("127.0.0.1"), "127.0.0.1");
        assert_eq!(loopback_for("192.168.1.5"), "192.168.1.5");
    }

    #[test]
    fn an_unused_port_is_free() {
        // Port 0 asks the OS for any free port, so this binds and releases one.
        let probe = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = probe.local_addr().unwrap().port();
        drop(probe);
        assert_eq!(port_status("127.0.0.1", port), PortStatus::Free);
    }

    /// The case that sent a Mac in circles: something else already on the port. It has
    /// to be distinguishable from our own gateway, because the two need opposite
    /// responses - move aside, or leave it alone.
    #[test]
    fn a_port_held_by_something_else_is_taken() {
        let squatter = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = squatter.local_addr().unwrap().port();
        assert_eq!(port_status("127.0.0.1", port), PortStatus::Taken);
        drop(squatter);
    }

    #[test]
    fn the_search_skips_a_port_someone_else_holds() {
        let squatter = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let taken = squatter.local_addr().unwrap().port();
        let found = usable_port("127.0.0.1", taken).expect("a free port above it");
        assert!(found > taken, "{found} should be past the occupied {taken}");
        drop(squatter);
    }

    #[test]
    fn waiting_for_a_gateway_that_never_arrives_gives_up() {
        let probe = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = probe.local_addr().unwrap().port();
        drop(probe);
        let began = std::time::Instant::now();
        assert!(!wait_until_answering(
            "127.0.0.1",
            port,
            std::time::Duration::from_millis(400)
        ));
        assert!(began.elapsed() < std::time::Duration::from_secs(5));
    }
}
