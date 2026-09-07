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

use crate::config::Config;
use crate::registry::{self, Fit};
use crate::serve;
use crate::vram;

pub struct AppState {
    pub key: String,
    pub config: Config,
    pub client: reqwest::Client,
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
    let presented = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(crate::auth::bearer);

    match presented {
        Some(token) if crate::auth::constant_time_eq(token, &state.key) => next.run(request).await,
        _ => ApiError::new(
            StatusCode::UNAUTHORIZED,
            "missing or invalid API key; send `Authorization: Bearer <key>` \
             (see `ailocal gateway key`)",
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

/// Forward a request to llama-server, preserving streaming.
async fn proxy(State(state): State<Arc<AppState>>, request: Request) -> ApiResult<Response> {
    let path = request.uri().path().to_owned();

    let bytes = axum::body::to_bytes(request.into_body(), usize::MAX)
        .await
        .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, format!("reading body: {e}")))?;

    let payload: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")))?;

    let model = payload["model"]
        .as_str()
        .ok_or_else(|| ApiError::new(StatusCode::BAD_REQUEST, "request has no \"model\" field"))?;

    let instance = ensure_loaded(&state, model).await?;

    let upstream = state
        .client
        .post(format!("{}{path}", instance.base_url()))
        .header(header::CONTENT_TYPE, "application/json")
        .body(bytes)
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

    #[test]
    fn api_errors_render_in_the_openai_shape() {
        let resp = ApiError::new(StatusCode::UNAUTHORIZED, "nope").into_response();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }
}
