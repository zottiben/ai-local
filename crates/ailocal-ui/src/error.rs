//! One error shape for the whole API.
//!
//! Every failure the console can hit is one the CLI would print - a config that does
//! not parse, a models directory that cannot be read - so the message is the CLI's
//! own, with its causes, and the window shows it as written.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

#[derive(Debug)]
pub enum Error {
    /// No token, or the wrong one.
    Unauthorized,
    /// The request itself is wrong: a reference that does not parse, a log offset
    /// that is not a number.
    BadRequest(String),
    /// No such job.
    NotFound(String),
    /// Not now: an install is already running, or the job has already finished.
    Busy(String),
    Internal(anyhow::Error),
}

impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Self::Internal(err)
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let (status, code, message) = match self {
            Self::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "missing or wrong token - open the link `ailocal ui` printed".to_owned(),
            ),
            Self::BadRequest(message) => (StatusCode::BAD_REQUEST, "bad_request", message),
            Self::NotFound(message) => (StatusCode::NOT_FOUND, "not_found", message),
            Self::Busy(message) => (StatusCode::CONFLICT, "busy", message),
            Self::Internal(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
                format!("{e:#}"),
            ),
        };
        (
            status,
            Json(serde_json::json!({ "error": message, "code": code })),
        )
            .into_response()
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Run one of the library's calls off the async threads.
///
/// They block on purpose - a launchctl query, a file read, a registry request - and
/// `reqwest`'s blocking client panics outright if it is used on a runtime thread.
pub async fn blocking<T, F>(f: F) -> Result<T>
where
    F: FnOnce() -> anyhow::Result<T> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| anyhow::anyhow!("a request handler stopped: {e}"))?
        .map_err(Error::from)
}
