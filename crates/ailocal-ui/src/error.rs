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
    /// that is not a number, a context below the smallest one worth loading.
    BadRequest(String),
    /// No such job, model or harness.
    NotFound(String),
    /// Not now: an install is already running, or the job has already finished.
    Busy(String),
    Internal(anyhow::Error),
}

impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        // The library marks the failures the caller caused, which over HTTP are a
        // different answer from the ones the machine caused. Without this a context
        // someone typed too small comes back as "internal error", which blames the
        // console for a typo and tells the page nothing it can act on.
        match err.downcast_ref::<ailocal::Refused>() {
            Some(ailocal::Refused::Missing(_)) => Self::NotFound(format!("{err:#}")),
            Some(ailocal::Refused::Invalid(_)) => Self::BadRequest(format!("{err:#}")),
            None => Self::Internal(err),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// 500 for a value someone typed is the console blaming itself for a typo, and a
    /// page cannot tell from it whether retrying would ever help.
    #[test]
    fn a_failure_the_caller_caused_is_not_an_internal_error() {
        let invalid: anyhow::Error =
            ailocal::Refused::Invalid("128 is too small".to_owned()).into();
        assert!(matches!(Error::from(invalid), Error::BadRequest(m) if m == "128 is too small"));

        let missing: anyhow::Error =
            ailocal::Refused::Missing("no model named x".to_owned()).into();
        assert!(matches!(Error::from(missing), Error::NotFound(m) if m == "no model named x"));

        let broke = anyhow::anyhow!("the models directory could not be read");
        assert!(matches!(Error::from(broke), Error::Internal(_)));
    }

    /// The context a refusal was given still has to reach the window: the library adds
    /// it with `.context()` above the refusal, and only `{e:#}` prints both.
    #[test]
    fn a_refusal_keeps_the_causes_stacked_on_top_of_it() {
        let err = anyhow::Error::from(ailocal::Refused::Invalid("port 8080 is in use".to_owned()))
            .context("could not move the gateway");
        let Error::BadRequest(message) = Error::from(err) else {
            panic!("a refusal must stay a refusal through a context");
        };
        assert_eq!(message, "could not move the gateway: port 8080 is in use");
    }
}
