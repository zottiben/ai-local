//! The per-run token (D3).
//!
//! Binding to loopback is not access control. Any page in any tab can POST to
//! `http://127.0.0.1:<port>`, and DNS rebinding can read the reply - and from PR3 this
//! API deletes models and rewrites harness credentials. So every request carries a
//! secret only the process that printed the URL knows. It is minted per run and never
//! written to disk: a console that outlives its process is one someone forgot was
//! listening.

use axum::body::to_bytes;
use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::Response;

use crate::error::{Error, Result};
use crate::state::AppState;

pub const TOKEN_HEADER: &str = "x-ailocal-token";
pub const TOKEN_QUERY: &str = "t";

/// 128 bits from the kernel.
///
/// # Errors
/// If `/dev/urandom` cannot be read.
pub fn mint() -> anyhow::Result<String> {
    ailocal::auth::random_hex(16)
}

/// How much of a rejected request's body to read before answering. Enough for anything
/// this API accepts, and a bound so an unauthenticated caller cannot make the server
/// buffer whatever it likes.
const DRAIN_LIMIT: usize = 64 * 1024;

pub async fn require_token(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response> {
    if presented(&request).is_some_and(|t| ailocal::auth::constant_time_eq(t, state.token())) {
        return Ok(next.run(request).await);
    }
    // Read the body it sent before turning it away. Closing a socket with bytes still
    // unread is a reset, not a clean end, so a rejected POST would reach the client as
    // "connection reset" instead of the 401 that explains what to do.
    let _ = to_bytes(request.into_body(), DRAIN_LIMIT).await;
    Err(Error::Unauthorized)
}

/// The header first, then the query. `EventSource` cannot set headers, so the event
/// stream has no other way in - which is why the query form exists at all.
fn presented(request: &Request) -> Option<&str> {
    if let Some(value) = request.headers().get(TOKEN_HEADER) {
        return value.to_str().ok();
    }
    request.uri().query()?.split('&').find_map(|pair| {
        let (key, value) = pair.split_once('=')?;
        (key == TOKEN_QUERY).then_some(value)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_minted_token_is_128_bits_of_hex_and_not_the_same_twice() {
        let a = mint().unwrap();
        let b = mint().unwrap();
        assert_eq!(a.len(), 32);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b);
    }
}
