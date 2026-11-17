//! Serving the frontend out of the binary (D7).
//!
//! `build.rs` writes the table; nothing here reads the filesystem at run time. That is
//! what makes `ailocal ui` work from an installed binary with no repo, no node and no
//! network.

use axum::http::{HeaderValue, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};

include!(concat!(env!("OUT_DIR"), "/assets.rs"));

/// Whether a real bundle was compiled in, as opposed to the placeholder.
#[must_use]
pub fn is_embedded() -> bool {
    !ASSETS.is_empty()
}

/// Anything that is not a file is one of the app's own routes, so it gets
/// `index.html` and the client router takes it from there. Without this, reloading
/// the page on `/models` is a 404.
pub async fn serve(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    match lookup(path).or_else(|| lookup("index.html")) {
        Some((body, mime)) => file(body, mime),
        None => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

fn lookup(path: &str) -> Option<(&'static [u8], &'static str)> {
    let wanted = if path.is_empty() { "index.html" } else { path };
    if let Some((_, mime, body)) = ASSETS.iter().find(|(name, _, _)| *name == wanted) {
        return Some((body, mime));
    }
    // No bundle compiled in. Say so in the browser rather than showing a blank page
    // and letting someone conclude the server is broken.
    (wanted == "index.html").then_some((MISSING_BUNDLE.as_bytes(), "text/html; charset=utf-8"))
}

fn file(body: &'static [u8], mime: &'static str) -> Response {
    (
        [
            (header::CONTENT_TYPE, HeaderValue::from_static(mime)),
            // The bundle changes with the binary and is served from memory, so a copy
            // a browser cached from the previous version is exactly what is not wanted.
            (
                header::CACHE_CONTROL,
                HeaderValue::from_static("no-cache, must-revalidate"),
            ),
        ],
        body,
    )
        .into_response()
}

const MISSING_BUNDLE: &str = r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark"><title>ai-local</title>
<style>
 body { font: 15px/1.6 ui-sans-serif, system-ui, sans-serif; margin: 0;
        display: grid; place-items: center; min-height: 100vh; padding: 2rem; }
 main { max-width: 34rem; text-align: center; }
 h1 { font-size: 1.15rem; margin: 0 0 .5rem; }
 p { opacity: .7; margin: 0 0 1rem; }
 code { font-family: ui-monospace, Menlo, monospace; font-size: .9em;
        background: color-mix(in srgb, currentColor 8%, transparent);
        padding: .15em .45em; border-radius: 4px; }
</style></head>
<body><main>
 <h1>No frontend bundle in this binary</h1>
 <p>The API is running, but the console was not compiled in. This happens when
    <code>ui/dist</code> was missing when <code>ailocal-ui</code> was built.</p>
 <p>From the repo: <code>cd ui &amp;&amp; npm ci &amp;&amp; npm run build</code>,
    then <code>cargo build -p ailocal-ui</code>.</p>
</main></body></html>
"#;
