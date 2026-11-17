//! Telling the browser that the machine changed (D4).
//!
//! The writers are other processes - `ailocal model install` in a terminal, launchd
//! restarting the gateway, the gateway loading a model on demand - so nothing here is
//! told about a change; it has to notice. One watcher per server compares a
//! fingerprint of everything the console shows and announces a difference. The event
//! carries no payload: the client refetches, because the machine is the state and a
//! second, granular model of it would be one more thing to keep honest.
//!
//! The jobs are the exception, and the same stream carries them in full. They exist
//! only in this process, so there is nothing to refetch them from, and a download's
//! progress is the one thing here that changes four times a second.

use std::convert::Infallible;
use std::path::Path;
use std::time::{Duration, SystemTime};

use axum::Router;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::get;
use futures_util::Stream;
use tokio::sync::broadcast::error::RecvError;

use ailocal::config::Config;
use ailocal::serve::{self, Instance};
use ailocal::status::{self, Gateway, Service};

use crate::state::AppState;

/// How often the cheap half is read: a config stat, a directory listing, the server
/// record and a pid check.
const TICK: Duration = Duration::from_secs(1);

/// Ticks between asks of the service manager and the gateway, which cost a few
/// subprocesses and a connection each time.
const PROBE_EVERY: u32 = 5;

pub fn routes() -> Router<AppState> {
    Router::new().route("/events", get(events))
}

/// Everything the console shows that another process can change.
#[derive(Debug, PartialEq)]
struct Fingerprint {
    config: Option<(SystemTime, u64)>,
    models: Vec<(String, u64, Option<SystemTime>)>,
    loaded: Option<Instance>,
    harnesses: Vec<&'static str>,
    probed: Probed,
}

/// The half that costs a subprocess or a connection to read.
#[derive(Debug, Clone, PartialEq)]
struct Probed {
    gateway: Gateway,
    services: Vec<(&'static str, Service)>,
}

impl Probed {
    fn now(cfg: &Config) -> Self {
        Self {
            gateway: status::gateway_state(cfg),
            services: status::service_states(),
        }
    }
}

/// Read the fingerprint, asking the service manager and the gateway again only when
/// `probed` is `None`.
fn fingerprint(probed: Option<Probed>) -> Fingerprint {
    // A config that does not parse still has a models directory worth watching, and
    // the window shows the parse error from its own request.
    let cfg = Config::load().unwrap_or_default();
    Fingerprint {
        config: Config::path()
            .ok()
            .and_then(|p| std::fs::metadata(p).ok())
            .map(|m| (m.modified().unwrap_or(SystemTime::UNIX_EPOCH), m.len())),
        models: model_files(&cfg.models_dir),
        loaded: serve::running().ok().flatten(),
        harnesses: ailocal::harness::configured(),
        probed: probed.unwrap_or_else(|| Probed::now(&cfg)),
    }
}

/// The `.gguf` files in `dir`, by name, size and modification time.
///
/// The same files `registry::scan` counts as models. A download in progress is a
/// `.part` beside them and is not listed, so a file growing for twenty minutes does
/// not announce a change every second.
fn model_files(dir: &Path) -> Vec<(String, u64, Option<SystemTime>)> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "gguf"))
        .filter_map(|e| {
            let meta = e.metadata().ok()?;
            Some((
                e.file_name().to_string_lossy().into_owned(),
                meta.len(),
                meta.modified().ok(),
            ))
        })
        .collect();
    files.sort();
    files
}

/// Watch the machine and announce every change. One task per server, not one per
/// client: ten open tabs must not mean ten pollers.
pub fn watch(state: AppState) {
    let changes = state.changes();
    tokio::spawn(async move {
        let mut last: Option<Fingerprint> = None;
        let mut generation = 0u64;
        let mut tick = 0u32;
        loop {
            let reuse = (!tick.is_multiple_of(PROBE_EVERY))
                .then(|| last.as_ref().map(|f| f.probed.clone()))
                .flatten();
            tick = tick.wrapping_add(1);

            // A read that failed outright is not a reason to stop watching for the
            // rest of the process's life.
            if let Ok(now) = tokio::task::spawn_blocking(move || fingerprint(reuse)).await {
                if last.as_ref().is_some_and(|before| *before != now) {
                    generation += 1;
                    // Err means nobody is listening, which is the normal state when no
                    // browser is open. Not a failure.
                    let _ = changes.send(generation);
                }
                last = Some(now);
            }
            tokio::time::sleep(TICK).await;
        }
    });
}

async fn events(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let mut changes = state.subscribe();
    let mut jobs = state.jobs().subscribe();
    let stream = async_stream::stream! {
        // The list as it is now, first: a page reloaded in the middle of a download
        // picks it up here rather than at the next chunk.
        let listed = jobs.borrow_and_update().clone();
        yield Ok(Event::default().event("jobs").data(&*listed));
        loop {
            let event = tokio::select! {
                change = changes.recv() => match change {
                    Ok(generation) => {
                        Some(Event::default().event("changed").data(generation.to_string()))
                    }
                    // This client fell behind the buffer. The event says only "something
                    // changed", so a missed one and a delivered one mean the same thing.
                    Err(RecvError::Lagged(_)) => {
                        Some(Event::default().event("changed").data("lagged"))
                    }
                    Err(RecvError::Closed) => None,
                },
                // Only the newest list is ever sent; a client too slow for every
                // update gets the latest one and loses nothing it needed.
                listed = jobs.changed() => listed.ok().map(|()| {
                    Event::default().event("jobs").data(&*jobs.borrow_and_update().clone())
                }),
            };
            match event {
                Some(event) => yield Ok(event),
                None => break,
            }
        }
    };

    Sse::new(stream).keep_alive(
        // Without this a laptop that slept leaves a silent connection that looks live
        // while being dead. EventSource reconnects by itself once the socket closes.
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3-byte model appears in the listing; a 5-byte download in progress beside it
    /// does not, and neither does anything that is not weights.
    #[test]
    fn only_finished_weights_count_as_a_change() {
        let dir = std::env::temp_dir().join(format!("ailocal-ui-models-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let before = model_files(&dir);

        std::fs::write(dir.join("qwen3.5-4b.gguf.part"), b"12345").unwrap();
        std::fs::write(dir.join("notes.txt"), b"hi").unwrap();
        assert_eq!(model_files(&dir), before);

        std::fs::write(dir.join("tiny.gguf"), b"abc").unwrap();
        let after = model_files(&dir);
        assert_eq!(after.len(), before.len() + 1);
        assert!(
            after
                .iter()
                .any(|(name, size, _)| name == "tiny.gguf" && *size == 3)
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_missing_models_directory_is_an_empty_listing_not_an_error() {
        assert!(model_files(Path::new("/nonexistent/ailocal/models")).is_empty());
    }
}
