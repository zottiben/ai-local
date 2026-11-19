//! The read half of the API.
//!
//! Every handler is one call into the library, the same call the CLI makes, and the
//! JSON is the library's own types serialising themselves. There is no layer here to
//! fall out of step: change what `status` reports and the window reports it too.

use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::{Path, PathBuf};

use axum::extract::rejection::QueryRejection;
use axum::extract::{self, Query, State};
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use ailocal::config::Config;
use ailocal::models::{self, Installed, Offer, Removal};
use ailocal::serve::Instance;
use ailocal::status::{self, Overview};
use ailocal::{harness, serve, service, setup};

use crate::error::{Error, Result, blocking};
use crate::state::AppState;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/meta", get(meta))
        .route("/overview", get(overview))
        .route("/models", get(installed))
        .route("/catalogue", get(catalogue))
        .route("/search", get(search))
        .route("/files", get(files))
        .route("/config", get(config))
        .route("/loaded", get(loaded))
        .route("/loaded/log", get(log))
        .route("/loaded/memory", get(memory))
        .route("/models/{name}/removal", get(removal))
        .route("/services", get(services))
        .route("/harnesses", get(harnesses))
        .route("/gateway", get(gateway))
}

/// What the client needs to word things for this machine.
#[derive(Debug, Serialize)]
struct Meta {
    version: &'static str,
    /// The file every setting comes from.
    config_path: PathBuf,
    /// Every harness ailocal can configure, whether or not it is configured.
    harnesses: [&'static str; 2],
    /// Where the gateway service writes its log, which differs by service manager.
    log_hint: String,
}

async fn meta() -> Result<Json<Meta>> {
    Ok(Json(
        blocking(|| {
            Ok(Meta {
                version: env!("CARGO_PKG_VERSION"),
                config_path: Config::path()?,
                harnesses: harness::NAMES,
                log_hint: service::log_hint(service::GATEWAY),
            })
        })
        .await?,
    ))
}

/// What `ailocal status` prints.
async fn overview() -> Result<Json<Overview>> {
    Ok(Json(
        blocking(|| Ok(status::overview(&Config::load()?))).await?,
    ))
}

/// What `ailocal model ls` prints.
async fn installed() -> Result<Json<Vec<Installed>>> {
    Ok(Json(
        blocking(|| models::inventory(&Config::load()?)).await?,
    ))
}

/// What `ailocal model pick` offers.
///
/// The registry's answer is kept between requests; what is on disk is read every
/// time, so a model installed from a terminal moves out of this list on the next
/// refetch.
async fn catalogue(State(state): State<AppState>) -> Result<Json<Offer>> {
    let (cfg, budget) = blocking(|| Ok((Config::load()?, serve::budget_or_estimate()))).await?;
    let remote = state.catalogue(&cfg, budget).await?;
    Ok(Json(
        blocking(move || Offer::from_candidates(&cfg, budget, remote)).await?,
    ))
}

/// How many repositories a search answers with.
///
/// The CLI defaults to the same number. Past about this many the list stops being a
/// shortlist and becomes something to search again, and every row costs a line in a
/// panel that has to share the page with what is already installed.
const SEARCH_LIMIT: usize = 20;

#[derive(Deserialize)]
struct SearchQuery {
    q: String,
}

/// What `ailocal model search` finds: Hugging Face repositories with GGUF files.
///
/// Hugging Face only. Ollama publishes no search endpoint, which is a fact about the
/// registry rather than a gap here - [`ailocal::source::search`] says so at length.
/// What Ollama has is in the catalogue, which this panel sits beside.
async fn search(
    query: std::result::Result<Query<SearchQuery>, QueryRejection>,
) -> Result<Json<Vec<ailocal::source::Repo>>> {
    let Query(query) = query.map_err(|e| Error::BadRequest(e.body_text()))?;
    if query.q.trim().is_empty() {
        return Err(Error::BadRequest("give me something to search for".into()));
    }
    Ok(Json(
        blocking(move || {
            ailocal::source::search(&ailocal::install::client()?, query.q.trim(), SEARCH_LIMIT)
        })
        .await?,
    ))
}

#[derive(Deserialize)]
struct FilesQuery {
    repo: String,
}

/// What `ailocal model files` lists: the quants in a repository, best first.
///
/// Measured against this machine here rather than in the page, so the verdict a
/// person reads is the one `model install` will act on.
async fn files(
    query: std::result::Result<Query<FilesQuery>, QueryRejection>,
) -> Result<Json<Vec<ailocal::source::Choice>>> {
    let Query(query) = query.map_err(|e| Error::BadRequest(e.body_text()))?;
    if query.repo.trim().is_empty() {
        return Err(Error::BadRequest("which repository?".into()));
    }
    Ok(Json(
        blocking(move || {
            let cfg = Config::load()?;
            let token = ailocal::source::hf_token(&cfg.hf_home);
            let found = ailocal::source::files(
                &ailocal::install::client()?,
                query.repo.trim(),
                token.as_deref(),
            )?;
            Ok(ailocal::source::weigh(
                found,
                serve::budget_or_estimate().available_mib(),
            ))
        })
        .await?,
    ))
}

/// The llama-server that is resident, as `ailocal ps` reports it, or `null`.
///
/// Its own route because the overview also asks the gateway and the service manager,
/// which is more than marking a row "loaded" needs.
async fn loaded() -> Result<Json<Option<Instance>>> {
    Ok(Json(blocking(serve::running).await?))
}

/// What the loaded server holds, with the word for it here: `memory` on unified
/// memory, where it is the server's own footprint, and `vram` on a card, where it is
/// the card's use, desktop included.
#[derive(Debug, Serialize)]
struct Memory {
    label: &'static str,
    mib: u64,
}

/// The figure `ailocal serve` prints once a model is up, or `null` when none is.
///
/// Its own route because it runs a probe - `footprint` takes about 70 ms on the M4
/// Max - which the badge marking a row "loaded" has no use for.
async fn memory() -> Result<Json<Option<Memory>>> {
    Ok(Json(
        blocking(|| {
            serve::running()?
                .map(|instance| {
                    let (label, mib) = ailocal::server_memory_mib(instance.pid)?;
                    Ok(Memory { label, mib })
                })
                .transpose()
        })
        .await?,
    ))
}

/// What deleting `name` would do, for the page to confirm before it asks (D6).
///
/// The same plan `ailocal model rm` prints before its prompt.
async fn removal(extract::Path(name): extract::Path<String>) -> Result<Json<Removal>> {
    Ok(Json(
        blocking(move || models::plan_removal(&Config::load()?, &name)).await?,
    ))
}

#[derive(Deserialize)]
struct LogQuery {
    /// Where the last answer ended, to continue from there.
    from: Option<u64>,
}

/// Where the loaded model's output is.
#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Log {
    /// Nothing is loaded.
    Idle,
    /// Not in a file ailocal keeps. A model the service runs logs where the service
    /// does, and one started with `--foreground` prints to its own terminal - the two
    /// cannot be told apart from here, so both are said.
    Elsewhere { service_log: String },
    /// Lines `from..to` of the file. A `from` other than the one asked for means the
    /// file was restarted or the page fell more than a window behind, so what it had
    /// is no longer continuous with this.
    File {
        path: PathBuf,
        from: u64,
        to: u64,
        text: String,
    },
}

/// The loaded model's log, from `from` or its last 64 KiB.
///
/// Only a file the server's record names. Every background start reuses one file,
/// so a model the service loaded would otherwise be shown some earlier server's
/// output as though it were its own.
async fn log(query: std::result::Result<Query<LogQuery>, QueryRejection>) -> Result<Json<Log>> {
    let Query(query) = query.map_err(|e| Error::BadRequest(e.body_text()))?;
    Ok(Json(
        blocking(move || {
            Ok(match serve::running()? {
                None => Log::Idle,
                Some(Instance {
                    log: Some(path), ..
                }) => {
                    let (from, to, text) = tail(&path, query.from)?;
                    Log::File {
                        path,
                        from,
                        to,
                        text,
                    }
                }
                Some(_) => Log::Elsewhere {
                    service_log: service::log_hint(service::MODEL),
                },
            })
        })
        .await?,
    ))
}

/// The most of the log one answer carries. llama-server writes a few hundred lines
/// while it loads and one per request after that, so this is the whole start and
/// then some; a page that falls further behind skips to the newest.
const LOG_WINDOW: u64 = 64 * 1024;

/// Whole lines of `path` from `from` to its last newline, as `(from, to, text)`.
///
/// Never part of a line: one being written is left for the next ask, and a start in
/// the middle of one - after skipping ahead - drops the fragment.
fn tail(path: &Path, from: Option<u64>) -> anyhow::Result<(u64, u64, String)> {
    use anyhow::Context as _;

    let mut file =
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let len = file.metadata()?.len();
    let newest = len.saturating_sub(LOG_WINDOW);
    // Past the end means the file was truncated by a new start since.
    let start = from.filter(|&f| f <= len).map_or(newest, |f| f.max(newest));

    file.seek(SeekFrom::Start(start))?;
    let mut buf = Vec::new();
    file.take(len - start)
        .read_to_end(&mut buf)
        .with_context(|| format!("reading {}", path.display()))?;

    let skip = if start > 0 && from != Some(start) {
        buf.iter()
            .position(|&b| b == b'\n')
            .map_or(buf.len(), |i| i + 1)
    } else {
        0
    };
    let end = buf
        .iter()
        .rposition(|&b| b == b'\n')
        .map_or(skip, |i| i + 1)
        .max(skip);
    Ok((
        start + skip as u64,
        start + end as u64,
        String::from_utf8_lossy(&buf[skip..end]).into_owned(),
    ))
}

/// The settings, as `ailocal config show` reads them.
async fn config() -> Result<Json<Config>> {
    Ok(Json(blocking(Config::load).await?))
}

/// What the service manager says about our units.
async fn services() -> Result<Json<setup::Services>> {
    Ok(Json(blocking(|| Ok(setup::services())).await?))
}

/// One harness, and what pointing it at the gateway would do.
#[derive(Debug, Serialize)]
struct Harness {
    name: &'static str,
    configured: bool,
    /// The file that points it at us. Claude Code's has to be sourced by hand, so the
    /// window has to be able to name it.
    path: Option<PathBuf>,
    /// What it would be pointed at, or why it cannot be. A machine with no runnable
    /// model has no plan for either harness, and saying which one is the whole point.
    #[serde(serialize_with = "ailocal::wire::result")]
    plan: anyhow::Result<setup::HarnessPlan>,
}

/// Both harnesses, planned against the gateway the config names.
///
/// Planned on every read rather than cached, because the answer moves on its own: a
/// model finishes downloading, the default changes, the gateway comes up.
async fn harnesses() -> Result<Json<Vec<Harness>>> {
    Ok(Json(
        blocking(|| {
            let cfg = Config::load()?;
            let url = gateway_url(&cfg);
            Ok(harness::NAMES
                .into_iter()
                .map(|name| Harness {
                    name,
                    configured: harness::is_configured(name),
                    path: harness_path(name).ok(),
                    plan: setup::plan_harness(&cfg, name, &url, None),
                })
                .collect())
        })
        .await?,
    ))
}

/// The gateway: where it is, whether it answers, and what is wrong if it does not.
#[derive(Debug, Serialize)]
struct GatewayView {
    url: String,
    port: u16,
    state: status::Gateway,
    trouble: Option<setup::Trouble>,
    /// The bearer token a harness needs. Loopback-only and already on this machine's
    /// disk, so showing it here reveals nothing the window could not read anyway.
    key: Option<String>,
}

async fn gateway() -> Result<Json<GatewayView>> {
    Ok(Json(
        blocking(|| {
            let cfg = Config::load()?;
            Ok(GatewayView {
                url: gateway_url(&cfg),
                port: cfg.gateway_port,
                state: status::gateway_state(&cfg),
                trouble: setup::trouble(&cfg),
                key: ailocal::auth::load()?,
            })
        })
        .await?,
    ))
}

/// Where a harness keeps what we wrote.
fn harness_path(name: &str) -> anyhow::Result<PathBuf> {
    match name {
        "claude-code" => harness::claude_code_env_path(),
        _ => harness::pi_models_json_path(),
    }
}

/// The URL a harness on this machine should use, which is always a loopback one.
pub(crate) fn gateway_url(cfg: &Config) -> String {
    format!(
        "http://{}:{}",
        ailocal::gateway::loopback_for(&cfg.gateway_host),
        cfg.gateway_port
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log_file(name: &str, contents: &[u8]) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("ailocal-ui-{name}-{}.log", std::process::id()));
        std::fs::write(&path, contents).unwrap();
        path
    }

    /// From the start: every whole line, and not the one still being written. Asked
    /// again from where that ended, only what has been finished since.
    #[test]
    fn a_tail_returns_whole_lines_and_continues_where_it_ended() {
        let path = log_file(
            "tail",
            b"load_model: loading\nsrv  init: ready\nslot 0: proc",
        );
        let (from, to, text) = tail(&path, None).unwrap();
        assert_eq!((from, to), (0, 37));
        assert_eq!(text, "load_model: loading\nsrv  init: ready\n");

        std::fs::write(
            &path,
            b"load_model: loading\nsrv  init: ready\nslot 0: processing\n",
        )
        .unwrap();
        let (from, to, text) = tail(&path, Some(to)).unwrap();
        assert_eq!((from, to), (37, 56));
        assert_eq!(text, "slot 0: processing\n");

        let (from, to, text) = tail(&path, Some(to)).unwrap();
        assert_eq!((from, to, text.as_str()), (56, 56, ""));
        std::fs::remove_file(&path).ok();
    }

    /// A file larger than the window is read from its newest 64 KiB, starting at the
    /// first whole line there - and so is one asked from an offset past its end, which
    /// is what a restarted server's truncated file looks like.
    #[test]
    fn a_long_or_restarted_log_starts_at_a_whole_line_near_its_end() {
        let line = format!("{}\n", "x".repeat(99));
        let path = log_file("long", line.repeat(1000).as_bytes());
        let len = 100 * 1000;

        let (from, to, text) = tail(&path, None).unwrap();
        assert_eq!(to, len);
        assert!(
            from > len - LOG_WINDOW && from.is_multiple_of(100),
            "{from}"
        );
        assert!(text.lines().all(|l| l.len() == 99));

        let (from, _, _) = tail(&path, Some(len + 500)).unwrap();
        assert!(
            from > len - LOG_WINDOW && from.is_multiple_of(100),
            "{from}"
        );
        std::fs::remove_file(&path).ok();
    }

    /// The page tells a log from its absence by `kind`.
    #[test]
    fn a_log_says_which_kind_it_is() {
        assert_eq!(serde_json::to_value(Log::Idle).unwrap()["kind"], "idle");
        let elsewhere = serde_json::to_value(Log::Elsewhere {
            service_log: "journalctl --user -u ailocal-model.service -n 50".to_owned(),
        })
        .unwrap();
        assert_eq!(elsewhere["kind"], "elsewhere");
        assert_eq!(
            elsewhere["service_log"],
            "journalctl --user -u ailocal-model.service -n 50"
        );
    }
}
