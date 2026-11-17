//! The read half of the API.
//!
//! Every handler is one call into the library, the same call the CLI makes, and the
//! JSON is the library's own types serialising themselves. There is no layer here to
//! fall out of step: change what `status` reports and the window reports it too.

use std::path::PathBuf;

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;

use ailocal::config::Config;
use ailocal::models::{self, Installed, Offer};
use ailocal::serve::Instance;
use ailocal::status::{self, Overview};
use ailocal::{harness, serve, service};

use crate::error::{Result, blocking};
use crate::state::AppState;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/meta", get(meta))
        .route("/overview", get(overview))
        .route("/models", get(installed))
        .route("/catalogue", get(catalogue))
        .route("/config", get(config))
        .route("/loaded", get(loaded))
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
                log_hint: service::log_hint(),
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

/// The llama-server that is resident, as `ailocal ps` reports it, or `null`.
///
/// Its own route because the overview also asks the gateway and the service manager,
/// which is more than marking a row "loaded" needs.
async fn loaded() -> Result<Json<Option<Instance>>> {
    Ok(Json(blocking(serve::running).await?))
}

/// The settings, as `ailocal config show` reads them.
async fn config() -> Result<Json<Config>> {
    Ok(Json(blocking(Config::load).await?))
}
