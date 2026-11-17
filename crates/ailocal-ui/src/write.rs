//! The write half of the API.
//!
//! As with the reads, each handler is the library call the matching command makes,
//! and the answer is the library's own type. What the CLI prints after a delete or a
//! change of default, the window shows from the same fields.

use std::path::PathBuf;

use axum::extract::rejection::{JsonRejection, PathRejection};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{delete, post, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use ailocal::config::Config;
use ailocal::models::{self, DefaultChange, Removed};
use ailocal::serve::{self, Instance};
use ailocal::source::Source;

use crate::error::{Error, Result, blocking};
use crate::state::AppState;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/install", post(install))
        .route("/serve", post(serve))
        .route("/jobs/{id}/cancel", post(cancel))
        .route("/jobs/{id}", delete(dismiss))
        .route("/models/{name}", delete(remove))
        .route("/default", put(set_default))
        .route("/stop", post(stop))
}

#[derive(Deserialize)]
struct InstallRequest {
    /// `ollama:<name>:<tag>` or `hf:<owner>/<repo>/<file>`, as on the command line.
    reference: String,
    #[serde(default)]
    force: bool,
}

#[derive(Deserialize)]
struct ServeRequest {
    name: String,
}

#[derive(Deserialize)]
struct DefaultRequest {
    name: String,
}

/// A job has started; its progress arrives on the event stream.
#[derive(Serialize)]
struct Started {
    id: u64,
}

/// `ailocal model install <reference>`, as a job.
async fn install(
    State(state): State<AppState>,
    body: std::result::Result<Json<InstallRequest>, JsonRejection>,
) -> Result<(StatusCode, Json<Started>)> {
    let Json(request) = body.map_err(rejected)?;
    let source: Source = request
        .reference
        .parse()
        .map_err(|e: anyhow::Error| Error::BadRequest(format!("{e:#}")))?;
    let id = state.jobs().install(source, request.force)?;
    Ok((StatusCode::ACCEPTED, Json(Started { id })))
}

/// `ailocal serve <name>`, as a job.
async fn serve(
    State(state): State<AppState>,
    body: std::result::Result<Json<ServeRequest>, JsonRejection>,
) -> Result<(StatusCode, Json<Started>)> {
    let Json(request) = body.map_err(rejected)?;
    let id = state.jobs().serve(request.name)?;
    Ok((StatusCode::ACCEPTED, Json(Started { id })))
}

async fn cancel(
    State(state): State<AppState>,
    id: std::result::Result<Path<u64>, PathRejection>,
) -> Result<StatusCode> {
    let Path(id) = id.map_err(|e| Error::BadRequest(e.body_text()))?;
    state.jobs().cancel(id)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn dismiss(
    State(state): State<AppState>,
    id: std::result::Result<Path<u64>, PathRejection>,
) -> Result<StatusCode> {
    let Path(id) = id.map_err(|e| Error::BadRequest(e.body_text()))?;
    state.jobs().dismiss(id)?;
    Ok(StatusCode::NO_CONTENT)
}

/// `ailocal model rm <name> --yes`. The page has already shown the plan from
/// `GET /models/{name}/removal` and had it confirmed (D6); this plans again rather
/// than trusting that one, as the library re-checks what is loaded, because the
/// machine may have moved while the dialog was open.
async fn remove(State(state): State<AppState>, Path(name): Path<String>) -> Result<Json<Removed>> {
    if state.jobs().loading(&name) {
        return Err(Error::Busy(format!(
            "{name} is loading; delete it once it is up, and it will be stopped first"
        )));
    }
    Ok(Json(
        blocking(move || {
            let cfg = Config::load()?;
            models::plan_removal(&cfg, &name)?.remove(&cfg, &unit_binary)
        })
        .await?,
    ))
}

/// What `ailocal model pick` does once a model is chosen.
async fn set_default(
    body: std::result::Result<Json<DefaultRequest>, JsonRejection>,
) -> Result<Json<DefaultChange>> {
    let Json(request) = body.map_err(rejected)?;
    Ok(Json(
        blocking(move || models::set_default(&request.name, &unit_binary)).await?,
    ))
}

/// `ailocal stop`: what was stopped, or `null` when nothing was running.
async fn stop() -> Result<Json<Option<Instance>>> {
    Ok(Json(blocking(serve::stop).await?))
}

/// The `ailocal` a service should run when a change made here rewrites one.
///
/// Never this binary: a unit that ran `ailocal-ui` would start a console where the
/// gateway belongs. The core binary is installed beside every extra, so that is the
/// one, unless this is a build directory - where the library's own rule applies.
fn unit_binary() -> anyhow::Result<PathBuf> {
    let beside = std::env::current_exe()?
        .canonicalize()?
        .with_file_name("ailocal");
    ailocal::update::unit_binary(&beside)
}

/// A body that is not the JSON a route takes. axum's own answer is plain text; this
/// API answers every error in one shape.
fn rejected(rejection: JsonRejection) -> Error {
    Error::BadRequest(rejection.body_text())
}
