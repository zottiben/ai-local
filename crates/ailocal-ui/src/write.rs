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
use ailocal::setup;
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
        .route("/settings", put(settings))
        .route("/services", post(install_services))
        .route("/services", delete(remove_services))
        .route("/services/{name}/{action}", post(service_action))
        .route("/gateway/check", post(gateway_check))
        .route("/harnesses/{name}", put(configure_harness))
        .route("/harnesses/{name}", delete(unconfigure_harness))
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

/// `ailocal config <setting> <value>`, for whichever settings the body names.
///
/// A patch rather than the whole config: the window sends the one field its input
/// changed, so a setting someone else edited in a terminal a second ago survives
/// instead of being quietly reverted to what the page loaded.
async fn settings(
    body: std::result::Result<Json<setup::Settings>, JsonRejection>,
) -> Result<Json<setup::Applied>> {
    let Json(settings) = body.map_err(rejected)?;
    Ok(Json(
        blocking(move || setup::apply_settings(&settings, &unit_binary)).await?,
    ))
}

#[derive(Deserialize)]
struct InstallServicesRequest {
    /// Write the units but leave them stopped, as `service install --no-start` does.
    #[serde(default)]
    no_start: bool,
}

/// `ailocal service install`.
async fn install_services(
    body: std::result::Result<Json<InstallServicesRequest>, JsonRejection>,
) -> Result<Json<setup::Installed>> {
    let Json(request) = body.map_err(rejected)?;
    Ok(Json(
        blocking(move || setup::install_services(&unit_binary()?, !request.no_start)).await?,
    ))
}

/// `ailocal service uninstall`: the units that were there.
async fn remove_services() -> Result<Json<Vec<String>>> {
    Ok(Json(blocking(setup::remove_services).await?))
}

/// `ailocal service start|stop <unit>`, and what the manager says afterwards.
///
/// The state is read back rather than assumed, because `start` only means the manager
/// accepted the job: a unit that exits immediately reports started and is not running.
async fn service_action(
    path: std::result::Result<Path<(String, String)>, PathRejection>,
) -> Result<Json<setup::Services>> {
    let Path((name, action)) = path.map_err(rejected)?;
    if !ailocal::service::service_names().contains(&name.as_str()) {
        return Err(Error::NotFound(format!("no unit named {name:?}")));
    }
    Ok(Json(
        blocking(move || {
            match action.as_str() {
                "start" => ailocal::service::start(&name)?,
                "stop" => ailocal::service::stop(&name)?,
                other => anyhow::bail!("unknown action {other:?}; expected `start` or `stop`"),
            }
            Ok(setup::services())
        })
        .await?,
    ))
}

/// `ailocal gateway check`: every assertion it makes, and what answered.
async fn gateway_check() -> Result<Json<ailocal::gateway::Check>> {
    Ok(Json(
        blocking(|| {
            let cfg = Config::load()?;
            let key = ailocal::auth::load_or_create()?;
            ailocal::gateway::check(&crate::read::gateway_url(&cfg), &key)
        })
        .await?,
    ))
}

#[derive(Deserialize)]
struct HarnessRequest {
    /// Which model to point it at. `None` lets the library choose, in the order a
    /// person would expect: configured, then loaded, then the first alphabetically.
    #[serde(default)]
    model: Option<String>,
}

/// `ailocal harness configure <name> [--model <model>]`.
async fn configure_harness(
    path: std::result::Result<Path<String>, PathRejection>,
    body: std::result::Result<Json<HarnessRequest>, JsonRejection>,
) -> Result<Json<setup::HarnessConfigured>> {
    let Path(name) = path.map_err(rejected)?;
    let Json(request) = body.map_err(rejected)?;
    Ok(Json(
        blocking(move || {
            let cfg = Config::load()?;
            let url = crate::read::gateway_url(&cfg);
            setup::configure_harness(&cfg, &name, &url, request.model.as_deref())
        })
        .await?,
    ))
}

/// `ailocal harness unconfigure <name>`: whether it was pointing at us.
async fn unconfigure_harness(
    path: std::result::Result<Path<String>, PathRejection>,
) -> Result<Json<bool>> {
    let Path(name) = path.map_err(rejected)?;
    Ok(Json(
        blocking(move || setup::unconfigure_harness(&name)).await?,
    ))
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
fn rejected(rejection: impl Rejected) -> Error {
    Error::BadRequest(rejection.body_text())
}

/// Axum gives each extractor its own rejection type with its own `body_text`, and no
/// trait over them, so one is declared here rather than repeating the conversion.
trait Rejected {
    fn body_text(self) -> String;
}

impl Rejected for JsonRejection {
    fn body_text(self) -> String {
        Self::body_text(&self)
    }
}

impl Rejected for PathRejection {
    fn body_text(self) -> String {
        Self::body_text(&self)
    }
}
