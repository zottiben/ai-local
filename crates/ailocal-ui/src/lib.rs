//! A local console over this machine's models, gateway and services.
//!
//! An extra (D2): the core `ailocal` binary is what runs all day under systemd or
//! launchd, and a web server with a frontend compiled into it is something a person
//! opens now and then. A library plus a thin binary, so the desktop app can start the
//! same server in its own process.
//!
//! Nothing here decides anything the CLI decides (D1). Every handler is a call into
//! `ailocal`, the same one the matching command makes.

mod assets;
mod auth;
mod error;
mod events;
mod jobs;
mod read;
mod state;
mod write;

use std::net::{Ipv4Addr, SocketAddr};

use axum::Router;
use tokio::net::TcpListener;

use crate::state::AppState;

pub use auth::{TOKEN_HEADER, TOKEN_QUERY};
pub use jobs::stop_owned_model;

/// Whether a real frontend was compiled in, rather than the page explaining that it
/// was not.
#[must_use]
pub fn bundle_embedded() -> bool {
    assets::is_embedded()
}

/// Bound, but not yet serving. Split in two so the caller can print and open the real
/// URL - which nobody knows until the OS has assigned the port - before it blocks.
pub struct Server {
    addr: SocketAddr,
    token: String,
    listener: TcpListener,
    router: Router,
}

impl Server {
    /// Bind 127.0.0.1 on `port`, or on a free port when it is 0.
    ///
    /// Loopback only, and deliberately not a parameter (D3). This API is about to
    /// delete models and rewrite harness credentials; it is not something to expose
    /// by flag, even when the gateway binds 0.0.0.0 for a tunnel.
    ///
    /// # Errors
    /// If no token can be minted or the port cannot be bound.
    pub async fn bind(port: u16) -> anyhow::Result<Self> {
        let token = auth::mint()?;
        let state = AppState::new(&token);

        let api = read::routes()
            .merge(write::routes())
            .merge(events::routes())
            .route_layer(axum::middleware::from_fn_with_state(
                state.clone(),
                auth::require_token,
            ));
        events::watch(state.clone());

        let router = Router::new()
            .nest("/api", api)
            .fallback(assets::serve)
            .with_state(state);

        let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, port)))
            .await
            .map_err(|e| anyhow::anyhow!("binding 127.0.0.1:{port}: {e}"))?;
        let addr = listener.local_addr()?;

        Ok(Self {
            addr,
            token,
            listener,
            router,
        })
    }

    #[must_use]
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    #[must_use]
    pub fn token(&self) -> &str {
        &self.token
    }

    /// The URL to open. The token rides in the query because that is the only channel
    /// a freshly opened browser tab has; the page moves it into memory and out of the
    /// address bar at once.
    #[must_use]
    pub fn url(&self) -> String {
        format!("http://{}/?{TOKEN_QUERY}={}", self.addr, self.token)
    }

    /// Serve until the process is stopped.
    ///
    /// # Errors
    /// If the listener fails.
    pub async fn serve(self) -> anyhow::Result<()> {
        axum::serve(self.listener, self.router).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ui/dist` is committed, so a build without it means it was deleted or never
    /// built - and that binary would serve a page explaining it has no frontend.
    #[test]
    fn the_bundle_is_compiled_in() {
        assert!(bundle_embedded(), "run `npm ci && npm run build` in ui/");
    }
}
