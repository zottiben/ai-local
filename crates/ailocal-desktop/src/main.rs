//! `ai-local.app` - the console in a window (D8).
//!
//! A shell, not a second implementation. It runs the same `ailocal_ui::Server` the
//! `ailocal ui` command runs, in its own process on a port the OS picks, and points a
//! webview at it. One frontend, one API, and the window and the terminal can only
//! disagree about chrome.
//!
//! What it owns that the terminal does not: a model started from here is this
//! process's child, and quitting has to stop it, because there is no shell to deliver
//! Ctrl-C to the process group (D9).

// Hides the console window a Windows release build would otherwise open. There is no
// Windows build (D8), and this is one line against the day someone tries.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use tauri::{RunEvent, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_dialog::{DialogExt, MessageDialogKind};
use tauri_plugin_window_state::{StateFlags, WindowExt};

/// How long to wait for the server to bind before giving up on it.
///
/// Binding a loopback port is immediate; this only has to cover a machine that is
/// thrashing, and it exists so a failure is a dialog rather than a window that never
/// appears.
const BIND_TIMEOUT: Duration = Duration::from_secs(15);

/// The window's size the first time, before the saved geometry exists.
///
/// Wide enough for the models table's eight columns without a horizontal scrollbar,
/// which is the widest thing the console draws.
const FIRST_SIZE: (f64, f64) = (1240.0, 880.0);
const MIN_SIZE: (f64, f64) = (760.0, 540.0);

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        // Saves every window's geometry on exit. Restoring is explicit below, because
        // this window is built at runtime rather than declared in tauri.conf.json.
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .setup(|app| {
            let url = match start_server() {
                Ok(url) => url,
                Err(e) => {
                    // No window yet, and no terminal behind it: a dialog is the only
                    // way this says anything at all before the process ends.
                    app.dialog()
                        .message(format!("{e:#}"))
                        .kind(MessageDialogKind::Error)
                        .title("ai-local could not start")
                        .blocking_show();
                    app.handle().exit(1);
                    return Ok(());
                }
            };

            let window = WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url))
                .title("ai-local")
                .inner_size(FIRST_SIZE.0, FIRST_SIZE.1)
                .min_inner_size(MIN_SIZE.0, MIN_SIZE.1)
                .build()?;
            // Geometry only. Restoring visibility would re-apply a window that was
            // closed to a minimised state nobody asked for.
            window.restore_state(StateFlags::SIZE | StateFlags::POSITION)?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("building the app")
        .run(|_app, event| {
            if matches!(event, RunEvent::Exit) {
                // Best effort, and deliberately quiet: the window is already going
                // away, so there is nowhere left to report a failure to stop it.
                let _ = ailocal_ui::stop_owned_model();
            }
        });
}

/// Start the console's server on a port the OS picks, and return the URL to open.
///
/// The runtime lives on a thread of its own for the life of the process: Tauri owns
/// the main thread's event loop, which on macOS is the only thread that may touch the
/// window, so the server cannot have it.
fn start_server() -> anyhow::Result<tauri::Url> {
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("ailocal-ui".to_owned())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(e) => {
                    let _ = tx.send(Err(anyhow::anyhow!("starting the runtime: {e}")));
                    return;
                }
            };
            runtime.block_on(async move {
                let server = match ailocal_ui::Server::bind(0).await {
                    Ok(server) => server,
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        return;
                    }
                };
                // Sent before serving, because `serve` only returns when the listener
                // fails - by which time the window has long since loaded.
                if tx.send(Ok(server.url())).is_err() {
                    return;
                }
                if let Err(e) = server.serve().await {
                    eprintln!("the console's server stopped: {e:#}");
                }
            });
        })?;

    let url = match rx.recv_timeout(BIND_TIMEOUT) {
        Ok(url) => url?,
        Err(RecvTimeoutError::Timeout) => {
            anyhow::bail!("the console's server did not start within {BIND_TIMEOUT:?}")
        }
        Err(RecvTimeoutError::Disconnected) => {
            anyhow::bail!("the console's server stopped before it was reachable")
        }
    };
    Ok(url.parse()?)
}
