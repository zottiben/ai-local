//! `ailocal ui` - open the console.

use std::process::{Command, Stdio};

use clap::Parser;

#[derive(Parser)]
#[command(
    name = "ailocal-ui",
    version,
    about = "A console for this machine's models, gateway and services, in the browser"
)]
struct Args {
    /// Port to listen on, on 127.0.0.1. 0 lets the OS pick a free one.
    #[arg(long, default_value_t = 0)]
    port: u16,

    /// Print the link without opening a browser.
    #[arg(long)]
    no_open: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async move {
            let server = ailocal_ui::Server::bind(args.port).await?;
            let url = server.url();
            println!("console  {url}");
            println!("         ctrl-c to stop");
            if !ailocal_ui::bundle_embedded() {
                eprintln!(
                    "note: this build has no frontend compiled in, so the page will say \
                     how to build one"
                );
            }

            if !args.no_open {
                // Opening is a convenience, never a reason to fail: a headless box
                // still has a perfectly good link printed above.
                if let Err(e) = open_browser(&url) {
                    eprintln!("note: could not open a browser ({e}) - open the link above");
                }
            }

            server.serve().await
        })
}

/// Hand `url` to the desktop's opener.
///
/// By absolute path, as everything ailocal spawns is: `open` is part of macOS and
/// `xdg-open` is where xdg-utils installs it, and a bare name resolves through
/// whatever `PATH` the caller happened to have.
fn open_browser(url: &str) -> anyhow::Result<()> {
    let opener = if cfg!(target_os = "macos") {
        "/usr/bin/open"
    } else {
        "/usr/bin/xdg-open"
    };
    Command::new(opener)
        .arg(url)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| anyhow::anyhow!("running {opener}: {e}"))?;
    Ok(())
}
