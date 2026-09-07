//! `ailocal update` - install the latest release in place.
//!
//! Deliberately owns no install logic of its own: it fetches the published
//! `install.sh` and runs it. That script already knows every platform detail - the
//! architecture matrix, checksum verification, the sudo fallback, restarting the
//! gateway - and keeping one copy of that knowledge means an update can never drift
//! from a fresh install. This contributes only what the script cannot know on its own:
//! the running binary's version, passed as `AILOCAL_CURRENT_VERSION` so the script can
//! answer "already up to date" without downloading a release.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context as _, bail};

/// Where the install/update script is published. `AILOCAL_INSTALL_URL` overrides it
/// (any URL curl accepts, including `file://`, which is how the tests run offline).
const INSTALL_URL: &str = "https://zottiben.github.io/ai-local/install.sh";

/// `ailocal update`: refresh the core binary and every extra installed beside it.
///
/// Extras are carried along rather than left behind, because a core and an extra from
/// different releases is the one failure this split CLI can create on its own. The
/// script's `--check` path exits before installing anything, so `update --check` still
/// only reports.
///
/// # Errors
/// If curl is missing, the script cannot be downloaded, or it cannot be run.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let mut all = args.to_vec();
    for extra in crate::extras::installed() {
        all.push("--extra".to_owned());
        all.push(extra.to_owned());
    }
    run_script(&all)
}

/// Fetch the published install script and run it, forwarding `args` and inheriting
/// stdio so the script's own progress output is what the user sees.
///
/// # Errors
/// If curl is missing, the script cannot be downloaded, or it cannot be run.
pub fn run_script(args: &[String]) -> anyhow::Result<i32> {
    if which("curl").is_none() {
        bail!(
            "installing needs curl on your PATH. Install curl, or download a \
             build from https://github.com/zottiben/ai-local/releases/latest"
        );
    }

    if let Some(exe) = std::env::current_exe().ok().filter(|p| is_build_dir(p)) {
        // A `cargo run` build updates the *installed* copy, not this one - say so
        // rather than let the version afterwards look like nothing happened.
        eprintln!(
            "Note: {} is a development build; this updates the installed copy.",
            exe.display()
        );
    }

    let url = std::env::var("AILOCAL_INSTALL_URL").unwrap_or_else(|_| INSTALL_URL.to_owned());
    let script = TempScript::new();

    let fetched = Command::new("curl")
        .args(["-fsSL", &url, "-o"])
        .arg(&script.0)
        .status()
        .context("running curl to fetch the install script")?;
    if !fetched.success() {
        bail!("could not download the install script from {url}");
    }

    let status = Command::new("sh")
        .arg(&script.0)
        .args(args)
        .env("AILOCAL_CURRENT_VERSION", env!("CARGO_PKG_VERSION"))
        .status()
        .context("running the install script")?;
    Ok(status.code().unwrap_or(1))
}

/// The downloaded script, removed when it goes out of scope, including on error paths.
struct TempScript(PathBuf);

impl TempScript {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!("ailocal-install-{}.sh", std::process::id())))
    }
}

impl Drop for TempScript {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).ok();
    }
}

/// Whether `exe` sits in a cargo build directory, including a cross-compiled
/// `target/<triple>/release/ailocal`.
///
/// Public because a systemd unit must never point into one: `cargo clean` would then
/// silently break the service.
#[must_use]
pub fn is_build_dir(exe: &Path) -> bool {
    let mut parents = exe.ancestors().skip(1);
    let Some(profile) = parents.next().and_then(|p| p.file_name()) else {
        return false;
    };
    if profile != "debug" && profile != "release" {
        return false;
    }
    parents.any(|p| p.file_name().is_some_and(|n| n == "target"))
}

/// Find `program` on `PATH`.
#[must_use]
pub fn which(program: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join(program))
            .find(|p| p.is_file())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_cargo_build_directories() {
        assert!(is_build_dir(Path::new(
            "/home/u/proj/target/release/ailocal"
        )));
        assert!(is_build_dir(Path::new("/home/u/proj/target/debug/ailocal")));
        assert!(is_build_dir(Path::new(
            "/home/u/proj/target/x86_64-unknown-linux-gnu/release/ailocal"
        )));
    }

    #[test]
    fn an_installed_binary_is_not_a_build_directory() {
        assert!(!is_build_dir(Path::new("/home/u/.local/bin/ailocal")));
        assert!(!is_build_dir(Path::new("/usr/local/bin/ailocal")));
        // "release" alone is not enough without a target/ ancestor.
        assert!(!is_build_dir(Path::new("/opt/release/ailocal")));
    }

    #[test]
    fn which_finds_a_program_that_exists() {
        assert!(which("sh").is_some());
        assert!(which("definitely-not-a-real-program-xyz").is_none());
    }
}
