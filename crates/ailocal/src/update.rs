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

/// The `ailocal` a service should run.
///
/// `beside` is the one the caller expects: itself, for the command line, and the
/// `ailocal` next to it for an extra. A unit that ran the console's own binary would
/// start a web server where the gateway belongs. A build directory is never the
/// answer, because `cargo clean` would break the service, so an installed `ailocal` on
/// `PATH` stands in for one.
///
/// # Errors
/// If `beside` is missing or in a build directory and no installed `ailocal` is on
/// `PATH`.
pub fn unit_binary(beside: &Path) -> anyhow::Result<PathBuf> {
    choose_unit_binary(beside, || {
        which("ailocal").and_then(|p| p.canonicalize().ok())
    })
}

fn choose_unit_binary(
    beside: &Path,
    installed: impl FnOnce() -> Option<PathBuf>,
) -> anyhow::Result<PathBuf> {
    if beside.is_file() && !is_build_dir(beside) {
        return Ok(beside.to_owned());
    }
    if let Some(path) = installed().filter(|p| !is_build_dir(p)) {
        return Ok(path);
    }
    if is_build_dir(beside) {
        anyhow::bail!(
            "running from {} - a service must not point into a build directory, \
             because `cargo clean` would break it.\n\
             Install it first, e.g. `install -Dm755 {} ~/.local/bin/ailocal`, then \
             re-run this from there.",
            beside.display(),
            beside.display()
        )
    }
    anyhow::bail!(
        "there is no ailocal at {} or on PATH for a service to run",
        beside.display()
    )
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

    /// The CLI's own binary when it is installed; the installed one when it runs from
    /// a build directory; an error, not a unit pointing into `target`, when there is
    /// no installed one.
    #[test]
    fn a_unit_never_points_into_a_build_directory() {
        let root = std::env::temp_dir().join(format!("ailocal-unit-{}", std::process::id()));
        let installed = root.join("bin/ailocal");
        let built = root.join("src/target/debug/ailocal");
        for file in [&installed, &built] {
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(file, b"").unwrap();
        }

        assert_eq!(choose_unit_binary(&installed, || None).unwrap(), installed);
        assert_eq!(
            choose_unit_binary(&built, || Some(installed.clone())).unwrap(),
            installed
        );
        let err = choose_unit_binary(&built, || Some(built.clone())).unwrap_err();
        assert!(err.to_string().contains("cargo clean"), "{err}");
        let err = choose_unit_binary(&root.join("bin/missing"), || None).unwrap_err();
        assert!(err.to_string().contains("no ailocal"), "{err}");

        std::fs::remove_dir_all(&root).ok();
    }

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
