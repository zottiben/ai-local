//! Optional companion binaries, installed only if you want them.
//!
//! The core `ailocal` is what a fresh machine downloads over `curl | sh` and what runs
//! under systemd all day, so it stays small and gains no dependency it does not need
//! to serve a model. Evaluation and training are neither of those things: they are
//! occasional, they carry their own task corpora and toolchains, and training will
//! eventually drag in a Python sidecar. Shipping them inside the core binary would make
//! every install pay for them.
//!
//! So they are separate binaries in this same repo, released from the same tag, and
//! reached through the core CLI as external subcommands - the pattern `git` and `cargo`
//! use. `ailocal eval ...` runs `ailocal-eval` if it is installed, and tells you how to
//! install it if it is not. Nothing about the core changes when an extra gains a
//! feature.
//!
//! Version skew is the one real hazard of splitting a CLI in two, so extras are
//! installed at the core's own version rather than at the latest, and `ailocal update`
//! refreshes whatever is installed alongside the core.

use std::path::{Path, PathBuf};

use crate::update::{is_build_dir, which};

/// One optional companion binary.
pub struct Extra {
    /// Subcommand name, as typed: `ailocal eval`.
    pub name: &'static str,
    /// Binary that implements it.
    pub binary: &'static str,
    /// One line for `ailocal extras`.
    pub summary: &'static str,
}

/// Every extra this release knows about.
pub const EXTRAS: &[Extra] = &[Extra {
    name: "eval",
    binary: "ailocal-eval",
    summary: "score models on a coding task set, and compare two side by side",
}];

/// Look up an extra by the subcommand name.
#[must_use]
pub fn find(name: &str) -> Option<&'static Extra> {
    EXTRAS.iter().find(|e| e.name == name)
}

/// Where an extra's binary is, if it is installed.
///
/// Looks beside the running binary before consulting `PATH`, so a locally built
/// `target/release/ailocal` finds the `ailocal-eval` built next to it rather than an
/// older installed copy - the alternative is testing a change against the wrong build.
#[must_use]
pub fn locate(extra: &Extra) -> Option<PathBuf> {
    let sibling = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(extra.binary)))
        .filter(|p| p.is_file());

    sibling.or_else(|| which(extra.binary))
}

/// Which extras are installed, by subcommand name.
///
/// Copies rather than borrows the names so callers can pass them straight to a command
/// line. Build-directory copies are excluded: `ailocal update` uses this to decide what
/// to refresh, and a sibling in `target/release` is not something a release can update.
#[must_use]
pub fn installed() -> Vec<&'static str> {
    EXTRAS
        .iter()
        .filter(|e| locate(e).is_some_and(|p| !is_build_dir(&p)))
        .map(|e| e.name)
        .collect()
}

/// Ask an installed extra for its version, so skew is visible rather than mysterious.
///
/// Returns `None` if the binary will not run or does not answer `--version` in the
/// conventional `<name> <version>` shape.
#[must_use]
pub fn version_of(path: &Path) -> Option<String> {
    let out = std::process::Command::new(path)
        .arg("--version")
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.split_whitespace().nth(1).map(str::to_owned)
}

/// Download and install an extra, at this binary's own version.
///
/// Pinned to our version rather than the latest release on purpose: an extra one
/// release ahead of the core is a subtle, hard-to-read failure, and `ailocal update`
/// is the supported way to move both forward together.
///
/// # Errors
/// If the install script cannot be fetched or run.
pub fn install(extra: &Extra) -> anyhow::Result<i32> {
    crate::update::run_script(&[
        "--version".to_owned(),
        format!("v{}", env!("CARGO_PKG_VERSION")),
        "--extras-only".to_owned(),
        "--extra".to_owned(),
        extra.name.to_owned(),
    ])
}

/// Delete an installed extra, returning where it was.
///
/// # Errors
/// If the binary is found but cannot be removed - most often because it lives in a
/// system directory this user cannot write to.
pub fn remove(extra: &Extra) -> anyhow::Result<Option<PathBuf>> {
    let Some(path) = locate(extra) else {
        return Ok(None);
    };
    std::fs::remove_file(&path).map_err(|e| anyhow::anyhow!("removing {}: {e}", path.display()))?;
    Ok(Some(path))
}

/// Run an extra, forwarding `args`, and return its exit code.
///
/// # Errors
/// If the extra is not installed, or its binary cannot be started.
pub fn dispatch(extra: &Extra, args: &[String]) -> anyhow::Result<i32> {
    let Some(path) = locate(extra) else {
        anyhow::bail!(
            "`ailocal {}` needs the {} extra, which is not installed.\n\
             Install it with:\n\
             \x20   ailocal extras install {}",
            extra.name,
            extra.name,
            extra.name
        );
    };

    // Spawn and wait rather than exec: the child joins this process group, so Ctrl-C
    // reaches it directly and a long eval stops when you expect it to.
    let status = std::process::Command::new(&path)
        .args(args)
        .status()
        .map_err(|e| anyhow::anyhow!("running {}: {e}", path.display()))?;

    // A signalled child has no exit code. Report the conventional 128+signal rather
    // than a bare 1, so `ailocal eval ... ; echo $?` distinguishes a failed run from
    // an interrupted one.
    Ok(status.code().unwrap_or_else(|| {
        use std::os::unix::process::ExitStatusExt as _;
        status.signal().map_or(1, |s| 128 + s)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_extra_is_addressable_by_name() {
        for e in EXTRAS {
            assert_eq!(find(e.name).map(|f| f.binary), Some(e.binary));
        }
        assert!(find("not-an-extra").is_none());
    }

    /// The binary name has to be derivable from the subcommand, because that is the
    /// contract install.sh relies on when it builds an asset name.
    #[test]
    fn binaries_are_named_after_their_subcommand() {
        for e in EXTRAS {
            assert_eq!(e.binary, format!("ailocal-{}", e.name));
        }
    }

    #[test]
    fn an_uninstalled_extra_explains_how_to_install_it() {
        let missing = Extra {
            name: "nope",
            binary: "ailocal-definitely-not-installed-xyz",
            summary: "",
        };
        let err = dispatch(&missing, &[]).unwrap_err().to_string();
        assert!(err.contains("ailocal extras install nope"), "got: {err}");
    }
}
