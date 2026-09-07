//! systemd units, so the gateway survives a reboot and a closed terminal.
//!
//! **User** units, not system ones. The model needs the GPU and writes under the user's
//! home, running as this user is what we want anyway, and it means installing the
//! service needs no root - which matters because `sudo` on this machine prompts for a
//! password an agent cannot supply.
//!
//! The one thing user units cannot do by themselves is start before login. That needs
//! `loginctl enable-linger`, which is a privileged call and therefore the user's to
//! make; [`linger_enabled`] reports whether it has been.

use std::path::{Path, PathBuf};

use anyhow::Context as _;

pub const GATEWAY_UNIT: &str = "ailocal-gateway.service";
pub const MODEL_UNIT: &str = "ailocal-model.service";

/// Directory systemd reads user units from.
///
/// # Errors
/// If neither `XDG_CONFIG_HOME` nor `HOME` is set.
pub fn unit_dir() -> anyhow::Result<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        return Ok(PathBuf::from(xdg).join("systemd/user"));
    }
    let home = std::env::var_os("HOME").ok_or_else(|| anyhow::anyhow!("HOME is not set"))?;
    Ok(PathBuf::from(home).join(".config/systemd/user"))
}

/// The gateway unit.
#[must_use]
pub fn gateway_unit(exe: &Path, host: &str, port: u16) -> String {
    format!(
        "[Unit]\n\
         Description=ailocal gateway (local models for coding harnesses)\n\
         Documentation=https://github.com/zottiben/ai-local\n\
         After=network-online.target\n\
         Wants=network-online.target\n\
         \n\
         [Service]\n\
         Type=simple\n\
         ExecStart={exe} gateway run --host {host} --port {port}\n\
         Restart=always\n\
         RestartSec=3\n\
         # The gateway is cheap to restart and useless when down, so keep trying rather\n\
         # than giving up after the default burst limit.\n\
         StartLimitIntervalSec=0\n\
         \n\
         [Install]\n\
         WantedBy=default.target\n",
        exe = exe.display(),
    )
}

/// The model unit, which keeps one model resident.
///
/// Runs llama-server in the foreground under systemd rather than detaching, so systemd
/// owns the process: it can restart it, collect its output, and kill it reliably.
#[must_use]
pub fn model_unit(exe: &Path, model: &str) -> String {
    format!(
        "[Unit]\n\
         Description=ailocal model server ({model})\n\
         Documentation=https://github.com/zottiben/ai-local\n\
         \n\
         [Service]\n\
         Type=simple\n\
         ExecStart={exe} serve {model} --foreground\n\
         Restart=on-failure\n\
         RestartSec=5\n\
         # A model load takes tens of seconds and VRAM has to be released between\n\
         # attempts, so do not hammer it.\n\
         StartLimitBurst=3\n\
         StartLimitIntervalSec=300\n\
         \n\
         [Install]\n\
         WantedBy=default.target\n",
        exe = exe.display(),
    )
}

/// Whether user services keep running when nobody is logged in.
#[must_use]
pub fn linger_enabled() -> bool {
    std::process::Command::new("loginctl")
        .args(["show-user", "--property=Linger", "--value"])
        .arg(whoami())
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).trim() == "yes")
}

fn whoami() -> String {
    std::env::var("USER").unwrap_or_else(|_| "root".into())
}

/// Install (or refresh) the unit files.
///
/// Returns the units written. `None` for the model unit means no default model is
/// configured, so nothing is preloaded and the gateway will pull one in on demand.
///
/// # Errors
/// If the unit directory or files cannot be written.
pub fn install(
    exe: &Path,
    host: &str,
    port: u16,
    default_model: Option<&str>,
) -> anyhow::Result<Vec<String>> {
    let dir = unit_dir()?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    let mut written = Vec::new();

    let gateway = dir.join(GATEWAY_UNIT);
    std::fs::write(&gateway, gateway_unit(exe, host, port))
        .with_context(|| format!("writing {}", gateway.display()))?;
    written.push(GATEWAY_UNIT.to_owned());

    let model_unit_path = dir.join(MODEL_UNIT);
    match default_model {
        Some(model) => {
            std::fs::write(&model_unit_path, model_unit(exe, model))
                .with_context(|| format!("writing {}", model_unit_path.display()))?;
            written.push(MODEL_UNIT.to_owned());
        }
        // Leaving a stale unit behind would keep loading a model the config no longer
        // names.
        None => {
            std::fs::remove_file(&model_unit_path).ok();
        }
    }

    Ok(written)
}

/// Remove the unit files.
///
/// # Errors
/// If a file exists but cannot be removed.
pub fn uninstall() -> anyhow::Result<Vec<String>> {
    let dir = unit_dir()?;
    let mut removed = Vec::new();
    for unit in [GATEWAY_UNIT, MODEL_UNIT] {
        let path = dir.join(unit);
        if path.exists() {
            std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
            removed.push(unit.to_owned());
        }
    }
    Ok(removed)
}

/// Run `systemctl --user` with the given arguments.
///
/// # Errors
/// If systemctl cannot be run.
pub fn systemctl(args: &[&str]) -> anyhow::Result<std::process::Output> {
    std::process::Command::new("systemctl")
        .arg("--user")
        .args(args)
        .output()
        .context("running systemctl --user")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gateway_unit_restarts_forever() {
        let unit = gateway_unit(Path::new("/usr/bin/ailocal"), "0.0.0.0", 8081);
        assert!(unit.contains("ExecStart=/usr/bin/ailocal gateway run --host 0.0.0.0 --port 8081"));
        assert!(unit.contains("Restart=always"));
        // Without this systemd stops retrying after a burst, which would leave the
        // gateway down after a transient failure at boot.
        assert!(unit.contains("StartLimitIntervalSec=0"));
        assert!(unit.contains("WantedBy=default.target"));
    }

    /// The model unit must supervise llama-server directly, not a command that forks
    /// and returns - otherwise systemd thinks the service ended the moment it started.
    #[test]
    fn the_model_unit_runs_in_the_foreground() {
        let unit = model_unit(Path::new("/usr/bin/ailocal"), "gemma4-12b");
        assert!(unit.contains("ExecStart=/usr/bin/ailocal serve gemma4-12b --foreground"));
        assert!(unit.contains("Type=simple"));
        // Model loads are slow and need VRAM released between attempts.
        assert!(unit.contains("Restart=on-failure"));
        assert!(unit.contains("StartLimitBurst=3"));
    }

    #[test]
    fn units_name_the_binary_by_absolute_path() {
        // systemd does not resolve PATH for ExecStart.
        let unit = gateway_unit(Path::new("/home/u/.local/bin/ailocal"), "127.0.0.1", 1);
        let exec = unit
            .lines()
            .find(|l| l.starts_with("ExecStart="))
            .expect("ExecStart");
        assert!(exec["ExecStart=".len()..].starts_with('/'), "{exec}");
    }
}
