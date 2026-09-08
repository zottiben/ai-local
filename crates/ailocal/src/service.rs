//! Keeping the gateway and a model running, across reboots.
//!
//! Two service managers, one shape. On Linux these are systemd **user** units; on macOS
//! they are launchd **agents**. Both run as the logged-in user rather than root, which
//! is what we want anyway - the model needs the GPU and writes under the user's home -
//! and it means installing the service needs no privileges.
//!
//! Each platform has one privileged loose end, and both are the user's to tie:
//! `loginctl enable-linger` on Linux so units start before login, and nothing at all on
//! macOS, where a `LaunchAgent` starts at login by design.

use std::path::{Path, PathBuf};

use anyhow::Context as _;

/// Names used on Linux.
pub const GATEWAY_UNIT: &str = "ailocal-gateway.service";
pub const MODEL_UNIT: &str = "ailocal-model.service";

/// Labels used on macOS. launchd identifies jobs by reverse-DNS label, not filename.
pub const GATEWAY_LABEL: &str = "io.github.zottiben.ailocal.gateway";
pub const MODEL_LABEL: &str = "io.github.zottiben.ailocal.model";

/// Which init system we are talking to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Manager {
    Systemd,
    Launchd,
}

/// The service manager on this platform.
#[must_use]
pub const fn manager() -> Manager {
    if cfg!(target_os = "macos") {
        Manager::Launchd
    } else {
        Manager::Systemd
    }
}

/// Names of the two services, as this platform knows them.
#[must_use]
pub fn service_names() -> [&'static str; 2] {
    match manager() {
        Manager::Systemd => [GATEWAY_UNIT, MODEL_UNIT],
        Manager::Launchd => [GATEWAY_LABEL, MODEL_LABEL],
    }
}

/// Directory holding the unit or plist files.
///
/// # Errors
/// If the home directory cannot be determined.
pub fn unit_dir() -> anyhow::Result<PathBuf> {
    let home = std::env::var_os("HOME").ok_or_else(|| anyhow::anyhow!("HOME is not set"))?;
    Ok(match manager() {
        Manager::Launchd => PathBuf::from(home).join("Library/LaunchAgents"),
        Manager::Systemd => std::env::var_os("XDG_CONFIG_HOME")
            .map_or_else(|| PathBuf::from(home).join(".config"), PathBuf::from)
            .join("systemd/user"),
    })
}

/// Where a launchd agent writes its output. launchd has no journal, so without this
/// the logs go nowhere.
fn log_dir() -> anyhow::Result<PathBuf> {
    let home = std::env::var_os("HOME").ok_or_else(|| anyhow::anyhow!("HOME is not set"))?;
    Ok(PathBuf::from(home).join("Library/Logs/ailocal"))
}

/// Filename for a service, including extension.
fn file_name(service: &str) -> String {
    match manager() {
        Manager::Systemd => service.to_owned(),
        Manager::Launchd => format!("{service}.plist"),
    }
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
/// Runs llama-server in the foreground under the service manager rather than detaching,
/// so the manager owns the process: it can restart it, collect its output, and kill it
/// reliably.
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

/// A launchd agent.
///
/// `KeepAlive` restarts it; `RunAtLoad` starts it at login. `ProcessType: Background`
/// asks macOS not to throttle it the way it throttles idle helpers, which matters for
/// something doing sustained GPU work.
#[must_use]
pub fn plist(label: &str, exe: &Path, args: &[&str], logs: &Path, keep_alive: bool) -> String {
    let arguments = std::iter::once(exe.display().to_string())
        .chain(args.iter().map(|a| (*a).to_owned()))
        .map(|a| format!("        <string>{}</string>\n", escape_xml(&a)))
        .collect::<String>();

    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \
         \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
         <plist version=\"1.0\">\n\
         <dict>\n\
         \x20   <key>Label</key>\n\
         \x20   <string>{label}</string>\n\
         \x20   <key>ProgramArguments</key>\n\
         \x20   <array>\n{arguments}\
         \x20   </array>\n\
         \x20   <key>RunAtLoad</key>\n\
         \x20   <true/>\n\
         \x20   <key>KeepAlive</key>\n\
         \x20   <{keep_alive}/>\n\
         \x20   <key>ProcessType</key>\n\
         \x20   <string>Background</string>\n\
         \x20   <key>StandardOutPath</key>\n\
         \x20   <string>{out}</string>\n\
         \x20   <key>StandardErrorPath</key>\n\
         \x20   <string>{out}</string>\n\
         </dict>\n\
         </plist>\n",
        keep_alive = keep_alive,
        out = escape_xml(&logs.join(format!("{label}.log")).display().to_string()),
    )
}

/// Escape the five XML entities. Paths can contain `&`, and a home directory with one
/// would otherwise produce a plist launchd refuses to parse.
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Whether user services keep running when nobody is logged in.
///
/// Only meaningful under systemd. launchd agents are tied to a login session by design,
/// so there is no equivalent to enable and this reports true.
#[must_use]
pub fn linger_enabled() -> bool {
    if manager() == Manager::Launchd {
        return true;
    }
    std::process::Command::new("loginctl")
        .args(["show-user", "--property=Linger", "--value"])
        .arg(whoami())
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).trim() == "yes")
}

fn whoami() -> String {
    std::env::var("USER").unwrap_or_else(|_| "root".into())
}

/// Install (or refresh) the service definitions.
///
/// Returns the services written. `None` for the model service means no default model is
/// configured, so nothing is preloaded and the gateway will pull one in on demand.
///
/// # Errors
/// If the service directory or files cannot be written.
pub fn install(
    exe: &Path,
    host: &str,
    port: u16,
    default_model: Option<&str>,
) -> anyhow::Result<Vec<String>> {
    let dir = unit_dir()?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    let [gateway, model] = service_names();
    let mut written = Vec::new();

    let gateway_body = match manager() {
        Manager::Systemd => gateway_unit(exe, host, port),
        Manager::Launchd => {
            let logs = log_dir()?;
            std::fs::create_dir_all(&logs)?;
            let port = port.to_string();
            plist(
                gateway,
                exe,
                &["gateway", "run", "--host", host, "--port", &port],
                &logs,
                true,
            )
        }
    };
    let gateway_path = dir.join(file_name(gateway));
    std::fs::write(&gateway_path, gateway_body)
        .with_context(|| format!("writing {}", gateway_path.display()))?;
    written.push(gateway.to_owned());

    let model_path = dir.join(file_name(model));
    match default_model {
        Some(name) => {
            let body = match manager() {
                Manager::Systemd => model_unit(exe, name),
                Manager::Launchd => {
                    let logs = log_dir()?;
                    plist(model, exe, &["serve", name, "--foreground"], &logs, true)
                }
            };
            std::fs::write(&model_path, body)
                .with_context(|| format!("writing {}", model_path.display()))?;
            written.push(model.to_owned());
        }
        // Leaving a stale definition behind would keep loading a model the config no
        // longer names.
        None => {
            std::fs::remove_file(&model_path).ok();
        }
    }

    Ok(written)
}

/// Remove the service definitions.
///
/// # Errors
/// If a file exists but cannot be removed.
pub fn uninstall() -> anyhow::Result<Vec<String>> {
    let dir = unit_dir()?;
    let mut removed = Vec::new();
    for service in service_names() {
        let path = dir.join(file_name(service));
        if path.exists() {
            std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
            removed.push(service.to_owned());
        }
    }
    Ok(removed)
}

/// Start a service and enable it at login.
///
/// # Errors
/// If the service manager cannot be run.
pub fn enable(service: &str) -> anyhow::Result<()> {
    match manager() {
        Manager::Systemd => {
            systemctl(&["enable", "--now", service])?;
        }
        Manager::Launchd => {
            let path = unit_dir()?.join(file_name(service));
            // bootstrap rather than the older `load`: it reports why a job was rejected
            // instead of failing silently. Re-bootstrapping an already-loaded job is an
            // error, so replace it.
            launchctl(&["bootout", &domain_target(service)]).ok();
            launchctl(&["bootstrap", &domain(), &path.display().to_string()])?;
        }
    }
    Ok(())
}

/// Stop a service and prevent it starting at login.
///
/// # Errors
/// If the service manager cannot be run.
pub fn disable(service: &str) -> anyhow::Result<()> {
    match manager() {
        Manager::Systemd => {
            systemctl(&["disable", "--now", service])?;
        }
        Manager::Launchd => {
            launchctl(&["bootout", &domain_target(service)])?;
        }
    }
    Ok(())
}

/// Stop a service without changing whether it starts on its own.
///
/// Distinct from [`disable`], which also un-enables it. This is for handing the model
/// over temporarily - the eval harness takes ownership of llama-server to control the
/// reasoning flag, and a supervisor that restarts it underneath would fight for it.
///
/// On launchd this unloads the job, because there is no stop that survives a
/// `KeepAlive`. [`start`] loads it again, so the pair round-trips.
///
/// # Errors
/// If the service manager cannot be run.
pub fn stop(service: &str) -> anyhow::Result<()> {
    match manager() {
        Manager::Systemd => {
            systemctl(&["stop", service])?;
        }
        Manager::Launchd => {
            launchctl(&["bootout", &domain_target(service)])?;
        }
    }
    Ok(())
}

/// Start a service that is already installed, without enabling it.
///
/// # Errors
/// If the service manager cannot be run.
pub fn start(service: &str) -> anyhow::Result<()> {
    match manager() {
        Manager::Systemd => {
            systemctl(&["start", service])?;
        }
        Manager::Launchd => {
            let path = unit_dir()?.join(file_name(service));
            launchctl(&["bootstrap", &domain(), &path.display().to_string()])?;
        }
    }
    Ok(())
}

/// Where to look when a service will not start.
///
/// The two platforms keep this in completely different places, and "check the logs" is
/// useless advice without the path.
#[must_use]
pub fn log_hint() -> String {
    match manager() {
        Manager::Systemd => format!("journalctl --user -u {GATEWAY_UNIT} -n 50"),
        Manager::Launchd => log_dir().map_or_else(
            |_| "~/Library/Logs/ailocal/".to_owned(),
            |dir| format!("{}/{GATEWAY_LABEL}.log", dir.display()),
        ),
    }
}

/// Whether a service is currently running.
#[must_use]
pub fn is_active(service: &str) -> bool {
    match manager() {
        Manager::Systemd => systemctl(&["is-active", service])
            .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).trim() == "active"),
        // `launchctl print` succeeds only for a job that is bootstrapped, and reports a
        // pid only while it is actually running.
        Manager::Launchd => launchctl(&["print", &domain_target(service)])
            .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains("pid = ")),
    }
}

/// Whether a service is set to start on its own.
#[must_use]
pub fn is_enabled(service: &str) -> bool {
    match manager() {
        Manager::Systemd => systemctl(&["is-enabled", service])
            .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).trim() == "enabled"),
        // A bootstrapped agent with RunAtLoad is enabled by definition.
        Manager::Launchd => {
            launchctl(&["print", &domain_target(service)]).is_ok_and(|o| o.status.success())
        }
    }
}

/// The launchd domain for this user's GUI session.
fn domain() -> String {
    format!("gui/{}", uid())
}

fn domain_target(label: &str) -> String {
    format!("{}/{label}", domain())
}

fn uid() -> u32 {
    rustix::process::getuid().as_raw()
}

/// Reload the manager's view of on-disk definitions.
///
/// # Errors
/// If the manager cannot be run.
pub fn reload() -> anyhow::Result<()> {
    if manager() == Manager::Systemd {
        systemctl(&["daemon-reload"])?;
    }
    // launchd has no equivalent: bootstrap reads the plist each time.
    Ok(())
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

/// Run `launchctl` with the given arguments.
///
/// # Errors
/// If launchctl cannot be run.
pub fn launchctl(args: &[&str]) -> anyhow::Result<std::process::Output> {
    let out = std::process::Command::new("launchctl")
        .args(args)
        .output()
        .context("running launchctl")?;
    anyhow::ensure!(
        out.status.success(),
        "launchctl {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr).trim()
    );
    Ok(out)
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

    /// The model service must supervise llama-server directly, not a command that forks
    /// and returns - otherwise the manager thinks the service ended as it started.
    #[test]
    fn the_model_unit_runs_in_the_foreground() {
        let unit = model_unit(Path::new("/usr/bin/ailocal"), "gemma4-12b");
        assert!(unit.contains("ExecStart=/usr/bin/ailocal serve gemma4-12b --foreground"));
        assert!(unit.contains("Type=simple"));
        assert!(unit.contains("Restart=on-failure"));
        assert!(unit.contains("StartLimitBurst=3"));
    }

    #[test]
    fn units_name_the_binary_by_absolute_path() {
        // Neither systemd nor launchd resolves PATH for the program it runs.
        let unit = gateway_unit(Path::new("/home/u/.local/bin/ailocal"), "127.0.0.1", 1);
        let exec = unit
            .lines()
            .find(|l| l.starts_with("ExecStart="))
            .expect("ExecStart");
        assert!(exec["ExecStart=".len()..].starts_with('/'), "{exec}");
    }

    #[test]
    fn a_plist_carries_the_binary_and_every_argument_separately() {
        let p = plist(
            GATEWAY_LABEL,
            Path::new("/opt/ailocal"),
            &["gateway", "run", "--host", "127.0.0.1", "--port", "8081"],
            Path::new("/tmp/logs"),
            true,
        );
        assert!(p.contains("<string>io.github.zottiben.ailocal.gateway</string>"));
        assert!(p.contains("<string>/opt/ailocal</string>"));
        // launchd does not run a shell, so arguments must be separate array entries
        // rather than one joined string.
        assert!(p.contains("<string>--host</string>"));
        assert!(p.contains("<string>127.0.0.1</string>"));
        assert!(!p.contains("gateway run --host"));
    }

    #[test]
    fn a_plist_starts_at_login_and_restarts() {
        let p = plist(
            MODEL_LABEL,
            Path::new("/opt/ailocal"),
            &["serve"],
            Path::new("/l"),
            true,
        );
        assert!(p.contains("<key>RunAtLoad</key>\n    <true/>"));
        assert!(p.contains("<key>KeepAlive</key>\n    <true/>"));
        // launchd has no journal, so output has to be pointed somewhere.
        assert!(p.contains("<key>StandardOutPath</key>"));
        assert!(p.contains("/l/io.github.zottiben.ailocal.model.log"));
    }

    /// A home directory containing `&` would otherwise produce a plist launchd refuses.
    #[test]
    fn plist_paths_are_xml_escaped() {
        let p = plist(
            "x",
            Path::new("/Users/a&b/bin/ailocal"),
            &["serve", "<model>"],
            Path::new("/l"),
            false,
        );
        assert!(p.contains("/Users/a&amp;b/bin/ailocal"));
        assert!(p.contains("&lt;model&gt;"));
        assert!(!p.contains("a&b"));
        assert!(p.contains("<key>KeepAlive</key>\n    <false/>"));
    }

    #[test]
    fn service_names_match_the_platform() {
        let [gateway, _] = service_names();
        match manager() {
            Manager::Systemd => {
                assert_eq!(gateway, GATEWAY_UNIT);
                assert_eq!(file_name(gateway), "ailocal-gateway.service");
            }
            Manager::Launchd => {
                assert_eq!(gateway, GATEWAY_LABEL);
                assert_eq!(
                    file_name(gateway),
                    "io.github.zottiben.ailocal.gateway.plist"
                );
            }
        }
    }

    /// launchd agents are bound to a login session, so there is nothing to enable.
    #[test]
    fn lingering_is_only_a_systemd_concern() {
        if manager() == Manager::Launchd {
            assert!(linger_enabled());
        }
    }
}
