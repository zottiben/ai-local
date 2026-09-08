//! Manage local LLMs and expose them to coding harnesses.
//!
//! The binary is a thin shell over this library, so the logic that decides whether a
//! model may be loaded is testable without a GPU present.

pub mod anthropic;
pub mod auth;
pub mod catalogue;
pub mod config;
pub mod device;
pub mod download;
pub mod extras;
pub mod gateway;
pub mod gguf;
pub mod harness;
pub mod prompt;
pub mod registry;
pub mod serve;
pub mod service;
pub mod source;
pub mod update;
pub mod vram;

/// Free space on the filesystem holding `path`, in MiB.
///
/// Walks up to the nearest ancestor that exists, so it answers for a directory that has
/// not been created yet - which is exactly when it is asked, while someone is choosing
/// where to put thirty gigabytes of weights.
///
/// Returns `None` rather than an error: this informs a choice, it does not gate one, and
/// a filesystem that will not answer `statvfs` is not a reason to refuse to continue.
#[must_use]
pub fn free_mib(path: &std::path::Path) -> Option<u64> {
    let existing = path.ancestors().find(|p| p.exists())?;
    let stat = rustix::fs::statvfs(existing).ok()?;
    // f_bavail is what a non-root user may actually use, which is smaller than f_bfree
    // and is the number that decides whether a download completes.
    Some(stat.f_bavail.saturating_mul(stat.f_frsize) / (1024 * 1024))
}

/// Render a byte count in whichever unit reads best, for space rather than for weights.
#[must_use]
pub fn format_mib(mib: u64) -> String {
    if mib >= 1024 {
        format!("{:.0} GB", mib as f64 / 1024.0)
    } else {
        format!("{mib} MB")
    }
}

/// VRAM currently in use, in MiB.
///
/// This is the footprint of everything that is not the model we are about to load, and
/// it is the number every budget calculation starts from.
///
/// Reads amdgpu's sysfs node where it exists, because that is cheap enough to poll from
/// the load watchdog. Everywhere else - macOS included - it falls back to asking
/// llama.cpp, which is slower but universal.
///
/// # Errors
/// If neither source can report memory usage.
pub fn vram_used_mib() -> anyhow::Result<u64> {
    use anyhow::Context as _;

    for card in 0..4 {
        let path = format!("/sys/class/drm/card{card}/device/mem_info_vram_used");
        if let Ok(raw) = std::fs::read_to_string(&path) {
            let bytes: u64 = raw
                .trim()
                .parse()
                .with_context(|| format!("parsing {path}"))?;
            return Ok(bytes / (1024 * 1024));
        }
    }

    let device = primary_device()?;
    Ok(device.total_mib.saturating_sub(device.free_mib))
}

/// Whether VRAM usage can be sampled cheaply enough to poll while a model loads.
///
/// The load watchdog exists because over-committing a discrete GPU takes the
/// compositor's framebuffer and kills the session. It needs a reading every 200 ms,
/// which rules out spawning llama-server - and on unified memory there is no such
/// catastrophe to guard against anyway.
#[must_use]
pub fn can_poll_vram() -> bool {
    (0..4).any(|card| {
        std::path::Path::new(&format!(
            "/sys/class/drm/card{card}/device/mem_info_vram_used"
        ))
        .exists()
    })
}

/// The device a model would load onto, probed once per process.
///
/// Memoised because probing spawns llama-server, and the answer cannot change while we
/// run - a card is not hot-plugged mid-command.
///
/// # Errors
/// If llama.cpp reports no usable device.
pub fn primary_device() -> anyhow::Result<device::Device> {
    static CACHE: std::sync::OnceLock<Option<device::Device>> = std::sync::OnceLock::new();

    CACHE
        .get_or_init(|| {
            device::probe()
                .ok()
                .and_then(|d| device::primary(&d).cloned())
        })
        .clone()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "llama.cpp reports no GPU. Install a backend (ggml-vulkan on Linux; \
                 Metal is built in on macOS) and check `llama-server --list-devices`"
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The point of this is to answer for a directory that does not exist yet, since
    /// that is the state it is asked about - while someone decides where to put a model.
    #[test]
    fn free_space_answers_for_a_directory_not_yet_created() {
        let absent = std::env::temp_dir().join("ailocal-definitely-absent/models/deeper");
        assert!(!absent.exists());
        assert!(
            free_mib(&absent).is_some(),
            "should have reported the filesystem holding its nearest existing parent"
        );
    }

    #[test]
    fn free_space_is_reported_for_a_real_directory() {
        let free = free_mib(std::path::Path::new("/")).expect("root filesystem");
        assert!(
            free > 0,
            "a mounted filesystem should report some free space"
        );
    }

    /// A path with no existing ancestor at all cannot be measured, and that is a
    /// missing answer rather than a failure.
    #[test]
    fn an_unrooted_relative_path_reports_nothing() {
        assert_eq!(free_mib(std::path::Path::new("")), None);
    }

    #[test]
    fn sizes_read_in_the_unit_that_suits_them() {
        assert_eq!(format_mib(512), "512 MB");
        assert_eq!(format_mib(1024), "1 GB");
        assert_eq!(format_mib(325 * 1024), "325 GB");
    }
}
