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
