//! Manage local LLMs and expose them to coding harnesses.
//!
//! The binary is a thin shell over this library, so the logic that decides whether a
//! model may be loaded is testable without a GPU present.

pub mod auth;
pub mod config;
pub mod download;
pub mod gateway;
pub mod gguf;
pub mod registry;
pub mod serve;
pub mod source;
pub mod vram;

/// VRAM currently in use, in MiB, read from the amdgpu sysfs node.
///
/// This is the desktop's footprint when no model is loaded, and it is the number every
/// budget calculation starts from.
///
/// # Errors
/// If no amdgpu card exposes `mem_info_vram_used`, or its contents do not parse.
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
    anyhow::bail!("no amdgpu card exposes mem_info_vram_used")
}
