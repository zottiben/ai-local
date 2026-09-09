//! What GPU is present and how much memory it will give us.
//!
//! Asked of llama.cpp rather than the operating system. `llama-server --list-devices`
//! prints the same shape on every backend:
//!
//! ```text
//! Vulkan0: AMD Radeon RX 7600 XT (RADV NAVI33) (16384 MiB, 4642 MiB free)
//! MTL0: Apple M1 Max (26000 MiB, 25999 MiB free)
//! ```
//!
//! That is worth more than a platform API, because the number llama.cpp reports is the
//! number it will actually honour. On macOS in particular the total reflects the
//! `iogpu.wired_limit_mb` cap rather than physical RAM, so this stays correct when the
//! user raises or lowers that limit - something reading `hw.memsize` would miss.

use std::process::Command;

use anyhow::Context as _;

/// Which llama.cpp backend a device belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Vulkan,
    Metal,
    Cuda,
    Rocm,
    Other,
}

impl Backend {
    fn from_id(id: &str) -> Self {
        // Device ids are `<Backend><index>`, e.g. `Vulkan0`, `MTL0`, `CUDA0`.
        let name: String = id.chars().take_while(|c| !c.is_ascii_digit()).collect();
        match name.to_ascii_uppercase().as_str() {
            "VULKAN" => Self::Vulkan,
            "MTL" | "METAL" => Self::Metal,
            "CUDA" => Self::Cuda,
            "ROCM" | "HIP" => Self::Rocm,
            _ => Self::Other,
        }
    }

    /// Whether this backend shares memory with the CPU.
    ///
    /// Unified-memory devices have no separate pool to exhaust: over-committing makes
    /// the system page rather than fail, so the consequences of a bad estimate are
    /// slowness rather than a dead graphical session.
    #[must_use]
    pub fn is_unified_memory(self) -> bool {
        self == Self::Metal
    }
}

/// A compute device llama.cpp can offload to.
#[derive(Debug, Clone)]
pub struct Device {
    pub id: String,
    pub name: String,
    pub backend: Backend,
    /// Memory the backend reports as belonging to this device.
    pub total_mib: u64,
    /// Memory free at the moment of the probe.
    pub free_mib: u64,
}

/// Ask llama.cpp what it can see.
///
/// # Errors
/// If llama-server is missing or cannot be run.
pub fn probe() -> anyhow::Result<Vec<Device>> {
    let exe = crate::llama_server()?;
    let output = Command::new(&exe)
        .arg("--list-devices")
        .output()
        .with_context(|| format!("running {} --list-devices", exe.display()))?;

    // Backends log to stderr and the device list to stdout, but which stream carries
    // the list has moved between releases, so parse both.
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push('\n');
    text.push_str(&String::from_utf8_lossy(&output.stderr));

    Ok(parse(&text))
}

/// Extract devices from `--list-devices` output.
#[must_use]
pub fn parse(text: &str) -> Vec<Device> {
    text.lines().filter_map(parse_line).collect()
}

fn parse_line(line: &str) -> Option<Device> {
    let line = line.trim();
    let (id, rest) = line.split_once(": ")?;
    // Device ids have no spaces; anything else on the line is prose.
    if id.is_empty() || id.contains(' ') || !id.chars().next()?.is_ascii_alphabetic() {
        return None;
    }

    // The trailing "(<total> MiB, <free> MiB free)" is the last parenthesised group -
    // device names contain their own brackets, e.g. "(RADV NAVI33)".
    let open = rest.rfind('(')?;
    let close = rest.rfind(')')?;
    if close < open {
        return None;
    }
    let (name, memory) = (rest[..open].trim(), &rest[open + 1..close]);

    let (total, free) = memory.split_once(',')?;
    let total_mib = mib(total)?;
    let free_mib = mib(free)?;

    Some(Device {
        id: id.to_owned(),
        name: name.to_owned(),
        backend: Backend::from_id(id),
        total_mib,
        free_mib,
    })
}

/// Parse a `"16384 MiB"` or `"4642 MiB free"` fragment.
fn mib(fragment: &str) -> Option<u64> {
    fragment.split_whitespace().next()?.parse().ok()
}

/// The device a model would be loaded onto: the one with the most memory.
///
/// llama.cpp offloads to the first device by default, but a machine with an integrated
/// and a discrete GPU lists both, and the big one is the one that matters.
#[must_use]
pub fn primary(devices: &[Device]) -> Option<&Device> {
    devices
        .iter()
        .filter(|d| d.backend != Backend::Other)
        .max_by_key(|d| d.total_mib)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real output from an RX 7600 XT with a model already resident.
    const VULKAN: &str = "\
Available devices:
  Vulkan0: AMD Radeon RX 7600 XT (RADV NAVI33) (16384 MiB, 4642 MiB free)
";

    /// Real output shape from Apple Silicon. The total is the iogpu wired limit, not
    /// physical RAM, which is exactly why this is read from llama.cpp.
    const METAL: &str = "\
Available devices:
  MTL0: Apple M1 Max (26000 MiB, 25999 MiB free)
";

    #[test]
    fn parses_a_vulkan_device() {
        let d = parse(VULKAN);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].id, "Vulkan0");
        assert_eq!(d[0].backend, Backend::Vulkan);
        // The device's own brackets must not be mistaken for the memory group.
        assert_eq!(d[0].name, "AMD Radeon RX 7600 XT (RADV NAVI33)");
        assert_eq!(d[0].total_mib, 16_384);
        assert_eq!(d[0].free_mib, 4642);
        assert!(!d[0].backend.is_unified_memory());
    }

    #[test]
    fn parses_a_metal_device() {
        let d = parse(METAL);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].backend, Backend::Metal);
        assert_eq!(d[0].name, "Apple M1 Max");
        assert_eq!(d[0].total_mib, 26_000);
        assert!(
            d[0].backend.is_unified_memory(),
            "Metal shares memory with the CPU, which changes the failure mode"
        );
    }

    #[test]
    fn ignores_prose_and_headers() {
        let noisy = "\
ggml_vulkan: Found 1 Vulkan devices:
load_backend: loaded Vulkan backend from /usr/lib/ggml/libggml-vulkan.so
Available devices:
  Vulkan0: AMD Radeon RX 7600 XT (RADV NAVI33) (16384 MiB, 4642 MiB free)
warning: asserts enabled, performance may be affected
";
        let d = parse(noisy);
        assert_eq!(d.len(), 1, "only the device line is a device");
        assert_eq!(d[0].id, "Vulkan0");
    }

    #[test]
    fn a_machine_with_no_gpu_yields_nothing() {
        assert!(parse("Available devices:\n").is_empty());
        assert!(parse("").is_empty());
    }

    #[test]
    fn the_largest_device_wins() {
        let mixed = "\
  Vulkan0: Intel integrated (2048 MiB, 2000 MiB free)
  Vulkan1: AMD Radeon RX 7600 XT (16384 MiB, 16000 MiB free)
";
        let devices = parse(mixed);
        assert_eq!(primary(&devices).unwrap().id, "Vulkan1");
    }

    #[test]
    fn backends_are_recognised_from_the_device_id() {
        assert_eq!(Backend::from_id("Vulkan0"), Backend::Vulkan);
        assert_eq!(Backend::from_id("MTL0"), Backend::Metal);
        assert_eq!(Backend::from_id("CUDA1"), Backend::Cuda);
        assert_eq!(Backend::from_id("ROCm0"), Backend::Rocm);
        assert_eq!(Backend::from_id("SYCL0"), Backend::Other);
    }
}
