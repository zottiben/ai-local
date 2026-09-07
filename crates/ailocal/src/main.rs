//! `ailocal` - manage local LLMs and expose them to coding harnesses.

use ailocal::vram;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "ailocal", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Report the VRAM budget this machine can spend on a model.
    Budget,
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Budget => budget(),
    }
}

fn budget() -> anyhow::Result<()> {
    let desktop = read_vram_used_mib()?;
    let budget = vram::Budget::new(desktop);

    println!("ceiling          {:>6} MiB", vram::CEILING_MIB);
    println!("desktop          {desktop:>6} MiB");
    println!("compute buffers  {:>6} MiB", vram::COMPUTE_BUFFER_MIB);
    println!("available        {:>6} MiB", budget.available_mib());

    Ok(())
}

/// VRAM currently in use, in MiB, read from the amdgpu sysfs node.
fn read_vram_used_mib() -> anyhow::Result<u64> {
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
