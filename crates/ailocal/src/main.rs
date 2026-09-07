//! `ailocal` - manage local LLMs and expose them to coding harnesses.

use ailocal::{config::Config, download, gguf, registry, registry::Fit, source::Source, vram};
use clap::{Args, Parser, Subcommand};

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
    /// Inspect models on disk.
    #[command(subcommand)]
    Model(ModelCmd),
    /// Inspect or write the config file.
    #[command(subcommand)]
    Config(ConfigCmd),
}

#[derive(Subcommand)]
enum ModelCmd {
    /// List models, with the largest context each can hold here.
    Ls,
    /// Download a model, checking it can actually run here first.
    Install(InstallArgs),
    /// Delete a model from disk.
    Rm { name: String },
}

#[derive(Args)]
struct InstallArgs {
    /// `ollama:<name>:<tag>` or `hf:<owner>/<repo>/<file.gguf>`.
    ///
    /// Prefer ollama where it carries the quant you want - it is far faster and its
    /// blobs verify against a published digest.
    reference: String,

    /// Download even if the model cannot hold a usable context on this machine.
    #[arg(long)]
    force: bool,
}

#[derive(Subcommand)]
enum ConfigCmd {
    /// Print the active configuration and where it came from.
    Show,
    /// Write the current configuration to disk.
    Init,
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Budget => budget(),
        Command::Model(ModelCmd::Ls) => model_ls(),
        Command::Model(ModelCmd::Install(args)) => model_install(&args),
        Command::Model(ModelCmd::Rm { name }) => model_rm(&name),
        Command::Config(ConfigCmd::Show) => config_show(),
        Command::Config(ConfigCmd::Init) => config_init(),
    }
}

fn budget() -> anyhow::Result<()> {
    let desktop = ailocal::vram_used_mib()?;
    let budget = vram::Budget::new(desktop);

    println!("ceiling          {:>6} MiB", vram::CEILING_MIB);
    println!("desktop          {desktop:>6} MiB");
    println!("compute buffers  {:>6} MiB", vram::COMPUTE_BUFFER_MIB);
    println!("available        {:>6} MiB", budget.available_mib());
    Ok(())
}

fn config_show() -> anyhow::Result<()> {
    let path = Config::path()?;
    let cfg = Config::load()?;
    let origin = if path.exists() { "file" } else { "defaults" };

    println!("# {} ({origin})", path.display());
    print!("{}", toml::to_string_pretty(&cfg)?);
    Ok(())
}

fn config_init() -> anyhow::Result<()> {
    let path = Config::load()?.save()?;
    println!("wrote {}", path.display());
    Ok(())
}

fn model_ls() -> anyhow::Result<()> {
    let cfg = Config::load()?;
    let cache: vram::CacheType = cfg.cache_type.parse()?;

    // Fall back to a nominal desktop footprint so the listing still works headless,
    // e.g. over SSH or in CI, where there is no amdgpu node to read.
    let desktop = ailocal::vram_used_mib().unwrap_or(900);
    let budget = vram::Budget::new(desktop);

    let models = registry::scan(&cfg.models_dir)?;
    if models.is_empty() {
        println!("no models in {}", cfg.models_dir.display());
        return Ok(());
    }

    println!(
        "{:<28} {:>7}  {:<10} {:>9} {:>10} {:>11}",
        "NAME", "SIZE", "ARCH", "KV/TOK", "TRAINED", "MAX CTX"
    );
    for m in &models {
        let kv_per_token = m.kv.map_or_else(
            || "-".to_owned(),
            |kv| format!("{:.0} KiB", kv.bytes_per_token(cache) / 1024.0),
        );
        let trained = m
            .trained_context
            .map_or_else(|| "-".to_owned(), format_count);
        let max_ctx = match m.max_context(&budget, cache) {
            Some(c) if c > 0 => format_count(c),
            _ => "WILL NOT FIT".to_owned(),
        };

        println!(
            "{:<28} {:>5} G  {:<10} {kv_per_token:>9} {trained:>10} {max_ctx:>11}{}",
            truncate(&m.name, 28),
            m.size_mib / 1024,
            m.arch.as_deref().unwrap_or("?"),
            if m.sliding_window { "  (swa)" } else { "" },
        );
    }

    if models.iter().any(|m| m.sliding_window) {
        println!(
            "\n(swa) sliding-window attention: most layers hold a fixed window rather than\n\
             growing with context, so KV/TOK is the marginal cost of the few global\n\
             layers. This is why a 12B reaches 256k where a 27B stops near 16k."
        );
    }
    Ok(())
}

/// How much of a remote GGUF to read before deciding whether to download it.
const PREFLIGHT_BYTES: u64 = 8 << 20;

/// Read a model's metadata from the network and judge whether it can run here.
fn preflight(
    client: &reqwest::blocking::Client,
    artifact: &ailocal::source::Artifact,
    budget: &vram::Budget,
    cache: vram::CacheType,
    weights_mib: u64,
) -> Fit {
    let Some(md) = download::head_bytes(client, artifact, PREFLIGHT_BYTES)
        .ok()
        .and_then(|head| gguf::parse_head(&head).ok())
    else {
        return Fit::Unknown;
    };

    println!(
        "  arch {}, trained to {}",
        md.architecture().unwrap_or("?"),
        md.context_length().map_or_else(|| "?".into(), format_count)
    );

    registry::assess(
        md.kv_layout(),
        md.context_length(),
        budget,
        cache,
        weights_mib,
    )
}

fn model_install(args: &InstallArgs) -> anyhow::Result<()> {
    let cfg = Config::load()?;
    let cache: vram::CacheType = cfg.cache_type.parse()?;
    let source: Source = args.reference.parse()?;

    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("ailocal/", env!("CARGO_PKG_VERSION")))
        .build()?;
    let token = ailocal::source::hf_token(&cfg.hf_home);

    println!("resolving {source}");
    let artifact = source.resolve(&client, token.as_deref())?;
    let weights_mib = artifact.size.map_or(0, |b| b / (1024 * 1024));
    println!(
        "  {} MiB{}",
        weights_mib,
        if artifact.sha256.is_some() {
            ", digest published"
        } else {
            ", no digest (length-checked only)"
        }
    );

    // Read just the head to learn the architecture. A model that cannot hold a usable
    // context here should cost seconds to reject, not an hour of downloading.
    let budget = vram::Budget::new(ailocal::vram_used_mib().unwrap_or(900));
    let fit = preflight(&client, &artifact, &budget, cache, weights_mib);

    match fit {
        Fit::Unknown => println!("  could not read metadata; skipping the fit check"),
        Fit::Fits(ctx) => println!("  fits here: up to {} context", format_count(ctx)),
        Fit::ContextTooSmall(ctx) => {
            println!("  will not fit: only {ctx} tokens of context on this machine");
            anyhow::ensure!(args.force, "refusing to download; pass --force to override");
        }
        Fit::WeightsTooLarge => {
            println!(
                "  will not fit: {weights_mib} MiB of weights against {} MiB available",
                budget.available_mib()
            );
            anyhow::ensure!(args.force, "refusing to download; pass --force to override");
        }
    }

    let (path, outcome) = download::fetch(&client, &artifact, &cfg.models_dir)?;
    match outcome {
        download::Fetched::AlreadyPresent => println!("already present: {}", path.display()),
        download::Fetched::Downloaded {
            bytes,
            resumed_from,
        } => {
            if resumed_from > 0 {
                println!("  resumed from {} MiB", resumed_from / (1024 * 1024));
            }
            println!(
                "installed {} ({} MiB)",
                path.display(),
                bytes / (1024 * 1024)
            );
        }
    }
    Ok(())
}

fn model_rm(name: &str) -> anyhow::Result<()> {
    let cfg = Config::load()?;
    let models = registry::scan(&cfg.models_dir)?;
    let model = models.iter().find(|m| m.name == name).ok_or_else(|| {
        anyhow::anyhow!("no model named {name:?} in {}", cfg.models_dir.display())
    })?;

    std::fs::remove_file(&model.path)?;
    println!("removed {} ({} MiB)", model.path.display(), model.size_mib);
    Ok(())
}

/// Render a token count as `32k` / `256k`, falling back to the exact number.
fn format_count(n: u64) -> String {
    if n >= 1024 && n.is_multiple_of(1024) {
        format!("{}k", n / 1024)
    } else {
        n.to_string()
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_owned();
    }
    let keep: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{keep}…")
}
