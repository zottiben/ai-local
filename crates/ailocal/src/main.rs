//! `ailocal` - manage local LLMs and expose them to coding harnesses.

use std::sync::Arc;

use ailocal::{
    auth, config::Config, download, gateway, gguf, registry, registry::Fit, serve, source::Source,
    vram,
};
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
    /// Run a model, evicting whatever is already running.
    Serve(ServeArgs),
    /// Show the running model, if any.
    Ps,
    /// Stop the running model.
    Stop,
    /// The authenticated OpenAI-compatible front end.
    #[command(subcommand)]
    Gateway(GatewayCmd),
}

#[derive(Subcommand)]
enum GatewayCmd {
    /// Run the gateway in the foreground.
    Run {
        #[arg(long, default_value_t = 8081)]
        port: u16,
        /// Bind address. Leave as loopback unless something else fronts it - the
        /// the tunnel host tunnel reaches this host over the LAN, so 0.0.0.0 is needed there.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },
    /// Print the API key, creating one if there is none.
    Key,
}

#[derive(Args)]
struct ServeArgs {
    /// Model name as shown by `ailocal model ls`.
    name: String,

    /// Context size. Defaults to the largest that fits; a larger request is refused
    /// rather than silently clamped.
    #[arg(long)]
    ctx: Option<u64>,

    #[arg(long, default_value_t = serve::DEFAULT_PORT)]
    port: u16,

    #[arg(long, default_value = serve::DEFAULT_HOST)]
    host: String,

    /// Whether the model may think before answering: auto, on or off.
    ///
    /// Defaults to the configured value. `off` forces a direct answer, which matters
    /// because these models can spend an entire token budget reasoning and return
    /// empty content.
    #[arg(long)]
    reasoning: Option<String>,
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
    quit_quietly_on_broken_pipe();

    match Cli::parse().command {
        Command::Budget => budget(),
        Command::Model(ModelCmd::Ls) => model_ls(),
        Command::Model(ModelCmd::Install(args)) => model_install(&args),
        Command::Model(ModelCmd::Rm { name }) => model_rm(&name),
        Command::Config(ConfigCmd::Show) => config_show(),
        Command::Config(ConfigCmd::Init) => config_init(),
        Command::Serve(args) => serve_model(&args),
        Command::Ps => ps(),
        Command::Stop => stop(),
        Command::Gateway(GatewayCmd::Run { port, host }) => gateway_run(&host, port),
        Command::Gateway(GatewayCmd::Key) => {
            println!("{}", auth::load_or_create()?);
            eprintln!("stored in {}", auth::key_path()?.display());
            Ok(())
        }
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

    // Budget against what would be free after evicting whatever is loaded. Using live
    // VRAM instead reports the running model as unable to fit, which is absurd on its
    // face and was exactly what this printed before.
    let budget = serve::budget_for_next_launch()
        .unwrap_or_else(|_| vram::Budget::new(ailocal::vram_used_mib().unwrap_or(900)));

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

fn serve_model(args: &ServeArgs) -> anyhow::Result<()> {
    let cfg = Config::load()?;
    let models = registry::scan(&cfg.models_dir)?;
    let model = models.iter().find(|m| m.name == args.name).ok_or_else(|| {
        anyhow::anyhow!(
            "no model named {:?}; `ailocal model ls` shows what is installed",
            args.name
        )
    })?;

    // Budget against what will be free after any eviction, not against current usage.
    let budget = serve::budget_for_next_launch()?;
    let opts = serve::Options {
        context: args.ctx,
        host: args.host.clone(),
        port: args.port,
        reasoning: args
            .reasoning
            .clone()
            .unwrap_or_else(|| cfg.reasoning.clone()),
        ..serve::Options::from_config(&cfg)?
    };

    // Resolve the context before announcing anything, so a refusal is not preceded by
    // a line claiming we started.
    let context = serve::plan_context(model, &budget, &opts)?;

    if let Some(old) = serve::running()? {
        println!("evicting {} (only one model fits at a time)", old.model);
    }
    println!(
        "starting {} with {} context ...",
        model.name,
        format_count(context)
    );

    let instance = serve::start(model, &budget, &opts)?;
    println!(
        "{} up at {} with {} context, {} KV, reasoning {}",
        instance.model,
        instance.base_url(),
        format_count(instance.context),
        instance.cache_type,
        opts.reasoning
    );
    println!("vram now {} MiB", ailocal::vram_used_mib()?);
    Ok(())
}

fn gateway_run(host: &str, port: u16) -> anyhow::Result<()> {
    let state = Arc::new(gateway::AppState {
        key: auth::load_or_create()?,
        config: Config::load()?,
        // No timeout: a cold model load can take tens of seconds and a streamed
        // completion runs for minutes. The upstream is a local process, so a hung
        // request is a bug to see rather than something to paper over with a deadline.
        client: reqwest::Client::builder().build()?,
    });

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind((host, port)).await?;
        println!("gateway listening on http://{host}:{port}");
        println!("  key in {}", auth::key_path()?.display());
        match serve::running()? {
            Some(i) => println!("  currently loaded: {} ({} ctx)", i.model, i.context),
            None => println!("  no model loaded; one will start on first request"),
        }

        axum::serve(listener, gateway::router(state))
            .with_graceful_shutdown(async {
                tokio::signal::ctrl_c().await.ok();
                println!("\nshutting down");
            })
            .await?;
        Ok::<_, anyhow::Error>(())
    })
}

fn ps() -> anyhow::Result<()> {
    match serve::running()? {
        None => println!("nothing running"),
        Some(i) => {
            println!("{:<12} {}", "model", i.model);
            println!("{:<12} {}", "url", i.base_url());
            println!("{:<12} {}", "context", format_count(i.context));
            println!("{:<12} {}", "kv cache", i.cache_type);
            println!("{:<12} {}", "pid", i.pid);
            println!("{:<12} {} MiB", "vram", ailocal::vram_used_mib()?);
        }
    }
    Ok(())
}

fn stop() -> anyhow::Result<()> {
    match serve::stop()? {
        None => println!("nothing running"),
        Some(i) => println!("stopped {} (pid {})", i.model, i.pid),
    }
    Ok(())
}

/// Exit cleanly when output is piped into something that stops reading.
///
/// Rust ignores SIGPIPE, so `ailocal model ls | head` makes `println!` panic on EPIPE
/// and prints a backtrace where a Unix tool should simply stop. Restoring the default
/// disposition needs `unsafe`, which the workspace forbids, so this catches the panic
/// instead and exits as if the write had succeeded.
fn quit_quietly_on_broken_pipe() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let broken_pipe = info
            .payload()
            .downcast_ref::<String>()
            .is_some_and(|m| m.contains("Broken pipe"));
        if broken_pipe {
            std::process::exit(0);
        }
        previous(info);
    }));
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
