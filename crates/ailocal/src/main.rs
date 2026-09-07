//! `ailocal` - manage local LLMs and expose them to coding harnesses.

use std::sync::Arc;

use ailocal::{
    auth, config::Config, download, gateway, gguf, harness, registry, registry::Fit, serve,
    service, source::Source, vram,
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
    /// Point a coding harness at this machine's gateway.
    #[command(subcommand)]
    Harness(HarnessCmd),
    /// systemd units, so this survives a reboot.
    #[command(subcommand)]
    Service(ServiceCmd),
    /// Check prerequisites and bring everything up in one go.
    Setup(SetupArgs),
    /// Install the latest release in place.
    ///
    /// Arguments are forwarded to the install script, e.g. `ailocal update --check`.
    /// Help is forwarded too, so `ailocal update --help` documents the script's own
    /// options rather than clap's view of them.
    #[command(disable_help_flag = true)]
    Update {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

#[derive(Args)]
struct SetupArgs {
    /// Model to install if none are present, e.g. `ollama:gemma4:12b`.
    #[arg(long)]
    model: Option<String>,

    /// Do not touch harness configuration.
    #[arg(long)]
    skip_harness: bool,

    /// Do not install systemd units.
    #[arg(long)]
    skip_service: bool,
}

#[derive(Subcommand)]
enum ServiceCmd {
    /// Write and enable the user units.
    Install {
        /// Write the units but do not enable or start them.
        #[arg(long)]
        no_start: bool,
    },
    /// Stop, disable and remove the units.
    Uninstall,
    /// Show unit state.
    Status,
}

#[derive(Subcommand)]
enum HarnessCmd {
    /// Write the gateway into a harness's configuration.
    Configure {
        /// Which harness. Currently `pi`.
        name: String,
        /// Gateway URL the harness should call.
        #[arg(long, default_value = "http://127.0.0.1:8081")]
        url: String,
    },
    /// Remove our entry from a harness's configuration.
    Unconfigure { name: String },
}

#[derive(Subcommand)]
enum GatewayCmd {
    /// Run the gateway in the foreground.
    Run {
        #[arg(long, default_value_t = 8081)]
        port: u16,
        /// Bind address. Leave as loopback unless something else fronts it - the
        /// tunnel reaches this host over the LAN, so 0.0.0.0 is needed to expose it.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },
    /// Print the API key, creating one if there is none.
    Key,
    /// Check a gateway is reachable and correctly authenticated.
    ///
    /// Point it at the public hostname to verify the tunnel end to end.
    Check {
        #[arg(default_value = "http://127.0.0.1:8081")]
        url: String,
    },
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

    /// Stay in the foreground until llama-server exits.
    ///
    /// What a service manager wants: one process to supervise and collect logs from,
    /// rather than a command that forks and returns.
    #[arg(long)]
    foreground: bool,
}

#[derive(Subcommand)]
enum ModelCmd {
    /// List models, with the largest context each can hold here.
    Ls,
    /// Pick a model interactively, ranked by what this machine can run.
    Pick,
    /// Search Hugging Face for models you could install.
    Search {
        /// Free text, e.g. "qwen3 coder" or "gemma4".
        query: Vec<String>,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// List the quantisations in a Hugging Face repo, and which of them fit here.
    Files {
        /// `<owner>/<repo>`, as printed by `ailocal model search`.
        repo: String,
    },
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
        Command::Model(ModelCmd::Pick) => model_pick().map(|_| ()),
        Command::Model(ModelCmd::Search { query, limit }) => model_search(&query.join(" "), limit),
        Command::Model(ModelCmd::Files { repo }) => model_files(&repo),
        Command::Model(ModelCmd::Install(args)) => model_install(&args),
        Command::Model(ModelCmd::Rm { name }) => model_rm(&name),
        Command::Config(ConfigCmd::Show) => config_show(),
        Command::Config(ConfigCmd::Init) => config_init(),
        Command::Serve(args) => serve_model(&args),
        Command::Ps => ps(),
        Command::Stop => stop(),
        Command::Gateway(GatewayCmd::Run { port, host }) => gateway_run(&host, port),
        Command::Harness(HarnessCmd::Configure { name, url }) => harness_configure(&name, &url),
        Command::Harness(HarnessCmd::Unconfigure { name }) => harness_unconfigure(&name),
        Command::Service(ServiceCmd::Install { no_start }) => service_install(no_start),
        Command::Service(ServiceCmd::Uninstall) => service_uninstall(),
        Command::Service(ServiceCmd::Status) => service_status(),
        Command::Setup(args) => setup(&args),
        Command::Update { args } => std::process::exit(ailocal::update::run(&args)?),
        Command::Gateway(GatewayCmd::Check { url }) => gateway_check(&url),
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

fn http_client() -> anyhow::Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .user_agent(concat!("ailocal/", env!("CARGO_PKG_VERSION")))
        .timeout(std::time::Duration::from_secs(30))
        .build()?)
}

/// Offer the curated shortlist, ranked for this machine, and install what is chosen.
///
/// Returns the installed model's name, so `setup` can carry straight on with it.
fn model_pick() -> anyhow::Result<Option<String>> {
    let budget = serve::budget_for_next_launch()
        .unwrap_or_else(|_| vram::Budget::new(ailocal::vram_used_mib().unwrap_or(900)));
    let available = budget.available_mib();

    let client = http_client()?;
    println!("Checking what fits in {available} MiB ...");
    let mut candidates = ailocal::catalogue::resolve(&client, available);
    anyhow::ensure!(!candidates.is_empty(), "could not reach the model registry");

    // Reads a few MiB of each fitting model for its real attention geometry, because
    // weight size does not predict usable context: a 12B with sliding-window attention
    // holds six times what a 14B with full attention does.
    println!("Measuring usable context ...");
    let cache = Config::load()?.cache_type.parse()?;
    ailocal::catalogue::measure(&client, &mut candidates, &budget, cache);

    let options: Vec<String> = candidates
        .iter()
        .map(ailocal::catalogue::Candidate::label)
        .collect();
    let header = format!(
        "Models for this machine ({available} MiB available). \
         Anything else: ailocal model search <query>"
    );

    let Some(index) = ailocal::prompt::choose(&header, &options)? else {
        println!("Nothing selected.");
        return Ok(None);
    };
    let chosen = &candidates[index];

    if !chosen.fits() {
        println!(
            "\n{} needs {} MiB but only {available} MiB is free.",
            chosen.entry.reference, chosen.size_mib
        );
        if !ailocal::prompt::confirm("Download it anyway?", false)? {
            return Ok(None);
        }
    }
    if !chosen.entry.note.is_empty() {
        println!("\nnote: {}", chosen.entry.note);
    }

    let reference = format!("ollama:{}", chosen.entry.reference);
    model_install(&InstallArgs {
        reference: reference.clone(),
        force: !chosen.fits(),
    })?;

    Ok(reference.parse::<Source>().ok().map(|s| {
        // The registry keys models by filename stem, which is what `serve` wants.
        s.file_name().trim_end_matches(".gguf").to_owned()
    }))
}

fn model_search(query: &str, limit: usize) -> anyhow::Result<()> {
    anyhow::ensure!(!query.trim().is_empty(), "give me something to search for");

    let repos = ailocal::source::search(&http_client()?, query, limit)?;
    if repos.is_empty() {
        println!("nothing matched {query:?}");
        return Ok(());
    }

    println!("{:<58} {:>12}", "REPO", "DOWNLOADS");
    for r in &repos {
        println!("{:<58} {:>12}", truncate(&r.id, 58), r.downloads);
    }
    println!("\nNext: ailocal model files {}", repos[0].id);
    Ok(())
}

fn model_files(repo: &str) -> anyhow::Result<()> {
    let cfg = Config::load()?;
    let budget = serve::budget_for_next_launch()
        .unwrap_or_else(|_| vram::Budget::new(ailocal::vram_used_mib().unwrap_or(900)));

    let client = http_client()?;
    let token = ailocal::source::hf_token(&cfg.hf_home);
    let files = ailocal::source::files(&client, repo, token.as_deref())?;

    anyhow::ensure!(!files.is_empty(), "{repo} has no .gguf files");

    // Weight size alone decides most of it, and the KV cache needs room on top. Show
    // the headroom rather than a bare yes/no so an almost-fitting quant is visible.
    let available = budget.available_mib();
    println!("{available} MiB available for weights + KV cache\n");
    println!("{:<52} {:>8}  VERDICT", "FILE", "SIZE");
    for f in &files {
        let verdict = if f.size_mib >= available {
            "too large".to_owned()
        } else {
            format!("{} MiB left for context", available - f.size_mib)
        };
        println!(
            "{:<52} {:>5} G  {verdict}",
            truncate(&f.path, 52),
            f.size_mib / 1024
        );
    }

    if let Some(best) = files.iter().rev().find(|f| f.size_mib + 2048 < available) {
        println!("\nNext: ailocal model install hf:{repo}/{}", best.path);
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

    if args.foreground {
        return serve::run_foreground(model, &budget, &opts);
    }

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

fn harness_configure(name: &str, url: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        matches!(name, "pi" | "claude-code"),
        "unknown harness {name:?}; expected `pi` or `claude-code`"
    );

    let cfg = Config::load()?;
    let cache: vram::CacheType = cfg.cache_type.parse()?;
    let budget = serve::budget_for_next_launch()
        .unwrap_or_else(|_| vram::Budget::new(ailocal::vram_used_mib().unwrap_or(900)));

    // Advertise only what can actually run, at the context it would actually get.
    // Offering a model Pi cannot load just moves the failure later.
    let models: Vec<harness::PiModel> = registry::scan(&cfg.models_dir)?
        .iter()
        .filter_map(|m| {
            match registry::assess(m.kv, m.trained_context, &budget, cache, m.size_mib) {
                Fit::Fits(ctx) => Some(harness::PiModel {
                    id: m.name.clone(),
                    context_window: ctx,
                    reasoning: cfg.reasoning != "off",
                }),
                _ => None,
            }
        })
        .collect();

    anyhow::ensure!(
        !models.is_empty(),
        "no installed model can run here; `ailocal model ls` shows why"
    );

    let key = auth::load_or_create()?;

    if name == "claude-code" {
        let first = &models[0];
        let (path, outcome) =
            harness::configure_claude_code(url, &key, &first.id, first.context_window)?;
        match outcome {
            harness::Outcome::AlreadyConfigured => println!("claude-code env already current"),
            harness::Outcome::Configured { .. } => println!("wrote {}", path.display()),
        }
        println!("  source it in a shell, then run claude there:");
        println!("    source {} && claude", path.display());
        println!(
            "  deliberately not written into ~/.claude/settings.json - that would\n\
             \x20 redirect every Claude Code session on this machine, not just the\n\
             \x20 ones you want on the local model."
        );
        return Ok(());
    }

    match harness::configure_pi(url, &key, &models)? {
        harness::Outcome::AlreadyConfigured => {
            println!("pi already points at {url}");
        }
        harness::Outcome::Configured { backup } => {
            println!("pi configured to use {url}");
            if let Some(b) = backup {
                println!("  backup: {}", b.display());
            }
        }
    }
    for m in &models {
        println!("  {} ({} context)", m.id, format_count(m.context_window));
    }
    println!(
        "  try: pi --provider {} --model {}",
        harness::PI_PROVIDER_ID,
        models[0].id
    );
    Ok(())
}

fn harness_unconfigure(name: &str) -> anyhow::Result<()> {
    if name == "claude-code" {
        let path = harness::claude_code_env_path()?;
        if path.exists() {
            std::fs::remove_file(&path)?;
            println!("removed {}", path.display());
        } else {
            println!("claude-code was not configured");
        }
        return Ok(());
    }
    anyhow::ensure!(
        name == "pi",
        "unknown harness {name:?}; expected `pi` or `claude-code`"
    );
    if harness::unconfigure_pi()? {
        println!("removed the {} entry from pi", harness::PI_PROVIDER_ID);
    } else {
        println!("pi was not configured");
    }
    Ok(())
}

/// Bring a fresh machine up: check what is needed, then do the rest.
fn setup(args: &SetupArgs) -> anyhow::Result<()> {
    let mut blocked = false;

    println!("1. prerequisites");
    match ailocal::update::which("llama-server") {
        Some(p) => println!("   ok    llama-server at {}", p.display()),
        None => {
            blocked = true;
            println!("   MISS  llama-server not on PATH");
            if cfg!(target_os = "macos") {
                println!("         macOS:  brew install llama.cpp");
            } else {
                println!("         Arch:   sudo pacman -Syu llama-cpp ggml-vulkan");
            }
            println!("         other:  https://github.com/ggml-org/llama.cpp");
        }
    }
    // Ask llama.cpp what it can see rather than looking for a backend library. On
    // Linux ggml ships backends as separate packages, and without the GPU one
    // llama-server silently runs on CPU - which looks like a broken GPU rather than a
    // missing 54 MB. On macOS Metal is built in and there is no library to look for.
    match ailocal::primary_device() {
        Ok(gpu) => {
            println!(
                "   ok    {} via {:?}, {} MiB",
                gpu.name, gpu.backend, gpu.total_mib
            );
            // Budget against what a launch would actually get, not against current
            // usage - otherwise a loaded model makes the machine look out of memory.
            let budget =
                serve::budget_for_next_launch().unwrap_or_else(|_| vram::Budget::for_device(&gpu));
            println!(
                "   ok    {} MiB available for a model",
                budget.available_mib()
            );
        }
        Err(e) => {
            blocked = true;
            println!("   MISS  {e}");
            if cfg!(target_os = "linux") {
                println!("         Arch: sudo pacman -Syu ggml-vulkan");
            }
        }
    }

    println!("\n2. config");
    let path = Config::path()?;
    if path.exists() {
        println!("   ok    {}", path.display());
    } else {
        println!("   wrote {}", Config::load()?.save()?.display());
    }
    let cfg = Config::load()?;
    if !cfg.models_dir.exists() {
        std::fs::create_dir_all(&cfg.models_dir)?;
        println!("   made  {}", cfg.models_dir.display());
    }

    anyhow::ensure!(
        !blocked,
        "prerequisites missing - install them and re-run `ailocal setup`"
    );

    println!("\n3. models");
    let mut models = registry::scan(&cfg.models_dir)?;
    if models.is_empty() {
        match &args.model {
            Some(reference) => {
                println!("   installing {reference} ...");
                model_install(&InstallArgs {
                    reference: reference.clone(),
                    force: false,
                })?;
                models = registry::scan(&cfg.models_dir)?;
            }
            None => {
                println!("   none installed - pick one:\n");
                if model_pick()?.is_none() {
                    println!("\nNothing installed. Re-run `ailocal setup` when ready.");
                    return Ok(());
                }
                models = registry::scan(&cfg.models_dir)?;
            }
        }
    }
    for m in &models {
        println!("   ok    {} ({} GiB)", m.name, m.size_mib / 1024);
    }

    if !args.skip_service {
        println!("\n4. services");
        service_install(false)?;
    }

    if !args.skip_harness {
        println!("\n5. harnesses");
        let url = format!("http://{}:{}", "127.0.0.1", cfg.gateway_port);
        for harness in ["pi", "claude-code"] {
            match harness_configure(harness, &url) {
                Ok(()) => {}
                // A harness that is not installed is not a failure of setup.
                Err(e) => println!("   skip  {harness}: {e}"),
            }
        }
    }

    println!("\nReady. `ailocal ps` shows what is loaded, `ailocal gateway check` verifies auth.");
    Ok(())
}

/// The binary a systemd unit should point at.
///
/// `current_exe` is wrong when running from `cargo run` or `target/release`: a unit
/// written with that path breaks on the next `cargo clean`, and quietly keeps running
/// a stale build until then. Prefer an installed copy on PATH, and refuse rather than
/// write a build path if there is none.
fn unit_binary() -> anyhow::Result<std::path::PathBuf> {
    let current = std::env::current_exe()?.canonicalize()?;
    if !ailocal::update::is_build_dir(&current) {
        return Ok(current);
    }

    let installed = ailocal::update::which("ailocal")
        .and_then(|p| p.canonicalize().ok())
        .filter(|p| !ailocal::update::is_build_dir(p));

    match installed {
        Some(path) => {
            eprintln!(
                "note: running from a build directory; units will point at {}",
                path.display()
            );
            Ok(path)
        }
        None => anyhow::bail!(
            "running from {} - a systemd unit must not point into a build directory, \
             because `cargo clean` would break the service.\n\
             Install it first, e.g. `install -Dm755 {} ~/.local/bin/ailocal`, then \
             re-run this from there.",
            current.display(),
            current.display()
        ),
    }
}

fn gateway_check(url: &str) -> anyhow::Result<()> {
    let base = url.trim_end_matches('/');
    let key = auth::load_or_create()?;
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let mut failures = 0;
    let mut check =
        |label: &str, expected: u16, send: &dyn Fn() -> anyhow::Result<u16>| match send() {
            Ok(status) if status == expected => println!("  ok    {label} ({status})"),
            Ok(status) => {
                failures += 1;
                println!("  FAIL  {label}: got {status}, expected {expected}");
            }
            Err(e) => {
                failures += 1;
                println!("  FAIL  {label}: {e}");
            }
        };

    println!("checking {base}");
    check("health, unauthenticated", 200, &|| {
        Ok(client
            .get(format!("{base}/health"))
            .send()?
            .status()
            .as_u16())
    });
    // If this returns 200 the endpoint is open to the internet.
    check("models rejects no credential", 401, &|| {
        Ok(client
            .get(format!("{base}/v1/models"))
            .send()?
            .status()
            .as_u16())
    });
    check("models rejects a wrong credential", 401, &|| {
        Ok(client
            .get(format!("{base}/v1/models"))
            .bearer_auth("ail_definitely_wrong")
            .send()?
            .status()
            .as_u16())
    });
    check("models accepts the real key", 200, &|| {
        Ok(client
            .get(format!("{base}/v1/models"))
            .bearer_auth(&key)
            .send()?
            .status()
            .as_u16())
    });

    anyhow::ensure!(failures == 0, "{failures} check(s) failed");
    println!("all checks passed");
    Ok(())
}

fn service_install(no_start: bool) -> anyhow::Result<()> {
    let cfg = Config::load()?;
    let exe = unit_binary()?;

    let units = service::install(
        &exe,
        &cfg.gateway_host,
        cfg.gateway_port,
        cfg.default_model.as_deref(),
    )?;
    println!(
        "wrote {} into {}",
        units.join(", "),
        service::unit_dir()?.display()
    );
    println!("  ExecStart uses {}", exe.display());

    service::reload()?;
    if !no_start {
        for unit in &units {
            service::enable(unit)?;
            println!("  enabled and started {unit}");
        }
    }

    if !service::linger_enabled() {
        println!(
            "\nnote: user services only run once you have logged in. For the gateway to\n\
             come up at boot and survive logout, enable lingering (needs root):\n\
             \x20   sudo loginctl enable-linger {}",
            std::env::var("USER").unwrap_or_default()
        );
    }
    Ok(())
}

fn service_uninstall() -> anyhow::Result<()> {
    for unit in service::service_names() {
        service::disable(unit).ok();
    }
    let removed = service::uninstall()?;
    service::reload()?;

    if removed.is_empty() {
        println!("no units were installed");
    } else {
        println!("removed {}", removed.join(", "));
    }
    Ok(())
}

fn service_status() -> anyhow::Result<()> {
    println!(
        "linger      {}",
        if service::linger_enabled() {
            "enabled (starts at boot, survives logout)"
        } else {
            "disabled (units start at login only)"
        }
    );
    for unit in service::service_names() {
        println!(
            "{unit:<40} {:<10} {}",
            if service::is_active(unit) {
                "active"
            } else {
                "inactive"
            },
            if service::is_enabled(unit) {
                "enabled"
            } else {
                "disabled"
            }
        );
    }
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
