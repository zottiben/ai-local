//! `ailocal` - manage local LLMs and expose them to coding harnesses.

use std::sync::Arc;

use ailocal::{
    auth, config::Config, download, extras, gateway, gguf, harness, registry, registry::Fit, serve,
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
    /// Optional companion tools, installed separately.
    #[command(subcommand)]
    Extras(ExtrasCmd),
    /// Score models on a coding task set. Needs the `eval` extra.
    ///
    /// Everything after `eval` goes to `ailocal-eval`, including `--help`, so its own
    /// options are what you see rather than clap's guess at them.
    #[command(disable_help_flag = true)]
    Eval {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
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
    /// Where weights, caches and datasets go. Asked for if not given.
    #[arg(long)]
    data_dir: Option<std::path::PathBuf>,

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
enum ExtrasCmd {
    /// Show which extras exist and which are installed.
    List,
    /// Download an extra at this binary's version.
    Install { name: String },
    /// Delete an installed extra.
    Remove { name: String },
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
        /// Which harness: `pi` or `claude-code`.
        name: String,
        /// Gateway URL the harness should call.
        #[arg(long, default_value = "http://127.0.0.1:8081")]
        url: String,
        /// Which model to point it at.
        ///
        /// Only Claude Code is pinned to one - Pi is given the whole list and picks
        /// per session, so this just decides which one it offers first. Defaults to
        /// `default_model`, then to whatever is loaded.
        #[arg(long)]
        model: Option<String>,
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
    /// Show or change where weights, caches and datasets live.
    ///
    /// Changing it does not move anything already downloaded - moving tens of
    /// gigabytes is `mv`, and doing it silently inside a config command would be a
    /// surprising amount of I/O. It says what to move instead.
    DataDir { path: Option<std::path::PathBuf> },
}

fn main() -> anyhow::Result<()> {
    quit_quietly_on_broken_pipe();

    match Cli::parse().command {
        Command::Budget => budget(),
        Command::Model(ModelCmd::Ls) => model_ls(),
        Command::Model(ModelCmd::Pick) => model_pick().map(|picked| {
            if let Some(name) = picked {
                harness_next_step(&name);
            }
        }),
        Command::Model(ModelCmd::Search { query, limit }) => model_search(&query.join(" "), limit),
        Command::Model(ModelCmd::Files { repo }) => model_files(&repo),
        Command::Model(ModelCmd::Install(args)) => model_install(&args).map(|name| {
            harness_next_step(&name);
        }),
        Command::Model(ModelCmd::Rm { name }) => model_rm(&name),
        Command::Config(ConfigCmd::Show) => config_show(),
        Command::Config(ConfigCmd::Init) => config_init(),
        Command::Config(ConfigCmd::DataDir { path }) => config_data_dir(path),
        Command::Serve(args) => serve_model(&args),
        Command::Ps => ps(),
        Command::Stop => stop(),
        Command::Gateway(GatewayCmd::Run { port, host }) => gateway_run(&host, port),
        Command::Harness(HarnessCmd::Configure { name, url, model }) => {
            harness_configure(&name, &url, model.as_deref())
        }
        Command::Harness(HarnessCmd::Unconfigure { name }) => harness_unconfigure(&name),
        Command::Service(ServiceCmd::Install { no_start }) => service_install(no_start),
        Command::Service(ServiceCmd::Uninstall) => service_uninstall(),
        Command::Service(ServiceCmd::Status) => service_status(),
        Command::Setup(args) => setup(&args),
        Command::Extras(ExtrasCmd::List) => extras_list(),
        Command::Extras(ExtrasCmd::Install { name }) => extras_install(&name),
        Command::Extras(ExtrasCmd::Remove { name }) => extras_remove(&name),
        Command::Eval { args } => {
            std::process::exit(extras::dispatch(named_extra("eval")?, &args)?)
        }
        Command::Update { args } => std::process::exit(ailocal::update::run(&args)?),
        Command::Gateway(GatewayCmd::Check { url }) => gateway_check(&url),
        Command::Gateway(GatewayCmd::Key) => {
            println!("{}", auth::load_or_create()?);
            eprintln!("stored in {}", auth::key_path()?.display());
            Ok(())
        }
    }
}

/// Resolve an extra by name, or explain what the names are.
fn named_extra(name: &str) -> anyhow::Result<&'static extras::Extra> {
    extras::find(name).ok_or_else(|| {
        let known: Vec<&str> = extras::EXTRAS.iter().map(|e| e.name).collect();
        anyhow::anyhow!(
            "no extra named {name:?}; known extras: {}",
            known.join(", ")
        )
    })
}

fn extras_list() -> anyhow::Result<()> {
    let core = env!("CARGO_PKG_VERSION");
    println!("{:<10} {:<12} SUMMARY", "EXTRA", "VERSION");
    for e in extras::EXTRAS {
        let state = match extras::locate(e) {
            None => "-".to_owned(),
            Some(path) => extras::version_of(&path).unwrap_or_else(|| "installed".to_owned()),
        };
        println!("{:<10} {state:<12} {}", e.name, e.summary);
    }

    // Skew is worth naming explicitly. The two binaries share a release tag, so a
    // mismatch means one of them was installed by hand and the report formats or task
    // corpus may not line up.
    let skewed: Vec<&str> = extras::EXTRAS
        .iter()
        .filter(|e| {
            extras::locate(e)
                .and_then(|p| extras::version_of(&p))
                .is_some_and(|v| v != core)
        })
        .map(|e| e.name)
        .collect();
    if skewed.is_empty() {
        println!("\nailocal {core}. Install one with `ailocal extras install <name>`.");
    } else {
        println!(
            "\nailocal is {core} but {} is not - run `ailocal update` to match them.",
            skewed.join(", ")
        );
    }
    Ok(())
}

fn extras_install(name: &str) -> anyhow::Result<()> {
    let extra = named_extra(name)?;
    let code = extras::install(extra)?;
    anyhow::ensure!(code == 0, "installing the {name} extra failed");
    // No closing hint here: the install script prints one already, and it is also what
    // the `curl | sh --extra` path relies on.
    Ok(())
}

fn extras_remove(name: &str) -> anyhow::Result<()> {
    match extras::remove(named_extra(name)?)? {
        Some(path) => println!("removed {}", path.display()),
        None => println!("the {name} extra is not installed"),
    }
    Ok(())
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

fn config_data_dir(path: Option<std::path::PathBuf>) -> anyhow::Result<()> {
    let mut cfg = Config::load()?;

    let Some(new_root) = path else {
        for (label, dir) in [
            ("data", &cfg.data_dir),
            ("models", &cfg.models_dir),
            ("hf cache", &cfg.hf_home),
            ("eval", &cfg.eval_dir),
        ] {
            println!(
                "{label:<9} {:<44} {}",
                dir.display(),
                ailocal::free_mib(dir).map_or_else(
                    || "?".into(),
                    |m| format!("{} free", ailocal::format_mib(m))
                )
            );
        }
        return Ok(());
    };

    let old_models = cfg.models_dir.clone();
    cfg.set_data_dir(new_root);
    std::fs::create_dir_all(&cfg.models_dir)
        .map_err(|e| anyhow::anyhow!("cannot create {}: {e}", cfg.models_dir.display()))?;
    println!("wrote {}", cfg.save()?.display());
    println!(
        "models now expected in {} ({} free)",
        cfg.models_dir.display(),
        ailocal::free_mib(&cfg.models_dir).map_or_else(|| "?".into(), ailocal::format_mib)
    );

    // Nothing is moved, so say plainly what is now in the wrong place rather than let
    // `ailocal model ls` come back empty and look like the models were lost.
    if old_models != cfg.models_dir
        && let Ok(existing) = registry::scan(&old_models)
        && !existing.is_empty()
    {
        println!(
            "\n{} model(s) are still in {}. Move them across to keep them:\n  mv {}/*.gguf {}/",
            existing.len(),
            old_models.display(),
            old_models.display(),
            cfg.models_dir.display()
        );
    }
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

/// Something the picker can offer.
enum Pickable {
    /// Already on disk. Choosing it costs nothing but a config write.
    Installed {
        name: String,
        size_mib: u64,
        context: u64,
        /// The catalogue's description, when this is a model the catalogue knows.
        summary: Option<&'static str>,
    },
    /// In the catalogue and not yet downloaded.
    Remote(Box<ailocal::catalogue::Candidate>),
}

impl Pickable {
    fn label(&self) -> String {
        match self {
            // Deliberately the same shape as Candidate::label, so the two halves of
            // the list read as one list.
            Self::Installed {
                name,
                size_mib,
                context,
                summary,
            } => format!(
                "{:<18} {:>5} GiB  {:>4}k context   on disk{}",
                name,
                size_mib / 1024,
                context / 1024,
                summary.map_or_else(String::new, |s| format!(" - {s}"))
            ),
            Self::Remote(c) => c.label(),
        }
    }
}

/// Offer what is installed and what could be, ranked for this machine, and make the
/// choice the default model.
///
/// This is the "which model am I using" command, not only the "download one" command.
/// A model already on disk is offered as itself rather than as something to fetch
/// again, so picking it is how you switch defaults - and re-picking the model you
/// already have costs nothing.
///
/// Returns the chosen model's name.
fn model_pick() -> anyhow::Result<Option<String>> {
    let cfg = Config::load()?;
    let cache: vram::CacheType = cfg.cache_type.parse()?;
    let budget = serve::budget_for_next_launch()
        .unwrap_or_else(|_| vram::Budget::new(ailocal::vram_used_mib().unwrap_or(900)));
    let available = budget.available_mib();

    let on_disk = registry::scan(&cfg.models_dir)?;

    let client = http_client()?;
    println!("Checking what fits in {available} MiB ...");
    let mut candidates = ailocal::catalogue::resolve(&client, available);

    // Anything already downloaded is offered below as itself, so drop it from the
    // download list rather than invite someone to fetch gigabytes they already have.
    // Keep what the catalogue called it, to describe it in the list.
    let mut described: Vec<(String, &'static str)> = Vec::new();
    candidates.retain(|c| {
        let Some(stem) = catalogue_stem(c) else {
            return true;
        };
        match registry::already_have(&on_disk, &stem, c.size_mib) {
            Some(have) => {
                described.push((have.name.clone(), c.entry.summary));
                false
            }
            None => true,
        }
    });

    let mut options: Vec<Pickable> = Vec::new();
    let mut unusable = 0;
    for m in &on_disk {
        match registry::assess(m.kv, m.trained_context, &budget, cache, m.size_mib) {
            Fit::Fits(context) => options.push(Pickable::Installed {
                name: m.name.clone(),
                size_mib: m.size_mib,
                context,
                summary: described
                    .iter()
                    .find(|(name, _)| *name == m.name)
                    .map(|(_, summary)| *summary),
            }),
            _ => unusable += 1,
        }
    }

    // The registry being unreachable is only fatal with nothing installed - otherwise
    // there is still a perfectly good choice to make offline.
    anyhow::ensure!(
        !candidates.is_empty() || !options.is_empty(),
        "could not reach the model registry, and nothing is installed yet"
    );

    if !candidates.is_empty() {
        // Reads a few MiB of each fitting model for its real attention geometry,
        // because weight size does not predict usable context: a 12B with
        // sliding-window attention holds six times what a 14B with full attention does.
        println!("Measuring usable context ...");
        ailocal::catalogue::measure(&client, &mut candidates, &budget, cache);
    }
    options.extend(
        candidates
            .into_iter()
            .map(|c| Pickable::Remote(Box::new(c))),
    );

    let labels: Vec<String> = options.iter().map(Pickable::label).collect();
    let header = format!(
        "Models for this machine ({available} MiB available). \
         Choosing one makes it the default. Anything else: ailocal model search <query>"
    );
    if unusable > 0 {
        println!("({unusable} installed model(s) cannot run here; `ailocal model ls` shows why)");
    }

    let Some(index) = ailocal::prompt::choose(&header, &labels)? else {
        println!("Nothing selected.");
        return Ok(None);
    };

    let name = match &options[index] {
        Pickable::Installed { name, .. } => {
            println!("\n{name} is already downloaded.");
            name.clone()
        }
        Pickable::Remote(chosen) => {
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
            model_install(&InstallArgs {
                reference: format!("ollama:{}", chosen.entry.reference),
                force: !chosen.fits(),
            })?
        }
    };

    set_default_model(&name)?;
    Ok(Some(name))
}

/// The filename stem a catalogue entry would be stored under, for spotting one that is
/// already downloaded.
fn catalogue_stem(candidate: &ailocal::catalogue::Candidate) -> Option<String> {
    format!("ollama:{}", candidate.entry.reference)
        .parse::<Source>()
        .ok()
        .map(|s| s.file_name().trim_end_matches(".gguf").to_owned())
}

/// Record `name` as the model this machine serves, and make that true.
fn set_default_model(name: &str) -> anyhow::Result<()> {
    let mut cfg = Config::load()?;
    if cfg.default_model.as_deref() == Some(name) {
        println!("{name} is already the default model.");
        return Ok(());
    }

    cfg.default_model = Some(name.to_owned());
    cfg.save()?;
    println!("default model is now {name}");
    repoint_model_service(&cfg);
    Ok(())
}

/// Point an installed model service at the configured default.
///
/// The unit embeds the model name in its `ExecStart`, so changing `default_model` in
/// the config does nothing on its own - the service would keep serving the old model
/// and would load it again at the next boot. Best-effort: a service that cannot be
/// updated must not turn a successful download into a failed command.
fn repoint_model_service(cfg: &Config) {
    let installed =
        service::is_enabled(service::MODEL_UNIT) || service::is_active(service::MODEL_UNIT);
    if !installed {
        return;
    }

    let outcome = (|| -> anyhow::Result<bool> {
        let exe = unit_binary()?;
        service::install(
            &exe,
            &cfg.gateway_host,
            cfg.gateway_port,
            cfg.default_model.as_deref(),
        )?;
        service::reload()?;
        let running = service::is_active(service::MODEL_UNIT);
        if running {
            service::stop(service::MODEL_UNIT)?;
            service::start(service::MODEL_UNIT)?;
        }
        Ok(running)
    })();

    match outcome {
        Ok(true) => println!("  restarted {} on it", service::MODEL_UNIT),
        Ok(false) => println!("  updated {}", service::MODEL_UNIT),
        Err(e) => println!(
            "  note: {} still points at the old model ({e:#})\n\
             \x20       run `ailocal service install` to update it",
            service::MODEL_UNIT
        ),
    }
}

/// The line that carries someone from "a model is on disk" to "my harness uses it".
///
/// Every other step of this flow ends by naming the next one - `model search` points at
/// `model files`, which points at `model install` - and this is where that chain used
/// to stop.
fn harness_next_step(name: &str) {
    println!("\nNext: ailocal harness configure pi --model {name}   (or claude-code)");
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

/// Download a model, returning the name the registry will know it by.
///
/// The name is what every later step needs - `serve`, `default_model`, a harness
/// catalogue entry - so it is returned rather than left for the caller to re-derive
/// from the reference.
fn model_install(args: &InstallArgs) -> anyhow::Result<String> {
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
    //
    // Budgeted against what a launch would get, not against live VRAM: with a model
    // already resident, live usage counts weights that serving this one would evict.
    // `model pick` sizes its list the same way, so using live usage here made the two
    // disagree - the picker offered a model as fitting and the installer then refused
    // to download it.
    let budget = serve::budget_for_next_launch()
        .unwrap_or_else(|_| vram::Budget::new(ailocal::vram_used_mib().unwrap_or(900)));
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

    Ok(path.file_stem().map_or_else(
        || artifact.file_name.clone(),
        |s| s.to_string_lossy().into_owned(),
    ))
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

/// Every installed model that could actually run here, alphabetically.
///
/// Offering a model a harness cannot load only moves the failure later, so anything
/// that does not fit is left out.
fn runnable_models(cfg: &Config) -> anyhow::Result<Vec<harness::PiModel>> {
    let cache: vram::CacheType = cfg.cache_type.parse()?;
    let budget = serve::budget_for_next_launch()
        .unwrap_or_else(|_| vram::Budget::new(ailocal::vram_used_mib().unwrap_or(900)));

    Ok(registry::scan(&cfg.models_dir)?
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
        .collect())
}

fn harness_configure(name: &str, url: &str, wanted: Option<&str>) -> anyhow::Result<()> {
    anyhow::ensure!(
        matches!(name, "pi" | "claude-code"),
        "unknown harness {name:?}; expected `pi` or `claude-code`"
    );

    let cfg = Config::load()?;
    let mut models = runnable_models(&cfg)?;
    anyhow::ensure!(
        !models.is_empty(),
        "no installed model can run here; `ailocal model ls` shows why"
    );

    let loaded = serve::running()?.map(|i| i.model);
    let (preferred, why) = harness::preferred(
        &models,
        wanted,
        cfg.default_model.as_deref(),
        loaded.as_deref(),
    )?;
    let (preferred, why) = (preferred.clone(), why);

    let key = auth::load_or_create()?;

    if name == "claude-code" {
        let (path, outcome) =
            harness::configure_claude_code(url, &key, &preferred.id, preferred.context_window)?;
        match outcome {
            harness::Outcome::AlreadyConfigured => println!("claude-code env already current"),
            harness::Outcome::Configured { .. } => println!("wrote {}", path.display()),
        }
        // Claude Code takes exactly one model, so which one it got is the single most
        // useful thing to say - and the thing that used to be decided invisibly.
        println!(
            "  model: {} ({} context) - {}",
            preferred.id,
            format_count(preferred.context_window),
            why.why()
        );
        if models.len() > 1 {
            println!(
                "  others installed: {}",
                others(&models, &preferred.id).join(", ")
            );
            println!("  change it with `ailocal harness configure claude-code --model <name>`");
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

    // Pi keeps the whole catalogue and chooses per session, so the preference only
    // decides the order it sees them in - and which one the hint below names.
    if let Some(at) = models.iter().position(|m| m.id == preferred.id) {
        let chosen = models.remove(at);
        models.insert(0, chosen);
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
        preferred.id
    );
    Ok(())
}

/// The other model names, for a "you also have these" line.
fn others(models: &[harness::PiModel], chosen: &str) -> Vec<String> {
    models
        .iter()
        .filter(|m| m.id != chosen)
        .map(|m| m.id.clone())
        .collect()
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

/// Settle where the large files go, writing a config if there is not one yet.
///
/// Asked rather than assumed, because the default cannot be right for everyone: model
/// weights are tens of gigabytes and the only machine that knows where there is room is
/// this one. `--data-dir` answers it for a script; a terminal gets a prompt; anything
/// else takes the default rather than blocking on a question nobody will see.
fn resolve_data_dir(args: &SetupArgs) -> anyhow::Result<Config> {
    let path = Config::path()?;
    let mut cfg = Config::load()?;

    if let Some(chosen) = &args.data_dir {
        cfg.set_data_dir(chosen.clone());
        println!("   wrote {}", cfg.save()?.display());
        return Ok(cfg);
    }

    if path.exists() {
        println!("   ok    {}", path.display());
        return Ok(cfg);
    }

    if ailocal::prompt::interactive() {
        let default = cfg.data_dir.display().to_string();
        println!(
            "   Models are large - a single one is 5-30 GB. Where should they live?\n\
             \x20  {default} has {} free.",
            ailocal::free_mib(&cfg.data_dir).map_or_else(|| "?".into(), ailocal::format_mib)
        );
        let answer = ailocal::prompt::ask("   Data directory", &default)?;
        cfg.set_data_dir(std::path::PathBuf::from(answer));
    }

    println!("   wrote {}", cfg.save()?.display());
    Ok(cfg)
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

    println!("\n2. where things go");
    let cfg = resolve_data_dir(args)?;
    if let Err(e) = std::fs::create_dir_all(&cfg.models_dir) {
        anyhow::bail!(
            "cannot create {}: {e}\n\
             Pick somewhere writable with `ailocal config data-dir <path>`, or re-run \
             `ailocal setup --data-dir <path>`.",
            cfg.models_dir.display()
        );
    }
    println!(
        "   ok    {} ({} free)",
        cfg.models_dir.display(),
        ailocal::free_mib(&cfg.models_dir).map_or_else(|| "?".into(), ailocal::format_mib)
    );

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

    // The model unit embeds a model name, so with no default there is no model service
    // at all - the gateway would come up with nothing behind it. Also covers a default
    // naming a model that has since been deleted.
    let installed = Config::load()?.default_model;
    if !installed.is_some_and(|d| models.iter().any(|m| m.name == d))
        && let Some(first) = models.first()
    {
        set_default_model(&first.name)?;
    }

    if !args.skip_service {
        println!("\n4. services");
        service_install(false)?;
    }

    if !args.skip_harness {
        println!("\n5. harnesses");
        let url = format!("http://{}:{}", "127.0.0.1", cfg.gateway_port);
        for harness in ["pi", "claude-code"] {
            match harness_configure(harness, &url, None) {
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
            println!("{:<12} {}", "reasoning", i.reasoning);
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
