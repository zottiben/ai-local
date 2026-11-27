//! `ailocal` - manage local LLMs and expose them to coding harnesses.

use std::sync::Arc;

use ailocal::{
    auth, capture, config::Config, distill, download, extras, gateway, harness, install, lessons,
    registry, registry::Fit, serve, service, setup, source::Source, vram,
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
    /// Lessons that local models get from your corrections.
    ///
    /// Only local models get them, through the Pi extension: a lesson's trigger holds
    /// the first matching call of each prompt and shows the lesson, and the lessons
    /// most like a prompt go along with it.
    #[command(subcommand)]
    Lessons(LessonsCmd),
    /// systemd units, so this survives a reboot.
    #[command(subcommand)]
    Service(ServiceCmd),
    /// Check prerequisites and bring everything up in one go.
    Setup(SetupArgs),
    /// Show where this machine is: model, gateway, services, harnesses.
    Status,
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
    /// Open a console for this machine in the browser. Needs the `ui` extra.
    ///
    /// Everything after `ui` goes to `ailocal-ui`, as with `eval`.
    #[command(disable_help_flag = true)]
    Ui {
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

    /// Port for the gateway. Defaults to the configured one, moving off it only if
    /// something else already has it.
    #[arg(long)]
    port: Option<u16>,

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
        ///
        /// Defaults to the configured `gateway_port` over loopback. It has to come
        /// from the config rather than a constant: changing the port is the fix for a
        /// port conflict, and a hardcoded default meant the harness kept being pointed
        /// at the old one.
        #[arg(long)]
        url: Option<String>,
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
enum LessonsCmd {
    /// List the lessons, active ones first.
    Ls,
    /// Add a lesson of your own.
    Add(LessonAddArgs),
    /// Delete a lesson.
    Rm { id: String },
    /// Learn lessons from your messages in Pi's sessions, for you to review.
    ///
    /// Reads what Pi has written since the last time - the first time, every session
    /// it has kept - and asks a model which of your corrections teach something to
    /// keep. What it learns waits for `ailocal lessons review`.
    Capture {
        /// Ask this model of Pi's instead of the one ailocal serves, as `provider/id`,
        /// e.g. claude-subscription/claude-haiku-5-5.
        #[arg(long, value_name = "PROVIDER/ID")]
        model: Option<String>,
    },
    /// Go through the pending lessons, accepting, editing or rejecting each.
    Review,
    /// Accept pending lessons, so local models get them.
    Accept {
        #[arg(required = true)]
        ids: Vec<String>,
    },
    /// Reject lessons. They are kept, so that capture does not propose them again.
    Reject {
        #[arg(required = true)]
        ids: Vec<String>,
    },
}

#[derive(Args)]
struct LessonAddArgs {
    /// What to do, as you would tell a colleague, e.g. "Fix a lint error at its
    /// cause instead of adding eslint-disable."
    #[arg(required = true, num_args = 1..)]
    text: Vec<String>,

    /// Hold the first command of each prompt that starts with these words, and show
    /// the lesson, e.g. `--command "gh pr create"`. Repeatable.
    #[arg(long = "command", value_name = "WORDS")]
    commands: Vec<String>,

    /// Hold the first edit or write of each prompt that adds this text, and show the
    /// lesson, e.g. `--adds eslint-disable`. Repeatable.
    #[arg(long = "adds", value_name = "TEXT")]
    adds: Vec<String>,

    /// Hold the first edit or write of each prompt to this file, directory or
    /// extension, and show the lesson, e.g. `--file CHANGELOG.md`. Repeatable.
    #[arg(long = "file", value_name = "PATH")]
    files: Vec<String>,

    /// Only in this repository: its remote URL, or `host/owner/repo`.
    #[arg(long, conflicts_with = "here")]
    repo: Option<String>,

    /// Only in the repository of the current directory.
    #[arg(long)]
    here: bool,
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
        /// Record every conversation to `<data_dir>/gateway-requests.jsonl`.
        ///
        /// For answering "what is my harness actually sending?" - whether an
        /// AGENTS.md reached the model, how large the system prompt really is, which
        /// tools were offered. Off by default: the log is a transcript of your work.
        #[arg(long)]
        log_requests: bool,
    },
    /// Print the API key, creating one if there is none.
    Key,
    /// Check a gateway is reachable and correctly authenticated.
    ///
    /// Point it at the public hostname to verify the tunnel end to end.
    Check { url: Option<String> },
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
    /// Delete a model's weights from disk, and stop advertising it.
    ///
    /// Frees the tens of gigabytes a model occupies, so it asks first unless told not
    /// to. Also unpicks what pointed at the model: the configured default, the
    /// capability entry in a harness's config, and the running server if it is this
    /// one - a deleted file that a harness still offers is a load failure later.
    #[command(visible_aliases = ["remove", "delete"])]
    Rm(RemoveArgs),
}

#[derive(Args)]
struct RemoveArgs {
    /// Model name as shown by `ailocal model ls`.
    name: String,

    /// Delete without asking. What a script wants; think before typing it.
    #[arg(long, short = 'y')]
    yes: bool,
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
    /// Show or change the port the gateway listens on.
    ///
    /// The escape hatch when something else already owns 8081.
    GatewayPort { port: Option<u16> },
    /// Show or change the largest context a launch will take.
    ///
    /// Only ever a cap - the model's training and this machine's memory still
    /// decide. Lower it to make a model answer faster, raise it to let a session run
    /// longer before the harness compacts it.
    MaxContext { tokens: Option<u64> },
    /// Show the built-in rules for coding agents, or turn them on or off.
    ///
    /// The gateway adds them to every request that offers tools, ahead of your own
    /// `system_prompt`. They ask a local model to run the project's checks before it
    /// commits and to use jq on JSON - habits a frontier model has without being told.
    AgentRules {
        #[arg(value_parser = ["on", "off"])]
        state: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    quit_quietly_on_broken_pipe();

    match Cli::parse().command {
        Command::Budget => budget(),
        Command::Model(ModelCmd::Ls) => model_ls(),
        Command::Model(ModelCmd::Pick) => model_pick().map(|picked| {
            if picked.is_some() {
                print_next_step();
            }
        }),
        Command::Model(ModelCmd::Search { query, limit }) => model_search(&query.join(" "), limit),
        Command::Model(ModelCmd::Files { repo }) => model_files(&repo),
        Command::Model(ModelCmd::Install(args)) => model_install(&args).map(|_| {
            print_next_step();
        }),
        Command::Model(ModelCmd::Rm(args)) => model_rm(&args),
        Command::Config(ConfigCmd::Show) => config_show(),
        Command::Config(ConfigCmd::Init) => config_init(),
        Command::Config(ConfigCmd::DataDir { path }) => config_data_dir(path),
        Command::Config(ConfigCmd::GatewayPort { port }) => config_gateway_port(port),
        Command::Config(ConfigCmd::MaxContext { tokens }) => config_max_context(tokens),
        Command::Config(ConfigCmd::AgentRules { state }) => config_agent_rules(state.as_deref()),
        Command::Serve(args) => serve_model(&args),
        Command::Ps => ps(),
        Command::Stop => stop(),
        Command::Gateway(GatewayCmd::Run {
            port,
            host,
            log_requests,
        }) => gateway_run(&host, port, log_requests),
        Command::Harness(HarnessCmd::Configure { name, url, model }) => {
            let url = url.map_or_else(local_gateway_url, Ok)?;
            harness_configure(&name, &url, model.as_deref())
        }
        Command::Harness(HarnessCmd::Unconfigure { name }) => harness_unconfigure(&name),
        Command::Service(ServiceCmd::Install { no_start }) => service_install(no_start),
        Command::Service(ServiceCmd::Uninstall) => service_uninstall(),
        Command::Service(ServiceCmd::Status) => service_status(),
        Command::Setup(args) => setup(&args),
        Command::Status => status(),
        Command::Lessons(LessonsCmd::Ls) => lessons_ls(),
        Command::Lessons(LessonsCmd::Add(args)) => lessons_add(args),
        Command::Lessons(LessonsCmd::Rm { id }) => lessons_rm(&id),
        Command::Lessons(LessonsCmd::Capture { model }) => lessons_capture(model),
        Command::Lessons(LessonsCmd::Review) => lessons_review(),
        Command::Lessons(LessonsCmd::Accept { ids }) => lessons_set(&ids, lessons::State::Active),
        Command::Lessons(LessonsCmd::Reject { ids }) => lessons_set(&ids, lessons::State::Rejected),
        Command::Extras(ExtrasCmd::List) => extras_list(),
        Command::Extras(ExtrasCmd::Install { name }) => extras_install(&name),
        Command::Extras(ExtrasCmd::Remove { name }) => extras_remove(&name),
        Command::Eval { args } => {
            std::process::exit(extras::dispatch(named_extra("eval")?, &args)?)
        }
        Command::Ui { args } => std::process::exit(extras::dispatch(named_extra("ui")?, &args)?),
        Command::Update { args } => std::process::exit(ailocal::update::run(&args)?),
        Command::Gateway(GatewayCmd::Check { url }) => {
            let url = url.map_or_else(local_gateway_url, Ok)?;
            gateway_check(&url)
        }
        Command::Gateway(GatewayCmd::Key) => {
            println!("{}", auth::load_or_create()?);
            eprintln!("stored in {}", auth::key_path()?.display());
            Ok(())
        }
    }
}

/// The URL a client on this machine should use to reach our gateway.
///
/// Always loopback, even when the gateway binds `0.0.0.0` for the tunnel: a wildcard
/// bind is not an address you can connect to.
fn local_gateway_url() -> anyhow::Result<String> {
    let cfg = Config::load()?;
    Ok(format!(
        "http://{}:{}",
        gateway::loopback_for(&cfg.gateway_host),
        cfg.gateway_port
    ))
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

fn config_gateway_port(port: Option<u16>) -> anyhow::Result<()> {
    let cfg = Config::load()?;

    let Some(wanted) = port else {
        let status = gateway::port_status(&cfg.gateway_host, cfg.gateway_port);
        println!("gateway port  {}", cfg.gateway_port);
        println!(
            "status        {}",
            match status {
                gateway::PortStatus::Ours => "an ailocal gateway is listening",
                gateway::PortStatus::Free => "nothing is listening",
                gateway::PortStatus::Taken => "IN USE by something that is not ailocal",
            }
        );
        if status == gateway::PortStatus::Taken {
            println!(
                "\nThe gateway cannot start while that is true. Move it:\n\
                 \x20   ailocal config gateway-port {}",
                gateway::usable_port(&cfg.gateway_host, cfg.gateway_port)
                    .unwrap_or(cfg.gateway_port + 1)
            );
        }
        return Ok(());
    };

    let applied = setup::apply_settings(
        &setup::Settings {
            gateway_port: Some(wanted),
            ..setup::Settings::default()
        },
        &unit_binary,
    )?;
    if applied.changed.is_empty() {
        println!("the gateway port is already {wanted}");
        return Ok(());
    }
    print_applied(&applied);
    Ok(())
}

fn config_max_context(tokens: Option<u64>) -> anyhow::Result<()> {
    let cfg = Config::load()?;

    let Some(wanted) = tokens else {
        println!("max context  {} tokens", cfg.max_context);
        if let Ok(Some(instance)) = serve::running() {
            println!(
                "loaded with  {} tokens ({})",
                instance.context, instance.model
            );
        }
        return Ok(());
    };

    let applied = setup::apply_settings(
        &setup::Settings {
            max_context: Some(wanted),
            ..setup::Settings::default()
        },
        &unit_binary,
    )?;
    if applied.changed.is_empty() {
        println!("max context is already {wanted} tokens");
        return Ok(());
    }
    print_applied(&applied);
    println!("\nAfter it reloads, refresh Pi's cached window:\n  ailocal harness configure pi");
    Ok(())
}

fn config_agent_rules(state: Option<&str>) -> anyhow::Result<()> {
    let Some(state) = state else {
        let cfg = Config::load()?;
        let on = if cfg.agent_rules { "on" } else { "off" };
        println!("agent rules  {on}");
        println!("\nAdded to every request that offers tools, ahead of your system_prompt:\n");
        for line in gateway::AGENT_RULES.lines() {
            println!("  {line}");
        }
        println!(
            "\nIn Pi they come with an extension that holds a local model's push once when\n\
             no test has passed since its last edit, holds Python over JSON once when jq is\n\
             installed, asks for the checks once before the model stops, sets git's\n\
             editors so that a rebase or a commit never waits on one, and gives it your\n\
             `ailocal lessons` where they apply:\n\n  {}",
            harness::pi_extension_path()?.display()
        );
        return Ok(());
    };

    let applied = setup::apply_settings(
        &setup::Settings {
            agent_rules: Some(state == "on"),
            ..setup::Settings::default()
        },
        &unit_binary,
    )?;
    if applied.changed.is_empty() {
        println!("agent rules are already {state}");
        return Ok(());
    }
    print_applied(&applied);
    println!("A running gateway applies it from the next request.");
    Ok(())
}

/// Say what a settings change wrote, and what it means for what is already running.
fn print_applied(applied: &setup::Applied) {
    if let Some(path) = &applied.path {
        println!("wrote {}", path.display());
    }

    match &applied.gateway_restarted {
        Some(Ok(())) => println!("restarted {}", service::GATEWAY),
        Some(Err(e)) => println!("note: could not restart {}: {e:#}", service::GATEWAY),
        None => {}
    }

    // The context is fixed when llama-server starts, so a resident model keeps the
    // window it was launched with - and the catalogue would then advertise one number
    // while the server serves another.
    match &applied.reload_model {
        Some(model) => {
            // The foreground parent exits with its child and launchd/systemd restarts
            // it from the new config. Starting a detached sibling would race the
            // supervisor for the same port.
            let command = if service::is_active(service::MODEL) {
                "ailocal stop".to_owned()
            } else {
                format!("ailocal serve {model}")
            };
            println!("\n{model} is loaded with the old context. Reload it:\n  {command}");
        }
        None if applied.changed.iter().any(|c| c == "max_context") => {
            println!("\nTakes effect the next time a model loads.");
        }
        None => {}
    }

    // Harnesses hold the URL in their own config, so they do not follow on their own.
    if applied.repoint_harnesses {
        println!("\nNext: re-point your harnesses at the new port");
        println!("  ailocal harness configure pi        (and claude-code)");
    }
}

fn budget() -> anyhow::Result<()> {
    let launch = serve::budget_for_next_launch()?;
    let unified = ailocal::is_unified_memory();

    println!("ceiling          {:>6} MiB", launch.ceiling_mib);
    // Unified memory has no machine-wide reading to show, only the probe's own ~1 MiB.
    // What the rest of the machine gets there is the reserve, so that is the line.
    let in_use = if unified {
        let total = ailocal::primary_device()?.total_mib;
        println!(
            "reserved         {:>6} MiB  of {total} for everything but the model",
            total.saturating_sub(launch.ceiling_mib)
        );
        None
    } else {
        let in_use = ailocal::vram_used_mib()?;
        println!("in use now       {in_use:>6} MiB");
        Some(in_use)
    };
    println!("compute buffers  {:>6} MiB", vram::COMPUTE_BUFFER_MIB);
    println!("available        {:>6} MiB", launch.available_mib());
    // The memory figures alone cannot explain a context smaller than they allow.
    if let Some(cap) = launch.context_cap {
        println!("context cap      {cap:>6} tokens");
    }

    // Reporting live usage alone reads as "almost nothing is free" whenever a model is
    // resident, which contradicts `model ls` saying that same model fits. The number
    // that decides what can be loaded is the one measured against the baseline from
    // before the current model loaded, since starting another evicts it.
    if let Ok(Some(instance)) = serve::running() {
        let held = match in_use {
            Some(in_use) => in_use.saturating_sub(launch.desktop_mib),
            None => ailocal::footprint_mib(instance.pid)?,
        };
        println!(
            "\n{} holds {held} MiB of that, and would be evicted by the next load - so\n\
             the figure above is what a launch actually gets, not what is idle now.",
            instance.model,
        );
    }
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
    let models = ailocal::models::inventory(&cfg)?;
    if models.is_empty() {
        println!("no models in {}", cfg.models_dir.display());
        return Ok(());
    }

    println!(
        "{:<28} {:>7}  {:<17} {:>9} {:>10} {:>11}  THINK",
        "NAME", "SIZE", "ARCH", "KV/TOK", "TRAINED", "MAX CTX"
    );
    for m in &models {
        let kv_per_token = m
            .kv_bytes_per_token
            .map_or_else(|| "-".to_owned(), |b| format!("{:.0} KiB", b / 1024.0));
        let trained = m
            .trained_context
            .map_or_else(|| "-".to_owned(), format_count);
        let max_ctx = m
            .max_context
            .map_or_else(|| "WILL NOT FIT".to_owned(), format_count);

        // The marker rides on ARCH rather than trailing the row: sliding-window
        // attention is a property of the architecture, and left at the end it read as
        // if it qualified whatever column happened to be last.
        let arch = format!(
            "{}{}",
            m.arch.as_deref().unwrap_or("?"),
            if m.sliding_window { " (swa)" } else { "" }
        );

        println!(
            "{:<28} {:>5} G  {arch:<17} {kv_per_token:>9} {trained:>10} {max_ctx:>11}  {}",
            truncate(&m.name, 28),
            m.size_mib / 1024,
            m.thinking.label(),
        );
    }

    if models.iter().any(|m| m.sliding_window) {
        println!(
            "\n(swa) sliding-window attention: most layers hold a fixed window rather than\n\
             growing with context, so KV/TOK is the marginal cost of the few global\n\
             layers. This is why a 12B reaches 256k on a 16 GB card."
        );
    }

    // The one column a harness's thinking controls depend on, and the only place the
    // answer is visible before `/effort` refuses. A `-` is the model, not a setting:
    // its chat template has no thinking mode, so nothing can switch one on.
    println!(
        "\nTHINK is what the model's own chat template offers: `on/off` a switch,\n\
         `effort` a level, `-` no thinking mode at all."
    );
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
    Installed(ailocal::models::Downloaded),
    /// In the catalogue and not yet downloaded.
    Remote(Box<ailocal::catalogue::Candidate>),
}

impl Pickable {
    fn label(&self) -> String {
        match self {
            // Deliberately the same shape as Candidate::label, so the two halves of
            // the list read as one list.
            Self::Installed(m) => format!(
                "{:<18} {:>5} GiB  {:>4}k context   on disk{}",
                m.name,
                m.size_mib / 1024,
                m.context / 1024,
                m.summary.map_or_else(String::new, |s| format!(" - {s}"))
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
    let budget = serve::budget_or_estimate();
    let available = budget.available_mib();

    let client = http_client()?;
    println!("Checking what fits in {available} MiB ...");
    let mut offer = ailocal::models::Offer::gather(&cfg, &client, budget)?;
    if !offer.remote.is_empty() {
        println!("Measuring usable context ...");
        offer.measure(&client);
    }

    let unusable = offer.unusable;
    let options: Vec<Pickable> = offer
        .downloaded
        .into_iter()
        .map(Pickable::Installed)
        .chain(
            offer
                .remote
                .into_iter()
                .map(|c| Pickable::Remote(Box::new(c))),
        )
        .collect();

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
        Pickable::Installed(m) => {
            println!("\n{} is already downloaded.", m.name);
            m.name.clone()
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

/// Record `name` as the model this machine serves, and make that true.
fn set_default_model(name: &str) -> anyhow::Result<()> {
    match ailocal::models::set_default(name, &unit_binary)? {
        ailocal::models::DefaultChange::Unchanged => {
            println!("{name} is already the default model.");
        }
        ailocal::models::DefaultChange::Changed(repoint) => {
            println!("default model is now {name}");
            print_repoint(&repoint);
        }
    }
    Ok(())
}

/// Say what became of the model service after the default changed.
fn print_repoint(repoint: &ailocal::models::Repoint) {
    use ailocal::models::Repoint;
    match repoint {
        Repoint::NoService => {}
        Repoint::Restarted => println!("  restarted {} on it", service::MODEL),
        Repoint::Cleared => println!(
            "  stopped {} - no default model left to preload",
            service::MODEL
        ),
        Repoint::Updated => println!("  updated {}", service::MODEL),
        Repoint::Failed(e) => println!(
            "  note: {} still points at the old model ({e:#})\n\
             \x20       run `ailocal service install` to update it",
            service::MODEL
        ),
    }
}

/// The whole picture in one command.
///
/// The pieces were all inspectable already - `ps`, `service status`, `gateway check`,
/// `model ls` - but only if you knew which to run and how to read them together.
/// Someone whose harness will not answer needs one command that says which link is
/// broken.
fn status() -> anyhow::Result<()> {
    use ailocal::status::{Gateway, Service};

    let cfg = Config::load()?;
    let o = ailocal::status::overview(&cfg);
    let mark = |ok: bool| if ok { "ok  " } else { "MISS" };

    println!(
        "{}  models      {}",
        mark(!o.models.is_empty()),
        if o.models.is_empty() {
            "none installed that can run here".to_owned()
        } else {
            format!(
                "{} runnable, default {}",
                o.models.len(),
                o.default_model.as_deref().unwrap_or("not set")
            )
        }
    );

    match &o.loaded {
        Some(i) => println!(
            "ok    loaded      {} at {} context, reasoning {}",
            i.model,
            format_count(i.context),
            i.reasoning
        ),
        None => println!("-     loaded      nothing resident (the gateway loads on demand)"),
    }

    println!(
        "{}  gateway     {} - {}",
        mark(o.gateway == Gateway::Answering),
        o.gateway_url,
        match o.gateway {
            Gateway::Answering => "answering",
            Gateway::PortTaken => "port held by something else",
            Gateway::NotAnswering => "not answering",
        }
    );

    // What the gateway advertises, not what is on disk. This is the list a harness
    // actually sees, and the two can disagree - a model that is installed but judged
    // not to fit is filtered out here, which looks from inside Pi like the model
    // simply vanishing from its picker.
    match &o.advertised {
        None => {}
        Some(Ok(names)) if names.is_empty() => println!(
            "MISS  advertised   none - the gateway is serving but offers no model\n\
             \x20                  `ailocal model ls` shows what it rejected and why"
        ),
        Some(Ok(names)) => println!("ok    advertised  {}", names.join(", ")),
        Some(Err(e)) => println!("MISS  advertised  could not ask the gateway: {e}"),
    }

    for (unit, state) in &o.services {
        println!(
            "{}  service     {unit} - {}",
            mark(*state == Service::Running),
            match state {
                Service::Running => "running",
                Service::InstalledNotRunning => "installed but not running",
                Service::NotInstalled => "not installed",
            }
        );
    }

    println!(
        "{}  harnesses   {}",
        mark(!o.harnesses.is_empty()),
        if o.harnesses.is_empty() {
            "none configured".to_owned()
        } else {
            o.harnesses.join(", ")
        }
    );

    let store = lessons::Store::load(&lessons::path()?)?;
    let count = |state| store.lessons.iter().filter(|l| l.state == state).count();
    let (active, pending) = (
        count(lessons::State::Active),
        count(lessons::State::Pending),
    );
    if pending > 0 {
        let wait = if pending == 1 { "waits" } else { "wait" };
        println!(
            "-     lessons     {active} active, {pending} learned from your corrections {wait} \
             for you: `ailocal lessons review`"
        );
    } else if active > 0 {
        println!("ok    lessons     {active} active");
    }

    match o.next.describe() {
        Some(line) => println!("\n{line}"),
        None => println!("\nEverything is in place."),
    }
    Ok(())
}

/// Print the next step, if there is one.
fn print_next_step() {
    let Ok(cfg) = Config::load() else { return };
    if let Some(line) = ailocal::status::next_step(&cfg).describe() {
        println!("\n{line}");
    }
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
    let budget = serve::budget_or_estimate();

    let client = http_client()?;
    let token = ailocal::source::hf_token(&cfg.hf_home);
    let files = ailocal::source::files(&client, repo, token.as_deref())?;

    anyhow::ensure!(!files.is_empty(), "{repo} has no .gguf files");

    // Weight size alone decides most of it, and the KV cache needs room on top. Show
    // the headroom rather than a bare yes/no so an almost-fitting quant is visible.
    let available = budget.available_mib();
    let choices = ailocal::source::weigh(files, available);

    println!("{available} MiB available for weights + KV cache\n");
    println!("{:<52} {:>8}  VERDICT", "FILE", "SIZE");
    for c in &choices {
        let verdict = if c.fits() {
            format!("{} MiB left for context", c.headroom_mib)
        } else if c.headroom_mib < 0 {
            format!("{} MiB too large", -c.headroom_mib)
        } else {
            "no room for a context".to_owned()
        };
        println!(
            "{:<52} {:>5} G  {verdict}",
            truncate(&c.file.path, 52),
            c.file.size_mib / 1024
        );
    }

    if let Some(best) = choices.iter().find(|c| c.fits()) {
        println!("\nNext: ailocal model install hf:{repo}/{}", best.file.path);
    }
    Ok(())
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
    let client = install::client()?;

    println!("resolving {source}");
    let resolved = install::resolve(&cfg, &client, source)?;
    println!(
        "  {} MiB{}",
        resolved.weights_mib(),
        if resolved.has_digest() {
            ", digest published"
        } else {
            ", no digest (length-checked only)"
        }
    );

    // Read just the head to learn the architecture. A model that cannot hold a usable
    // context here should cost seconds to reject, not an hour of downloading.
    let budget = serve::budget_or_estimate();
    refuse_unless_forced(
        install::check(&client, &resolved, &budget, cache),
        args.force,
    )?;

    let landed = install::fetch(&cfg, &client, &resolved, &mut download::Bar::default())?;
    match landed.outcome {
        download::Fetched::AlreadyPresent => {
            println!("already present: {}", landed.path.display());
        }
        download::Fetched::Downloaded {
            bytes,
            resumed_from,
        } => {
            if resumed_from > 0 {
                println!("  resumed from {} MiB", resumed_from / (1024 * 1024));
            }
            println!(
                "installed {} ({} MiB)",
                landed.path.display(),
                bytes / (1024 * 1024)
            );
        }
    }
    match &landed.sampling {
        None => {}
        Some(Ok(s)) if s.is_empty() => {}
        Some(Ok(s)) => println!("  sampling: {s} (from the publisher)"),
        Some(Err(e)) => eprintln!(
            "warning: {e:#}\n  it serves on llama.cpp's defaults until the install is run again"
        ),
    }
    Ok(landed.name)
}

/// Report what the head of the file said, and stop unless it may be downloaded.
///
/// # Errors
/// If the check blocked the download and `force` is not set.
fn refuse_unless_forced(check: install::Check, force: bool) -> anyhow::Result<()> {
    let encoder = check.encoder();
    match check.header {
        install::Header::Unreadable(e) if !force => {
            return Err(e.context(
                "could not read the model's header to check that it loads here; \
                 try again, or pass --force to download it unchecked",
            ));
        }
        install::Header::Unreadable(e) => {
            println!("  could not read its header ({e:#}); downloading unchecked");
        }
        install::Header::NotGguf => {}
        install::Header::Read {
            arch,
            trained_context,
            tensor_count,
            ..
        } => {
            println!(
                "  arch {}, trained to {}",
                arch.as_deref().unwrap_or("?"),
                trained_context.map_or_else(|| "?".into(), format_count)
            );
            if let Some((encoder, _)) = encoder {
                println!(
                    "  will not load: {encoder} of its {tensor_count} tensors are a vision or audio \
                     encoder\n  packed into the weights, and llama.cpp only loads one from a \
                     separate\n  mmproj file - as GGUFs converted by llama.cpp itself ship them"
                );
                anyhow::ensure!(force, "refusing to download; pass --force to override");
            }
        }
    }

    match check.fit {
        Fit::Unknown => println!("  could not read metadata; skipping the fit check"),
        Fit::Fits(ctx) => println!("  fits here: up to {} context", format_count(ctx)),
        Fit::ContextTooSmall(ctx) => {
            println!("  will not fit: only {ctx} tokens of context on this machine");
            anyhow::ensure!(force, "refusing to download; pass --force to override");
        }
        Fit::WeightsTooLarge => {
            println!(
                "  will not fit: {} MiB of weights against {} MiB available",
                check.weights_mib, check.available_mib
            );
            anyhow::ensure!(force, "refusing to download; pass --force to override");
        }
    }
    Ok(())
}

/// Delete a model's weights, and everything that pointed at them.
fn model_rm(args: &RemoveArgs) -> anyhow::Result<()> {
    let name = args.name.as_str();
    let cfg = Config::load()?;
    let plan = ailocal::models::plan_removal(&cfg, name)?;

    println!("{} ({})", plan.path.display(), format_size(plan.size_mib));
    if plan.loaded {
        println!("  it is loaded right now, and will be stopped first");
    }
    if plan.is_default {
        println!("  it is the default model, which will be cleared");
    }

    // A re-download is 17 GB over a link that is not always fast, so the default is to
    // ask. Without a terminal there is nobody to ask, and guessing yes on the strength
    // of an empty stdin is not a thing to do to a model directory.
    if !args.yes {
        anyhow::ensure!(
            ailocal::prompt::interactive(),
            "refusing to delete {name:?} without confirmation; pass --yes"
        );
        if !ailocal::prompt::confirm("Delete it?", false)? {
            println!("left alone");
            return Ok(());
        }
    }

    let removed = plan.remove(&cfg, &unit_binary)?;
    if removed.stopped {
        println!("stopped {name}");
    }
    println!(
        "removed {}, freeing {}",
        plan.path.display(),
        format_size(plan.size_mib)
    );

    match &removed.default {
        None => {}
        Some(Ok(repoint)) => {
            println!("cleared default_model");
            print_repoint(repoint);
        }
        Some(Err(e)) => eprintln!("warning: could not clear default_model: {e:#}"),
    }
    match &removed.pi {
        Ok(true) => println!("removed pi's entry for it"),
        Ok(false) => {}
        Err(e) => eprintln!("warning: could not update pi's models.json: {e:#}"),
    }
    // Claude Code's env file pins exactly one model, and it is sourced by hand rather
    // than read through us, so it cannot be quietly repointed - say what to run.
    if removed.claude_code_pinned {
        println!(
            "claude-code is still pointed at {name} - rerun\n\
             \x20 `ailocal harness configure claude-code` to move it"
        );
    }
    Ok(())
}

fn serve_model(args: &ServeArgs) -> anyhow::Result<()> {
    let cfg = Config::load()?;

    // Resolve the context before announcing anything, so a refusal is not preceded by
    // a line claiming we started.
    let prepared = serve::prepare(
        &cfg,
        &serve::Launch {
            name: args.name.clone(),
            context: args.ctx,
            host: args.host.clone(),
            port: args.port,
            reasoning: args.reasoning.clone(),
        },
    )?;

    if let Some(old) = &prepared.evicting {
        println!("evicting {old} (only one model fits at a time)");
    }
    println!(
        "starting {} with {} context ...",
        prepared.model.name,
        format_count(prepared.context)
    );

    if args.foreground {
        return serve::run_foreground(&prepared.model, &prepared.budget, &prepared.opts);
    }

    let instance = prepared.start()?;
    println!(
        "{} up at {} with {} context, {} KV, reasoning {}",
        instance.model,
        instance.base_url(),
        format_count(instance.context),
        instance.cache_type,
        prepared.opts.reasoning
    );
    let (label, mib) = ailocal::server_memory_mib(instance.pid)?;
    println!("{label} now {mib} MiB");
    Ok(())
}

fn harness_configure(name: &str, url: &str, wanted: Option<&str>) -> anyhow::Result<()> {
    // Planned before the gateway offer so a typo in --model refuses outright, rather
    // than first installing a service and only then refusing.
    setup::plan_harness(&Config::load()?, name, url, wanted)?;

    // A harness stores this URL and only finds out it is wrong when a prompt fails
    // with a bare "Connection error", which says nothing about the port. Far better to
    // say it here, while there is still context - and better still to offer the fix.
    let cfg = offer_to_start_the_gateway(Config::load()?)?;
    let done = setup::configure_harness(&cfg, name, url, wanted)?;
    let plan = &done.plan;

    if let Some(trouble) = &plan.trouble {
        print_trouble(trouble, url);
        println!("  Configuring anyway - the harness will work once the gateway is up.\n");
    }

    if name == "claude-code" {
        let path = done.path.clone().unwrap_or_default();
        if done.unchanged {
            println!("claude-code env already current");
        } else {
            println!("wrote {}", path.display());
        }
        // Claude Code takes exactly one model, so which one it got is the single most
        // useful thing to say - and the thing that used to be decided invisibly.
        println!(
            "  model: {} ({} context) - {}",
            plan.model,
            format_count(plan.context),
            plan.why
        );
        if !plan.others.is_empty() {
            println!("  others installed: {}", plan.others.join(", "));
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

    if done.unchanged {
        println!("pi already points at {url}");
    } else {
        println!("pi configured to use {url}");
        if let Some(b) = &done.backup {
            println!("  backup: {}", b.display());
        }
    }
    for m in ailocal::status::runnable_models(&cfg)? {
        println!(
            "  {} ({} context, thinking {})",
            m.id,
            format_count(m.context_window),
            m.thinking.label()
        );
    }
    if cfg.agent_rules {
        println!("  extension: {}", harness::pi_extension_path()?.display());
        println!("    holds a local model's untested push; `ailocal config agent-rules` says more");
    }
    // Said here because the alternative is finding out from `/effort`, which reports
    // it as the harness refusing a level rather than the model not having one.
    if let Some(id) = plan.without_thinking.first() {
        println!(
            "  {id} has no thinking mode - its chat template has no `enable_thinking`,\n\
             \x20 so `/thinking` and `/effort` in pi can only ever offer `off` for it."
        );
    }
    println!(
        "  try: pi --provider {} --model {}",
        harness::PI_PROVIDER_ID,
        plan.model
    );
    Ok(())
}

/// Make sure there is a gateway before pointing a harness at one, offering to start it.
///
/// This is the moment the whole flow used to fall apart: the harness gets configured
/// against a URL nothing answers, and the first symptom is "Connection error" from
/// inside Pi, which mentions neither the gateway nor the port. Since the fix is one
/// command, ask rather than describe.
///
/// Returns the config, reloaded if starting the service changed it.
fn offer_to_start_the_gateway(cfg: Config) -> anyhow::Result<Config> {
    if setup::trouble(&cfg).is_none() {
        return Ok(cfg);
    }

    let installed = service::is_enabled(service::GATEWAY) || service::is_active(service::GATEWAY);
    if !installed && ailocal::prompt::interactive() {
        println!("The gateway is not running, so a harness pointed at it cannot answer.");
        if ailocal::prompt::confirm("Install and start it now?", true)? {
            service_install(false)?;
            println!();
            return Config::load();
        }
        println!();
    }
    Ok(cfg)
}

/// Say so, in terms that name the cause, when the gateway is not going to answer.
///
/// Not a refusal: writing a harness config is still the right thing when the gateway is
/// merely not started yet, and `setup` configures harnesses as part of a sequence that
/// brings everything up.
fn warn_if_gateway_is_not_answering(cfg: &Config, url: &str) {
    if let Some(trouble) = setup::trouble(cfg) {
        print_trouble(&trouble, url);
    }
}

fn print_trouble(trouble: &setup::Trouble, url: &str) {
    println!("WARNING: nothing is answering as a gateway on {url}");
    match trouble {
        setup::Trouble::PortTaken { free } => println!(
            "  That port is held by something that is not ailocal, so the gateway\n\
             \x20 cannot start. Move it with `ailocal config gateway-port {free}` and\n\
             \x20 re-run this."
        ),
        setup::Trouble::WrongPort { expected } => println!(
            "  {} is running, but not on port {expected}. If you changed the port, the\n\
             \x20 unit still has the old one - `ailocal service install` rewrites it.\n\
             \x20 Otherwise it may just be starting; `ailocal gateway check` will say.",
            service::GATEWAY
        ),
        setup::Trouble::NotRunning => println!(
            "  The gateway is not running. Start it with `ailocal service install`,\n\
             \x20 or `ailocal gateway run` in another terminal."
        ),
    }
}

fn harness_unconfigure(name: &str) -> anyhow::Result<()> {
    let was = setup::unconfigure_harness(name)?;
    match (name, was) {
        ("claude-code", true) => println!("removed {}", harness::claude_code_env_path()?.display()),
        ("claude-code", false) => println!("claude-code was not configured"),
        (_, true) => println!("removed the {} entry from pi", harness::PI_PROVIDER_ID),
        (_, false) => println!("pi was not configured"),
    }
    Ok(())
}

fn lessons_ls() -> anyhow::Result<()> {
    use lessons::State;

    let path = lessons::path()?;
    let store = lessons::Store::load(&path)?;
    if store.lessons.is_empty() {
        println!("no lessons yet, in {}", path.display());
        println!(
            "\nAdd one of your own with\n  \
             ailocal lessons add \"<what to do>\" [--command \"gh pr create\"] [--adds eslint-disable]"
        );
        return Ok(());
    }

    let mut all: Vec<_> = store.lessons.iter().collect();
    all.sort_by_key(|l| match l.state {
        State::Active => 0,
        State::Pending => 1,
        State::Rejected => 2,
    });
    for lesson in &all {
        let mut head = describe(lesson);
        if lesson.state != State::Active {
            head.push_str(&format!("  {}", lesson.state.as_str()));
        }
        println!("{head}\n  {}\n", lesson.text);
    }
    let active = all.iter().filter(|l| l.state == State::Active).count();
    println!("{active} of {} active, in {}", all.len(), path.display());
    Ok(())
}

fn lessons_add(args: LessonAddArgs) -> anyhow::Result<()> {
    let repo = if args.here {
        Some(repo_here()?)
    } else {
        args.repo
            .map(|r| lessons::repo_id(&r).unwrap_or_else(|| r.trim().trim_matches('/').to_owned()))
    };
    let path = lessons::path()?;
    let mut store = lessons::Store::load(&path)?;
    let id = store.add(lessons::Lesson {
        id: String::new(),
        text: args.text.join(" "),
        repo,
        commands: args.commands,
        adds: args.adds,
        files: args.files,
        state: lessons::State::Active,
        source: "added by hand".to_owned(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
        corrections: Vec::new(),
    })?;
    store.save(&path)?;

    println!("added {id}");
    if harness::pi_extension_path().is_ok_and(|p| p.is_file()) {
        println!("Local models in Pi get it from their next prompt.");
    } else if !Config::load()?.agent_rules {
        println!(
            "No model gets it yet: Pi's ailocal extension is off with the agent rules.\n\
             `ailocal config agent-rules on` turns both on."
        );
    } else {
        println!(
            "No model gets it yet: Pi's ailocal extension is not installed.\n\
             `ailocal harness configure pi` installs it."
        );
    }
    Ok(())
}

/// The repository of the current directory, as lessons name it.
fn repo_here() -> anyhow::Result<String> {
    let out = std::process::Command::new("git")
        .args(["config", "--get", "remote.origin.url"])
        .output()
        .map_err(|e| anyhow::anyhow!("running git: {e}"))?;
    anyhow::ensure!(
        out.status.success(),
        "the current directory is not in a repository with a remote named origin"
    );
    let remote = String::from_utf8_lossy(&out.stdout);
    lessons::repo_id(&remote)
        .ok_or_else(|| anyhow::anyhow!("origin is {}, which names no host", remote.trim()))
}

fn lessons_rm(id: &str) -> anyhow::Result<()> {
    let path = lessons::path()?;
    let mut store = lessons::Store::load(&path)?;
    let lesson = store.remove(id)?;
    store.save(&path)?;
    println!("removed {}\n  {}", lesson.id, lesson.text);
    Ok(())
}

fn lessons_capture(model: Option<String>) -> anyhow::Result<()> {
    use capture::Outcome;
    use std::io::Write as _;

    let distiller = match model {
        Some(model) => distill::Distiller::Pi {
            pi: harness::find_pi()?,
            model,
        },
        None => {
            let instance = serve::running()?.ok_or_else(|| {
                anyhow::anyhow!(
                    "no model is running to learn with; start one with `ailocal serve <name>`, \
                     or name one of Pi's with --model <provider/id>"
                )
            })?;
            distill::Distiller::Local {
                base: instance.base_url(),
                model: instance.model,
            }
        }
    };
    let (path, progress) = (lessons::path()?, capture::Progress::path()?);
    let files = capture::session_files()?;
    if !progress.exists() {
        let name = distiller.name();
        match files.len() {
            0 => println!("Pi has kept no sessions to learn from yet."),
            1 => println!("Reading Pi's one session for the first time, with {name}."),
            n => println!("Reading all {n} of Pi's sessions for the first time, with {name}."),
        }
    }

    let (mut read, mut learned, mut repeated) = (0, 0, 0);
    let counting = std::io::IsTerminal::is_terminal(&std::io::stderr());
    capture::pass(
        &distiller,
        &files,
        &path,
        &progress,
        || true,
        |candidate, outcome| {
            read += 1;
            if counting {
                eprint!("\r\x1b[K");
            }
            match outcome {
                Outcome::Learned(id) => {
                    learned += 1;
                    println!("learned  {id}");
                }
                Outcome::Repeated(id) => {
                    repeated += 1;
                    println!("again    {id}");
                }
                Outcome::Nothing => {}
                Outcome::Failed(e) => eprintln!(
                    "could not read {} in {}: {e:#}",
                    candidate.entry,
                    candidate.session.display()
                ),
            }
            if counting {
                eprint!("{read} of your messages read");
                std::io::stderr().flush().ok();
            }
        },
    )?;
    if counting {
        eprint!("\r\x1b[K");
    }

    let noun = |n| if n == 1 { "lesson" } else { "lessons" };
    println!(
        "{read} of your messages read: {learned} new {}, {repeated} taught again",
        noun(learned)
    );
    let pending = lessons::Store::load(&path)?
        .lessons
        .iter()
        .filter(|l| l.state == lessons::State::Pending)
        .count();
    if pending > 0 {
        let wait = if pending == 1 { "waits" } else { "wait" };
        println!(
            "{pending} {} {wait} for you: `ailocal lessons review`",
            noun(pending)
        );
    }
    Ok(())
}

fn lessons_review() -> anyhow::Result<()> {
    use lessons::State;

    let path = lessons::path()?;
    let mut queue: Vec<lessons::Lesson> = lessons::Store::load(&path)?
        .lessons
        .into_iter()
        .filter(|l| l.state == State::Pending)
        .collect();
    if queue.is_empty() {
        println!("no lessons wait for review");
        return Ok(());
    }
    // The most corrected first: those are the ones that cost the most steering.
    queue.sort_by_key(|l| std::cmp::Reverse(l.corrections.len()));

    let total = queue.len();
    for (n, lesson) in queue.into_iter().enumerate() {
        println!("\n[{}/{total}] {}", n + 1, describe(&lesson));
        println!("  {}", lesson.text);
        for c in lesson.corrections.iter().rev().take(3) {
            println!("    {}  \"{}\"", c.at.get(..10).unwrap_or(&c.at), c.said);
        }

        let mut options = vec!["Accept".to_owned()];
        if lesson.repo.is_some() {
            options.push("Accept for every repository".to_owned());
        }
        if !(lesson.commands.is_empty() && lesson.adds.is_empty() && lesson.files.is_empty()) {
            options.push("Accept without its holds".to_owned());
        }
        options.extend(["Edit, then accept", "Reject", "Skip"].map(str::to_owned));
        let Some(choice) = ailocal::prompt::choose("What should happen to it?", &options)? else {
            break;
        };

        let mut store = lessons::Store::load(&path)?;
        let Ok(kept) = store.lesson_mut(&lesson.id) else {
            continue;
        };
        match options[choice].as_str() {
            "Accept" => kept.state = State::Active,
            "Accept for every repository" => {
                kept.repo = None;
                kept.state = State::Active;
            }
            "Accept without its holds" => {
                kept.commands.clear();
                kept.adds.clear();
                kept.files.clear();
                kept.state = State::Active;
            }
            "Edit, then accept" => {
                let text = ailocal::prompt::ask("The lesson", &kept.text)?;
                kept.text = text.trim().to_owned();
                kept.state = State::Active;
            }
            "Reject" => kept.state = State::Rejected,
            _ => continue,
        }
        println!("  {}", kept.state.as_str());
        store.save(&path)?;
    }
    Ok(())
}

/// Where a lesson applies and what it holds, on one line.
fn describe(lesson: &lessons::Lesson) -> String {
    let mut head = vec![
        lesson.id.clone(),
        lesson
            .repo
            .clone()
            .unwrap_or_else(|| "everywhere".to_owned()),
    ];
    let holds: Vec<String> = lesson
        .commands
        .iter()
        .map(|c| format!("`{c}`"))
        .chain(lesson.adds.iter().map(|a| format!("edits adding `{a}`")))
        .chain(lesson.files.iter().map(|f| format!("edits to `{f}`")))
        .collect();
    if !holds.is_empty() {
        head.push(format!("holds {}", holds.join(", ")));
    }
    if !lesson.corrections.is_empty() {
        head.push(format!("taught {}x", lesson.corrections.len()));
    }
    head.join("  ")
}

fn lessons_set(ids: &[String], state: lessons::State) -> anyhow::Result<()> {
    let path = lessons::path()?;
    let mut store = lessons::Store::load(&path)?;
    for id in ids {
        store.lesson_mut(id)?.state = state;
    }
    store.save(&path)?;
    for id in ids {
        println!("{}  {id}", state.as_str());
    }
    Ok(())
}

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
        let answer = ailocal::prompt::ask_path("   Data directory", &default)?;
        cfg.set_data_dir(std::path::PathBuf::from(answer));
    }

    println!("   wrote {}", cfg.save()?.display());
    Ok(cfg)
}

/// Make sure the gateway has a port it can actually bind before anything is built on
/// top of it.
///
/// 8081 is a popular default - Metro uses it, among others - and a machine where
/// something else already owns it used to fail invisibly: the service manager reported
/// the agent started, the harness was configured against a port nothing was listening
/// on, and the only symptom was a bare "Connection error" the first time a prompt was
/// sent. Moving aside automatically is better than that, and saying so is the whole
/// point.
fn settle_gateway_port(args: &SetupArgs) -> anyhow::Result<()> {
    let mut cfg = Config::load()?;
    let wanted = args.port.unwrap_or(cfg.gateway_port);

    println!("\n3. gateway port");
    let port = match gateway::port_status(&cfg.gateway_host, wanted) {
        gateway::PortStatus::Free => wanted,
        gateway::PortStatus::Ours => {
            println!("   ok    {wanted}, where a gateway of ours is already listening");
            wanted
        }
        gateway::PortStatus::Taken => {
            // An explicitly requested port is a decision, so report it rather than
            // quietly using a different one.
            anyhow::ensure!(
                args.port.is_none(),
                "port {wanted} is in use by something else; pick another with --port"
            );
            let free = gateway::usable_port(&cfg.gateway_host, wanted).ok_or_else(|| {
                anyhow::anyhow!(
                    "port {wanted} is in use and so are the next {} - free one, or pass \
                     --port",
                    20
                )
            })?;
            println!("   busy  {wanted} is held by something else, moving to {free}");
            free
        }
    };

    if port != cfg.gateway_port {
        cfg.gateway_port = port;
        cfg.save()?;
    }
    if port == wanted && gateway::port_status(&cfg.gateway_host, port) == gateway::PortStatus::Free
    {
        println!("   ok    {port} is free");
    }
    Ok(())
}

/// Bring a fresh machine up: check what is needed, then do the rest.
fn setup(args: &SetupArgs) -> anyhow::Result<()> {
    let mut blocked = false;

    println!("1. prerequisites");
    // The same resolver the rest of the process uses, or setup passes on a machine
    // where every later command fails.
    match ailocal::llama_server() {
        Ok(p) => println!("   ok    llama-server at {}", p.display()),
        Err(_) => {
            blocked = true;
            println!("   MISS  llama-server not found");
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

    settle_gateway_port(args)?;

    println!("\n4. models");
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
        println!("\n5. services");
        service_install(false)?;
    }

    // Re-read: the port may have moved at step 3, and the model at step 4.
    let cfg = Config::load()?;
    let url = local_gateway_url()?;

    if !args.skip_harness {
        println!("\n6. harnesses");
        for harness in ["pi", "claude-code"] {
            match harness_configure(harness, &url, None) {
                Ok(()) => {}
                // A harness that is not installed is not a failure of setup.
                Err(e) => println!("   skip  {harness}: {e}"),
            }
        }
    }

    if args.skip_service {
        println!("\nReady. Start the gateway with `ailocal gateway run`, then check it");
        println!("with `ailocal gateway check`.");
        return Ok(());
    }

    // The step that turns "everything reported success" into "it actually works".
    // Without it a fresh machine could finish setup with a dead gateway and only find
    // out when a harness answered a prompt with "Connection error".
    println!("\n7. checking it works");
    if !gateway::wait_until_answering(
        &cfg.gateway_host,
        cfg.gateway_port,
        std::time::Duration::from_secs(20),
    ) {
        println!("   FAIL  no gateway answering on {url}");
        println!("         logs: {}", service::log_hint(service::GATEWAY));
        anyhow::bail!(
            "the gateway did not come up. Everything else is configured, so fixing it \
             and re-running `ailocal setup` will finish the job."
        );
    }
    gateway_check(&url)?;

    println!("\nReady. `ailocal ps` shows what is loaded.");
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
    let unit = ailocal::update::unit_binary(&current)?;
    if unit != current {
        eprintln!(
            "note: running from a build directory; units will point at {}",
            unit.display()
        );
    }
    Ok(unit)
}

fn gateway_check(url: &str) -> anyhow::Result<()> {
    let key = auth::load_or_create()?;
    let check = gateway::check(url, &key)?;

    println!("checking {}", check.base);
    if !check.reachable {
        println!("  FAIL  nothing is listening on {}", check.base);
        println!();
        warn_if_gateway_is_not_answering(&Config::load()?, &check.base);
        anyhow::bail!("the gateway is not reachable");
    }

    for probe in &check.probes {
        match &probe.got {
            Ok(status) if probe.passed() => println!("  ok    {} ({status})", probe.label),
            Ok(status) => println!(
                "  FAIL  {}: got {status}, expected {}",
                probe.label, probe.expected
            ),
            Err(e) => println!("  FAIL  {}: {e:#}", probe.label),
        }
    }

    let failures = check.failures();
    anyhow::ensure!(failures == 0, "{failures} check(s) failed");
    println!("all checks passed");
    Ok(())
}

fn service_install(no_start: bool) -> anyhow::Result<()> {
    let installed = setup::install_services(&unit_binary()?, !no_start)?;

    if let Some(model) = &installed.chose_default {
        println!("default model set to {model} so it is loaded before you need it");
    }
    if installed.gateway_only {
        println!(
            "note: no installed model can run here, so only the gateway is installed.\n\
             \x20     Nothing will be resident, and the first request will wait for a\n\
             \x20     model to load. `ailocal model pick` fixes that."
        );
    }

    println!(
        "wrote {} into {}",
        installed.units.join(", "),
        installed.unit_dir.display()
    );
    println!("  ExecStart uses {}", installed.exe.display());

    match installed.answering {
        None => {}
        Some(true) => {
            for unit in &installed.units {
                println!("  enabled and started {unit}");
            }
            let cfg = Config::load()?;
            println!(
                "  answering on http://{}:{}",
                gateway::loopback_for(&cfg.gateway_host),
                cfg.gateway_port
            );
        }
        Some(false) => {
            let cfg = Config::load()?;
            println!(
                "\nWARNING: {} started but nothing is answering on port {}.\n\
                 \x20        {}",
                service::GATEWAY,
                cfg.gateway_port,
                service::log_hint(service::GATEWAY)
            );
        }
    }

    if !installed.linger {
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
    let removed = setup::remove_services()?;
    if removed.is_empty() {
        println!("no units were installed");
    } else {
        println!("removed {}", removed.join(", "));
    }
    Ok(())
}

fn service_status() -> anyhow::Result<()> {
    let services = setup::services();
    println!(
        "linger      {}",
        if services.linger {
            "enabled (starts at boot, survives logout)"
        } else {
            "disabled (units start at login only)"
        }
    );
    for unit in &services.units {
        println!(
            "{:<40} {:<10} {}",
            unit.name,
            if unit.active { "active" } else { "inactive" },
            if unit.enabled { "enabled" } else { "disabled" }
        );
    }
    Ok(())
}

fn gateway_run(host: &str, port: u16, log_requests: bool) -> anyhow::Result<()> {
    // Checked before the runtime starts, so the failure is a sentence rather than
    // "Address already in use (os error 48)" in a log file under ~/Library/Logs that
    // nobody thinks to open. 8081 is a busy port - Metro, among others, defaults to it.
    match gateway::port_status(host, port) {
        gateway::PortStatus::Free => {}
        gateway::PortStatus::Ours => anyhow::bail!(
            "an ailocal gateway is already listening on {host}:{port}.\n\
             Use it as it is, or stop it first ({}).",
            if service::is_active(service::GATEWAY) {
                "it is running as a service"
            } else {
                "it was started by hand"
            }
        ),
        gateway::PortStatus::Taken => anyhow::bail!(
            "port {port} is already in use by something else, so the gateway cannot \
             start.\n\
             Free it, or move the gateway:\n\
             \x20   ailocal config gateway-port {}\n\
             That updates the service too; re-run `ailocal harness configure` \
             afterwards so your harnesses follow.",
            gateway::usable_port(host, port).unwrap_or(port + 1)
        ),
    }

    let config = ailocal::config::Live::follow(Config::path()?)?;
    let request_log = log_requests.then(|| config.get().data_dir.join("gateway-requests.jsonl"));
    if let Some(path) = &request_log {
        println!("  recording requests to {}", path.display());
    }

    // `ailocal update` restarts the gateway, so this is where a new release's Pi
    // extension reaches a Pi that an older one configured. A failure here must not keep
    // the gateway down.
    match harness::sync_pi_extension(config.get().agent_rules) {
        Ok(true) => println!("  brought the Pi extension up to date"),
        Ok(false) => {}
        Err(e) => eprintln!("  could not update the Pi extension: {e:#}"),
    }

    let state = Arc::new(gateway::AppState {
        key: auth::load_or_create()?,
        request_log,
        config,
        client: gateway::upstream_client()?,
    });

    // Lessons are for the Pi extension to give, so they are learned while it is there.
    let watched = Arc::clone(&state);
    std::thread::spawn(move || {
        capture::keep_capturing(|| {
            watched.config.get().agent_rules && harness::is_configured("pi")
        });
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
            if let Some(slots) = i.slots {
                println!("{:<12} {slots}, each with the whole context", "slots");
            }
            println!("{:<12} {}", "kv cache", i.cache_type);
            // The one allocation that grows after the launch, so it is worth seeing
            // next to the context that sizes it.
            if let Some(mib) = i.cache_ram_mib {
                println!("{:<12} {mib} MiB", "prompt cache");
            }
            println!("{:<12} {}", "reasoning", i.reasoning);
            match &i.sampling {
                Some(s) => println!("{:<12} {s} (from the publisher)", "sampling"),
                None => println!(
                    "{:<12} the model file's, else llama.cpp's defaults",
                    "sampling"
                ),
            }
            println!("{:<12} {}", "pid", i.pid);
            let (label, mib) = ailocal::server_memory_mib(i.pid)?;
            println!("{label:<12} {mib} MiB");
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

/// Render a file size for a human about to decide whether to delete it.
///
/// Whole GiB above a gigabyte, MiB below it. Integer division alone renders a 400 MiB
/// file as `0 GiB`, which is the wrong impression to give at a confirmation prompt.
fn format_size(size_mib: u64) -> String {
    if size_mib >= 1024 {
        format!("{:.1} GiB", size_mib as f64 / 1024.0)
    } else {
        format!("{size_mib} MiB")
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_owned();
    }
    let keep: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{keep}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the command says when the registry cannot be read. A failed read once let
    /// the installer download 1801 MiB of qwen3-vl:2b, which llama.cpp refuses, so it
    /// stops and names the override; with `--force` it goes on.
    #[test]
    fn a_header_that_cannot_be_read_stops_the_install() {
        let unreadable = || install::Check {
            header: install::Header::Unreadable(anyhow::anyhow!("connection refused")),
            fit: Fit::Unknown,
            weights_mib: 1801,
            available_mib: 1013,
        };

        let err = refuse_unless_forced(unreadable(), false).unwrap_err();
        assert!(
            format!("{err:#}").contains("--force"),
            "unexpected: {err:#}"
        );
        refuse_unless_forced(unreadable(), true).unwrap();
    }

    /// devstral-small-2:24b packs 222 encoder tensors among its 585; it is refused
    /// even though its weights fit, and `--force` overrides that like any other refusal.
    #[test]
    fn a_packed_encoder_is_refused_unless_forced() {
        let packed = || install::Check {
            header: install::Header::Read {
                arch: Some("mistral3".into()),
                trained_context: Some(131_072),
                encoder_tensors: Some(222),
                tensor_count: 585,
            },
            fit: Fit::Fits(131_072),
            weights_mib: 14_336,
            available_mib: 39_312,
        };
        let err = refuse_unless_forced(packed(), false).unwrap_err();
        assert!(err.to_string().contains("--force"), "unexpected: {err}");
        refuse_unless_forced(packed(), true).unwrap();
    }

    /// A 400 MiB file must not read as `0 GiB` at a prompt asking to delete it.
    #[test]
    fn sizes_below_a_gigabyte_are_reported_in_mib() {
        assert_eq!(format_size(400), "400 MiB");
        assert_eq!(format_size(0), "0 MiB");
        assert_eq!(format_size(1024), "1.0 GiB");
        assert_eq!(format_size(17_699), "17.3 GiB");
    }

    /// clap's own consistency check, so a duplicated flag fails the build rather than
    /// the first run.
    #[test]
    fn the_cli_is_well_formed() {
        use clap::CommandFactory as _;
        Cli::command().debug_assert();
    }
}
