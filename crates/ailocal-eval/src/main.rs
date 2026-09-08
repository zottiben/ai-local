//! `ailocal eval` - score local models on a coding task set.
//!
//! An optional companion to `ailocal`, installed with `ailocal extras install eval`.
//! It exists because the choices this project has already made - reasoning off by
//! default, which models are worth listing, whether a heavily quantised larger model
//! beats a clean smaller one - were made on judgement rather than measurement, and
//! because no adapter should be trained until there is something for it to beat.
//!
//! An arm is one model in one reasoning mode. `run` measures one or several, saves a
//! report each, and prints them side by side.

mod check;
mod report;
mod run;
mod task;

use std::path::{Path, PathBuf};

use ailocal::config::Config;
use clap::{Args, Parser, Subcommand};

use report::Report;
use task::{Split, Task};

#[derive(Parser)]
#[command(name = "ailocal-eval", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Measure one or more arms and save a report for each.
    Run(RunArgs),
    /// List saved runs.
    Ls,
    /// Show one run task by task.
    Show {
        run: String,
        /// Show everything the model said for one task, rather than the summary.
        ///
        /// This is how you tell a model that got it wrong from a task whose wording
        /// was ambiguous - which need opposite responses and are indistinguishable
        /// from a score alone.
        #[arg(long = "task")]
        only: Option<String>,
    },
    /// Put two runs side by side.
    Compare { a: String, b: String },
    /// Print the task set without running anything.
    Tasks {
        /// Task file to read instead of the built-in suite.
        #[arg(long)]
        tasks: Option<PathBuf>,
    },
}

#[derive(Args)]
struct RunArgs {
    /// Model to measure, as named by `ailocal model ls`. Repeat to compare several.
    ///
    /// Defaults to whichever model is loaded, then to the configured default.
    #[arg(long)]
    model: Vec<String>,

    /// Reasoning mode to measure: `off`, `on` or `auto`. Repeat to compare several.
    ///
    /// Each mode is a separate arm, because llama-server takes it at launch - so
    /// comparing two means reloading the model between them.
    #[arg(long)]
    reasoning: Vec<String>,

    /// Task file to use instead of the built-in suite.
    #[arg(long)]
    tasks: Option<PathBuf>,

    /// Only tasks in this split: `dev` or `held-out`.
    #[arg(long)]
    split: Option<Split>,

    /// Only these task ids. Repeatable.
    #[arg(long = "task")]
    only: Vec<String>,

    /// Stop after this many tasks, for a quick smoke test.
    #[arg(long)]
    limit: Option<usize>,

    /// Compile the model's code but do not run it.
    ///
    /// `rust_test` checks then report as unjudged rather than passing or failing, and
    /// the score is computed over what remains.
    #[arg(long)]
    no_exec: bool,

    #[arg(long, default_value_t = run::DEFAULT_SEED)]
    seed: i64,

    /// Token budget for every task, overriding the per-task one in the corpus.
    ///
    /// Reasoning arms need this. A model that thinks for a thousand tokens before
    /// writing anything returns an empty answer on the default budget, and "was cut
    /// off" is a different finding from "is worse" - so measure both.
    #[arg(long)]
    max_tokens: Option<u32>,

    /// Cap thinking at this many tokens, forcing the model to answer.
    ///
    /// The setting between `on` and `off`: llama.cpp closes the thought once the cap
    /// is reached. Worth measuring before concluding that reasoning cannot pay - a
    /// model that never stops thinking and one that reasons badly look identical from
    /// the outside, and only this tells them apart.
    #[arg(long)]
    reasoning_budget: Option<i64>,

    /// Print the reports as JSON instead of tables.
    #[arg(long)]
    json: bool,
}

fn main() -> anyhow::Result<()> {
    quit_quietly_on_broken_pipe();

    match Cli::parse().command {
        Command::Run(args) => run_arms(&args),
        Command::Ls => list_runs(),
        Command::Show { run, only } => show_run(&run, only.as_deref()),
        Command::Compare { a, b } => compare_runs(&a, &b),
        Command::Tasks { tasks } => show_tasks(tasks.as_deref()),
    }
}

/// Where compiled answers go while they are being judged.
///
/// Under the configured eval directory, which is on the big disk - a run compiles a
/// test binary per task and hard rule 1 is that nothing sizeable lands on `/` or
/// `/home`.
fn scratch_dir(config: &Config) -> PathBuf {
    config.eval_dir.join("scratch")
}

fn run_arms(args: &RunArgs) -> anyhow::Result<()> {
    let config = Config::load()?;
    let corpus = task::load(args.tasks.as_deref())?;
    let selected = select(&corpus.tasks, args)?;

    let models = resolve_models(&config, args)?;
    let reasonings = if args.reasoning.is_empty() {
        vec![config.reasoning.clone()]
    } else {
        args.reasoning.clone()
    };
    for mode in &reasonings {
        anyhow::ensure!(
            matches!(mode.as_str(), "off" | "on" | "auto"),
            "unknown reasoning mode {mode:?}; expected `off`, `on` or `auto`"
        );
    }

    if args.no_exec {
        eprintln!("--no-exec: code will be compiled but not run.\n");
    } else {
        // Worth stating plainly every time rather than only in the docs: this compiles
        // and executes code a language model wrote, with this user's privileges.
        eprintln!(
            "This runs code the model writes, under a {}s timeout and no sandbox.\n\
             Use --no-exec to compile only.\n",
            10
        );
    }

    eprintln!(
        "{} task(s) from {} ({}), {} arm(s)\n",
        selected.len(),
        corpus.origin,
        corpus.digest,
        models.len() * reasonings.len()
    );

    // Taken before anything reads the loaded-model state, so that nothing can change it
    // underneath the run. See ModelServiceGuard for why the reverse order is a race.
    let _guard = run::ModelServiceGuard::take()?;

    let mut reports = Vec::new();
    // Model-major so a model is loaded once and then only reloaded for each reasoning
    // mode, rather than swapping weights on every arm.
    for model in &models {
        for mode in &reasonings {
            let plan = run::Plan {
                model: model.clone(),
                reasoning: mode.clone(),
                seed: args.seed,
                allow_exec: !args.no_exec,
                max_tokens: args.max_tokens,
                reasoning_budget: args.reasoning_budget,
            };
            eprintln!("== {model}, reasoning {mode}");
            let report = run::arm(&config, &plan, &corpus, &selected)?;
            let path = report.save(&config.eval_dir)?;
            eprintln!("   saved {}\n", path.display());
            reports.push(report);
        }
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&reports)?);
        return Ok(());
    }

    print!(
        "{}",
        report::summary_table(&reports.iter().collect::<Vec<_>>())
    );
    if let [only] = reports.as_slice() {
        println!();
        print!("{}", report::detail_table(only));
    }
    println!("\n`ailocal eval show <run>` for detail, `ailocal eval compare <a> <b>` for two.");
    Ok(())
}

/// Which models to measure, and a useful error when there is no obvious one.
fn resolve_models(config: &Config, args: &RunArgs) -> anyhow::Result<Vec<String>> {
    if !args.model.is_empty() {
        return Ok(args.model.clone());
    }

    // Whatever is loaded is almost always what you meant, and measuring it costs no
    // reload.
    if let Some(instance) = ailocal::serve::running()? {
        return Ok(vec![instance.model]);
    }
    if let Some(default) = &config.default_model {
        return Ok(vec![default.clone()]);
    }

    let installed = ailocal::registry::scan(&config.models_dir)?;
    anyhow::bail!(
        "no model is loaded and none is configured as the default, so there is nothing \
         to measure. Pass --model <name>; installed: {}",
        if installed.is_empty() {
            "none".to_owned()
        } else {
            installed
                .iter()
                .map(|m| m.name.clone())
                .collect::<Vec<_>>()
                .join(", ")
        }
    )
}

/// Apply `--split`, `--task` and `--limit`, in that order.
fn select<'a>(tasks: &'a [Task], args: &RunArgs) -> anyhow::Result<Vec<&'a Task>> {
    let mut chosen: Vec<&Task> = tasks
        .iter()
        .filter(|t| args.split.is_none_or(|s| t.split == s))
        .filter(|t| args.only.is_empty() || args.only.contains(&t.id))
        .collect();

    // A typo in --task would otherwise silently measure nothing, or worse, measure the
    // rest of the suite and look like a valid result.
    for wanted in &args.only {
        anyhow::ensure!(
            tasks.iter().any(|t| &t.id == wanted),
            "no task with id {wanted:?}; `ailocal eval tasks` lists them"
        );
    }

    if let Some(limit) = args.limit {
        chosen.truncate(limit);
    }
    anyhow::ensure!(!chosen.is_empty(), "the filters selected no tasks");
    Ok(chosen)
}

fn list_runs() -> anyhow::Result<()> {
    let config = Config::load()?;
    let paths = report::list(&config.eval_dir)?;
    if paths.is_empty() {
        println!(
            "no runs yet in {}",
            report::runs_dir(&config.eval_dir).display()
        );
        return Ok(());
    }

    println!(
        "{:<38} {:>6} {:>6} {:>7}  ARM",
        "RUN", "SCORE", "PASS", "TASKS"
    );
    for path in &paths {
        // A report that will not parse should not hide the ones that will.
        let Ok(r) = Report::read(path) else {
            println!(
                "{:<38} {:>6} {:>6} {:>7}  unreadable",
                stem(path),
                "-",
                "-",
                "-"
            );
            continue;
        };
        let s = report::Summary::of(&r);
        println!(
            "{:<38} {:>5.0}% {:>5.0}% {:>7}  {}",
            r.id,
            s.score * 100.0,
            s.pass_rate * 100.0,
            s.scored,
            r.arm.label()
        );
    }
    Ok(())
}

fn show_run(needle: &str, only: Option<&str>) -> anyhow::Result<()> {
    let config = Config::load()?;
    let r = report::resolve(&config.eval_dir, needle)?;

    if let Some(id) = only {
        return show_task(&r, id);
    }

    let s = report::Summary::of(&r);

    println!("{}", r.id);
    println!("  model      {}", r.arm.model);
    println!("  reasoning  {}", r.arm.reasoning);
    println!("  context    {}", r.arm.context);
    println!("  kv cache   {}", r.arm.cache_type);
    println!(
        "  sampling   temperature {}, seed {}",
        r.arm.temperature, r.arm.seed
    );
    println!(
        "  code       {}",
        if r.arm.exec {
            "compiled and run"
        } else {
            "compiled only"
        }
    );
    println!("  tasks      {} ({})", r.corpus.origin, r.corpus.digest);
    println!("  started    {}", r.started_at);
    println!(
        "\n  score {:.0}%, pass {:.0}%, {} judged, {} unjudged, {:.0}s at {:.1} tok/s",
        s.score * 100.0,
        s.pass_rate * 100.0,
        s.scored,
        s.unjudgeable,
        s.seconds,
        s.tokens_per_second()
    );
    if s.thought_out > 0 {
        println!(
            "  {} answer(s) lost entirely to reasoning_content",
            s.thought_out
        );
    }
    if s.errors > 0 {
        println!("  {} request(s) failed and were not scored", s.errors);
    }
    println!();
    print!("{}", report::detail_table(&r));
    println!("\n`--task <id>` shows what the model actually wrote.");
    Ok(())
}

/// Everything recorded about one task: the checks, the answer, and the thinking.
fn show_task(r: &report::Report, id: &str) -> anyhow::Result<()> {
    let t = r.tasks.iter().find(|t| t.id == id).ok_or_else(|| {
        anyhow::anyhow!(
            "run {} has no task {id:?}; it has: {}",
            r.id,
            r.tasks
                .iter()
                .map(|t| t.id.clone())
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;

    println!("{} / {}", r.id, t.id);
    println!(
        "  {:.0}% in {:.1}s, {} tokens, finish_reason {}",
        t.score.unwrap_or(0.0) * 100.0,
        t.seconds,
        t.completion_tokens,
        if t.finish_reason.is_empty() {
            "-"
        } else {
            &t.finish_reason
        }
    );
    if let Some(error) = &t.error {
        println!("  request failed: {error}");
    }

    println!("\nchecks");
    for c in &t.checks {
        let verdict = match &c.outcome {
            check::Outcome::Pass => "pass".to_owned(),
            check::Outcome::Fail { why } => format!("FAIL - {why}"),
            check::Outcome::Skip { why } => format!("skip - {why}"),
        };
        println!("  {:<16} weight {:<4} {verdict}", c.label, c.weight);
    }

    // Thinking first, because it is what came first and because when the answer is
    // empty this is the only thing that explains why.
    if !t.reasoning.is_empty() {
        println!(
            "\nreasoning_content ({} chars, never seen by a harness)\n{}",
            t.reasoning.chars().count(),
            indent(&t.reasoning)
        );
    }
    if t.answer.is_empty() {
        println!("\ncontent was empty");
    } else {
        println!("\ncontent\n{}", indent(&t.answer));
    }
    Ok(())
}

fn indent(text: &str) -> String {
    text.lines()
        .map(|l| format!("  | {l}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn compare_runs(a: &str, b: &str) -> anyhow::Result<()> {
    let config = Config::load()?;
    let left = report::resolve(&config.eval_dir, a)?;
    let right = report::resolve(&config.eval_dir, b)?;
    print!("{}", report::comparison(&left, &right));
    Ok(())
}

fn show_tasks(path: Option<&Path>) -> anyhow::Result<()> {
    let corpus = task::load(path)?;
    println!("{} ({})\n", corpus.origin, corpus.digest);
    println!("{:<28} {:<14} {:<10} CHECKS", "ID", "CATEGORY", "SPLIT");
    for t in &corpus.tasks {
        let kinds: Vec<&str> = t.checks.iter().map(check::Check::label).collect();
        println!(
            "{:<28} {:<14} {:<10} {}",
            t.id,
            t.category,
            t.split.to_string(),
            kinds.join(", ")
        );
    }
    Ok(())
}

fn stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Exit cleanly when output is piped into something that stops reading.
///
/// Same reason as the core CLI: Rust ignores SIGPIPE, so `ailocal eval ls | head`
/// panics on EPIPE where a Unix tool should simply stop, and restoring the default
/// disposition needs `unsafe`, which the workspace forbids.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> RunArgs {
        RunArgs {
            model: Vec::new(),
            reasoning: Vec::new(),
            tasks: None,
            split: None,
            only: Vec::new(),
            limit: None,
            no_exec: false,
            seed: run::DEFAULT_SEED,
            max_tokens: None,
            reasoning_budget: None,
            json: false,
        }
    }

    fn corpus() -> Vec<Task> {
        task::load(None).unwrap().tasks
    }

    #[test]
    fn no_filters_selects_everything() {
        let all = corpus();
        assert_eq!(select(&all, &args()).unwrap().len(), all.len());
    }

    #[test]
    fn a_split_narrows_the_selection() {
        let all = corpus();
        let held = select(
            &all,
            &RunArgs {
                split: Some(Split::HeldOut),
                ..args()
            },
        )
        .unwrap();
        assert!(held.iter().all(|t| t.split == Split::HeldOut));
        assert!(held.len() < all.len(), "the suite needs some dev tasks too");
    }

    #[test]
    fn limit_truncates() {
        let all = corpus();
        let some = select(
            &all,
            &RunArgs {
                limit: Some(3),
                ..args()
            },
        )
        .unwrap();
        assert_eq!(some.len(), 3);
    }

    /// A mistyped id must stop the run. Silently measuring the whole suite instead
    /// would produce a number that answers a different question than the one asked.
    #[test]
    fn an_unknown_task_id_is_an_error() {
        let all = corpus();
        let err = select(
            &all,
            &RunArgs {
                only: vec!["not-a-task".to_owned()],
                ..args()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no task with id"), "got: {err}");
    }

    #[test]
    fn naming_tasks_selects_exactly_those() {
        let all = corpus();
        let first = all[0].id.clone();
        let picked = select(
            &all,
            &RunArgs {
                only: vec![first.clone()],
                ..args()
            },
        )
        .unwrap();
        assert_eq!(picked.len(), 1);
        assert_eq!(picked[0].id, first);
    }

    #[test]
    fn filters_that_match_nothing_are_an_error_rather_than_an_empty_run() {
        let all = vec![];
        assert!(select(&all, &args()).is_err());
    }

    #[test]
    fn scratch_stays_under_the_configured_eval_directory() {
        let config = Config::default();
        assert!(scratch_dir(&config).starts_with(&config.eval_dir));
    }

    /// clap's own consistency check. Catches a duplicated flag or a bad default at
    /// build time rather than the first time someone runs the command.
    #[test]
    fn the_cli_is_well_formed() {
        use clap::CommandFactory as _;
        Cli::command().debug_assert();
    }
}
