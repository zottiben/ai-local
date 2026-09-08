//! Choosing from a list, interactively.
//!
//! Uses [gum](https://github.com/charmbracelet/gum) when it is on PATH, and a plain
//! numbered prompt otherwise. gum is not a dependency: the whole point of a single
//! static binary installed by `curl | sh` is that a fresh machine needs nothing else,
//! so a missing gum degrades the presentation rather than the function.

use std::io::{BufRead as _, IsTerminal as _, Write as _};

use anyhow::Context as _;

/// Present `options` and return the index chosen, or `None` if the user backed out.
///
/// # Errors
/// If the terminal cannot be read, or gum fails for a reason other than cancellation.
pub fn choose(header: &str, options: &[String]) -> anyhow::Result<Option<usize>> {
    if options.is_empty() {
        return Ok(None);
    }
    // gum drives a full-screen UI and needs a real terminal: without one it fails with
    // "could not open TTY" and we would report a cancellation the user never made.
    // Checking first means a pipe or a CI run gets the plain prompt instead.
    let interactive = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    match crate::update::which("gum") {
        Some(_) if interactive => choose_with_gum(header, options),
        _ => choose_with_stdin(header, options),
    }
}

fn choose_with_gum(header: &str, options: &[String]) -> anyhow::Result<Option<usize>> {
    use std::process::{Command, Stdio};

    let mut child = Command::new("gum")
        .arg("choose")
        .args(["--header", header])
        // Keep the whole list on screen where it fits; gum defaults to a short window.
        .args(["--height", &(options.len() + 2).to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .context("running gum")?;

    child
        .stdin
        .take()
        .context("gum stdin")?
        .write_all(options.join("\n").as_bytes())
        .context("writing options to gum")?;

    let out = child.wait_with_output().context("waiting for gum")?;
    // A non-zero exit is how gum reports Esc / Ctrl-C, which is a choice, not an error.
    if !out.status.success() {
        return Ok(None);
    }

    let picked = String::from_utf8_lossy(&out.stdout).trim().to_owned();
    Ok(options.iter().position(|o| o == &picked))
}

fn choose_with_stdin(header: &str, options: &[String]) -> anyhow::Result<Option<usize>> {
    println!("{header}");
    for (i, option) in options.iter().enumerate() {
        println!("  {:>2}) {option}", i + 1);
    }
    print!("\nPick a number (or Enter to cancel): ");
    std::io::stdout().flush().ok();

    let mut line = String::new();
    std::io::stdin()
        .lock()
        .read_line(&mut line)
        .context("reading your choice")?;

    Ok(parse_choice(&line, options.len()))
}

/// Parse a typed selection into a zero-based index.
///
/// Anything that is not a number in range is a cancellation rather than an error: at a
/// prompt, a typo should not abort a long-running command.
#[must_use]
pub fn parse_choice(input: &str, len: usize) -> Option<usize> {
    let n: usize = input.trim().parse().ok()?;
    (1..=len).contains(&n).then(|| n - 1)
}

/// Ask for a line of text, falling back to `default` on an empty answer.
///
/// Expands a leading `~/`, because someone typing a path at a prompt is not running a
/// shell and will not get it expanded for them - and a literal `~` directory appearing
/// in the home directory is a confusing way to find that out.
///
/// # Errors
/// If the terminal cannot be read.
pub fn ask(question: &str, default: &str) -> anyhow::Result<String> {
    print!("{question} [{default}] ");
    std::io::stdout().flush().ok();

    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    let answer = line.trim();
    Ok(if answer.is_empty() {
        default.to_owned()
    } else {
        expand_home(answer)
    })
}

/// Expand a leading `~` against `HOME`.
#[must_use]
pub fn expand_home(input: &str) -> String {
    let Some(rest) = input.strip_prefix('~') else {
        return input.to_owned();
    };
    // `~foo` is another user's home, which we do not resolve - leave it be rather than
    // silently turn it into a subdirectory of this user's.
    if !(rest.is_empty() || rest.starts_with('/')) {
        return input.to_owned();
    }
    match std::env::var_os("HOME") {
        Some(home) => format!("{}{rest}", home.to_string_lossy()),
        None => input.to_owned(),
    }
}

/// Whether there is a human present to answer a prompt.
///
/// Without this a script or a CI run blocks forever on a question nobody will see, or
/// takes an empty stdin as a deliberate choice.
#[must_use]
pub fn interactive() -> bool {
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

/// Ask a yes/no question, defaulting to `default` on an empty answer.
///
/// # Errors
/// If the terminal cannot be read.
pub fn confirm(question: &str, default: bool) -> anyhow::Result<bool> {
    let hint = if default { "Y/n" } else { "y/N" };
    print!("{question} [{hint}] ");
    std::io::stdout().flush().ok();

    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(match line.trim().to_ascii_lowercase().as_str() {
        "" => default,
        "y" | "yes" => true,
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_valid_selection() {
        assert_eq!(parse_choice("1", 3), Some(0));
        assert_eq!(parse_choice("3", 3), Some(2));
        assert_eq!(parse_choice("  2  \n", 3), Some(1));
    }

    #[test]
    fn anything_out_of_range_or_unparseable_is_a_cancellation() {
        for input in ["", "\n", "0", "4", "abc", "-1", "1.5"] {
            assert_eq!(parse_choice(input, 3), None, "for {input:?}");
        }
    }

    #[test]
    fn a_leading_tilde_becomes_the_home_directory() {
        let home = std::env::var("HOME").unwrap();
        assert_eq!(expand_home("~/models"), format!("{home}/models"));
        assert_eq!(expand_home("~"), home);
    }

    #[test]
    fn paths_without_a_tilde_are_untouched() {
        assert_eq!(
            expand_home("/mnt/kingston/ailocal"),
            "/mnt/kingston/ailocal"
        );
        assert_eq!(expand_home("relative/path"), "relative/path");
    }

    /// `~other` is another user's home. We do not resolve it, and must not quietly
    /// rewrite it into a subdirectory of this user's.
    #[test]
    fn another_users_home_is_left_alone() {
        assert_eq!(expand_home("~other/models"), "~other/models");
    }
}
