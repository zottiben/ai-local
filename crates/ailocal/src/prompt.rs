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
}
