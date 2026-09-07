//! Resumable, verified downloads.
//!
//! Model files are 7-30 GB over links that measurably stall, so a download is assumed
//! to be interrupted rather than treated as an exceptional case. Bytes land in a
//! `.part` file that is only renamed into place once the length and digest check out,
//! which means an interrupted run can never leave something that looks like a usable
//! model.

use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_LENGTH, RANGE};
use sha2::{Digest as _, Sha256};

use crate::source::Artifact;

/// Read buffer. Large enough that a ~90 MB/s link is not syscall-bound.
const CHUNK: usize = 1 << 20;

/// Outcome of a fetch, so the caller can tell work from a no-op.
#[derive(Debug, PartialEq, Eq)]
pub enum Fetched {
    /// The file was already present and verified.
    AlreadyPresent,
    Downloaded {
        bytes: u64,
        resumed_from: u64,
    },
}

/// Fetch the first `limit` bytes of an artifact.
///
/// Used to read GGUF metadata before committing to a multi-gigabyte transfer, so a
/// model that could never fit in VRAM is rejected in seconds rather than after an hour.
///
/// # Errors
/// If the request fails or the server rejects the range.
pub fn head_bytes(client: &Client, artifact: &Artifact, limit: u64) -> anyhow::Result<Vec<u8>> {
    let mut req = client
        .get(&artifact.url)
        .header(RANGE, format!("bytes=0-{}", limit.saturating_sub(1)));
    if let Some(auth) = &artifact.auth {
        req = req.header(AUTHORIZATION, auth);
    }

    let mut resp = req
        .send()
        .with_context(|| format!("fetching head of {}", artifact.url))?
        .error_for_status()?;

    let mut buf = Vec::with_capacity(usize::try_from(limit).unwrap_or(CHUNK));
    // Bound the read: a server that ignores Range answers with the whole file.
    resp.by_ref()
        .take(limit)
        .read_to_end(&mut buf)
        .context("reading head")?;
    Ok(buf)
}

/// Download `artifact` into `dir`, resuming and verifying.
///
/// # Errors
/// If the transfer fails, or the completed file does not match its expected size or
/// digest.
pub fn fetch(
    client: &Client,
    artifact: &Artifact,
    dir: &Path,
) -> anyhow::Result<(PathBuf, Fetched)> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;

    let dest = dir.join(&artifact.file_name);
    let part = dir.join(format!("{}.part", artifact.file_name));

    if dest.exists() {
        verify(&dest, artifact)?;
        return Ok((dest, Fetched::AlreadyPresent));
    }

    // Always ask to resume when there is a partial file, and let the response say
    // whether it worked. Probing `Accept-Ranges` first is unreliable: the Ollama
    // registry serves 206 for a Range request but does not advertise the header at
    // all, so trusting it restarted every interrupted download from zero.
    let resume_from = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);

    let mut req = client.get(&artifact.url);
    if let Some(auth) = &artifact.auth {
        req = req.header(AUTHORIZATION, auth);
    }
    if resume_from > 0 {
        req = req.header(RANGE, format!("bytes={resume_from}-"));
    }

    let mut resp = req
        .send()
        .with_context(|| format!("downloading {}", artifact.url))?
        .error_for_status()?;

    // A 200 in reply to a Range request means the server ignored it and is sending the
    // whole file, so anything already on disk has to be discarded.
    let restarting = resume_from > 0 && resp.status() != reqwest::StatusCode::PARTIAL_CONTENT;
    let start = if restarting { 0 } else { resume_from };

    let remaining = resp
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    let total = artifact.size.or_else(|| remaining.map(|r| r + start));

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(start == 0)
        .open(&part)
        .with_context(|| format!("opening {}", part.display()))?;
    if start > 0 {
        file.seek(SeekFrom::Start(start))
            .context("seeking to resume point")?;
    }

    let bar = progress_bar(total, start, &artifact.file_name);
    let mut written = start;
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = resp.read(&mut buf).context("reading response body")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).context("writing to disk")?;
        written += n as u64;
        bar.set_position(written);
    }
    file.flush().context("flushing")?;
    drop(file);
    bar.finish_and_clear();

    verify(&part, artifact)?;
    std::fs::rename(&part, &dest)
        .with_context(|| format!("moving {} into place", part.display()))?;

    Ok((
        dest,
        Fetched::Downloaded {
            bytes: written,
            resumed_from: start,
        },
    ))
}

/// Check a finished file against the artifact's expected size and digest.
fn verify(path: &Path, artifact: &Artifact) -> anyhow::Result<()> {
    let len = std::fs::metadata(path)
        .with_context(|| format!("stat {}", path.display()))?
        .len();

    if let Some(expected) = artifact.size {
        anyhow::ensure!(
            len == expected,
            "{} is {len} bytes, expected {expected}",
            path.display()
        );
    }

    let Some(expected) = &artifact.sha256 else {
        // Hugging Face exposes no plain digest, so length is all we can check.
        return Ok(());
    };

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    // sha2 0.11 returns a plain byte array, which has no hex formatting of its own.
    let actual: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    anyhow::ensure!(
        actual == *expected,
        "checksum mismatch for {}:\n  expected sha256:{expected}\n  got      sha256:{actual}",
        path.display()
    );
    Ok(())
}

fn progress_bar(total: Option<u64>, start: u64, name: &str) -> ProgressBar {
    let bar = match total {
        Some(t) => {
            let b = ProgressBar::new(t);
            b.set_style(
                ProgressStyle::with_template(
                    "  {msg} [{bar:32}] {bytes}/{total_bytes} {binary_bytes_per_sec} eta {eta}",
                )
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("=> "),
            );
            b
        }
        None => ProgressBar::new_spinner(),
    };
    bar.set_message(name.to_owned());
    bar.set_position(start);
    bar
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(sha256: Option<&str>, size: Option<u64>) -> Artifact {
        Artifact {
            url: "https://example.invalid/x.gguf".into(),
            size,
            sha256: sha256.map(str::to_owned),
            file_name: "x.gguf".into(),
            auth: None,
        }
    }

    fn temp_file(contents: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "ailocal-test-{}-{:?}.bin",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, contents).unwrap();
        path
    }

    /// sha256 of "hello"
    const HELLO: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

    #[test]
    fn accepts_a_matching_digest() {
        let path = temp_file(b"hello");
        assert!(verify(&path, &artifact(Some(HELLO), Some(5))).is_ok());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_a_corrupt_file() {
        let path = temp_file(b"hellp");
        let err = verify(&path, &artifact(Some(HELLO), Some(5))).unwrap_err();
        assert!(
            err.to_string().contains("checksum mismatch"),
            "unexpected error: {err}"
        );
        std::fs::remove_file(&path).ok();
    }

    /// A truncated download must fail on length before it is ever hashed, so a partial
    /// file cannot be renamed into place as a usable model.
    #[test]
    fn rejects_a_truncated_file() {
        let path = temp_file(b"hel");
        let err = verify(&path, &artifact(Some(HELLO), Some(5))).unwrap_err();
        assert!(err.to_string().contains("expected 5"), "unexpected: {err}");
        std::fs::remove_file(&path).ok();
    }

    /// Hugging Face publishes no plain digest, so length alone has to be enough.
    #[test]
    fn digestless_artifacts_are_length_checked_only() {
        let path = temp_file(b"hello");
        assert!(verify(&path, &artifact(None, Some(5))).is_ok());
        assert!(verify(&path, &artifact(None, Some(6))).is_err());
        std::fs::remove_file(&path).ok();
    }
}
