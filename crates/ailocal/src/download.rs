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

/// How many times to try a head read before reporting it failed.
///
/// The read is what stops an unloadable or oversized model before it is downloaded, so
/// one failure must not be the last word. On 2026-11-17 a single failed read from the
/// Ollama registry was followed by three that worked and twenty curl range requests
/// that all returned 206 in about a second.
const HEAD_ATTEMPTS: u64 = 3;

/// Fetch the first `limit` bytes of an artifact.
///
/// Used to read GGUF metadata before committing to a multi-gigabyte transfer, so a
/// model that could never fit in VRAM is rejected in seconds rather than after an hour.
/// A dropped connection or a server error is retried; a client error is not, because
/// it will be the same error the next time.
///
/// # Errors
/// If every attempt fails, or the server rejects the request outright.
pub fn head_bytes(client: &Client, artifact: &Artifact, limit: u64) -> anyhow::Result<Vec<u8>> {
    let mut attempt = 1;
    loop {
        match head_once(client, artifact, limit) {
            Err(e) if attempt < HEAD_ATTEMPTS && is_transient(&e) => {
                std::thread::sleep(std::time::Duration::from_secs(attempt));
                attempt += 1;
            }
            result => return result,
        }
    }
}

/// Whether asking again could get a different answer.
///
/// Anything without a status is the connection or the body failing part way, which is
/// exactly what a retry is for.
fn is_transient(err: &anyhow::Error) -> bool {
    err.chain()
        .find_map(|e| e.downcast_ref::<reqwest::Error>())
        .and_then(reqwest::Error::status)
        .is_none_or(|s| s.is_server_error() || s == reqwest::StatusCode::TOO_MANY_REQUESTS)
}

fn head_once(client: &Client, artifact: &Artifact, limit: u64) -> anyhow::Result<Vec<u8>> {
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

/// Where a download reports how far it has got.
///
/// A terminal draws a bar from these numbers; a window streams them to the page. The
/// transfer itself does not care which, so it takes the trait rather than drawing.
pub trait Progress {
    /// The body is about to be read, with `start` bytes of `total` already on disk.
    fn begin(&mut self, name: &str, total: Option<u64>, start: u64);
    /// `written` bytes of the file are now on disk, counting any resumed from.
    fn advance(&mut self, written: u64);
    /// The body has been read. Verifying the digest follows, and on a 20 GB file that
    /// is several seconds of reading it back.
    fn finish(&mut self);
}

/// The terminal's progress bar, drawn on stderr and hidden when that is not a terminal.
#[derive(Default)]
pub struct Bar(Option<ProgressBar>);

impl Progress for Bar {
    fn begin(&mut self, name: &str, total: Option<u64>, start: u64) {
        self.0 = Some(progress_bar(total, start, name));
    }

    fn advance(&mut self, written: u64) {
        if let Some(bar) = &self.0 {
            bar.set_position(written);
        }
    }

    fn finish(&mut self) {
        if let Some(bar) = self.0.take() {
            bar.finish_and_clear();
        }
    }
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
    progress: &mut dyn Progress,
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

    let resp = req
        .send()
        .with_context(|| format!("downloading {}", artifact.url))?;

    // Nothing lies past the end of the `.part`: it is already whole, as one is when an
    // install is stopped while its digest is checked. The registry says so with a 416
    // rather than an empty 206, and failing on it failed every retry of that install
    // for good. Checking what is on disk is the answer, and the digest has the last
    // word - a `.part` longer than the file fails it just as a corrupt one does.
    if resume_from > 0 && resp.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        progress.begin(&artifact.file_name, artifact.size, resume_from);
        progress.finish();
        move_into_place(&part, &dest, artifact)?;
        return Ok((
            dest,
            Fetched::Downloaded {
                bytes: resume_from,
                resumed_from: resume_from,
            },
        ));
    }
    let mut resp = resp.error_for_status()?;

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

    progress.begin(&artifact.file_name, total, start);
    let mut written = start;
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = resp.read(&mut buf).context("reading response body")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).context("writing to disk")?;
        written += n as u64;
        progress.advance(written);
    }
    file.flush().context("flushing")?;
    drop(file);
    progress.finish();

    move_into_place(&part, &dest, artifact)?;
    Ok((
        dest,
        Fetched::Downloaded {
            bytes: written,
            resumed_from: start,
        },
    ))
}

/// Verify a finished `.part` and give it its real name, which is what makes it a model.
fn move_into_place(part: &Path, dest: &Path, artifact: &Artifact) -> anyhow::Result<()> {
    verify(part, artifact)?;
    std::fs::rename(part, dest).with_context(|| format!("moving {} into place", part.display()))
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

    /// What the test server does with each connection, in the order they arrive.
    enum Reply {
        /// Read the request and close without a word, as a registry hiccup does.
        Drop,
        Respond(&'static str),
    }

    /// Serve one canned reply per connection on a free loopback port.
    fn server(replies: Vec<Reply>) -> (Artifact, std::thread::JoinHandle<()>) {
        use std::io::BufRead as _;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            for reply in replies {
                let (stream, _) = listener.accept().unwrap();
                let mut reader = std::io::BufReader::new(&stream);
                let mut line = String::new();
                while reader.read_line(&mut line).is_ok_and(|n| n > 2) {
                    line.clear();
                }
                if let Reply::Respond(response) = reply {
                    (&stream).write_all(response.as_bytes()).unwrap();
                }
            }
        });
        let mut artifact = artifact(None, None);
        artifact.url = format!("http://127.0.0.1:{port}/x.gguf");
        (artifact, handle)
    }

    /// One registry read in this session failed and was never seen again in three
    /// reruns or twenty range requests. A failure that rare is a reason to ask again,
    /// not to give up on the check it was feeding.
    #[test]
    fn a_dropped_connection_is_read_on_the_retry() {
        let (artifact, server) = server(vec![
            Reply::Drop,
            Reply::Respond(
                "HTTP/1.1 206 Partial Content\r\nContent-Length: 4\r\nConnection: close\r\n\r\nGGUF",
            ),
        ]);
        let head = head_bytes(&Client::new(), &artifact, 4).unwrap();
        assert_eq!(head, b"GGUF");
        server.join().unwrap();
    }

    /// A 404 will be a 404 the second time too. Asking again would only delay the
    /// error, so the second reply here, which would succeed, must never be read.
    #[test]
    fn a_client_error_is_not_retried() {
        let (artifact, _server) = server(vec![
            Reply::Respond(
                "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            ),
            Reply::Respond(
                "HTTP/1.1 206 Partial Content\r\nContent-Length: 4\r\nConnection: close\r\n\r\nGGUF",
            ),
        ]);
        let err = head_bytes(&Client::new(), &artifact, 4).unwrap_err();
        assert!(format!("{err:#}").contains("404"), "unexpected: {err:#}");
    }

    /// What a [`Progress`] was told, in place of a bar.
    #[derive(Default)]
    struct Recorded {
        begun: Option<(Option<u64>, u64)>,
        last: u64,
        finished: bool,
    }

    impl Progress for Recorded {
        fn begin(&mut self, _name: &str, total: Option<u64>, start: u64) {
            self.begun = Some((total, start));
        }
        fn advance(&mut self, written: u64) {
            self.last = written;
        }
        fn finish(&mut self) {
            self.finished = true;
        }
    }

    /// A window draws its bar from these calls, so they report what is on disk rather
    /// than what this transfer added: resuming "hello" from a two-byte `.part` begins
    /// at 2 of 5 and ends at 5, and the result verifies against the published digest.
    #[test]
    fn progress_counts_the_bytes_on_disk_across_a_resume() {
        let (mut artifact, server) = server(vec![Reply::Respond(
            "HTTP/1.1 206 Partial Content\r\nContent-Length: 3\r\nConnection: close\r\n\r\nllo",
        )]);
        artifact.sha256 = Some(HELLO.to_owned());
        artifact.size = Some(5);
        let dir = std::env::temp_dir().join(format!("ailocal-resume-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("x.gguf.part"), b"he").unwrap();

        let mut seen = Recorded::default();
        let (path, outcome) = fetch(&Client::new(), &artifact, &dir, &mut seen).unwrap();
        server.join().unwrap();

        assert_eq!(
            outcome,
            Fetched::Downloaded {
                bytes: 5,
                resumed_from: 2
            }
        );
        assert_eq!(seen.begun, Some((Some(5), 2)));
        assert_eq!(seen.last, 5);
        assert!(seen.finished);
        assert_eq!(std::fs::read(&path).unwrap(), b"hello");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An install stopped while the digest was being checked leaves a `.part` that is
    /// already whole. Asked for the bytes after its end, the Ollama registry answers
    /// 416 Range Not Satisfiable - measured on qwen3:0.6b's 522640096-byte blob - and
    /// every later install of that model failed on it until the file was deleted by
    /// hand. What is on disk is checked instead, and here it is "hello" in full.
    #[test]
    fn a_whole_part_file_is_verified_rather_than_fetched_again() {
        let (mut artifact, server) = server(vec![Reply::Respond(
            "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        )]);
        artifact.sha256 = Some(HELLO.to_owned());
        artifact.size = Some(5);
        let dir = std::env::temp_dir().join(format!("ailocal-whole-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("x.gguf.part"), b"hello").unwrap();

        let mut seen = Recorded::default();
        let (path, outcome) = fetch(&Client::new(), &artifact, &dir, &mut seen).unwrap();
        server.join().unwrap();

        assert_eq!(
            outcome,
            Fetched::Downloaded {
                bytes: 5,
                resumed_from: 5
            }
        );
        assert_eq!(seen.begun, Some((Some(5), 5)));
        assert!(seen.finished);
        assert_eq!(std::fs::read(&path).unwrap(), b"hello");
        assert!(!dir.join("x.gguf.part").exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
