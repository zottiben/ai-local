//! Getting a model onto disk: resolve the reference, check it can load here, fetch it.
//!
//! The check reads only the head of the file, so a model that cannot run here costs
//! seconds to reject rather than an hour of downloading. It answers rather than
//! refuses: whether to go ahead anyway is the caller's decision, made with `--force`
//! on the command line and a button in the window.

use std::path::PathBuf;

use reqwest::blocking::Client;

use crate::config::Config;
use crate::download::{self, Progress};
use crate::gguf;
use crate::registry::{self, Fit};
use crate::source::{self, Artifact, Source};
use crate::vram::{Budget, CacheType};

/// How much of a remote GGUF to read before deciding whether to download it.
const PREFLIGHT_BYTES: u64 = 8 << 20;

/// The most the check reads looking for the tensor index.
///
/// The index follows the tokeniser and chat template, so where it ends depends on the
/// vocabulary: 6.3 MiB into `qwen3-vl:2b`, 11.4 into `qwen3.6:35b-a3b`, 15.1 into
/// `gemma4:31b`. The read doubles from [`PREFLIGHT_BYTES`] until it gets there.
const PREFLIGHT_MAX_BYTES: u64 = 64 << 20;

/// The client a download uses.
///
/// No overall timeout: a 20 GB model takes minutes from the Ollama registry and hours
/// from Hugging Face, and a deadline would cut off every one of them part way.
///
/// # Errors
/// If the TLS backend cannot be initialised.
pub fn client() -> anyhow::Result<Client> {
    Ok(Client::builder()
        .user_agent(concat!("ailocal/", env!("CARGO_PKG_VERSION")))
        .build()?)
}

/// A reference resolved to the file it names.
#[derive(Debug, Clone)]
pub struct Resolved {
    pub source: Source,
    pub artifact: Artifact,
}

impl Resolved {
    #[must_use]
    pub fn weights_mib(&self) -> u64 {
        self.artifact.size.map_or(0, |b| b / (1024 * 1024))
    }

    /// Whether the registry publishes a digest, so the download can be verified rather
    /// than only length-checked.
    #[must_use]
    pub const fn has_digest(&self) -> bool {
        self.artifact.sha256.is_some()
    }
}

/// Ask the registry what `source` refers to: its URL, size and digest.
///
/// # Errors
/// If the registry cannot be reached or does not know the reference.
pub fn resolve(cfg: &Config, client: &Client, source: Source) -> anyhow::Result<Resolved> {
    let token = source::hf_token(&cfg.hf_home);
    let artifact = source.resolve(client, token.as_deref())?;
    Ok(Resolved { source, artifact })
}

/// What the head of the file said.
#[derive(Debug)]
pub enum Header {
    Read {
        arch: Option<String>,
        trained_context: Option<u64>,
        /// Vision or audio encoder tensors packed into the weights; `None` when the
        /// tensor index lies beyond what was read.
        encoder_tensors: Option<u64>,
        tensor_count: u64,
    },
    /// The bytes are not a GGUF this parser understands, so nothing could be judged.
    NotGguf,
    /// The head could not be fetched. The model may be one llama.cpp refuses, and
    /// nothing has looked.
    Unreadable(anyhow::Error),
}

/// Whether a model can load and run here, judged from the head of its file.
#[derive(Debug)]
pub struct Check {
    pub header: Header,
    pub fit: Fit,
    pub weights_mib: u64,
    pub available_mib: u64,
}

impl Check {
    /// Encoder tensors packed into the weights, when there are any.
    #[must_use]
    pub fn encoder(&self) -> Option<(u64, u64)> {
        match self.header {
            Header::Read {
                encoder_tensors: Some(n),
                tensor_count,
                ..
            } if n > 0 => Some((n, tensor_count)),
            _ => None,
        }
    }

    /// Whether this should only be downloaded on an explicit override.
    ///
    /// A head that was not GGUF is not a reason: that has always meant "could not
    /// judge", and a model in a newer format than this parser should still install.
    #[must_use]
    pub fn blocked(&self) -> bool {
        matches!(self.header, Header::Unreadable(_))
            || self.encoder().is_some()
            || matches!(self.fit, Fit::ContextTooSmall(_) | Fit::WeightsTooLarge)
    }
}

/// Read the head of a model's file and judge whether it can run here.
///
/// Budgeted against what a launch would get, not against live VRAM: with a model
/// already resident, live usage counts weights that serving this one would evict.
/// `model pick` sizes its list the same way, so using live usage here made the two
/// disagree - the picker offered a model as fitting and the installer then refused to
/// download it.
#[must_use]
pub fn check(client: &Client, resolved: &Resolved, budget: &Budget, cache: CacheType) -> Check {
    let weights_mib = resolved.weights_mib();
    let judged = |header, fit| Check {
        header,
        fit,
        weights_mib,
        available_mib: budget.available_mib(),
    };

    let mut limit = PREFLIGHT_BYTES;
    let md = loop {
        let head = match download::head_bytes(client, &resolved.artifact, limit) {
            Ok(head) => head,
            Err(e) => return judged(Header::Unreadable(e), Fit::Unknown),
        };
        let Ok(md) = gguf::parse_head(&head) else {
            return judged(Header::NotGguf, Fit::Unknown);
        };
        if md.encoder_tensors.is_some() || limit >= PREFLIGHT_MAX_BYTES {
            break md;
        }
        limit *= 2;
    };

    let fit = registry::assess(
        md.kv_layout(),
        md.context_length(),
        budget,
        cache,
        weights_mib,
    );
    judged(
        Header::Read {
            arch: md.architecture().map(str::to_owned),
            trained_context: md.context_length(),
            encoder_tensors: md.encoder_tensors,
            tensor_count: md.tensor_count,
        },
        fit,
    )
}

/// A model now on disk.
#[derive(Debug)]
pub struct OnDisk {
    /// What the registry, `serve`, `default_model` and a harness will call it.
    pub name: String,
    pub path: PathBuf,
    pub outcome: download::Fetched,
}

/// Download a resolved model into the models directory, resuming and verifying.
///
/// The name is returned because every later step needs it, and it is not the
/// reference: `ollama:qwen3.6:35b-a3b` lands as `qwen3.6-35b-a3b`.
///
/// # Errors
/// If the transfer fails or the file does not verify.
pub fn fetch(
    cfg: &Config,
    client: &Client,
    resolved: &Resolved,
    progress: &mut dyn Progress,
) -> anyhow::Result<OnDisk> {
    let (path, outcome) = download::fetch(client, &resolved.artifact, &cfg.models_dir, progress)?;
    Ok(OnDisk {
        name: path.file_stem().map_or_else(
            || resolved.artifact.file_name.clone(),
            |s| s.to_string_lossy().into_owned(),
        ),
        path,
        outcome,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolved(url: String, size_mib: u64) -> Resolved {
        Resolved {
            source: "ollama:qwen3-vl:2b".parse().unwrap(),
            artifact: Artifact {
                url,
                size: Some(size_mib << 20),
                sha256: None,
                file_name: "x.gguf".into(),
                auth: None,
            },
        }
    }

    /// The check's own answer when the registry cannot be read. A failed read once let
    /// the installer download 1801 MiB of qwen3-vl:2b, which llama.cpp refuses, so it
    /// must block rather than read as "could not judge".
    #[test]
    fn a_header_that_cannot_be_read_blocks() {
        let port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };
        let check = check(
            &Client::new(),
            &resolved(format!("http://127.0.0.1:{port}/x.gguf"), 1801),
            &Budget::new(1013),
            CacheType::Q8_0,
        );
        assert!(matches!(check.header, Header::Unreadable(_)), "{check:?}");
        assert_eq!(check.fit, Fit::Unknown);
        assert_eq!(check.weights_mib, 1801);
        assert!(check.blocked());
    }

    /// devstral-small-2:24b packs 222 encoder tensors among its 585, and llama.cpp
    /// will not load it; a fitting model with none is not blocked.
    #[test]
    fn packed_encoder_tensors_block_and_none_do_not() {
        let read = |encoder_tensors| Check {
            header: Header::Read {
                arch: Some("mistral3".into()),
                trained_context: Some(131_072),
                encoder_tensors,
                tensor_count: 585,
            },
            fit: Fit::Fits(131_072),
            weights_mib: 14_336,
            available_mib: 39_312,
        };
        assert_eq!(read(Some(222)).encoder(), Some((222, 585)));
        assert!(read(Some(222)).blocked());
        assert_eq!(read(Some(0)).encoder(), None);
        assert!(!read(Some(0)).blocked());
        assert!(!read(None).blocked());
    }

    /// Bytes that do not parse have always meant "could not judge", not "refuse".
    #[test]
    fn a_head_that_is_not_gguf_does_not_block() {
        let check = Check {
            header: Header::NotGguf,
            fit: Fit::Unknown,
            weights_mib: 258,
            available_mib: 39_312,
        };
        assert!(!check.blocked());
    }
}
