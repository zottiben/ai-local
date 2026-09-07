//! Where a model can be fetched from, and how to turn a short reference into a URL.
//!
//! Two registries, because neither alone is sufficient from this network. Measured from
//! Adelaide on 2026-09-07: `registry.ollama.ai` sustains 79-97 MB/s and its blobs are
//! plain content-addressed GGUF files with a digest we can verify against. Hugging Face
//! managed 0.016-2.7 MB/s over the same period, wildly variable, and its `hf_xet`
//! client stalled outright - but it is the only source with full quant coverage.
//!
//! So: prefer Ollama for anything it carries, fall back to Hugging Face for the low
//! quants it does not.

use std::str::FromStr;

use anyhow::Context as _;
use reqwest::blocking::Client;

const OLLAMA_REGISTRY: &str = "https://registry.ollama.ai/v2/library";
const HF_ENDPOINT: &str = "https://huggingface.co";

/// A model reference, as typed on the command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// `ollama:<name>:<tag>`, e.g. `ollama:gemma4:12b`.
    Ollama { name: String, tag: String },
    /// `hf:<owner>/<repo>/<file.gguf>`, e.g. `hf:unsloth/Qwen3.8-27B-GGUF/Q3_K_XL.gguf`.
    HuggingFace { repo: String, file: String },
}

impl FromStr for Source {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (scheme, rest) = s
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("expected <source>:<ref>, got {s:?}"))?;

        match scheme {
            "ollama" => {
                let (name, tag) = rest
                    .split_once(':')
                    .ok_or_else(|| anyhow::anyhow!("expected ollama:<name>:<tag>, got {s:?}"))?;
                anyhow::ensure!(
                    !name.is_empty() && !tag.is_empty(),
                    "ollama reference has an empty name or tag: {s:?}"
                );
                Ok(Self::Ollama {
                    name: name.to_owned(),
                    tag: tag.to_owned(),
                })
            }
            "hf" => {
                let parts: Vec<&str> = rest.rsplitn(2, '/').collect();
                let [file, repo] = parts.as_slice() else {
                    anyhow::bail!("expected hf:<owner>/<repo>/<file.gguf>, got {s:?}")
                };
                anyhow::ensure!(
                    repo.contains('/') && !file.is_empty(),
                    "expected hf:<owner>/<repo>/<file.gguf>, got {s:?}"
                );
                Ok(Self::HuggingFace {
                    repo: (*repo).to_owned(),
                    file: (*file).to_owned(),
                })
            }
            other => anyhow::bail!("unknown source {other:?}, expected 'ollama' or 'hf'"),
        }
    }
}

impl std::fmt::Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ollama { name, tag } => write!(f, "ollama:{name}:{tag}"),
            Self::HuggingFace { repo, file } => write!(f, "hf:{repo}/{file}"),
        }
    }
}

/// A resolved, downloadable artifact.
#[derive(Debug, Clone)]
pub struct Artifact {
    pub url: String,
    pub size: Option<u64>,
    /// Expected content hash, when the registry publishes one. Ollama addresses blobs
    /// by digest so this is always present there; Hugging Face does not expose a plain
    /// sha256, so downloads from it can only be length-checked.
    pub sha256: Option<String>,
    /// Filename to store it under, derived from the reference.
    pub file_name: String,
    pub auth: Option<String>,
}

impl Source {
    /// Local filename for this reference, always ending in `.gguf`.
    #[must_use]
    pub fn file_name(&self) -> String {
        match self {
            Self::Ollama { name, tag } => format!("{name}-{tag}.gguf"),
            Self::HuggingFace { file, .. } => file.clone(),
        }
    }

    /// Turn the reference into a URL, with a size and digest where available.
    ///
    /// # Errors
    /// If the registry is unreachable, the reference does not exist, or an Ollama
    /// manifest contains no model layer.
    pub fn resolve(&self, client: &Client, hf_token: Option<&str>) -> anyhow::Result<Artifact> {
        match self {
            Self::Ollama { name, tag } => {
                let manifest: serde_json::Value = client
                    .get(format!("{OLLAMA_REGISTRY}/{name}/manifests/{tag}"))
                    .send()
                    .with_context(|| format!("fetching manifest for {self}"))?
                    .error_for_status()
                    .with_context(|| format!("no such model {self}"))?
                    .json()
                    .context("parsing manifest")?;

                // A manifest carries several layers - template, params, licence. The
                // weights are the one whose mediaType mentions the model.
                let layer = manifest["layers"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .find(|l| l["mediaType"].as_str().is_some_and(|m| m.contains("model")))
                    .ok_or_else(|| anyhow::anyhow!("manifest for {self} has no model layer"))?;

                let digest = layer["digest"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("model layer has no digest"))?;

                Ok(Artifact {
                    url: format!("{OLLAMA_REGISTRY}/{name}/blobs/{digest}"),
                    size: layer["size"].as_u64(),
                    sha256: digest.strip_prefix("sha256:").map(str::to_owned),
                    file_name: self.file_name(),
                    auth: None,
                })
            }

            Self::HuggingFace { repo, file } => {
                let url = format!("{HF_ENDPOINT}/{repo}/resolve/main/{file}");
                let auth = hf_token.map(|t| format!("Bearer {t}"));

                let mut head = client.head(&url);
                if let Some(a) = &auth {
                    head = head.header(reqwest::header::AUTHORIZATION, a);
                }
                let resp = head
                    .send()
                    .with_context(|| format!("resolving {self}"))?
                    .error_for_status()
                    .with_context(|| format!("no such file {self}"))?;

                let size = resp
                    .headers()
                    .get(reqwest::header::CONTENT_LENGTH)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse().ok());

                Ok(Artifact {
                    url,
                    size,
                    // HF serves through a Xet bridge that exposes no plain sha256, so
                    // there is nothing to verify against beyond the length.
                    sha256: None,
                    file_name: self.file_name(),
                    auth,
                })
            }
        }
    }
}

/// Whether a filename is one part of a split GGUF, e.g. `...-00002-of-00003.gguf`.
///
/// Listing these is worse than useless: a shard cannot be loaded on its own, and its
/// size makes a 60 GB model look like it would fit. Fetching a split model means
/// fetching every part, which this downloader does not do.
fn is_shard(path: &str) -> bool {
    let stem = path.strip_suffix(".gguf").unwrap_or(path);
    let mut parts = stem.rsplit('-');
    let (Some(total), Some(of), Some(index)) = (parts.next(), parts.next(), parts.next()) else {
        return false;
    };
    of == "of"
        && total.len() == 5
        && index.len() == 5
        && total.bytes().all(|b| b.is_ascii_digit())
        && index.bytes().all(|b| b.is_ascii_digit())
}

/// A repository found by [`search`].
#[derive(Debug, Clone)]
pub struct Repo {
    pub id: String,
    pub downloads: u64,
}

/// A downloadable file within a repository.
#[derive(Debug, Clone)]
pub struct RepoFile {
    pub path: String,
    pub size_mib: u64,
}

/// Search Hugging Face for repositories containing GGUF files.
///
/// Hugging Face is the only one of the two registries with a usable search API -
/// Ollama publishes no JSON endpoint for listing or searching its library, only web
/// pages. Since HF also has far better quant coverage, that is where searching belongs.
///
/// # Errors
/// If the request fails or the response is not the expected shape.
pub fn search(client: &Client, query: &str, limit: usize) -> anyhow::Result<Vec<Repo>> {
    let response: serde_json::Value = client
        .get(format!("{HF_ENDPOINT}/api/models"))
        .query(&[
            ("search", query),
            ("filter", "gguf"),
            ("sort", "downloads"),
            ("direction", "-1"),
            ("limit", &limit.to_string()),
        ])
        .send()
        .context("searching Hugging Face")?
        .error_for_status()?
        .json()
        .context("parsing search results")?;

    Ok(response
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| {
            Some(Repo {
                id: m["id"].as_str()?.to_owned(),
                downloads: m["downloads"].as_u64().unwrap_or(0),
            })
        })
        .collect())
}

/// List the GGUF files in a Hugging Face repository, smallest first.
///
/// Quantisations are files rather than tags, so choosing one means choosing a file -
/// which is also how you find out whether any of them fit.
///
/// # Errors
/// If the repository cannot be listed.
pub fn files(client: &Client, repo: &str, token: Option<&str>) -> anyhow::Result<Vec<RepoFile>> {
    let mut request = client.get(format!("{HF_ENDPOINT}/api/models/{repo}/tree/main"));
    request = request.query(&[("recursive", "true")]);
    if let Some(t) = token {
        request = request.header(reqwest::header::AUTHORIZATION, format!("Bearer {t}"));
    }

    let response: serde_json::Value = request
        .send()
        .with_context(|| format!("listing {repo}"))?
        .error_for_status()
        .with_context(|| format!("no such repository {repo}"))?
        .json()
        .context("parsing file list")?;

    let mut out: Vec<RepoFile> = response
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|f| {
            let path = f["path"].as_str()?;
            if !path.to_ascii_lowercase().ends_with(".gguf") || is_shard(path) {
                return None;
            }
            Some(RepoFile {
                path: path.to_owned(),
                size_mib: f["size"].as_u64().unwrap_or(0) / (1024 * 1024),
            })
        })
        .collect();
    out.sort_by_key(|f| f.size_mib);
    Ok(out)
}

/// Read the Hugging Face token from `$HF_HOME/token`, if present.
///
/// Deliberately file-based rather than an environment variable: it is the path the
/// official client already uses, it keeps the secret out of process listings and shell
/// history, and it lives on the big disk outside the repo.
#[must_use]
pub fn hf_token(hf_home: &std::path::Path) -> Option<String> {
    let raw = std::fs::read_to_string(hf_home.join("token")).ok()?;
    let trimmed = raw.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_an_ollama_reference() {
        let s: Source = "ollama:gemma4:12b".parse().unwrap();
        assert_eq!(
            s,
            Source::Ollama {
                name: "gemma4".into(),
                tag: "12b".into()
            }
        );
        assert_eq!(s.file_name(), "gemma4-12b.gguf");
    }

    /// The file name contains slashes and the repo contains one too, so the split has
    /// to come from the right.
    #[test]
    fn parses_a_hugging_face_reference() {
        let s: Source = "hf:unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf"
            .parse()
            .unwrap();
        assert_eq!(
            s,
            Source::HuggingFace {
                repo: "unsloth/Qwen3.8-27B-GGUF".into(),
                file: "Qwen3.8-27B-UD-Q3_K_XL.gguf".into(),
            }
        );
        assert_eq!(s.file_name(), "Qwen3.8-27B-UD-Q3_K_XL.gguf");
    }

    #[test]
    fn round_trips_through_display() {
        for r in [
            "ollama:gemma4:12b",
            "hf:unsloth/Qwen3.8-27B-GGUF/model.gguf",
        ] {
            assert_eq!(r.parse::<Source>().unwrap().to_string(), r);
        }
    }

    #[test]
    fn rejects_malformed_references() {
        for bad in [
            "gemma4:12b",        // no scheme
            "ollama:gemma4",     // no tag
            "ollama::12b",       // empty name
            "hf:justrepo.gguf",  // no owner
            "docker:gemma4:12b", // unknown scheme
        ] {
            assert!(bad.parse::<Source>().is_err(), "should reject {bad:?}");
        }
    }

    #[test]
    fn missing_token_file_is_none_not_an_error() {
        assert!(hf_token(std::path::Path::new("/nonexistent/hf")).is_none());
    }

    /// A shard's size makes a 60 GB model look like it fits, and it cannot be loaded
    /// alone, so listing one is actively misleading.
    #[test]
    fn split_gguf_shards_are_recognised() {
        assert!(is_shard("BF16/Qwen3-Coder-BF16-00002-of-00003.gguf"));
        assert!(is_shard("model-00001-of-00009.gguf"));
    }

    #[test]
    fn ordinary_quant_files_are_not_shards() {
        for path in [
            "Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf",
            "gemma4-12b-Q4_K_M.gguf",
            "mmproj-F16.gguf",
            "model.gguf",
        ] {
            assert!(!is_shard(path), "{path} is not a shard");
        }
    }
}
