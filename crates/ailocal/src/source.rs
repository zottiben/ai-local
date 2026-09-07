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
}
