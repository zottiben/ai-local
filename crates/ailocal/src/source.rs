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
                // Split from the left, at exactly two segments: a Hugging Face
                // repository id is always `owner/name`, and everything after it is
                // the path within the repository. Taking the last slash instead
                // would read a quant in a subdirectory as a deeper repository that
                // does not exist, and listings are recursive so those paths occur.
                let mut parts = rest.splitn(3, '/');
                let (Some(owner), Some(name), Some(file)) =
                    (parts.next(), parts.next(), parts.next())
                else {
                    anyhow::bail!("expected hf:<owner>/<repo>/<file.gguf>, got {s:?}")
                };
                anyhow::ensure!(
                    !owner.is_empty() && !name.is_empty() && !file.is_empty(),
                    "expected hf:<owner>/<repo>/<file.gguf>, got {s:?}"
                );
                Ok(Self::HuggingFace {
                    repo: format!("{owner}/{name}"),
                    file: file.to_owned(),
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
    ///
    /// The last segment only. A repository path can carry a directory, and that
    /// directory is a fact about how the repository is arranged rather than anything
    /// the models directory should reproduce - a name with a slash in it would be a
    /// subdirectory nothing creates, and a model name no later command could repeat.
    #[must_use]
    pub fn file_name(&self) -> String {
        match self {
            Self::Ollama { name, tag } => format!("{name}-{tag}.gguf"),
            Self::HuggingFace { file, .. } => file.rsplit('/').next().unwrap_or(file).to_owned(),
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

/// Whether a filename is a multimodal projector rather than a model.
///
/// Same problem as a shard, from the other direction. `mmproj-F16.gguf` is the vision
/// tower a multimodal model is served *alongside*, through `--mmproj`; on its own it
/// has no layers to generate with. It is also small, so it lists as the thing with
/// the most room to spare on the machine - the most inviting row in the list, and the
/// one guaranteed not to work.
fn is_projector(path: &str) -> bool {
    path.rsplit('/')
        .next()
        .unwrap_or(path)
        .to_ascii_lowercase()
        .contains("mmproj")
}

/// A repository found by [`search`].
#[derive(Debug, Clone, serde::Serialize)]
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

    let sent = request.send().with_context(|| format!("listing {repo}"))?;

    // A name nobody can fetch is a typo, not a fault here, so it comes back as a 404
    // rather than a 500 that a page cannot tell from a registry outage. Hugging Face
    // answers 401 rather than 404 for a repository that is private or absent, since
    // saying which would leak the existence of private ones - both read the same way
    // from here, and both are fixed by checking the name.
    let status = sent.status();
    if status == reqwest::StatusCode::NOT_FOUND || status == reqwest::StatusCode::UNAUTHORIZED {
        anyhow::bail!(crate::Refused::Missing(format!(
            "no repository {repo} you can read - check the name, or set a Hugging Face \
             token if it is private"
        )));
    }

    let response: serde_json::Value = sent
        .error_for_status()
        .with_context(|| format!("listing {repo}"))?
        .json()
        .context("parsing file list")?;

    let mut out: Vec<RepoFile> = response
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|f| {
            let path = f["path"].as_str()?;
            if !path.to_ascii_lowercase().ends_with(".gguf") || is_shard(path) || is_projector(path)
            {
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

/// A repository file with this machine's answer about it attached.
///
/// Size alone is not the question anyone is asking. A 9 GiB quant on a 10 GiB budget
/// is not "nearly fine": the weights fit and then there is nowhere to put the KV
/// cache, so it will not load. The headroom is what decides it, and it travels with
/// the file so a list cannot be drawn without it.
#[derive(Debug, Clone)]
pub struct Choice {
    pub file: RepoFile,
    /// MiB left for KV cache and compute buffers once the weights are resident.
    pub headroom_mib: i64,
}

impl Choice {
    #[must_use]
    pub fn fits(&self) -> bool {
        self.headroom_mib >= crate::catalogue::MIN_HEADROOM_MIB
    }
}

/// The fields, and whether it fits.
///
/// Written out by hand for the same reason [`crate::catalogue::Candidate`]'s is:
/// [`Choice::fits`] is where the line is drawn, and a window that worked it out again
/// from `headroom_mib` could draw it somewhere else.
impl serde::Serialize for Choice {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct as _;

        let mut out = serializer.serialize_struct("Choice", 4)?;
        out.serialize_field("path", &self.file.path)?;
        out.serialize_field("size_mib", &self.file.size_mib)?;
        out.serialize_field("headroom_mib", &self.headroom_mib)?;
        out.serialize_field("fits", &self.fits())?;
        out.end()
    }
}

/// Measure each file against what this machine has, largest first.
///
/// Reversed from [`files`]'s smallest-first order because the question in front of
/// anyone reading a quant list is "what is the best one I can run", and the best one
/// that fits is the largest - so the answer should be at the top rather than found by
/// scanning to the last row that still says yes.
#[must_use]
pub fn weigh(files: Vec<RepoFile>, available_mib: u64) -> Vec<Choice> {
    let mut out: Vec<Choice> = files
        .into_iter()
        .map(|file| Choice {
            headroom_mib: i64::try_from(available_mib).unwrap_or(i64::MAX)
                - i64::try_from(file.size_mib).unwrap_or(i64::MAX),
            file,
        })
        .collect();
    out.reverse();
    out
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

    /// The best runnable quant is the first row, and a file that would leave no room
    /// for a KV cache is refused rather than shown as a near miss.
    #[test]
    fn the_largest_quant_that_fits_is_offered_first() {
        let files = vec![
            RepoFile {
                path: "q3.gguf".into(),
                size_mib: 4000,
            },
            RepoFile {
                path: "q4.gguf".into(),
                size_mib: 8000,
            },
            RepoFile {
                path: "q8.gguf".into(),
                size_mib: 9500,
            },
        ];
        let weighed = weigh(files, 10_000);

        let order: Vec<&str> = weighed.iter().map(|c| c.file.path.as_str()).collect();
        assert_eq!(order, ["q8.gguf", "q4.gguf", "q3.gguf"], "largest first");

        // 500 MiB spare is under the 1024 the catalogue calls the minimum, so the
        // weights fitting is not the same as the model running.
        assert_eq!(weighed[0].headroom_mib, 500);
        assert!(!weighed[0].fits(), "q8 leaves no room for a KV cache");
        assert!(weighed[1].fits(), "q4 leaves 2000 MiB");
    }

    /// A quant larger than the whole budget reports how far past it is, rather than
    /// saturating at zero: "3 GiB too large" tells someone to look at a smaller repo,
    /// where "will not fit" does not.
    #[test]
    fn a_quant_past_the_budget_keeps_its_negative_headroom() {
        let files = vec![RepoFile {
            path: "f16.gguf".into(),
            size_mib: 13_000,
        }];
        let weighed = weigh(files, 10_000);
        assert_eq!(weighed[0].headroom_mib, -3000);
        assert!(!weighed[0].fits());
    }

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

    /// A quant in a subdirectory still belongs to a two-segment repository.
    ///
    /// Repository listings are recursive, so a path can carry a directory, and the
    /// console builds a reference out of whatever it listed. Splitting on the last
    /// slash would make the directory part of the repository name and fetch from
    /// somewhere that does not exist.
    #[test]
    fn a_quant_in_a_subdirectory_keeps_the_repository_two_segments() {
        let s: Source = "hf:unsloth/Qwen3-235B-GGUF/UD-Q2_K_XL/model.gguf"
            .parse()
            .unwrap();
        assert_eq!(
            s,
            Source::HuggingFace {
                repo: "unsloth/Qwen3-235B-GGUF".into(),
                file: "UD-Q2_K_XL/model.gguf".into(),
            }
        );
        // The name on disk is the file's, not the whole path.
        assert_eq!(s.file_name(), "model.gguf");
    }

    #[test]
    fn round_trips_through_display() {
        for r in [
            "ollama:gemma4:12b",
            "hf:unsloth/Qwen3.8-27B-GGUF/model.gguf",
            "hf:unsloth/Qwen3.8-27B-GGUF/UD-Q2_K_XL/model.gguf",
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
            "hf:owner/repo",     // a repository, with no file in it
            "hf:owner/repo/",    // the same, with the slash typed
            "docker:gemma4:12b", // unknown scheme
        ] {
            assert!(bad.parse::<Source>().is_err(), "should reject {bad:?}");
        }
    }

    /// A projector is a companion file, so offering it as a model offers something
    /// that cannot be loaded - and it is small, so it would sort to the row that
    /// looks most comfortable on this machine.
    #[test]
    fn multimodal_projectors_are_not_models() {
        for p in [
            "mmproj-F16.gguf",
            "mmproj-BF16.gguf",
            "gemma-3-12b-mmproj-f16.gguf",
        ] {
            assert!(is_projector(p), "should skip {p}");
        }
        for p in ["gemma-3-12b-it-Q4_K_M.gguf", "model.gguf"] {
            assert!(!is_projector(p), "should keep {p}");
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
