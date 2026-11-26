//! Lessons a local model learns from the person's corrections.
//!
//! A lesson is one short rule, optionally scoped to a repository, with optional
//! triggers. The Pi extension reads this store itself and gives lessons to a local
//! model only: a trigger holds the first matching call of each prompt with the lesson,
//! and the lessons most like the prompt ride along with it. A frontier model never
//! sees one.
//!
//! Triggers are literals rather than patterns. The extension would run a pattern in
//! another regex dialect than any check here, and a literal is what a person - or a
//! model distilling a correction - writes correctly the first time.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

/// The store's schema version. A newer one is refused rather than misread.
pub const VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Store {
    pub version: u32,
    #[serde(default)]
    pub lessons: Vec<Lesson>,
}

impl Default for Store {
    fn default() -> Self {
        Self {
            version: VERSION,
            lessons: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lesson {
    pub id: String,
    pub text: String,
    /// Only in this repository, as `host/owner/repo`. Everywhere when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    /// Commands, as their first words, that the lesson holds: `gh pr create`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub commands: Vec<String>,
    /// Text that the lesson holds an edit or a write for adding: `eslint-disable`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub adds: Vec<String>,
    /// Files, directories or extensions that the lesson holds an edit or a write to:
    /// `CHANGELOG.md`, `database/migrations`, `.tsx`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<String>,
    pub state: State,
    /// Where it came from: the correction it was learned from, or that a person wrote it.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source: String,
    /// When it was made, in seconds since the Unix epoch.
    pub created: u64,
}

/// Only an active lesson reaches a model. A rejected one is kept so that the same
/// lesson is not proposed again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum State {
    Active,
    Pending,
    Rejected,
}

impl State {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Pending => "pending",
            Self::Rejected => "rejected",
        }
    }
}

/// Where lessons are kept: beside the config file, since they are as small and as
/// much the person's own, or `AILOCAL_LESSONS`, which is how the eval seeds its own.
/// The Pi extension works out the same path.
///
/// # Errors
/// If none of `AILOCAL_LESSONS`, `XDG_CONFIG_HOME` and `HOME` is set.
pub fn path() -> anyhow::Result<PathBuf> {
    if let Some(path) = std::env::var_os("AILOCAL_LESSONS").filter(|p| !p.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    Ok(crate::config::Config::path()?.with_file_name("lessons.json"))
}

impl Store {
    /// Read the store, or an empty one if there is no file yet.
    ///
    /// # Errors
    /// If the file cannot be read or parsed, or a newer ailocal wrote it.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        let store: Self =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        anyhow::ensure!(
            store.version <= VERSION,
            "{} was written by a newer ailocal (version {}); update this one",
            path.display(),
            store.version
        );
        Ok(store)
    }

    /// Write the store whole, so the extension never reads half of it.
    ///
    /// # Errors
    /// If the directory or the file cannot be written.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let mut text = serde_json::to_string_pretty(self)?;
        text.push('\n');
        crate::write_atomically(path, &text)
    }

    /// Add a lesson under an id made from its text, and return that id.
    ///
    /// # Errors
    /// If the text is empty, or a trigger is blank.
    pub fn add(&mut self, mut lesson: Lesson) -> anyhow::Result<String> {
        lesson.text = lesson.text.trim().to_owned();
        anyhow::ensure!(!lesson.text.is_empty(), "a lesson needs some text");
        for trigger in lesson
            .commands
            .iter_mut()
            .chain(lesson.adds.iter_mut())
            .chain(lesson.files.iter_mut())
        {
            *trigger = trigger.trim().to_owned();
            anyhow::ensure!(!trigger.is_empty(), "a trigger cannot be blank");
        }
        lesson.id = self.free_id(&slug(&lesson.text));
        let id = lesson.id.clone();
        self.lessons.push(lesson);
        Ok(id)
    }

    /// Remove a lesson, returning it.
    ///
    /// # Errors
    /// If there is no lesson with that id.
    pub fn remove(&mut self, id: &str) -> anyhow::Result<Lesson> {
        let index = self
            .lessons
            .iter()
            .position(|l| l.id == id)
            .ok_or_else(|| {
                anyhow::anyhow!("there is no lesson {id:?}; `ailocal lessons ls` lists them")
            })?;
        Ok(self.lessons.remove(index))
    }

    fn free_id(&self, base: &str) -> String {
        let taken = |id: &str| self.lessons.iter().any(|l| l.id == id);
        if !taken(base) {
            return base.to_owned();
        }
        (2..)
            .map(|n| format!("{base}-{n}"))
            .find(|id| !taken(id))
            .expect("an unbounded range has a free number")
    }
}

/// Words that say nothing about which lesson it is.
const FILLER: [&str; 20] = [
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "its", "of",
    "on", "or", "the", "to", "with",
];

/// A short id from the first telling words of a lesson: `use-pr-template-always`.
fn slug(text: &str) -> String {
    let words: Vec<String> = text
        .split(|c: char| !c.is_ascii_alphanumeric())
        .map(str::to_ascii_lowercase)
        .filter(|w| !w.is_empty() && !FILLER.contains(&w.as_str()))
        .take(4)
        .collect();
    if words.is_empty() {
        "lesson".to_owned()
    } else {
        words.join("-")
    }
}

/// A repository as lessons name it, `host/owner/repo`, from any form of its remote
/// URL, so every clone and worktree of it is the same repository.
///
/// Both this and the Pi extension read origin as configured, before an `insteadOf`
/// rewrites it, and both are tested against `pi/remotes.json`.
#[must_use]
pub fn repo_id(remote: &str) -> Option<String> {
    let remote = remote.trim();
    let (authority, path) = match remote.split_once("://") {
        Some((_, rest)) => rest.split_once('/')?,
        // scp-like: `git@github.com:owner/repo.git`. A local path has no colon.
        None => remote.split_once(':')?,
    };
    let host = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    let host = host
        .split(':')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let path = path.trim_matches('/');
    let path = path.strip_suffix(".git").unwrap_or(path);
    if host.is_empty() || path.is_empty() {
        return None;
    }
    Some(format!("{host}/{path}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lesson(text: &str) -> Lesson {
        Lesson {
            id: String::new(),
            text: text.to_owned(),
            repo: None,
            commands: Vec::new(),
            adds: Vec::new(),
            files: Vec::new(),
            state: State::Active,
            source: String::new(),
            created: 1_795_600_000,
        }
    }

    #[test]
    fn a_missing_store_is_empty_and_a_saved_one_reads_back() {
        let dir = std::env::temp_dir().join(format!("ailocal-lessons-{}", std::process::id()));
        let path = dir.join("lessons.json");
        assert_eq!(Store::load(&path).unwrap(), Store::default());

        let mut store = Store::default();
        let mut l = lesson("Use the PR template in .github when writing a PR body.");
        l.commands = vec!["gh pr create".to_owned()];
        store.add(l).unwrap();
        store.save(&path).unwrap();
        assert_eq!(Store::load(&path).unwrap(), store);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_store_from_a_newer_ailocal_is_refused() {
        let dir = std::env::temp_dir().join(format!("ailocal-lessons-new-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lessons.json");
        std::fs::write(&path, r#"{"version": 2, "lessons": []}"#).unwrap();
        let error = Store::load(&path).unwrap_err().to_string();
        assert!(error.contains("newer ailocal"), "{error}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ids_come_from_the_text_and_never_collide() {
        let mut store = Store::default();
        let a = store
            .add(lesson("  Use the PR template, always.  "))
            .unwrap();
        let b = store.add(lesson("Use the PR template, always!")).unwrap();
        assert_eq!(a, "use-pr-template-always");
        assert_eq!(b, "use-pr-template-always-2");
        assert_eq!(store.lessons[0].text, "Use the PR template, always.");
        assert_eq!(store.remove(&a).unwrap().id, a);
        assert!(store.remove(&a).is_err());
    }

    #[test]
    fn an_empty_lesson_or_a_blank_trigger_is_refused() {
        let mut store = Store::default();
        assert!(store.add(lesson("   ")).is_err());
        let mut l = lesson("Fix lint errors at their cause.");
        l.adds = vec![" ".to_owned()];
        assert!(store.add(l).is_err());
        assert!(store.lessons.is_empty());
    }

    #[test]
    fn every_form_of_a_remote_names_the_same_repository() {
        // The Pi extension's test reads the same cases.
        let cases: Vec<(String, Option<String>)> =
            serde_json::from_str(include_str!("../pi/remotes.json")).unwrap();
        assert!(cases.len() > 5);
        for (remote, id) in cases {
            assert_eq!(repo_id(&remote), id, "{remote:?}");
        }
    }
}
