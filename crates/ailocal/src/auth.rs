//! The bearer key the gateway requires.
//!
//! Every OpenAI-compatible client already sends `Authorization: Bearer <key>`, so a
//! bearer token is the one credential that works across Pi, Codex and Claude Code
//! without any of them needing to know about us. Cloudflare Access cannot fill this
//! role - its login is a browser redirect, and a CLI harness cannot complete one.
//!
//! The key lives in a 0600 file outside the repo, never in the config, so that
//! `ailocal config show` can be pasted into an issue without leaking it.

use std::path::PathBuf;

use anyhow::Context as _;

/// Bytes of entropy in a generated key.
const KEY_BYTES: usize = 32;

/// Prefix so a leaked key is recognisable in logs and greppable in a codebase.
const PREFIX: &str = "ail_";

/// Path to the key file.
///
/// # Errors
/// If neither `XDG_CONFIG_HOME` nor `HOME` is set.
pub fn key_path() -> anyhow::Result<PathBuf> {
    Ok(crate::config::Config::path()?.with_file_name("gateway.key"))
}

/// Read the key, if one has been created.
///
/// # Errors
/// If the file exists but cannot be read.
pub fn load() -> anyhow::Result<Option<String>> {
    let path = key_path()?;
    match std::fs::read_to_string(&path) {
        Ok(s) => {
            let key = s.trim().to_owned();
            Ok((!key.is_empty()).then_some(key))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(anyhow::Error::new(e).context(format!("reading {}", path.display()))),
    }
}

/// Return the existing key, creating one if there is none.
///
/// # Errors
/// If the key cannot be read, generated or written.
pub fn load_or_create() -> anyhow::Result<String> {
    if let Some(key) = load()? {
        return Ok(key);
    }

    let key = generate()?;
    let path = key_path()?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&path, format!("{key}\n"))
        .with_context(|| format!("writing {}", path.display()))?;

    // Written before anyone can read it would be better, but create_new plus a mode is
    // not portable through std; setting it immediately is the practical equivalent for
    // a single-user machine.
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("locking down {}", path.display()))?;

    Ok(key)
}

/// Generate a key from the kernel CSPRNG.
///
/// `/dev/urandom` rather than a crate: it is the right source on this platform and
/// avoids a dependency for thirty-two bytes.
///
/// # Errors
/// If `/dev/urandom` cannot be read.
pub fn generate() -> anyhow::Result<String> {
    use std::io::Read as _;

    let mut buf = [0u8; KEY_BYTES];
    std::fs::File::open("/dev/urandom")
        .context("opening /dev/urandom")?
        .read_exact(&mut buf)
        .context("reading /dev/urandom")?;

    let hex: String = buf.iter().map(|b| format!("{b:02x}")).collect();
    Ok(format!("{PREFIX}{hex}"))
}

/// Compare a presented credential against the expected one in constant time.
///
/// A short-circuiting comparison leaks the length of the matching prefix, which is
/// enough to recover a key one byte at a time over a network.
#[must_use]
pub fn constant_time_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    // Fold every byte into the accumulator so the loop cannot exit early.
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Pull the credential out of an `Authorization` header value.
///
/// Accepts `Bearer <key>` case-insensitively on the scheme, which is what RFC 7235
/// requires and what clients actually vary on.
#[must_use]
pub fn bearer(header: &str) -> Option<&str> {
    let (scheme, token) = header.split_once(' ')?;
    scheme
        .eq_ignore_ascii_case("bearer")
        .then(|| token.trim())
        .filter(|t| !t.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_keys_are_prefixed_and_unique() {
        let a = generate().unwrap();
        let b = generate().unwrap();
        assert!(a.starts_with(PREFIX));
        assert_eq!(a.len(), PREFIX.len() + KEY_BYTES * 2);
        assert_ne!(a, b, "two calls must not return the same key");
    }

    #[test]
    fn constant_time_eq_matches_normal_equality() {
        assert!(constant_time_eq("ail_abc", "ail_abc"));
        assert!(!constant_time_eq("ail_abc", "ail_abd"));
        assert!(!constant_time_eq("ail_abc", "ail_abcd"));
        assert!(!constant_time_eq("", "x"));
        assert!(constant_time_eq("", ""));
    }

    #[test]
    fn parses_a_bearer_header() {
        assert_eq!(bearer("Bearer ail_xyz"), Some("ail_xyz"));
        assert_eq!(bearer("bearer ail_xyz"), Some("ail_xyz"));
        assert_eq!(bearer("BEARER ail_xyz"), Some("ail_xyz"));
    }

    #[test]
    fn rejects_headers_that_are_not_bearer() {
        assert_eq!(bearer("Basic dXNlcjpwYXNz"), None);
        assert_eq!(bearer("ail_xyz"), None);
        assert_eq!(bearer("Bearer "), None);
        assert_eq!(bearer(""), None);
    }
}
