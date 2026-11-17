//! How the library's answers serialise for the console.
//!
//! Most types derive `Serialize` and need nothing from here. What they cannot derive is
//! an `anyhow::Error`, and several answers carry one on purpose - a step that failed
//! after the step that mattered had already succeeded. Every one of them goes out as
//! the CLI would print it, with the whole chain of causes, because the window shows the
//! message as written.

use serde::{Serialize, Serializer};

/// An error as `{e:#}`: the message and every cause, on one line.
pub(crate) fn error<S: Serializer>(e: &anyhow::Error, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_str(&format_args!("{e:#}"))
}

/// A result as its value when it succeeded, and as `{"error": "..."}` when it did not.
pub(crate) fn result<T: Serialize, S: Serializer>(
    result: &anyhow::Result<T>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    Answer(result).serialize(serializer)
}

/// [`result`], or `null` for a step that was not taken at all.
pub(crate) fn optional_result<T: Serialize, S: Serializer>(
    result: &Option<anyhow::Result<T>>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    result.as_ref().map(Answer).serialize(serializer)
}

struct Answer<'a, T>(&'a anyhow::Result<T>);

impl<T: Serialize> Serialize for Answer<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Failed {
            error: String,
        }

        match self.0 {
            Ok(value) => value.serialize(serializer),
            Err(e) => Failed {
                error: format!("{e:#}"),
            }
            .serialize(serializer),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct Outcome {
        #[serde(serialize_with = "result")]
        forgotten: anyhow::Result<bool>,
        #[serde(serialize_with = "optional_result")]
        default: Option<anyhow::Result<u8>>,
        #[serde(serialize_with = "error")]
        cause: anyhow::Error,
    }

    /// The window tells a failed step from a successful one by the `error` key, and
    /// shows the chain after it - both causes here, not only the outermost.
    #[test]
    fn a_failed_step_carries_its_whole_chain_and_a_skipped_one_is_null() {
        let failing = || anyhow::anyhow!("permission denied").context("writing models.json");
        let json = |forgotten, default| {
            serde_json::to_value(Outcome {
                forgotten,
                default,
                cause: failing(),
            })
            .unwrap()
        };

        let worked = json(Ok(true), Some(Ok(7)));
        assert_eq!(worked["forgotten"], true);
        assert_eq!(worked["default"], 7);
        assert_eq!(worked["cause"], "writing models.json: permission denied");

        let failed = json(Err(failing()), None);
        assert_eq!(
            failed["forgotten"]["error"],
            "writing models.json: permission denied"
        );
        assert!(failed["default"].is_null());
    }
}
