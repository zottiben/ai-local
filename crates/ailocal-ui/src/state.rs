//! What every handler shares.
//!
//! Nothing about the machine is kept here (D4): every read goes to the config, the
//! models directory, the server record and the service manager when it is asked. The
//! one thing held is the measured catalogue, which describes the registry rather than
//! the machine and costs ten seconds of network to ask again. The other is the jobs
//! this process is running, which exist nowhere else.

use std::sync::Arc;

use ailocal::catalogue::{self, Candidate};
use ailocal::config::Config;
use ailocal::vram::{Budget, CacheType};
use tokio::sync::{Mutex, broadcast};

use crate::error::{Result, blocking};
use crate::jobs::Jobs;

#[derive(Clone)]
pub struct AppState {
    token: Arc<str>,
    changes: broadcast::Sender<u64>,
    catalogue: Arc<Mutex<Option<Measured>>>,
    jobs: Jobs,
}

impl AppState {
    #[must_use]
    pub fn new(token: &str) -> Self {
        // Small on purpose. An event says only "something changed", so a client that
        // falls behind loses nothing by hearing it once instead of eight times.
        let (changes, _) = broadcast::channel(16);
        Self {
            token: token.into(),
            changes,
            catalogue: Arc::new(Mutex::new(None)),
            jobs: Jobs::new(),
        }
    }

    #[must_use]
    pub fn token(&self) -> &str {
        &self.token
    }

    #[must_use]
    pub fn changes(&self) -> broadcast::Sender<u64> {
        self.changes.clone()
    }

    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<u64> {
        self.changes.subscribe()
    }

    #[must_use]
    pub fn jobs(&self) -> &Jobs {
        &self.jobs
    }

    /// The catalogue resolved and measured against `budget`.
    ///
    /// Measured once and kept: asking the registry about ten models and reading the
    /// head of each took 10.5 to 10.9 s on the M4 Max, and the models directory - which
    /// this must not cache - is folded in by the caller on every request. The lock is
    /// held while measuring, so a second tab waits for the first measurement rather
    /// than starting its own.
    ///
    /// An empty answer is not kept. It means the registry could not be reached, and
    /// the next request should try again rather than be told the same thing.
    ///
    /// # Errors
    /// If the configured cache type is unknown or no HTTP client can be built.
    pub async fn catalogue(&self, cfg: &Config, budget: Budget) -> Result<Vec<Candidate>> {
        let mut held = self.catalogue.lock().await;
        if let Some(measured) = held.as_ref().filter(|m| m.answers_for(cfg, &budget)) {
            return Ok(measured.remote.clone());
        }

        let cache: CacheType = cfg.cache_type.parse()?;
        let remote = blocking(move || {
            let client = ailocal::install::client()?;
            let mut remote = catalogue::resolve(&client, budget.available_mib());
            catalogue::measure(&client, &mut remote, &budget, cache);
            Ok(remote)
        })
        .await?;

        if !remote.is_empty() {
            *held = Some(Measured {
                cache_type: cfg.cache_type.clone(),
                context_cap: budget.context_cap,
                available_mib: budget.available_mib(),
                remote: remote.clone(),
            });
        }
        Ok(remote)
    }
}

/// How far the budget may move before a measured catalogue no longer answers for it.
///
/// Not exact equality. On a discrete card the desktop's share of VRAM moves as it
/// allocates framebuffers, and re-reading every model's head because of that would
/// make the catalogue ten seconds slow on most visits while changing what it shows by
/// less than the GiB it reports sizes in. Whether a download really fits is checked
/// again, against the live budget, before any byte of it is fetched.
const SAME_BUDGET_MIB: u64 = 1024;

/// A catalogue measured against one budget.
struct Measured {
    cache_type: String,
    context_cap: Option<u64>,
    available_mib: u64,
    remote: Vec<Candidate>,
}

impl Measured {
    fn answers_for(&self, cfg: &Config, budget: &Budget) -> bool {
        self.cache_type == cfg.cache_type
            && self.context_cap == budget.context_cap
            && self.available_mib.abs_diff(budget.available_mib()) < SAME_BUDGET_MIB
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured(available_mib: u64) -> Measured {
        Measured {
            cache_type: "q8_0".to_owned(),
            context_cap: Some(131_072),
            available_mib,
            remote: Vec::new(),
        }
    }

    /// Measured with the desktop holding its idle ~900 MiB of the 14400 MiB ceiling. A
    /// desktop that takes 1023 MiB more is still answered from memory; one more MiB, a
    /// different context cap or a different cache type measures again.
    #[test]
    fn a_measurement_answers_until_the_budget_moves_a_gib() {
        let cfg = Config {
            cache_type: "q8_0".to_owned(),
            ..Config::default()
        };
        let budget = |desktop_mib: u64| Budget::new(desktop_mib).capped_at(Some(131_072));
        let at = measured(budget(900).available_mib());

        assert!(at.answers_for(&cfg, &budget(900)));
        assert!(at.answers_for(&cfg, &budget(900 + 1023)));
        assert!(!at.answers_for(&cfg, &budget(900 + 1024)));
        assert!(!at.answers_for(&cfg, &Budget::new(900).capped_at(Some(65_536))));
        assert!(!at.answers_for(
            &Config {
                cache_type: "f16".to_owned(),
                ..cfg.clone()
            },
            &budget(900)
        ));
    }
}
