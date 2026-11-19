//! Work that outlives the request that started it (D5, D10).
//!
//! An install is minutes of downloading and a model takes up to five minutes to load,
//! so neither can be a request the page waits on. Each runs on a thread of its own in
//! this process; the request that starts one returns its id at once, and the page
//! follows it over the event stream. Because the job lives here and not in the page, a
//! reload re-attaches to it instead of losing it.
//!
//! One of each kind at a time. Two downloads share one link and one disk and finish no
//! sooner, and two models cannot share the budget - starting a second would evict the
//! first.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::sync::watch;

use ailocal::config::Config;
use ailocal::download::{Cancelled, Progress};
use ailocal::install::{self, Check, OnDisk};
use ailocal::serve::{self, Instance, Launch};
use ailocal::source::Source;

use crate::error::{Error, Result};

/// How often a download's progress reaches the page.
///
/// A chunk is 1 MiB, so a link at the 79-97 MB/s measured from the Ollama registry
/// finishes about ninety a second. Every one of them is recorded; four a second are
/// published, which is smooth to watch and costs the page nothing.
const PUBLISH_EVERY: Duration = Duration::from_millis(250);

/// Finished jobs kept until the page dismisses them. Enough to read back what the
/// last few installs did; a bound so a console left open for a week does not grow.
const KEEP_FINISHED: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Install,
    Serve,
}

#[derive(Debug, Serialize)]
pub struct Job {
    pub id: u64,
    pub kind: Kind,
    /// The reference being installed, or the model being served.
    pub target: String,
    /// Downloading even if the check blocks it, as `--force` does.
    pub force: bool,
    /// What the head of the file said, once it has been read. Kept for the rest of
    /// the job, so a refusal and a forced download both show the reason.
    pub check: Option<Check>,
    pub state: State,
    #[serde(skip)]
    cancel: Arc<AtomicBool>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum State {
    /// Asking the registry what the reference names.
    Resolving,
    /// Reading the head of the file to judge whether it can run here.
    Checking {
        weights_mib: u64,
    },
    /// `written` counts the bytes already on disk, so a resumed download starts part
    /// of the way along rather than at zero.
    Downloading {
        written: u64,
        total: Option<u64>,
        resumed_from: u64,
    },
    /// Reading the file back to check its digest: seconds on a 20 GB file, and past
    /// the point where a cancel can stop it.
    Verifying {
        total: Option<u64>,
    },
    /// Sizing the context and finding what it will evict.
    Preparing,
    /// llama-server is loading the weights.
    Starting {
        context: u64,
        evicting: Option<String>,
    },
    Installed(OnDisk),
    Served(Instance),
    /// The check blocked the download, as it blocks `ailocal model install`.
    Refused,
    /// Stopped by the page. The `.part` is kept, so installing it again resumes.
    Cancelled {
        written: u64,
        total: Option<u64>,
    },
    Failed {
        error: String,
    },
}

impl State {
    const fn finished(&self) -> bool {
        matches!(
            self,
            Self::Installed(_)
                | Self::Served(_)
                | Self::Refused
                | Self::Cancelled { .. }
                | Self::Failed { .. }
        )
    }
}

/// Every job, and the page's view of them.
#[derive(Clone)]
pub struct Jobs {
    inner: Arc<Mutex<Inner>>,
    published: watch::Sender<Arc<str>>,
}

struct Inner {
    next_id: u64,
    jobs: Vec<Job>,
}

impl Jobs {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                next_id: 1,
                jobs: Vec::new(),
            })),
            published: watch::Sender::new(Arc::from("[]")),
        }
    }

    /// The job list as JSON, now and whenever it changes. A watch rather than a
    /// broadcast: a client that falls behind needs the latest list, never every list
    /// in between, and one that connects gets the current one at once.
    #[must_use]
    pub fn subscribe(&self) -> watch::Receiver<Arc<str>> {
        self.published.subscribe()
    }

    /// Download `source` into the models directory, as `ailocal model install` does.
    ///
    /// # Errors
    /// If an install is already running.
    pub fn install(&self, source: Source, force: bool) -> Result<u64> {
        let job = self.start(Kind::Install, source.to_string(), force)?;
        let id = job.id;
        std::thread::spawn(move || {
            let state = match fetch(&job, source, force) {
                Ok(Some(on_disk)) => State::Installed(on_disk),
                Ok(None) => State::Refused,
                Err(e) => match e.downcast_ref::<Cancelled>() {
                    Some(c) => State::Cancelled {
                        written: c.written,
                        total: c.total,
                    },
                    None => State::Failed {
                        error: format!("{e:#}"),
                    },
                },
            };
            job.set(state);
        });
        Ok(id)
    }

    /// Load `name`, as `ailocal serve <name>` does.
    ///
    /// # Errors
    /// If a model is already starting.
    pub fn serve(&self, name: String) -> Result<u64> {
        let job = self.start(Kind::Serve, name.clone(), false)?;
        let id = job.id;
        std::thread::spawn(move || {
            let state = match load(&job, name) {
                Ok(instance) => State::Served(instance),
                Err(e) => State::Failed {
                    error: format!("{e:#}"),
                },
            };
            job.set(state);
        });
        Ok(id)
    }

    /// Ask an install to stop before its next chunk.
    ///
    /// # Errors
    /// If there is no such job, it has finished, or it is a model starting - which
    /// has no point to stop at short of the server itself.
    pub fn cancel(&self, id: u64) -> Result<()> {
        let inner = self.lock();
        let job = inner.find(id)?;
        if job.state.finished() {
            return Err(Error::Busy(format!("job {id} has already finished")));
        }
        if job.kind == Kind::Serve {
            return Err(Error::Busy(
                "a model that is loading cannot be cancelled; stop it once it is up".to_owned(),
            ));
        }
        job.cancel.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Forget a finished job.
    ///
    /// # Errors
    /// If there is no such job, or it is still running.
    pub fn dismiss(&self, id: u64) -> Result<()> {
        let mut inner = self.lock();
        if !inner.find(id)?.state.finished() {
            return Err(Error::Busy(format!(
                "job {id} is still running; cancel it first"
            )));
        }
        inner.jobs.retain(|j| j.id != id);
        self.publish(&inner);
        Ok(())
    }

    /// Whether `name` is being loaded right now, which is no time to delete it.
    #[must_use]
    pub fn loading(&self, name: &str) -> bool {
        self.lock()
            .jobs
            .iter()
            .any(|j| j.kind == Kind::Serve && j.target == name && !j.state.finished())
    }

    fn start(&self, kind: Kind, target: String, force: bool) -> Result<Handle> {
        let mut inner = self.lock();
        if let Some(running) = inner
            .jobs
            .iter()
            .find(|j| j.kind == kind && !j.state.finished())
        {
            return Err(Error::Busy(match kind {
                Kind::Install => format!(
                    "{} is still downloading; one install at a time",
                    running.target
                ),
                Kind::Serve => format!("{} is still loading", running.target),
            }));
        }

        let id = inner.next_id;
        inner.next_id += 1;
        let cancel = Arc::new(AtomicBool::new(false));
        inner.jobs.push(Job {
            id,
            kind,
            target,
            force,
            check: None,
            state: match kind {
                Kind::Install => State::Resolving,
                Kind::Serve => State::Preparing,
            },
            cancel: Arc::clone(&cancel),
        });
        self.publish(&inner);
        Ok(Handle {
            jobs: self.clone(),
            id,
            cancel,
        })
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        // A job thread that panicked holding the lock left the list as it was, which
        // is still a list worth showing.
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn publish(&self, inner: &Inner) {
        // Every field is a string, a number, a path or a library type built from them,
        // so this cannot fail in practice. Not a panic if it does: this runs while a
        // job's handle is dropped, and a panic there would abort the console.
        match serde_json::to_string(&inner.jobs) {
            Ok(json) => {
                self.published.send_replace(Arc::from(json));
            }
            Err(e) => eprintln!("error: listing jobs for the page: {e}"),
        }
    }

    #[cfg(test)]
    fn snapshot(&self) -> serde_json::Value {
        serde_json::from_str(&self.published.borrow()).unwrap()
    }
}

impl Default for Jobs {
    fn default() -> Self {
        Self::new()
    }
}

impl Inner {
    fn find(&self, id: u64) -> Result<&Job> {
        self.jobs
            .iter()
            .find(|j| j.id == id)
            .ok_or_else(|| Error::NotFound(format!("no job {id}")))
    }

    /// Drop the oldest finished jobs beyond [`KEEP_FINISHED`].
    fn prune(&mut self) {
        let finished = self.jobs.iter().filter(|j| j.state.finished()).count();
        let mut excess = finished.saturating_sub(KEEP_FINISHED);
        self.jobs.retain(|j| {
            let drop = excess > 0 && j.state.finished();
            excess -= usize::from(drop);
            !drop
        });
    }
}

/// A running job's own view of itself.
struct Handle {
    jobs: Jobs,
    id: u64,
    cancel: Arc<AtomicBool>,
}

impl Handle {
    /// Change the job and tell the page.
    fn set(&self, state: State) {
        self.update(true, |job| job.state = state);
    }

    /// Change the job, telling the page only when `publish` says so.
    fn update(&self, publish: bool, change: impl FnOnce(&mut Job)) {
        let mut inner = self.jobs.lock();
        let Some(job) = inner.jobs.iter_mut().find(|j| j.id == self.id) else {
            return;
        };
        change(job);
        if job.state.finished() {
            inner.prune();
        }
        if publish {
            self.jobs.publish(&inner);
        }
    }
}

impl Drop for Handle {
    /// A job thread that ends without saying how - a panic in the library - would
    /// otherwise stay "downloading" forever, and as one install at a time is the
    /// rule, no other install could ever start.
    fn drop(&mut self) {
        self.update(true, |job| {
            if !job.state.finished() {
                job.state = State::Failed {
                    error: "the job stopped without finishing".to_owned(),
                };
            }
        });
    }
}

/// Resolve, check and download, as `ailocal model install` does. `None` when the
/// check blocked it.
fn fetch(job: &Handle, source: Source, force: bool) -> anyhow::Result<Option<OnDisk>> {
    let cfg = Config::load()?;
    let cache = cfg.cache_type.parse()?;
    let client = install::client()?;
    let resolved = install::resolve(&cfg, &client, source)?;

    job.set(State::Checking {
        weights_mib: resolved.weights_mib(),
    });
    let check = install::check(&client, &resolved, &serve::budget_or_estimate(), cache);
    let blocked = check.blocked();
    job.update(true, |j| j.check = Some(check));
    if blocked && !force {
        return Ok(None);
    }

    install::fetch(&cfg, &client, &resolved, &mut Reporter { job, last: None }).map(Some)
}

/// Prepare and start, as `ailocal serve <name>` does, with the port and context it
/// would choose.
fn load(job: &Handle, name: String) -> anyhow::Result<Instance> {
    let cfg = Config::load()?;
    let prepared = serve::prepare(
        &cfg,
        &Launch {
            name,
            context: None,
            host: serve::DEFAULT_HOST.to_owned(),
            port: serve::DEFAULT_PORT,
            reasoning: None,
        },
    )?;
    job.set(State::Starting {
        context: prepared.context,
        evicting: prepared.evicting.clone(),
    });

    let (instance, mut child) = prepared.start_owned()?;
    OURS.store(instance.pid, Ordering::Relaxed);
    // This process goes on running, so the server is its child until it exits, and an
    // exited child nobody waits on is a zombie that `kill(pid, 0)` still answers for
    // (D9). Waiting is all this thread does; the server's record is the library's.
    std::thread::spawn(move || child.wait());
    Ok(instance)
}

/// The pid of the last model this process started, or 0.
///
/// A process id, not a flag: between starting one and quitting, the model may have
/// been stopped and another started from a terminal, and killing that one on the way
/// out would stop something this window never touched.
static OURS: AtomicU32 = AtomicU32::new(0);

/// Stop a model this process started, if it is still the one that is running.
///
/// `ailocal ui` never needs this: a terminal delivers Ctrl-C to the whole process
/// group, and the model is in it. The desktop app has no terminal to deliver that
/// signal, so quitting the window has to do it (D9) - otherwise closing the window
/// leaves gigabytes resident with nothing left on screen to stop them.
///
/// # Errors
/// If the running instance cannot be read or the process cannot be signalled.
pub fn stop_owned_model() -> anyhow::Result<Option<Instance>> {
    let ours = OURS.load(Ordering::Relaxed);
    if ours == 0 {
        return Ok(None);
    }
    match serve::running()? {
        Some(instance) if instance.pid == ours => serve::stop(),
        _ => Ok(None),
    }
}

/// A download's progress, recorded on its job.
struct Reporter<'a> {
    job: &'a Handle,
    last: Option<Instant>,
}

impl Progress for Reporter<'_> {
    fn begin(&mut self, _name: &str, total: Option<u64>, start: u64) {
        self.job.set(State::Downloading {
            written: start,
            total,
            resumed_from: start,
        });
        self.last = Some(Instant::now());
    }

    fn advance(&mut self, written: u64) {
        let due = self.last.is_none_or(|t| t.elapsed() >= PUBLISH_EVERY);
        self.job.update(due, |job| {
            if let State::Downloading { written: w, .. } = &mut job.state {
                *w = written;
            }
        });
        if due {
            self.last = Some(Instant::now());
        }
    }

    fn finish(&mut self) {
        self.job.update(true, |job| {
            let total = match job.state {
                State::Downloading { total, .. } => total,
                _ => None,
            };
            job.state = State::Verifying { total };
        });
    }

    fn cancelled(&self) -> bool {
        self.job.cancel.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quitting stops only a model this process started and still owns.
    ///
    /// One test rather than two, because `OURS` is a static that the whole test
    /// binary shares: two tests asserting on it race each other, and the second one
    /// to run sees whatever the first left.
    ///
    /// Both halves matter. The window is one of several ways to start a model, and
    /// the common case on a fresh console is that it started none - at which point
    /// the only safe answer is to leave whatever is resident alone. And between
    /// starting one and quitting, it can be stopped and another started from a
    /// terminal, so the pid it recorded names either nothing or, once the number is
    /// reused, something else entirely.
    #[test]
    fn only_a_model_this_process_still_owns_is_stopped_on_the_way_out() {
        assert_eq!(OURS.load(Ordering::Relaxed), 0, "no test starts a model");
        assert!(stop_owned_model().unwrap().is_none());

        // 1 is init on both platforms, so it is certainly running and certainly not a
        // model this process started.
        OURS.store(1, Ordering::Relaxed);
        let left_alone = stop_owned_model().unwrap();
        OURS.store(0, Ordering::Relaxed);
        assert!(
            left_alone.is_none(),
            "stopped a model that was not the one we started"
        );
    }

    /// A second install is turned away while the first runs, but a model can load
    /// meanwhile - and once the install has finished, the next one may start.
    #[test]
    fn one_job_of_each_kind_at_a_time() {
        let jobs = Jobs::new();
        let first = jobs
            .start(Kind::Install, "ollama:qwen3.5:4b".into(), false)
            .unwrap();

        let Err(Error::Busy(busy)) = jobs.start(Kind::Install, "ollama:qwen3:0.6b".into(), false)
        else {
            panic!("a second install was not turned away");
        };
        assert!(
            busy.starts_with("ollama:qwen3.5:4b is still downloading"),
            "{busy}"
        );
        let serving = jobs.start(Kind::Serve, "gemma4-12b".into(), false).unwrap();
        assert!(jobs.loading("gemma4-12b"));
        assert!(!jobs.loading("qwen3-coder-30b"));

        first.set(State::Refused);
        assert!(
            jobs.start(Kind::Install, "ollama:qwen3:0.6b".into(), false)
                .is_ok()
        );
        drop(serving);
    }

    /// The page reads these names: the job's kind and target, the check once read,
    /// and the phase with its numbers. 498 MiB is qwen3:0.6b's blob.
    #[test]
    fn the_job_list_is_published_as_the_page_reads_it() {
        let jobs = Jobs::new();
        let job = jobs
            .start(Kind::Install, "ollama:qwen3:0.6b".into(), false)
            .unwrap();
        assert_eq!(jobs.snapshot()[0]["state"]["phase"], "resolving");

        let mut reporter = Reporter {
            job: &job,
            last: None,
        };
        reporter.begin("qwen3-0.6b.gguf", Some(522_640_096), 1 << 20);
        let listed = &jobs.snapshot()[0];
        assert_eq!(listed["id"], job.id);
        assert_eq!(listed["kind"], "install");
        assert_eq!(listed["target"], "ollama:qwen3:0.6b");
        assert_eq!(listed["state"]["phase"], "downloading");
        assert_eq!(listed["state"]["written"], 1 << 20);
        assert_eq!(listed["state"]["total"], 522_640_096);
        assert_eq!(listed["state"]["resumed_from"], 1 << 20);
        assert!(listed["check"].is_null());
        assert!(listed.get("cancel").is_none());
    }

    /// Ninety chunks a second are all recorded, but the page hears about them at most
    /// every 250 ms. The phase changes either side are always published.
    #[test]
    fn progress_is_recorded_every_chunk_and_published_four_times_a_second() {
        let jobs = Jobs::new();
        let job = jobs
            .start(Kind::Install, "ollama:qwen3:0.6b".into(), false)
            .unwrap();
        let mut reporter = Reporter {
            job: &job,
            last: None,
        };
        reporter.begin("qwen3-0.6b.gguf", Some(100 << 20), 0);
        reporter.advance(1 << 20);
        reporter.advance(2 << 20);
        assert_eq!(jobs.snapshot()[0]["state"]["written"], 0);
        assert_eq!(
            serde_json::to_value(&jobs.lock().jobs[0].state).unwrap()["written"],
            2 << 20
        );

        reporter.last = Some(Instant::now() - PUBLISH_EVERY);
        reporter.advance(3 << 20);
        assert_eq!(jobs.snapshot()[0]["state"]["written"], 3 << 20);

        reporter.finish();
        let verifying = &jobs.snapshot()[0]["state"];
        assert_eq!(verifying["phase"], "verifying");
        assert_eq!(verifying["total"], 100 << 20);
    }

    /// A cancel reaches the download through its reporter. A finished job cannot be
    /// cancelled, a loading model cannot be cancelled at all, and an unknown id is
    /// not found.
    #[test]
    fn cancel_reaches_a_running_install_and_nothing_else() {
        let jobs = Jobs::new();
        let job = jobs
            .start(Kind::Install, "ollama:qwen3:0.6b".into(), false)
            .unwrap();
        let reporter = Reporter {
            job: &job,
            last: None,
        };
        assert!(!reporter.cancelled());
        jobs.cancel(job.id).unwrap();
        assert!(reporter.cancelled());

        let serving = jobs.start(Kind::Serve, "gemma4-12b".into(), false).unwrap();
        assert!(matches!(jobs.cancel(serving.id), Err(Error::Busy(_))));

        job.set(State::Cancelled {
            written: 3 << 20,
            total: Some(100 << 20),
        });
        assert!(matches!(jobs.cancel(job.id), Err(Error::Busy(_))));
        assert!(matches!(jobs.cancel(99), Err(Error::NotFound(_))));
        drop(serving);
    }

    /// Only a finished job can be dismissed, and the oldest finished ones go by
    /// themselves beyond eight.
    #[test]
    fn finished_jobs_are_kept_until_dismissed_and_bounded() {
        let jobs = Jobs::new();
        let running = jobs
            .start(Kind::Install, "ollama:qwen3:0.6b".into(), false)
            .unwrap();
        assert!(matches!(jobs.dismiss(running.id), Err(Error::Busy(_))));
        running.set(State::Refused);
        jobs.dismiss(running.id).unwrap();
        assert_eq!(jobs.snapshot(), serde_json::json!([]));

        let ids: Vec<u64> = (0..10)
            .map(|_| {
                let job = jobs.start(Kind::Serve, "m".into(), false).unwrap();
                job.set(State::Failed {
                    error: "x".to_owned(),
                });
                job.id
            })
            .collect();
        let kept: Vec<u64> = jobs.lock().jobs.iter().map(|j| j.id).collect();
        assert_eq!(kept, ids[2..]);
    }

    /// A job thread that unwinds - here, one that simply drops its handle - leaves a
    /// failed job behind rather than an install that blocks every other forever.
    #[test]
    fn a_job_that_ends_without_an_answer_is_reported_failed() {
        let jobs = Jobs::new();
        drop(
            jobs.start(Kind::Install, "ollama:qwen3:0.6b".into(), false)
                .unwrap(),
        );
        let listed = &jobs.snapshot()[0];
        assert_eq!(listed["state"]["phase"], "failed");
        assert_eq!(
            listed["state"]["error"],
            "the job stopped without finishing"
        );
        assert!(
            jobs.start(Kind::Install, "ollama:qwen3:0.6b".into(), false)
                .is_ok()
        );
    }
}
