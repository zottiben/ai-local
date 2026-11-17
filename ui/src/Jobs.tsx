// Work the console is doing, as it does it (D5).
//
// An install is minutes and a model takes up to five to load, so neither can be a
// button that goes grey and comes back. Each is a job on the server with a phase and,
// while it downloads, a count of bytes; the page shows that and offers the one thing a
// person can do about it.

import { api } from "./api";
import { describeCheck, describeJob, formatBytes } from "./format";
import { type Actions } from "./hooks";
import type { Job } from "./types";

/** Phases where there is still something to wait for. */
const RUNNING = new Set(["resolving", "checking", "downloading", "verifying", "preparing", "starting"]);

export function isRunning(job: Job): boolean {
  return RUNNING.has(job.state.phase);
}

export function Jobs({ jobs, actions }: { jobs: Job[]; actions: Actions }) {
  if (jobs.length === 0) return null;
  return (
    <section className="panel" aria-label="Jobs">
      <h2 className="panel-title">Running here</h2>
      <ul className="jobs">
        {jobs.map((job) => (
          <JobRow key={job.id} job={job} actions={actions} />
        ))}
      </ul>
    </section>
  );
}

function JobRow({ job, actions }: { job: Job; actions: Actions }) {
  const running = isRunning(job);
  const refused = job.state.phase === "refused";
  const bad = job.state.phase === "failed" || refused;
  const verdict = job.check === null ? null : describeCheck(job.check);
  const failure = actions.failed?.key === `job-${job.id}` ? actions.failed.message : null;

  return (
    <li className={bad ? "job bad-job" : "job"}>
      <div className="job-head">
        <code className="reference">{job.target}</code>
        <span className="muted small">{job.kind === "install" ? "install" : "load"}</span>
        {job.force && <span className="badge warn">forced</span>}
        <div className="job-buttons">
          {running && job.kind === "install" && (
            <button
              className="small"
              disabled={actions.busy !== null}
              onClick={() => actions.run(`job-${job.id}`, () => api.cancel(job.id))}
            >
              Stop
            </button>
          )}
          {refused && (
            // The same override `--force` is on the command line, and it starts a
            // fresh job so the check is read again rather than taken from this one.
            <button
              className="small"
              disabled={actions.busy !== null}
              onClick={() =>
                actions.run(`job-${job.id}`, async () => {
                  await api.install(job.target, true);
                  await api.dismiss(job.id);
                })
              }
            >
              Download anyway
            </button>
          )}
          {!running && (
            <button
              className="small ghost"
              aria-label={`Dismiss ${job.target}`}
              disabled={actions.busy !== null}
              onClick={() => actions.run(`job-${job.id}`, () => api.dismiss(job.id))}
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {job.state.phase === "downloading" && (
        <Bar written={job.state.written} total={job.state.total} />
      )}

      <p className={bad ? "bad small" : "muted small"}>{describeJob(job)}</p>

      {verdict?.detail && (
        <p className={verdict.blocked ? "bad small" : "muted small"}>{verdict.detail}</p>
      )}
      {failure && <p className="bad small">{failure}</p>}
    </li>
  );
}

/**
 * The bytes on disk against the whole file.
 *
 * A total the registry did not publish leaves the bar indeterminate rather than
 * inventing a denominator: a bar that claims a percentage it cannot know is worse than
 * one that admits it is only moving.
 */
function Bar({ written, total }: { written: number; total: number | null }) {
  return (
    <div className="bar">
      <progress value={total === null ? undefined : written} max={total ?? undefined} />
      <span className="muted small bar-count">
        {formatBytes(written)}
        {total !== null && ` of ${formatBytes(total)}`}
      </span>
    </div>
  );
}
