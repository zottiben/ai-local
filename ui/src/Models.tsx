import { useRef, useState } from "react";

import { api } from "./api";
import { formatCount, formatDownloads, formatSize, thinkingLabel } from "./format";
import { type Actions, useActions, useResource } from "./hooks";
import { Jobs, isRunning } from "./Jobs";
import { Loaded } from "./Loaded";
import { RemoveDialog } from "./RemoveDialog";
import type { Candidate, Choice, Installed, Job, Offer, Repo } from "./types";

/** An answer that has to be asked for, and can be slow enough to need a state. */
type Fetched<T> =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ok"; value: T }
  | { kind: "failed"; message: string };

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** What `ailocal model ls` lists, what `model pick` offers, and the buttons for both. */
export function Models({ generation, jobs }: { generation: number; jobs: Job[] }) {
  const models = useResource(() => api.models(), [generation]);
  const config = useResource(() => api.config(), [generation]);
  const loaded = useResource(() => api.loaded(), [generation]);
  // Refetched on every change too. The server keeps the registry's answer for the run,
  // so this costs a directory scan, and a model installed from a terminal leaves this
  // list for the one above without a reload.
  const catalogue = useResource(() => api.catalogue(), [generation]);

  const actions = useActions();
  const [removing, setRemoving] = useState<string | null>(null);

  // The server takes one job of each kind at a time and answers 409 for a second, so
  // the buttons that would earn that 409 are the ones to hold shut. Per kind, as there:
  // a model loading is no reason to stop a download being started.
  const running = (kind: Job["kind"]) =>
    jobs.some((job) => job.kind === kind && isRunning(job));
  const installing = running("install");
  const loading = running("serve");

  return (
    <>
      {loaded.data && <Loaded instance={loaded.data} actions={actions} />}
      <Jobs jobs={jobs} actions={actions} />

      <section className="panel">
        <h2 className="panel-title">Installed</h2>
        {models.error ? (
          <p className="bad">{models.error.message}</p>
        ) : !models.data ? (
          <p className="muted">Reading the models directory…</p>
        ) : (
          <InstalledTable
            models={models.data}
            defaultModel={config.data?.default_model ?? null}
            loadedModel={loaded.data?.model ?? null}
            actions={actions}
            loading={loading}
            onRemove={setRemoving}
          />
        )}
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Add a model</h2>
          {catalogue.data && (
            <span className="muted small">
              measured against {formatSize(catalogue.data.available_mib)} free for a model
            </span>
          )}
        </div>

        <InstallByReference actions={actions} busy={installing} />

        <h3>From the catalogue</h3>
        {catalogue.error ? (
          <p className="bad">{catalogue.error.message}</p>
        ) : !catalogue.data ? (
          <p className="muted">
            Asking the registry about each model and reading the head of each file. The first
            time takes about ten seconds; after that it is kept until the console stops.
          </p>
        ) : (
          <CatalogueList offer={catalogue.data} actions={actions} busy={installing} />
        )}

        <h3>Search Hugging Face</h3>
        <SearchHuggingFace actions={actions} busy={installing} />
      </section>

      {removing !== null && (
        <RemoveDialog name={removing} onClose={() => setRemoving(null)} />
      )}
    </>
  );
}

/**
 * Any reference `ailocal model install` takes.
 *
 * The catalogue is ten models chosen for machines like this one; the registries hold
 * everything else, and a quant the catalogue does not carry is the ordinary reason to
 * reach past it.
 */
export function InstallByReference({ actions, busy }: { actions: Actions; busy: boolean }) {
  const [reference, setReference] = useState("");
  const failure = actions.failed?.key === "install-form" ? actions.failed.message : null;
  const trimmed = reference.trim();

  return (
    <form
      className="install-form"
      onSubmit={(event) => {
        event.preventDefault();
        if (trimmed === "") return;
        actions.run("install-form", async () => {
          await api.install(trimmed);
          setReference("");
        });
      }}
    >
      <input
        aria-label="Model reference"
        placeholder="ollama:qwen3.5:4b  or  hf:owner/repo/file.gguf"
        value={reference}
        onChange={(event) => setReference(event.target.value)}
        spellCheck={false}
      />
      <button
        className="primary"
        type="submit"
        disabled={trimmed === "" || busy || actions.busy !== null}
      >
        Install
      </button>
      {/* One install at a time, as on the server. Saying so beats a 409 after a click. */}
      {busy && <span className="muted small">another install is running</span>}
      {failure && <span className="bad small">{failure}</span>}
    </form>
  );
}

export function InstalledTable({
  models,
  defaultModel,
  loadedModel,
  actions,
  loading,
  onRemove,
}: {
  models: Installed[];
  defaultModel: string | null;
  loadedModel: string | null;
  actions: Actions;
  /** Another model is already on its way up, and the server takes one at a time. */
  loading: boolean;
  onRemove: (name: string) => void;
}) {
  if (models.length === 0) {
    return (
      <p className="muted">
        No models yet. The catalogue below offers the ones that fit this machine.
      </p>
    );
  }

  // Matched exactly, never by prefix: `job-1` is a prefix of `job-10`, and a failure
  // shown against the wrong row is worse than one shown nowhere.
  const rowKeys = models.flatMap((model) => [`serve-${model.name}`, `default-${model.name}`]);
  const failure =
    actions.failed !== null && rowKeys.includes(actions.failed.key) ? actions.failed.message : null;

  return (
    <>
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th>Model</th>
              <th className="num">Size</th>
              <th className="num">Context here</th>
              <th className="num">Trained</th>
              <th>Thinking</th>
              <th>Architecture</th>
              <th aria-label="Actions" />
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr key={model.name}>
                <td>
                  <span className="model-name" title={model.path}>
                    {model.name}
                  </span>
                  {model.name === defaultModel && <span className="badge accent">default</span>}
                  {model.name === loadedModel && <span className="badge ok">loaded</span>}
                </td>
                <td className="num">{formatSize(model.size_mib)}</td>
                <td className="num">
                  {model.max_context === null ? (
                    <span className="bad">cannot run here</span>
                  ) : (
                    formatCount(model.max_context)
                  )}
                </td>
                <td className="num">
                  {model.trained_context === null ? "-" : formatCount(model.trained_context)}
                </td>
                <td className={model.thinking === "none" ? "muted" : undefined}>
                  {thinkingLabel(model.thinking)}
                </td>
                <td>
                  {model.arch === null ? (
                    <span className="bad">not a readable GGUF</span>
                  ) : (
                    <>
                      {model.arch}
                      {model.sliding_window && (
                        <abbr className="muted" title="sliding-window attention">
                          {" "}
                          (swa)
                        </abbr>
                      )}
                    </>
                  )}
                </td>
                <td className="row-actions">
                  {model.name !== loadedModel && (
                    <button
                      className="small"
                      // The library has already said this one cannot hold a usable
                      // context here, so there is nothing for the button to try.
                      disabled={actions.busy !== null || model.max_context === null || loading}
                      title={
                        model.max_context === null
                          ? "This model cannot hold a usable context on this machine"
                          : loading
                            ? "Another model is loading"
                            : undefined
                      }
                      onClick={() => actions.run(`serve-${model.name}`, () => api.serve(model.name))}
                    >
                      Load
                    </button>
                  )}
                  {model.name !== defaultModel && (
                    <button
                      className="small"
                      disabled={actions.busy !== null}
                      onClick={() =>
                        actions.run(`default-${model.name}`, () => api.setDefault(model.name))
                      }
                    >
                      Make default
                    </button>
                  )}
                  <button
                    className="small"
                    disabled={actions.busy !== null}
                    onClick={() => onRemove(model.name)}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {failure && <p className="bad small footnote">{failure}</p>}
      {models.some((model) => model.sliding_window) && (
        <p className="muted small footnote">
          (swa) sliding-window attention: most layers hold a fixed window rather than growing
          with context, which is how a 12B reaches 256k.
        </p>
      )}
    </>
  );
}

/** The picker's line for one candidate: its context when it fits, or that it will not. */
export function fitLabel(candidate: Candidate): string {
  if (!candidate.fits) return "will not fit";
  if (candidate.context !== null) return `${formatCount(candidate.context)} context`;
  return `${formatSize(candidate.headroom_mib)} spare`;
}

export function CatalogueList({
  offer,
  actions,
  busy,
}: {
  offer: Offer;
  actions: Actions;
  /** An install is already running, and the server takes one at a time. */
  busy: boolean;
}) {
  if (offer.remote.length === 0) {
    return (
      <p className="muted">
        Nothing to add. Either every catalogue model that fits is installed, or the registry
        could not be reached - <code>ailocal model pick</code> says which.
      </p>
    );
  }

  return (
    <ul className="catalogue">
      {offer.remote.map((candidate) => {
        const reference = `ollama:${candidate.entry.reference}`;
        return (
          <li key={candidate.entry.reference} className={candidate.fits ? undefined : "unfit"}>
            <div className="catalogue-head">
              <code className="reference">{candidate.entry.reference}</code>
              <span className="muted small">{formatSize(candidate.size_mib)}</span>
              <span className={`badge ${candidate.fits ? "ok" : "bad"}`}>
                {fitLabel(candidate)}
              </span>
              <button
                className="small"
                aria-label={`Install ${candidate.entry.reference}`}
                // One that does not fit is downloadable, as it is from the picker with
                // --force. The header check runs either way and the job says what it found.
                disabled={busy || actions.busy !== null}
                title={busy ? "Another install is running" : undefined}
                onClick={() =>
                  actions.run(`install-${candidate.entry.reference}`, () =>
                    api.install(reference, !candidate.fits),
                  )
                }
              >
                Install
              </button>
            </div>
            <p>{candidate.entry.summary}</p>
            {candidate.entry.note && <p className="muted small">{candidate.entry.note}</p>}
            {actions.failed?.key === `install-${candidate.entry.reference}` && (
              <p className="bad small">{actions.failed.message}</p>
            )}
          </li>
        );
      })}
    </ul>
  );
}

/**
 * What `ailocal model search` and `model files` do: find a repository, then choose a
 * quant from it.
 *
 * Two steps rather than one list, because a search answers with repositories and a
 * repository holds a dozen quants of the same model. Flattening that would be a
 * hundred rows where twenty were asked for, and the choice between them is the only
 * one that matters here: which quantisation this machine can actually run.
 *
 * Hugging Face only. Ollama publishes no search endpoint, so what it has is the
 * catalogue above rather than anything findable from here.
 */
export function SearchHuggingFace({ actions, busy }: { actions: Actions; busy: boolean }) {
  const [query, setQuery] = useState("");
  const [repos, setRepos] = useState<Fetched<Repo[]>>({ kind: "idle" });
  // Which repository's quants are showing. One at a time: this is a drill-down into a
  // result, not twenty expandable rows to scroll past.
  const [open, setOpen] = useState<string | null>(null);
  const [files, setFiles] = useState<Fetched<Choice[]>>({ kind: "idle" });
  // Only the newest request may write its answer. Hugging Face is slow and variable
  // enough from here that a second search often overtakes the first.
  const latest = useRef(0);
  const trimmed = query.trim();

  function search() {
    const mine = ++latest.current;
    setRepos({ kind: "loading" });
    setOpen(null);
    setFiles({ kind: "idle" });
    api
      .search(trimmed)
      .then((value) => mine === latest.current && setRepos({ kind: "ok", value }))
      .catch((err: unknown) => {
        if (mine === latest.current) setRepos({ kind: "failed", message: message(err) });
      });
  }

  function show(repo: string) {
    if (open === repo) {
      setOpen(null);
      return;
    }
    const mine = ++latest.current;
    setOpen(repo);
    setFiles({ kind: "loading" });
    api
      .files(repo)
      .then((value) => mine === latest.current && setFiles({ kind: "ok", value }))
      .catch((err: unknown) => {
        if (mine === latest.current) setFiles({ kind: "failed", message: message(err) });
      });
  }

  return (
    <>
      <form
        className="install-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (trimmed !== "") search();
        }}
      >
        <input
          aria-label="Search Hugging Face"
          placeholder="qwen3 coder  or  gemma 12b"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          spellCheck={false}
        />
        <button type="submit" disabled={trimmed === "" || repos.kind === "loading"}>
          {repos.kind === "loading" ? "Searching" : "Search"}
        </button>
      </form>

      {repos.kind === "failed" && <p className="bad">{repos.message}</p>}
      {repos.kind === "ok" && repos.value.length === 0 && (
        <p className="muted">Nothing matched. Hugging Face matches on names, not on what a model does.</p>
      )}

      {repos.kind === "ok" && repos.value.length > 0 && (
        <ul className="catalogue">
          {repos.value.map((repo) => (
            <li key={repo.id}>
              <div className="catalogue-head">
                <code className="reference">{repo.id}</code>
                <span className="muted small">{formatDownloads(repo.downloads)} downloads</span>
                <button
                  className="small"
                  aria-expanded={open === repo.id}
                  // Named for the repository it belongs to. Twenty buttons all
                  // called "Quants" is one button as far as a screen reader is
                  // concerned, and the visible word is still the first of the name.
                  aria-label={`${open === repo.id ? "Hide quants" : "Quants"} in ${repo.id}`}
                  onClick={() => show(repo.id)}
                >
                  {open === repo.id ? "Hide quants" : "Quants"}
                </button>
              </div>
              {open === repo.id && (
                <Quants state={files} repo={repo.id} actions={actions} busy={busy} />
              )}
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

/** The quants in one repository, measured against this machine by the server. */
function Quants({
  state,
  repo,
  actions,
  busy,
}: {
  state: Fetched<Choice[]>;
  repo: string;
  actions: Actions;
  busy: boolean;
}) {
  if (state.kind === "loading") return <p className="muted small">Listing the files.</p>;
  if (state.kind === "failed") return <p className="bad small">{state.message}</p>;
  if (state.kind !== "ok") return null;
  if (state.value.length === 0) {
    return (
      <p className="muted small">
        No single-file GGUF here. A model split into shards cannot be loaded a piece at a
        time, so those are left out rather than offered and then refused.
      </p>
    );
  }

  return (
    <ul className="quants">
      {state.value.map((choice) => {
        const reference = `hf:${repo}/${choice.path}`;
        return (
          <li key={choice.path} className={choice.fits ? undefined : "unfit"}>
            <code>{choice.path}</code>
            <span className="muted small">{formatSize(choice.size_mib)}</span>
            <span className={`badge ${choice.fits ? "ok" : "bad"}`}>{quantFit(choice)}</span>
            <button
              className="small"
              aria-label={`Install ${choice.path}`}
              // Downloadable even when it does not fit, as from the catalogue: the
              // header check runs either way and the job reports what it found.
              disabled={busy || actions.busy !== null}
              title={busy ? "Another install is running" : undefined}
              onClick={() => actions.run(`install-${reference}`, () => api.install(reference, !choice.fits))}
            >
              Install
            </button>
            {actions.failed?.key === `install-${reference}` && (
              <p className="bad small">{actions.failed.message}</p>
            )}
          </li>
        );
      })}
    </ul>
  );
}

/**
 * What a quant's headroom means, in the words the CLI uses.
 *
 * A quant that is merely smaller than the budget still has to leave room for the KV
 * cache, so "fits" and "smaller than what I have" are different questions and the
 * middle case has to be sayable.
 */
export function quantFit(choice: Choice): string {
  if (choice.fits) return `${formatSize(choice.headroom_mib)} spare`;
  if (choice.headroom_mib < 0) return `${formatSize(-choice.headroom_mib)} too large`;
  return "no room for a context";
}
