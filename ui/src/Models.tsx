import { api } from "./api";
import { formatCount, formatSize, thinkingLabel } from "./format";
import { useResource } from "./hooks";
import type { Candidate, Installed, Offer } from "./types";

/** What `ailocal model ls` lists, then what `ailocal model pick` offers to add. */
export function Models({ generation }: { generation: number }) {
  const models = useResource(() => api.models(), [generation]);
  const config = useResource(() => api.config(), [generation]);
  const loaded = useResource(() => api.loaded(), [generation]);
  // Refetched on every change too. The server keeps the registry's answer for the run,
  // so this costs a directory scan, and a model installed from a terminal leaves this
  // list for the one above without a reload.
  const catalogue = useResource(() => api.catalogue(), [generation]);

  return (
    <>
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
          />
        )}
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Catalogue</h2>
          {catalogue.data && (
            <span className="muted small">
              measured against {formatSize(catalogue.data.available_mib)} free for a model
            </span>
          )}
        </div>
        {catalogue.error ? (
          <p className="bad">{catalogue.error.message}</p>
        ) : !catalogue.data ? (
          <p className="muted">
            Asking the registry about each model and reading the head of each file. The first
            time takes about ten seconds; after that it is kept until the console stops.
          </p>
        ) : (
          <CatalogueList offer={catalogue.data} />
        )}
      </section>
    </>
  );
}

export function InstalledTable({
  models,
  defaultModel,
  loadedModel,
}: {
  models: Installed[];
  defaultModel: string | null;
  loadedModel: string | null;
}) {
  if (models.length === 0) {
    return (
      <p className="muted">
        No models yet. <code>ailocal model pick</code> offers the ones that fit this machine.
      </p>
    );
  }

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
              </tr>
            ))}
          </tbody>
        </table>
      </div>
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

export function CatalogueList({ offer }: { offer: Offer }) {
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
      {offer.remote.map((candidate) => (
        <li key={candidate.entry.reference} className={candidate.fits ? undefined : "unfit"}>
          <div className="catalogue-head">
            <code className="reference">{candidate.entry.reference}</code>
            <span className="muted small">{formatSize(candidate.size_mib)}</span>
            <span className={`badge ${candidate.fits ? "ok" : "bad"}`}>{fitLabel(candidate)}</span>
          </div>
          <p>{candidate.entry.summary}</p>
          {candidate.entry.note && <p className="muted small">{candidate.entry.note}</p>}
        </li>
      ))}
    </ul>
  );
}
