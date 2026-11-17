// Show the plan, then delete (D6).
//
// The plan is the library's own `plan_removal`, the one `ailocal model rm` prints
// before its prompt - not a sentence this page composes. A 17 GB re-download over a
// link that is not always fast is the cost of getting this wrong, and the things that
// also change - the default model, the model service, Pi's entry - are each a
// confusing failure minutes or a reboot away from the delete that caused them.

import { useEffect, useState } from "react";

import { api } from "./api";
import { describeRemoved, formatSize, tilde } from "./format";
import { useResource } from "./hooks";
import type { Removed } from "./types";

interface Props {
  name: string;
  onClose: () => void;
}

export function RemoveDialog({ name, onClose }: Props) {
  const plan = useResource(() => api.removal(name), [name]);
  const [removed, setRemoved] = useState<Removed | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Escape closes, as it does in every dialog. Not while the delete is in flight:
  // there is nothing left to cancel by then, and closing would hide what it did.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !deleting) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [deleting, onClose]);

  async function remove() {
    setDeleting(true);
    setError(null);
    try {
      setRemoved(await api.remove(name));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setDeleting(false);
    }
  }

  return (
    <div className="scrim" onClick={() => !deleting && onClose()} role="presentation">
      <div
        className="dialog"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={`Delete ${name}`}
      >
        <header>
          <h2>{removed ? `Deleted ${name}` : `Delete ${name}?`}</h2>
          <button className="ghost" onClick={onClose} disabled={deleting} aria-label="Close">
            ✕
          </button>
        </header>

        <div className="dialog-body">
          {removed ? (
            <Outcome removed={removed} name={name} />
          ) : plan.error ? (
            <p className="bad">{plan.error.message}</p>
          ) : !plan.data ? (
            <p className="muted">Working out what would change…</p>
          ) : (
            <>
              <ul className="actions">
                <li>
                  delete {tilde(plan.data.path)}, freeing {formatSize(plan.data.size_mib)}
                </li>
                {plan.data.loaded && <li>stop it first - it is loaded right now</li>}
                {plan.data.is_default && <li>clear default_model, which names it</li>}
              </ul>
              <p className="muted small">
                The weights have to be downloaded again to undo this.
              </p>
            </>
          )}
          {error && <p className="bad">{error}</p>}
        </div>

        <footer>
          {removed ? (
            <button className="primary" onClick={onClose}>
              Close
            </button>
          ) : (
            <>
              <button onClick={onClose} disabled={deleting}>
                Keep it
              </button>
              <button
                className="primary danger"
                onClick={() => void remove()}
                disabled={deleting || !plan.data}
              >
                {deleting ? "Deleting…" : "Delete"}
              </button>
            </>
          )}
        </footer>
      </div>
    </div>
  );
}

/** Everything the delete also did, in the words `model rm` prints after it. */
function Outcome({ removed, name }: { removed: Removed; name: string }) {
  const lines = describeRemoved(removed, name);
  return (
    <>
      <p>The weights are gone.</p>
      {lines.length > 0 && (
        <ul className="actions">
          {lines.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      )}
    </>
  );
}
