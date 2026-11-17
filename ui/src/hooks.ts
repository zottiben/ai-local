// Loading data, without a data library.
//
// Three states and a refetch is the whole requirement. The previous value stays on
// screen while the next one loads, which is what keeps a live update from blinking a
// panel out of existence every time a file changes.

import { useEffect, useRef, useState } from "react";

export interface Resource<T> {
  data: T | undefined;
  error: Error | undefined;
  /** True only until the first answer. A background refresh must not blank the screen. */
  loading: boolean;
}

/**
 * @param deps Fetch again when any of these change. Pass the live generation here to
 *   follow changes other processes make.
 */
export function useResource<T>(fetcher: () => Promise<T>, deps: readonly unknown[]): Resource<T> {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<Error>();
  const [loading, setLoading] = useState(true);

  // A slow answer that lands after a newer one would otherwise overwrite fresher data
  // with staler data. The catalogue's first answer takes ten seconds, so this happens.
  const generation = useRef(0);

  useEffect(() => {
    const mine = ++generation.current;
    let cancelled = false;
    const current = () => !cancelled && mine === generation.current;

    fetcher()
      .then((value) => {
        if (!current()) return;
        setData(value);
        setError(undefined);
      })
      .catch((err: unknown) => {
        if (!current()) return;
        setError(err instanceof Error ? err : new Error(String(err)));
      })
      .finally(() => {
        if (current()) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
    // The caller names what the fetcher depends on; the fetcher itself is a new
    // closure on every render.
  }, deps);

  return { data, error, loading };
}

export interface Actions {
  /** Start one, remembering which is in flight and what it said if it failed. */
  run: (key: string, action: () => Promise<unknown>) => void;
  /** The key of the action in flight, or `null`. */
  busy: string | null;
  /** The last failure, with the key of the button that caused it. */
  failed: { key: string; message: string } | null;
}

/**
 * One action at a time, for buttons that change the machine.
 *
 * Everything they do is machine-wide and single-tenant - one model is resident, one
 * install runs, one default is set - so serialising them costs nothing real and makes
 * a double-click impossible. The key is the button's, so a failure appears beside the
 * one that caused it rather than at the top of the page.
 */
export function useActions(): Actions {
  const [busy, setBusy] = useState<string | null>(null);
  const [failed, setFailed] = useState<{ key: string; message: string } | null>(null);
  const live = useRef(true);

  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
    };
  }, []);

  function run(key: string, action: () => Promise<unknown>) {
    if (busy !== null) return;
    setBusy(key);
    setFailed(null);
    action()
      .catch((err: unknown) => {
        if (live.current) {
          setFailed({ key, message: err instanceof Error ? err.message : String(err) });
        }
      })
      .finally(() => {
        if (live.current) setBusy(null);
      });
  }

  return { run, busy, failed };
}
