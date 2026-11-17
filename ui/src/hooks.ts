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
