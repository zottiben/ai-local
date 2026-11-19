// Two routes, on the History API. A router library would add more to the committed
// bundle than the feature is worth.

import { useSyncExternalStore } from "react";

export type View = "overview" | "models" | "setup";

export const PATHS: Record<View, string> = {
  overview: "/",
  models: "/models",
  setup: "/setup",
};

const listeners = new Set<() => void>();

function subscribe(listener: () => void) {
  listeners.add(listener);
  window.addEventListener("popstate", listener);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("popstate", listener);
  };
}

export function navigate(view: View) {
  const path = PATHS[view];
  if (path === window.location.pathname) return;
  window.history.pushState(null, "", path);
  for (const listener of listeners) listener();
}

/** Anything that is not a known path is the overview, as the server serves it. */
export function parse(path: string): View {
  const trimmed = path.replace(/\/+$/, "");
  if (trimmed === PATHS.models) return "models";
  if (trimmed === PATHS.setup) return "setup";
  return "overview";
}

export function useView(): View {
  return parse(
    useSyncExternalStore(
      subscribe,
      () => window.location.pathname,
      () => "/",
    ),
  );
}
