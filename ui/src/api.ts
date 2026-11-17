// Talking to the console server.
//
// The token arrives in the URL because that is the only channel a freshly opened tab
// has. It is moved into memory and stripped from the address bar at once: left there it
// puts a live credential into the history, into any screenshot, and into every Referer
// header the page sends.

import type {
  Config,
  DefaultChange,
  Installed,
  Instance,
  LogView,
  Memory,
  Meta,
  Offer,
  Overview,
  Removal,
  Removed,
} from "./types";

const TOKEN_KEY = "ailocal.token";

function takeToken(): string {
  const url = new URL(window.location.href);
  const fromUrl = url.searchParams.get("t");
  if (fromUrl) {
    // Session storage, not local: the token dies with the process that minted it, so
    // keeping it beyond the tab would only ever produce a confusing 401 later.
    sessionStorage.setItem(TOKEN_KEY, fromUrl);
    url.searchParams.delete("t");
    window.history.replaceState(null, "", url.pathname + url.search + url.hash);
    return fromUrl;
  }
  return sessionStorage.getItem(TOKEN_KEY) ?? "";
}

export const token = takeToken();

/** A refusal from the server, with the message the CLI would have printed. */
export class ApiError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(status: number, body: { error?: string; code?: string }) {
    super(body.error ?? `request failed (${status})`);
    this.name = "ApiError";
    this.status = status;
    this.code = body.code ?? "unknown";
  }
}

async function request<T>(path: string, method = "GET", body?: unknown): Promise<T> {
  const response = await fetch(`/api${path}`, {
    method,
    headers: {
      "x-ailocal-token": token,
      ...(body === undefined ? {} : { "content-type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    // A body that is not JSON means something upstream of the handler broke. Say that
    // rather than throwing a parse error over the top of it.
    const failure = await response.json().catch(() => ({
      error: `${response.status} ${response.statusText}`,
    }));
    throw new ApiError(response.status, failure);
  }
  // 204 for the routes that only act. Reading a body that is not there would throw
  // over the top of a request that succeeded.
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

/** A job has started. Its progress arrives on the event stream, not here. */
interface Started {
  id: number;
}

export const api = {
  meta: () => request<Meta>("/meta"),
  overview: () => request<Overview>("/overview"),
  models: () => request<Installed[]>("/models"),
  catalogue: () => request<Offer>("/catalogue"),
  config: () => request<Config>("/config"),
  loaded: () => request<Instance | null>("/loaded"),
  memory: () => request<Memory | null>("/loaded/memory"),
  /** From `from`, or the log's last window when it is not given. */
  log: (from?: number) =>
    request<LogView>(`/loaded/log${from === undefined ? "" : `?from=${from}`}`),
  removal: (name: string) => request<Removal>(`/models/${encodeURIComponent(name)}/removal`),

  install: (reference: string, force = false) =>
    request<Started>("/install", "POST", { reference, force }),
  serve: (name: string) => request<Started>("/serve", "POST", { name }),
  cancel: (id: number) => request<void>(`/jobs/${id}/cancel`, "POST"),
  dismiss: (id: number) => request<void>(`/jobs/${id}`, "DELETE"),
  remove: (name: string) => request<Removed>(`/models/${encodeURIComponent(name)}`, "DELETE"),
  setDefault: (name: string) => request<DefaultChange>("/default", "PUT", { name }),
  stop: () => request<Instance | null>("/stop", "POST"),
};
