// Talking to the console server.
//
// The token arrives in the URL because that is the only channel a freshly opened tab
// has. It is moved into memory and stripped from the address bar at once: left there it
// puts a live credential into the history, into any screenshot, and into every Referer
// header the page sends.

import type { Config, Installed, Instance, Meta, Offer, Overview } from "./types";

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

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`/api${path}`, { headers: { "x-ailocal-token": token } });
  if (!response.ok) {
    // A body that is not JSON means something upstream of the handler broke. Say that
    // rather than throwing a parse error over the top of it.
    const body = await response.json().catch(() => ({
      error: `${response.status} ${response.statusText}`,
    }));
    throw new ApiError(response.status, body);
  }
  return (await response.json()) as T;
}

export const api = {
  meta: () => request<Meta>("/meta"),
  overview: () => request<Overview>("/overview"),
  models: () => request<Installed[]>("/models"),
  catalogue: () => request<Offer>("/catalogue"),
  config: () => request<Config>("/config"),
  loaded: () => request<Instance | null>("/loaded"),
};
