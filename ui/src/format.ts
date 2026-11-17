// The CLI's words, for the window.
//
// Every decision here was made by the library - which step is next, whether a model
// fits - and these functions only say it. Where the terminal has a phrasing, this uses
// it, so the two never describe the same machine differently.

import type { Gateway, Meta, NextStep, Service, Thinking } from "./types";

/** `131072` -> `128k`, as `model ls` prints a context. An odd size is printed whole. */
export function formatCount(n: number): string {
  return n >= 1024 && n % 1024 === 0 ? `${n / 1024}k` : String(n);
}

/** `7039` MiB -> `6.9 GiB`. MiB below a GiB, because `0 GiB` is the wrong impression. */
export function formatSize(mib: number): string {
  return mib >= 1024 ? `${(mib / 1024).toFixed(1)} GiB` : `${mib} MiB`;
}

/** The THINK column of `model ls`. */
export function thinkingLabel(thinking: Thinking): string {
  return { none: "no thinking mode", toggle: "on/off", effort: "effort levels" }[thinking];
}

export function gatewayLabel(gateway: Gateway): string {
  return {
    answering: "answering",
    port_taken: "port held by something else",
    not_answering: "not answering",
  }[gateway];
}

export function serviceLabel(service: Service): string {
  return {
    running: "running",
    installed_not_running: "installed but not running",
    not_installed: "not installed",
  }[service];
}

/**
 * `io.github.zottiben.ailocal.gateway` -> `gateway`, `ailocal-model.service` -> `model`.
 *
 * launchd and systemd name the same two services differently, and the part a person
 * reads is the last word once systemd's suffix is off.
 */
export function serviceShortName(name: string): string {
  return name.replace(/\.service$/, "").split(/[.-]/).pop() ?? name;
}

export interface Step {
  title: string;
  /** The commands, as `ailocal status` prints them. */
  commands: string[];
  why: string;
  /** Where the gateway's log is: a `journalctl` command on Linux, a file on macOS. */
  log?: string;
}

/** What `NextStep::describe` says, laid out for a panel. `null` when nothing is missing. */
export function describeStep(next: NextStep, meta: Pick<Meta, "log_hint" | "harnesses">): Step | null {
  switch (next) {
    case "install_model":
      return {
        title: "Choose a model for this machine",
        commands: ["ailocal model pick"],
        why: "Nothing installed can run here yet.",
      };
    case "start_gateway":
      return {
        title: "Start the gateway",
        commands: ["ailocal service install"],
        why: "Nothing is serving yet. This starts the gateway and keeps it running.",
      };
    case "fix_gateway":
      return {
        title: "The gateway service is installed but not answering",
        commands: ["ailocal gateway check"],
        why: "The check says what is wrong.",
        log: meta.log_hint,
      };
    case "configure_harness":
      return {
        title: "Point a harness at the gateway",
        commands: meta.harnesses.map((h) => `ailocal harness configure ${h}`),
        why: "The gateway is serving, but no harness is configured to use it.",
      };
    case "ready":
      return null;
  }
}

/** `/Users/me/.ailocal` -> `~/.ailocal`, which is how people say it. */
export function tilde(path: string): string {
  const home = /^(\/(?:Users|home)\/[^/]+)(?=\/|$)/.exec(path)?.[1];
  return home ? `~${path.slice(home.length)}` : path;
}
