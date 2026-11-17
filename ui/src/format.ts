// The CLI's words, for the window.
//
// Every decision here was made by the library - which step is next, whether a model
// fits - and these functions only say it. Where the terminal has a phrasing, this uses
// it, so the two never describe the same machine differently.

import type {
  Check,
  Failed,
  Gateway,
  Job,
  JobState,
  Meta,
  NextStep,
  Removed,
  Repoint,
  Service,
  Thinking,
} from "./types";

/** `131072` -> `128k`, as `model ls` prints a context. An odd size is printed whole. */
export function formatCount(n: number): string {
  return n >= 1024 && n % 1024 === 0 ? `${n / 1024}k` : String(n);
}

/** `7039` MiB -> `6.9 GiB`. MiB below a GiB, because `0 GiB` is the wrong impression. */
export function formatSize(mib: number): string {
  return mib >= 1024 ? `${(mib / 1024).toFixed(1)} GiB` : `${mib} MiB`;
}

/**
 * Bytes as a size. A download reports bytes, and `0 MiB` for the first second of one
 * is a bar that looks broken, so below 10 MiB this keeps a decimal.
 */
export function formatBytes(bytes: number): string {
  const mib = bytes / (1024 * 1024);
  if (mib >= 1024) return `${(mib / 1024).toFixed(1)} GiB`;
  if (mib >= 10) return `${Math.round(mib)} MiB`;
  return `${mib.toFixed(1)} MiB`;
}

/** A step that failed, as `wire::result` writes it, or `null` when it succeeded. */
export function failure(value: unknown): string | null {
  return typeof value === "object" && value !== null && "error" in value
    ? String((value as Failed).error)
    : null;
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

/** What the head of the file said, and why it will not run when it will not. */
export interface Verdict {
  /** The library's own answer. Never worked out again here. */
  blocked: boolean;
  /** What was read, as `model install` prints it. */
  summary: string;
  /** Why it is blocked, when it is. */
  detail?: string;
}

/**
 * `refuse_unless_forced`, for a window.
 *
 * The same verdict in the same order the command prints it: what the header said, then
 * the encoder, then the fit. Reflowed rather than hard-wrapped, and "pass --force"
 * becomes the button beside it, because that is what the override is here.
 */
export function describeCheck(check: Check): Verdict {
  const blocked = check.blocked;

  if (typeof check.header === "object" && "unreadable" in check.header) {
    return {
      blocked,
      summary: "Could not read its header",
      detail: `${check.header.unreadable}. Nothing has looked at whether it loads here; downloading it is unchecked.`,
    };
  }
  if (check.header === "not_gguf") {
    return { blocked, summary: "Not a GGUF this version understands, so nothing was judged" };
  }

  const { arch, trained_context, encoder_tensors, tensor_count } = check.header.read;
  const summary = `arch ${arch ?? "?"}, trained to ${
    trained_context === null ? "?" : formatCount(trained_context)
  }`;

  if (encoder_tensors !== null && encoder_tensors > 0) {
    return {
      blocked,
      summary,
      detail:
        `Will not load: ${encoder_tensors} of its ${tensor_count} tensors are a vision or ` +
        "audio encoder packed into the weights, and llama.cpp only loads one from a " +
        "separate mmproj file - as GGUFs converted by llama.cpp itself ship them.",
    };
  }
  return { blocked, summary, detail: describeFit(check) };
}

function describeFit(check: Check): string | undefined {
  if (check.fit === "unknown") return "Could not read its metadata, so the fit was not checked.";
  if (check.fit === "weights_too_large") {
    return `Will not fit: ${check.weights_mib} MiB of weights against ${check.available_mib} MiB available.`;
  }
  if ("context_too_small" in check.fit) {
    return `Will not fit: only ${check.fit.context_too_small} tokens of context on this machine.`;
  }
  return undefined;
}

/** The line for a job's phase. A download also draws a bar, from the same numbers. */
export function describeJob(job: Job): string {
  const state: JobState = job.state;
  switch (state.phase) {
    case "resolving":
      return "Asking the registry what this is";
    case "checking":
      return `Reading its header to see whether it runs here (${state.weights_mib} MiB)`;
    case "downloading":
      return state.resumed_from > 0
        ? `Resumed from ${formatBytes(state.resumed_from)}`
        : "Downloading";
    case "verifying":
      return "Checking its digest, which reads the whole file back";
    case "preparing":
      return "Sizing the context against what is free";
    case "starting":
      return (
        `Loading at ${formatCount(state.context)} context` +
        (state.evicting === null ? "" : `, evicting ${state.evicting}`)
      );
    case "installed":
      return state.outcome === "already_present"
        ? `Already on disk as ${state.name}`
        : `Installed as ${state.name} (${formatBytes(state.outcome.downloaded.bytes)})`;
    case "served":
      return `Up at ${state.host}:${state.port} with ${formatCount(state.context)} context`;
    case "refused":
      return "Not downloaded: it will not load or fit here";
    case "cancelled":
      return `Stopped at ${formatBytes(state.written)}${
        state.total === null ? "" : ` of ${formatBytes(state.total)}`
      } - installing it again carries on from there`;
    case "failed":
      return state.error;
  }
}

/** What became of the model service, as `print_repoint` says it. */
export function repointLabel(repoint: Repoint): string | null {
  if (repoint === "no_service") return null;
  if (repoint === "restarted") return "restarted the model service on it";
  if (repoint === "cleared") return "stopped the model service - no default model left to preload";
  if (repoint === "updated") return "updated the model service";
  return (
    `the model service still points at the old model (${repoint.failed}) - ` +
    "run `ailocal service install` to update it"
  );
}

/** Everything a delete also did, as `model rm` prints it after the file is gone. */
export function describeRemoved(removed: Removed, name: string): string[] {
  const lines: string[] = [];
  if (removed.stopped) lines.push(`stopped ${name}`);

  if (removed.default !== null) {
    const failed = failure(removed.default);
    if (failed !== null) {
      lines.push(`could not clear default_model: ${failed}`);
    } else {
      lines.push("cleared default_model");
      const repoint = repointLabel(removed.default as Repoint);
      if (repoint !== null) lines.push(repoint);
    }
  }

  const piFailed = failure(removed.pi);
  if (piFailed !== null) lines.push(`could not update pi's models.json: ${piFailed}`);
  else if (removed.pi === true) lines.push("removed pi's entry for it");

  // Claude Code's env file pins one model by name and is sourced by hand, so it cannot
  // be repointed from here - say what to run.
  if (removed.claude_code_pinned) {
    lines.push(
      `claude-code is still pointed at ${name} - run \`ailocal harness configure claude-code\` to move it`,
    );
  }
  return lines;
}

/** `/Users/me/.ailocal` -> `~/.ailocal`, which is how people say it. */
export function tilde(path: string): string {
  const home = /^(\/(?:Users|home)\/[^/]+)(?=\/|$)/.exec(path)?.[1];
  return home ? `~${path.slice(home.length)}` : path;
}
