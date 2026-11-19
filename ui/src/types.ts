// The wire, as the library's own types serialise it.
//
// There is no translation layer on the server: these mirror the Rust structs field for
// field, so a field renamed there is a type error here rather than a blank cell.

/** `gguf::Thinking`: what the model's own chat template offers. */
export type Thinking = "none" | "toggle" | "effort";

/** `status::Gateway`. */
export type Gateway = "answering" | "port_taken" | "not_answering";

/** `status::Service`. */
export type Service = "running" | "installed_not_running" | "not_installed";

/** `status::NextStep`. */
export type NextStep =
  | "install_llama_server"
  | "install_model"
  | "start_gateway"
  | "fix_gateway"
  | "configure_harness"
  | "ready";

export interface Meta {
  version: string;
  config_path: string;
  harnesses: string[];
  log_hint: string;
}

/** `harness::PiModel`: a model that can run here, as a harness is told about it. */
export interface RunnableModel {
  id: string;
  context_window: number;
  thinking: Thinking;
}

/** `serve::Instance`: the llama-server that is resident, from its state file. */
export interface Instance {
  pid: number;
  model: string;
  path: string;
  host: string;
  port: number;
  context: number;
  cache_type: string;
  desktop_mib: number | null;
  cache_ram_mib: number | null;
  reasoning: string;
  /** `null` for a model a service runs, whose output goes to the service's log. */
  log: string | null;
}

/** `ailocal::server_memory_mib`: what the loaded server holds, and the word for it. */
export interface Memory {
  /** `memory` on unified memory, where it is the server's own footprint; `vram` on a card. */
  label: string;
  mib: number;
}

/** `null` when the gateway was not asked because it is not answering. */
export type Advertised = { models: string[] } | { error: string } | null;

/** `status::Overview`: what `ailocal status` reports. */
export interface Overview {
  models: RunnableModel[];
  default_model: string | null;
  loaded: Instance | null;
  gateway_url: string;
  gateway: Gateway;
  advertised: Advertised;
  services: [string, Service][];
  harnesses: string[];
  /** Why llama.cpp could not be found, with the command that installs it. */
  llama_server: string | null;
  next: NextStep;
}

/** `models::Installed`: one row of `ailocal model ls`. */
export interface Installed {
  name: string;
  path: string;
  size_mib: number;
  /** `null` when the file could not be parsed as GGUF. */
  arch: string | null;
  sliding_window: boolean;
  kv_bytes_per_token: number | null;
  trained_context: number | null;
  /** `null` when it cannot run here at all. */
  max_context: number | null;
  thinking: Thinking;
}

/** `registry::Fit`: whether a model can hold a usable context here. */
export type Fit =
  | { fits: number }
  | { context_too_small: number }
  | "weights_too_large"
  | "unknown";

/** `install::Header`: what the head of the file said. */
export type Header =
  | {
      read: {
        arch: string | null;
        trained_context: number | null;
        /** Vision or audio encoder tensors packed into the weights. */
        encoder_tensors: number | null;
        tensor_count: number;
      };
    }
  /** Not a GGUF this parser understands, which has always meant "could not judge". */
  | "not_gguf"
  /** The head could not be fetched, with the message the CLI would print. */
  | { unreadable: string };

/** `install::Check`: whether a model can load and run here, judged before downloading. */
export interface Check {
  header: Header;
  fit: Fit;
  weights_mib: number;
  available_mib: number;
  /** The CLI's own verdict, never worked out again here. `--force` is what overrides it. */
  blocked: boolean;
}

/** `download::Fetched`. */
export type Fetched = "already_present" | { downloaded: { bytes: number; resumed_from: number } };

/** `install::OnDisk`: a model now in the models directory. */
export interface OnDisk {
  name: string;
  path: string;
  outcome: Fetched;
}

/** `jobs::Kind`. */
export type JobKind = "install" | "serve";

/**
 * `jobs::State`, internally tagged on `phase`. The `installed` and `served` phases carry
 * their payload's fields alongside the tag, as serde writes a newtype variant.
 */
export type JobState =
  | { phase: "resolving" }
  | { phase: "checking"; weights_mib: number }
  | { phase: "downloading"; written: number; total: number | null; resumed_from: number }
  | { phase: "verifying"; total: number | null }
  | { phase: "preparing" }
  | { phase: "starting"; context: number; evicting: string | null }
  | ({ phase: "installed" } & OnDisk)
  | ({ phase: "served" } & Instance)
  | { phase: "refused" }
  | { phase: "cancelled"; written: number; total: number | null }
  | { phase: "failed"; error: string };

/** `jobs::Job`: work running in the console, followed over the event stream. */
export interface Job {
  id: number;
  kind: JobKind;
  /** The reference being installed, or the model being served. */
  target: string;
  force: boolean;
  check: Check | null;
  state: JobState;
}

/** `models::Repoint`: what became of the model service after the default changed. */
export type Repoint = "no_service" | "restarted" | "cleared" | "updated" | { failed: string };

/** `models::DefaultChange`. */
export type DefaultChange = "unchanged" | { changed: Repoint };

/** A step that failed, as `wire::result` writes it. */
export interface Failed {
  error: string;
}

/** `models::Removal`: what deleting a model would involve. */
export interface Removal {
  name: string;
  path: string;
  size_mib: number;
  /** llama-server has it open, so it will be stopped first. */
  loaded: boolean;
  /** It is `default_model`, which will be cleared. */
  is_default: boolean;
}

/** `models::Removed`: what deleting it did, beyond the delete. */
export interface Removed {
  stopped: boolean;
  /** `null` when it was not the default. */
  default: Repoint | Failed | null;
  /** Whether Pi had an entry for it, which is now gone. */
  pi: boolean | Failed;
  claude_code_pinned: boolean;
}

/** `read::Log`: where the loaded model's output is. */
export type LogView =
  | { kind: "idle" }
  /** A model a service runs, or one in the foreground: not in a file ailocal keeps. */
  | { kind: "elsewhere"; service_log: string }
  | { kind: "file"; path: string; from: number; to: number; text: string };

/** `catalogue::Entry`. */
export interface Entry {
  reference: string;
  summary: string;
  note: string;
  /** A judgement, 1 (tiny) to 5 (specialist), not a measurement. */
  tier: number;
}

/** `catalogue::Candidate`: a catalogue model not on disk, measured against this machine. */
export interface Candidate {
  entry: Entry;
  size_mib: number;
  headroom_mib: number;
  /** The context a launch would get, once its header has been read. */
  context: number | null;
  /** Whether it clears the picker's line. Decided by the library, never here. */
  fits: boolean;
}

/** `models::Downloaded`: an installed model the picker offers. */
export interface Downloaded {
  name: string;
  size_mib: number;
  context: number;
  summary: string | null;
}

/** `models::Offer`: what `ailocal model pick` offers. */
export interface Offer {
  available_mib: number;
  downloaded: Downloaded[];
  remote: Candidate[];
  unusable: number;
}

/** `config::Config`, as the config file spells it. Paths derived from `data_dir` are omitted. */
export interface Config {
  data_dir: string;
  models_dir?: string;
  hf_home?: string;
  eval_dir?: string;
  cache_type: string;
  reasoning: string;
  reasoning_budget: number;
  max_context: number;
  default_model: string | null;
  system_prompt?: string;
  gateway_host: string;
  gateway_port: number;
}

// --- setup ---

/** `setup::Unit`: one service, as the manager reports it. */
export interface Unit {
  name: string;
  active: boolean;
  enabled: boolean;
}

/** `setup::Services`. */
export interface Services {
  /** Without it the units only start after a login, so a headless reboot leaves the
   * gateway down. */
  linger: boolean;
  units: Unit[];
}

/** `setup::Installed`: what writing the units did. */
export interface ServicesInstalled {
  units: string[];
  unit_dir: string;
  exe: string;
  chose_default: string | null;
  gateway_only: boolean;
  /** `null` when the units were written but not started. */
  answering: boolean | null;
  linger: boolean;
}

/** `setup::Trouble`: why nothing answers as a gateway. */
export type Trouble =
  | "not_running"
  | { port_taken: { free: number } }
  | { wrong_port: { expected: number } };

/** What `GET /gateway` reports. */
export interface GatewayView {
  url: string;
  port: number;
  state: Gateway;
  trouble: Trouble | null;
  key: string | null;
}

/** `gateway::Probe`: one assertion `gateway check` makes. */
export interface Probe {
  label: string;
  expected: number;
  /** The status it gave, or `{ error }` when the request did not finish. */
  got: number | { error: string };
}

/** `gateway::Check`. */
export interface GatewayCheck {
  base: string;
  reachable: boolean;
  probes: Probe[];
}

/** `setup::HarnessPlan`: what pointing a harness at the gateway would do. */
export interface HarnessPlan {
  harness: string;
  url: string;
  model: string;
  context: number;
  why: string;
  others: string[];
  /** Models whose chat template offers no thinking mode at all. */
  without_thinking: string[];
  trouble: Trouble | null;
  already_configured: boolean;
}

/** One harness, as `GET /harnesses` lists it. */
export interface HarnessView {
  name: string;
  configured: boolean;
  /** The file that points it at us. Claude Code's has to be sourced by hand. */
  path: string | null;
  /** `{ error }` when nothing here could be pointed at it. */
  plan: HarnessPlan | { error: string };
}

/** `setup::HarnessConfigured`: the plan, plus what was written. */
export type HarnessConfigured = HarnessPlan & {
  unchanged: boolean;
  path: string | null;
  backup: string | null;
};

/** `setup::Settings`: only the fields being changed. */
export interface SettingsPatch {
  cache_type?: string;
  reasoning?: string;
  reasoning_budget?: number;
  max_context?: number;
  gateway_port?: number;
  /** `null` clears the house rule; leaving it out keeps it. */
  system_prompt?: string | null;
}

/** `setup::Applied`: what changed, and what it means for what is running. */
export interface Applied {
  changed: string[];
  path: string | null;
  gateway_restarted: null | { error: string } | Record<string, never>;
  /** The resident model still on the old context, which has to reload to pick it up. */
  reload_model: string | null;
  repoint_harnesses: boolean;
}
