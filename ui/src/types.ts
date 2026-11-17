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
