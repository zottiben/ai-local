# ai-local - project knowledge

A Rust CLI and gateway to download, serve, fine-tune and expose local LLMs to my coding
harnesses (Pi, Codex, Claude Code), locally and remotely via a Cloudflare tunnel.

**Stack:** Rust (CLI, registry, gateway) + a pinned Python sidecar for training only.
Inference is llama.cpp's `llama-server`, Vulkan backend. Models are GGUF from Hugging Face.

**Layout**
- `crates/` - Rust workspace (CLI, registry, gateway)
- `python/` - training sidecar, its own uv-managed venv, never the system interpreter
- `$data_dir` - all weights, adapters, datasets and caches. Not in this repo. Defaults
  to `~/.ailocal`; **on this machine it is `/mnt/kingston/ailocal`** (see rule 1).

**Commands** (from repo root)
- Build / lint / test: `cargo build`, `cargo clippy -- -D warnings`, `cargo test`
- Plan: `aip status` (the build plan is in ai-planner, not in a markdown file)
- Install a model: `ailocal model install ollama:<name>:<tag>` (or `hf:<owner>/<repo>/<file>`)
- Delete one: `ailocal model rm <name>` (aliases `remove`, `delete`; `--yes` to skip the prompt)
- Choose which model to use: `ailocal model pick` (also sets `default_model`)
- What is on disk, what fits, and which can think: `ailocal model ls`
- Run one: `ailocal serve <name>`, then `ailocal ps` / `ailocal stop`
- Where am I, and what is missing: `ailocal status`
- Expose it to harnesses: `ailocal gateway run` (key via `ailocal gateway key`)
- Move the gateway off a busy port: `ailocal config gateway-port <n>`
- Point a harness at it: `ailocal harness configure pi|claude-code [--model <name>]`
- Add a house rule to every harness: `system_prompt` in the config
- Run it 24/7: `ailocal service install`, then `ailocal service status`
- Score a model: `ailocal extras install eval`, then `ailocal eval run`
- Score its habits as a coding agent in Pi: `ailocal eval agent`
- Open the console: `ailocal extras install ui`, then `ailocal ui`
- Build the desktop app: `cd crates/ailocal-desktop && npx @tauri-apps/cli@^2 build`
  (`--no-bundle` for just the binary). It is a workspace member but **not** a default
  member, so plain `cargo build` / `clippy` / `test` skip it - that is deliberate, and
  why those commands must not be given `--workspace`
- After editing `ui/src`: `cd ui && npm ci && npm run build`. `ui/dist` is committed and
  compiled into `ailocal-ui`; CI fails if it is not what the sources build
- Check a model's real context limit: `scripts/bench/ctx_probe.sh <model.gguf> q8_0`

The Rust workspace is established by PR0; until then the cargo commands have nothing to
run against.

## Hard rules

### 1. Nothing large is written outside the configured data dir
Every large artifact goes under `Config::data_dir`, never a library default and never a
hardcoded path. `models_dir`, `hf_home` and `eval_dir` derive from it, so one setting
moves everything. A new absolute path in the source is a bug: this runs on more than one
machine and their disks differ.

Any code path touching the Hugging Face hub must set `HF_HOME` from the config. The
library default is `~/.cache/huggingface`, which is how a single download fills a home
partition.

**On this machine** `data_dir = /mnt/kingston/ailocal`, because `/` has ~17 GB free and
`/home` ~13 GB while kingston has 325 GB. A single model fills either of the first two.
That is a fact about this box, not a default - `~/.ailocal` is, and `ailocal setup` asks.

### 1b. Hugging Face is slow here; prefer the Ollama registry
Measured from Adelaide: `registry.ollama.ai` 79-97 MB/s, `huggingface.co` 0.016-2.7 MB/s
and wildly variable even with a Pro token. The `hf_xet` client stalls outright. Ollama
registry blobs are plain content-addressed GGUFs and need no ollama install - see
`scripts/bench/pull_ollama.sh`. Its limit is quant coverage: default Q4_K_M only, so
IQ3/Q3 still has to come from HF (use plain authenticated `curl -C -`, not `hf`).

### 2. Never use the system Python
It is 3.14.6 and no ML library ships wheels for it. The sidecar gets its own 3.11/3.12
venv under `$data_dir/venv`, created and verified by the Rust CLI.

### 3. Never exceed 14400 MiB of VRAM, and size weights against ~10 GiB
The card has 16368 MiB and the desktop holds ~900 MiB at idle, but it allocates *new*
framebuffers on demand. Over-committing does not fail gracefully: the compositor loses
its framebuffer (`amdgpu pin failed`, `-12`) and the graphical session dies. This
crashed the machine once already.

```
total ceiling      14400 MiB   (enforce this, never probe past it)
- desktop            ~900 MiB
= llama-server     ~13500 MiB   for weights + KV cache + compute buffers
=> usable weights   ~10 GiB     not 14
```

So a 24B at Q4 (13.3 GiB) cannot hold even a 4096 context and is unusable. Models must
fit entirely in VRAM - there are only 8.5 GB of system RAM, so CPU offload thrashes.
Use `--cache-type-k q8_0 --cache-type-v q8_0`: it roughly halves the KV cache at no
measurable throughput cost.

`ailocal serve` enforces this: it sizes the context from the model's own KV geometry,
refuses a larger `--ctx` rather than clamping it, and watches VRAM while loading so a
mis-estimate costs a failed start instead of the session.

### 3b. Both current models are reasoning models
gemma4-12b and qwen3-14b emit chain-of-thought into `reasoning_content` and only fill
`content` afterwards. A small `max_tokens` therefore returns an **empty answer** with
`finish_reason: length` - gemma4 burned ~700 tokens thinking about "say hello in three
words" and never reached an answer.

`reasoning = "off"` is therefore the config default, and it is the measured answer
rather than a guess: on the PR11 suite gemma4-12b scored 95% off against 45% on, at
4.6x the wall clock, with 7 of 14 answers empty. Re-measure per model with
`ailocal eval run --reasoning off --reasoning on` before changing it.

### 3c. The advertised window is per slot, so every slot must own it
With `-np` unset llama-server runs 4 slots over **one** KV pool of `-c` cells, and
`/props` reports the whole `-c` for each. Two pi sessions at 71k and 56k on a 128k
server both died in the same second with `Context size has been exceeded.` - a 500
that pi does not treat as an overflow, so it stops rather than compacts. That pool
also clears idle slots on every new task, so concurrent sessions re-prefill each
other from zero.

`serve` therefore pins `-np N --no-kv-unified -c N*context`, where N is
`vram::SLOTS` or fewer when a whole window per slot does not fit. Never advertise
more than one slot owns, and never split one window to make room for a second slot.
An over-window request then gets the 400 `exceeds the available context size`,
which pi does compact on.

### 4. ROCm never touches the inference path
Inference is Vulkan. gfx1102 ROCm is flaky (upstream segfaults, hipBLASLt reports the
arch unsupported). ROCm is for training only, so a training-side breakage can never take
down the daily driver.

### 5. Services are systemd **user** units
They need the GPU and the user's home, and installing them needs no root - which matters
because `sudo` here prompts for a password. The one privileged part is lingering: without
`sudo loginctl enable-linger <user>` the units only start after a login, so a headless
reboot leaves the gateway down.

`ailocal serve --foreground` exists for this: systemd must supervise llama-server
directly, not a command that forks and returns.

**Never spawn a tool by bare name.** A service gets whatever `PATH` its manager hands
it, and launchd hands out `/usr/bin:/bin:/usr/sbin:/sbin` - no `/opt/homebrew/bin`, no
`~/.local/bin`. Spawning `llama-server` by name worked from a shell and crash-looped the
model unit, reporting "llama.cpp reports no GPU" the whole time because the device probe
could not tell a binary it failed to run from a machine with no card. Resolve through
`ailocal::llama_server()`, which tries `PATH` then the usual install directories.
Reproduce a service's view with
`env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin ailocal budget`.

### 6. Root steps go to the user, and Arch is never partially upgraded
`sudo` needs a password, so the agent cannot install packages. Hand the user an exact
command. It always syncs first:

```
sudo pacman -Syu <packages>
```

`pacman -S` into a stale database is how a rolling-release install gets broken.

### 7. Pi needs three files, and only one of them is ours to keep
Pi's llama.cpp provider is registered from the *cached model catalogue*, not from the
credential. Writing only `~/.pi/agent/auth.json` leaves `--provider llama.cpp` failing
with "Unknown provider" and `auth check` reporting `provider_not_found`. The catalogue
entry in `~/.pi/agent/models-store.json` needs `api: "openai-completions"` and
`provider: "llama.cpp"`, and Pi strips a trailing `/v1` from `LLAMA_BASE_URL` before
storing it - so store the stripped form or every run looks like a change.

**`models-store.json` is a cache, so nothing about capability can live there.** Pi
rebuilds it from our `<base>/models` on every refresh, through a function that hardcodes
`reasoning: false`, `compat.supportsReasoningEffort: false` and
`contextWindow: meta.n_ctx ?? 128000`. Anything we write there is gone the next time
`/model` opens. Capability goes in `~/.pi/agent/models.json` under
`providers["llama.cpp"].modelOverrides.<id>`, which Pi composes *over* the provider on
every read. Proof: with the cache saying `reasoning: false, contextWindow: 128000`,
`pi --list-models` reported `thinking yes` and `262.1K`.

Our router listing must therefore also report `meta.n_ctx`, or Pi's own refresh caches a
flat 128000 and auto-compacts a 256k session at half its window.

All three files hold live credentials or hand-written config for other providers. Always
merge, never rewrite, and back up first. `pi --list-models` prints a `thinking` column
and the context window, which verifies both in one command.

**"Configured" therefore means the catalogue too, not just the credential.** Pi empties
its own cache whenever a refresh finds nothing to list - which a gateway answers
honestly while a download is still a `.part`, since the digest of 15 GB takes long
enough to click something in the meantime. The credential survives that, so a check
that reads only `auth.json` reports a harness that Pi no longer offers: measured here,
`ailocal status` said `ok harnesses pi` and the console showed a `configured` badge
while `pi --list-models` listed anthropic, claude-subscription and openai-codex and no
llama.cpp at all. Worse, the badge said the same thing before and after, so the one
button that would have fixed it looked like it did nothing. `harness::is_configured`
requires a non-empty `models` array beside the credential, and `configure_pi` refuses
to write an empty catalogue rather than unregistering the provider itself.

Claude Code is configured by **environment**, and `ailocal harness configure claude-code`
writes a file to `source` rather than touching `~/.claude/settings.json` - settings there
apply to every Claude Code session on the machine, including ones meant for the real
Anthropic API.

### 7b. A harness's capabilities come from the model, not from our settings
Reporting `reasoning` from the config's launch flag told Pi the model was incapable of
thinking whenever `reasoning = "off"`, so `thinking` offered only `off` and `effort` was
rejected. Detect from the model: `gguf::thinking_support` searches the chat template for
`enable_thinking` and `reasoning_effort` (which sit past the 8 MB metadata read for
gemma4) and answers `None`, `Toggle` or `Effort`. It agrees with llama-server's own
`/props` `chat_template_caps` for both models here.

**Some models have no thinking mode, and that is not a bug to configure around.**
qwen3-coder-30b's template mentions neither name, so `off` is the only level Pi can
offer for it and no override changes that. `ailocal model ls` has a THINK column so this
is visible before `/effort` refuses.

`--reasoning` at launch is a default, not a lock. Per request llama.cpp acts on exactly
two fields, both measured:

- `chat_template_kwargs.enable_thinking` - the switch. It type-checks the value, which
  is how you can tell it is read.
- `reasoning_budget_tokens` - a real cap on the thinking. **Not** `thinking_budget_tokens`,
  which Pi's docs name and llama-server accepts and silently ignores on
  `/v1/chat/completions`. Measured on gemma4-12b: no budget 554 characters of
  `reasoning_content`, budget 24 gives 88.

Pi 1.2 validates `thinkingTokenBudgetField` against a closed enum that excludes
`reasoning_budget_tokens`. Advertise `thinking_budget_tokens` to Pi and translate it
in the gateway to `reasoning_budget_tokens`; otherwise Pi rejects the entire shared
`models.json`. An explicit native budget wins, including zero. Keep this translation
independent of effort/template overrides, and preserve native-budget clients.

`reasoning_effort` is accepted and ignored unless the template reads it, so the gateway
translates it (and Anthropic's `thinking` block) into `enable_thinking` and passes the
native budget through untouched. That is what makes a level ladder mean something rather than
being five spellings of "on": Pi pairs each level with a budget of 1024 / 2048 / 8192 /
16384, clamped to `maxTokens` less 1024 kept back for the answer - so cap the advertised
output below 17408 and `medium` and `high` collapse into one thing on the wire.

Two consequences for the overrides we write:

- `thinkingLevelMap.off` must be spelled `"none"`. With the key absent Pi sends *no*
  field when thinking is off, which is indistinguishable from a harness that never
  mentioned it, so llama-server stays on its launch default and `/thinking off` does
  nothing.
- `xhigh` and `max` are mapped to `null` for a toggle model. Pi folds both onto `high`
  before choosing a budget, so offering them would be two controls that do nothing.
  `/effort max` still works - it is an alias that resolves to the highest real level.

### 8. Retrieval for facts, fine-tuning for behaviour
Codebase knowledge is a retrieval problem, not a QLoRA problem. An adapter trained on a
repo produces confident wrong API signatures and is stale on the next commit. No
fine-tuning lands before the eval harness can prove it helped - it exists now, as the
`eval` extra, and `held-out` is the split an adapter must be measured on.

### 8b. The desktop app is a shell, and it owns what it starts
`crates/ailocal-desktop` runs the same `ailocal_ui::Server` in its own process and
points a webview at it. There is no second frontend and no second API - a feature
lands in `ui/src` and both surfaces have it.

It is a workspace member but not a *default* member, so `cargo build`, `clippy` and
`test` skip the webview unless asked for it by `-p ailocal-desktop`. Adding
`--workspace` back to those gates undoes that and costs every build minutes.

**A model started from the window has to be stopped when the window closes.** A
terminal delivers Ctrl-C to the whole process group and the model is in it; a window
has no such signal, so `RunEvent::Exit` calls `ailocal_ui::stop_owned_model`. That
compares the recorded pid against the running instance rather than trusting a flag,
because the model can be replaced from a terminal in between and killing a pid that
was reused stops something the window never touched.

Tauri reads the version from `Cargo.toml` when `tauri.conf.json` omits it - leave it
omitted, or the `.dmg` name is a second place for the version to drift from the tag.
macOS ad-hoc signing happens through `bundle.macOS.signingIdentity: "-"`, during
bundling, so the signature is inside the `.dmg`. **Tauri deletes
`bundle/macos/ai-local.app` once the installer is built**, so anything that touches
the `.app` afterwards - signing it, verifying it, copying it - is working on a path
that no longer exists. This failed a release. Mount the `.dmg` and work on the copy
inside it, which is the one that ships.

### 8c. A repository listing is not a list of models
Hugging Face lists everything in the repository, and two kinds of `.gguf` in it cannot
be loaded as a model: **split shards** (`...-00002-of-00003.gguf`) and **multimodal
projectors** (`mmproj-*.gguf`, the vision tower served alongside a model via
`--mmproj`). `source::files` drops both. A projector is the dangerous one - it is a
few hundred MiB, so it sorts to the row with the most headroom and reads as the safest
choice in the list.

A Hugging Face repository id is always exactly two segments, so `hf:` references split
from the **left** at two, not from the right at the last slash. Listings are recursive
and a quant can sit in a subdirectory: `hf:owner/repo/UD-Q2_K_XL/model.gguf` is that
repository and that path, not a repository called `owner/repo/UD-Q2_K_XL`. The name on
disk is the last segment only.

### 9. The core binary stays small; eval and training are opt-in extras
`ailocal` is what a fresh machine curl-pipes and what runs 24/7 under systemd. Anything
occasional or heavy is a separate binary in `crates/ailocal-<name>`, released from the
same tag as its own asset and reached as an external subcommand (`ailocal eval ...` runs
`ailocal-eval`). Adding one means: a workspace member, an entry in `extras::EXTRAS`, the
name in install.sh's `KNOWN_EXTRAS`, and the binary in release.yml's build loop.

Versions move in lockstep - `ailocal extras install` pins to the core's own version and
`ailocal update` refreshes whatever is installed - because a core and an extra from
different releases is the one failure this split can create on its own.

**The desktop app is not an extra.** An `Extra` is a binary implementing a subcommand,
which `locate` finds beside `ailocal` or on `PATH`; the app is an application bundle,
answers to no subcommand, and lives in `/Applications` or as an AppImage. It gets
install.sh's own `--app` flag, and `update::app_installed` decides whether
`ailocal update` carries that flag along. Those two paths must agree: if they drift,
`update` silently leaves an old window beside a new core.

**An app bundle cannot resolve `ailocal` by `PATH` alone** (rule 5 again). There is no
`ailocal` inside the bundle, and a launcher-started process gets launchd's
`/usr/bin:/bin:/usr/sbin:/sbin` - without `~/.local/bin`, which is where install.sh
puts it. Measured on that PATH: installing the services answered "there is no ailocal
at ... or on PATH for a service to run" while `~/.local/bin/ailocal` was there the
whole time. `update::installed_ailocal` searches the install directories after `PATH`.
A window opened with `open` from a terminal inherits that terminal's PATH and never
sees this, so reproduce it with the bare PATH rather than by launching the app.

**install.sh never touches `com.apple.quarantine`.** The flag is set by the program
that downloads, never by the host, and curl sets none, so clearing it "in case of a
re-host" removed nothing. It is also the Gatekeeper-bypass pattern endpoint security
alerts on - download, strip quarantine, run - and on a managed Mac it did, once per
update. A quarantined copy is the user's to allow in System Settings.
