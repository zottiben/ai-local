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

`reasoning_effort` is accepted and ignored unless the template reads it, so the gateway
translates it (and Anthropic's `thinking` block) into `enable_thinking` and passes the
budget through untouched. That is what makes a level ladder mean something rather than
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

### 9. The core binary stays small; eval and training are opt-in extras
`ailocal` is what a fresh machine curl-pipes and what runs 24/7 under systemd. Anything
occasional or heavy is a separate binary in `crates/ailocal-<name>`, released from the
same tag as its own asset and reached as an external subcommand (`ailocal eval ...` runs
`ailocal-eval`). Adding one means: a workspace member, an entry in `extras::EXTRAS`, the
name in install.sh's `KNOWN_EXTRAS`, and the binary in release.yml's build loop.

Versions move in lockstep - `ailocal extras install` pins to the core's own version and
`ailocal update` refreshes whatever is installed - because a core and an extra from
different releases is the one failure this split can create on its own.
