# ailocal

Download, serve and expose local LLMs to your coding harness - Pi, Codex and Claude
Code - on your own GPU, reachable from anywhere.

```
curl -fsSL https://zottiben.github.io/ai-local/install.sh | sh
ailocal setup --model ollama:gemma4:12b
```

`setup` is the whole thing: it picks a model, starts the gateway as a background
service, points your harnesses at it, and then proves it works by calling it. If you
assemble it by hand instead, every command ends by naming the next one, and
`ailocal status` says which link is missing:

```
$ ailocal status
ok    models      2 runnable, default gemma4-12b-Q4_K_M
MISS  gateway     http://127.0.0.1:8081 - not answering
MISS  service     ailocal-gateway.service - not installed
MISS  harnesses   none configured

Next: ailocal service install   (nothing is serving yet - this starts the gateway
                                and keeps it running)
```

Linux (x86_64/aarch64) or macOS (Apple Silicon and Intel). Needs
[llama.cpp](https://github.com/ggml-org/llama.cpp):

```
brew install llama.cpp                          # macOS, Metal is built in
sudo pacman -Syu llama-cpp ggml-vulkan          # Arch, Vulkan backend
```

On macOS the memory ceiling comes from `iogpu.wired_limit_mb`, which is what llama.cpp
reports as device memory. Raising it lets larger models load at the cost of the headroom
macOS uses to stay responsive:

```
sudo sysctl iogpu.wired_limit_mb=28000   # session only; =0 restores the default
```

## Where things go

Models are large - one is 5-30 GB - so `ailocal setup` asks where to put them before it
downloads anything, and shows the free space where it is about to write.

One setting decides all of it:

```
ailocal config data-dir                    # show it, with free space
ailocal config data-dir /mnt/big/ailocal   # move it
ailocal setup --data-dir /mnt/big/ailocal  # or choose up front, unattended
```

It defaults to `~/.ailocal`, holding `models/`, `hf/` and `eval/`. Any of those can be
pinned individually in `~/.config/ailocal/config.toml` if, say, only the weights belong
on a scratch volume - setting the root afterwards leaves a path you chose yourself alone.

`HF_HOME` is always set from this, never left to the Hugging Face default of
`~/.cache/huggingface` - which is how one download fills a home partition.

Changing the root does not move what is already downloaded; it tells you what to `mv`.

## What it does

```
$ ailocal model ls
NAME                  SIZE  ARCH          KV/TOK   TRAINED   MAX CTX  THINK
gemma4-12b-Q4_K_M      6 G  gemma4 (swa)  11 KiB      256k      256k  on/off
qwen3-14b              8 G  qwen3        106 KiB       40k     39572  -
```

The interesting column is MAX CTX. "Does it fit in VRAM" is the wrong question -
weights and KV cache have to fit *together*, and the KV cache is what actually decides
how much context you get. ailocal computes that from the model's own attention geometry
and refuses loads that cannot work, rather than discovering the limit at runtime.

THINK is the model's own thinking control, read from its chat template: a switch, an
effort level, or nothing at all. It is what a harness's `/thinking` and `/effort` can
offer, so a model with no thinking mode says so here rather than at the prompt.

That matters more than it sounds: on AMD there is no graceful VRAM exhaustion. A model
that over-allocates does not get an error - the compositor loses its framebuffer and the
desktop session dies.

## Commands

| | |
| --- | --- |
| `ailocal setup` | check prerequisites, install a model, start services, configure harnesses |
| `ailocal status` | where you are, and the one command that gets you further |
| `ailocal config data-dir [path]` | where weights, caches and datasets live |
| `ailocal config gateway-port [n]` | which port the gateway listens on |
| `ailocal model pick` | choose your model: what you have, plus what would fit |
| `ailocal model search <query>` | find GGUF repos on Hugging Face |
| `ailocal model files <owner/repo>` | list quantisations and which of them fit |
| `ailocal model install <ref>` | resumable, checksum-verified download |
| `ailocal model ls` | what is on disk, what fits, and what can think |
| `ailocal model rm <model>` | delete its weights, and stop advertising it |
| `ailocal serve <model>` | run it, sized to the VRAM budget |
| `ailocal ps` / `stop` | what is loaded |
| `ailocal gateway run` / `key` / `check` | the authenticated front end |
| `ailocal harness configure pi\|claude-code [--model X]` | point a harness at it |
| `ailocal service install` / `status` | systemd user units for 24/7 |
| `ailocal update` | install the latest release in place |
| `ailocal extras list` / `install <name>` | optional companions, see below |
| `ailocal eval run` | score a model on a coding suite (needs the `eval` extra) |

## Choosing a model

`ailocal model pick` is the "which model am I using" command, not only the "download
one" command. It lists what is already on disk alongside what the catalogue offers,
ranked by what this machine can actually run:

```
 1) gemma4-12b-Q4_K_M   6 GiB   256k context   on disk - general + coding, very long context
 2) qwen3-14b           8 GiB    38k context   on disk - holds up well under long prompts
 3) mistral-nemo:12b    6 GiB    58k context   general purpose, modest footprint
 4) qwen3-coder:30b    17 GiB    will not fit  code specialist, mixture-of-experts
```

Picking one makes it the default: the config is updated and, if a model service is
installed, its unit is rewritten and restarted onto it. Picking something you already
have downloads nothing, so this is also how you switch between models.

A model already on disk is recognised even when its filename does not match the
reference - `ollama:gemma4:12b` would store `gemma4-12b.gguf`, and a
`gemma4-12b-Q4_K_M.gguf` of exactly the same size is the same blob, so it is offered as
itself rather than as seven gigabytes to fetch again.

## Pointing a harness at it

```
ailocal harness configure pi
ailocal harness configure claude-code --model qwen3-14b
```

Pi is given the whole catalogue and chooses per session. Claude Code takes exactly one
`ANTHROPIC_MODEL`, so `--model` decides it; without one it uses `default_model`, then
whatever is loaded, and only then falls back to alphabetical order. It prints which it
chose and why, because pinning one model out of several is a decision worth showing.

Re-run it after installing a model: the harness keeps a cached catalogue, so a new model
is not visible to it until that is rewritten. The gateway needs no such nudge - it
rescans on every request.

## Why the first prompt is slow, and what to do about it

Three costs stack on a first turn, and two of them are avoidable. Measured on an
RX 7600 XT with gemma4-12b and a 9,700-token system prompt, which is the size a coding
harness actually sends:

| | first turn | same prompt again |
| --- | --- | --- |
| `reasoning = "off"` | 31.5s | 15.9s |
| `reasoning = "on"` | 52.2s | 36.0s |

Plus a cold model load if nothing is resident - about 5s here with the weights in page
cache, considerably longer reading 6 GB off an SSD for the first time.

- **Keep the model resident.** `ailocal service install` loads it at boot and holds it
  there, so no request ever pays for the load. Without it the gateway loads on demand
  and the first prompt waits.
- **Leave reasoning off.** It is the default because the eval measured it, and the
  measurement is stronger than the latency headline: across three reasoning settings,
  reasoning never beat `off` on a single task. See below.
- **The prompt cache does the rest.** llama.cpp reuses the common prefix between turns,
  which is why the second turn above is half the first. Only the first turn of a session
  pays full prompt processing.

That leaves prompt processing of the harness's system prompt as the irreducible part.
It is why a local model feels slower than a hosted one on the first turn and comparable
afterwards.

## The gateway

One port serves both protocols, because harnesses disagree about which to speak:

- **OpenAI** - `/v1/chat/completions`, `/v1/models`
- **Anthropic** - `/v1/messages`, so `ANTHROPIC_BASE_URL` works for Claude Code
- **llama.cpp router** - `/models`, `/models/load`, so Pi's model picker drives it

Requests for a model that is not resident swap it in; only one fits at a time.
Authentication is a bearer key, which is what every harness already sends, so the
gateway can sit behind a tunnel without anything else in front of it. See
[docs/tunnel.md](docs/tunnel.md).

### A house rule for every harness

Harnesses build their own system prompt and have no notion of a per-machine one, so the
gateway is the only place to add an instruction once and have every harness get it:

```toml
# ~/.config/ailocal/config.toml
system_prompt = """
If AGENTS.md or CLAUDE.md exists in the working directory, read it with your
file-reading tool before your first substantive action, and follow it.
"""
```

It is appended to whatever system prompt the harness sends rather than replacing it -
the harness's own prompt is what makes its tools work - and inserted as a system message
only when there is none.

Every token here is processed on the first turn of every session, and a small model given
instructions that argue with the harness's follows neither well. A sentence or two.

### What actually happens to AGENTS.md

The file is loaded by the *harness*, not the model - Codex's docs say "Codex reads
`AGENTS.md` files before doing any work", and Claude Code's say the files "are read at
session start and delivered to Claude". Measured here with `--log-requests`, Pi does the
same: it inlines the file into a `developer` message wrapped in `<project_instructions
path="...">`, and gemma4-12b then answers from it correctly.

So a local model does get your AGENTS.md. What differs is fidelity: asked to repeat a
distinctive token from that file, gemma4-12b corrupted it on two runs out of three. That
is a model-quality difference rather than a plumbing one, and it is what the
`instruction-from-context` eval task exists to measure.

If you want to see it for yourself:

```
ailocal gateway run --log-requests    # writes <data_dir>/gateway-requests.jsonl
```

It records what the harness sent, before this gateway changes anything. Off by default -
it is a transcript of your work.

### Thinking, per request

`--reasoning` is a llama-server launch flag, but it is a *default*, not a lock. The
gateway translates a harness's thinking control into the one thing llama.cpp acts on:

| what a harness sends | what llama.cpp needs |
| --- | --- |
| `reasoning_effort: "high"` (OpenAI) | `chat_template_kwargs: {enable_thinking: true}` |
| `thinking: {type: "enabled"}` (Anthropic) | same |
| `reasoning_effort: "none"`/`"minimal"` | `enable_thinking: false` |

llama.cpp accepts `reasoning_effort` and ignores it, so without this translation a
harness's thinking control does nothing at all - which looks exactly like a model that
refuses to think. An explicit `chat_template_kwargs` from the caller always wins.

Whether a model is offered the control at all comes from the model itself: the chat
template is searched for an `enable_thinking` toggle. It used to come from the config's
`reasoning` setting, which meant `reasoning = "off"` told harnesses the model was
incapable of thinking rather than merely not doing it by default.

### If the port is taken

8081 is a popular default - React Native's Metro bundler uses it, among others - so on
a development machine something may already own it. `ailocal setup` checks before it
builds anything on top, and moves aside:

```
3. gateway port
   busy  8081 is held by something else, moving to 8082
```

It only moves for a port held by *something else*; a gateway of ours already listening
is left exactly where it is. `--port` picks one explicitly, and is an error rather than
a suggestion if that port is occupied too.

Afterwards, `ailocal config gateway-port` reports who holds the port and `ailocal config
gateway-port <n>` moves it, rewriting and restarting the service. Harnesses store the
URL themselves, so re-run `ailocal harness configure` to bring them along - and
`harness configure` warns when it is about to write a URL nothing is answering on,
since the alternative is discovering it as a bare "Connection error" at the first
prompt.

Setup finishes by actually calling the gateway rather than assuming the service manager
starting a job means it worked.

## Finding a model

The quickest route is to let it rank a shortlist against your card:

```
$ ailocal model pick
Checking what fits in 12887 MiB ...
Measuring usable context ...
Models for this machine (12887 MiB available)
  1) gemma4:12b          6 GiB   256k context   general + coding, very long context
  2) qwen3:14b           8 GiB    38k context   general + coding, holds up under long prompts
  3) mistral-nemo:12b    6 GiB    57k context   general purpose, modest footprint
  ...
  7) qwen3-coder:30b    17 GiB   will not fit   code specialist, mixture-of-experts
```

Pick one and it downloads, verifies and registers it. `ailocal setup` runs this for you
when nothing is installed yet.

The context figures are measured, not guessed: it reads each model's attention geometry
from the first few MiB of the real file. Ordering is capability first among models that
can hold a *usable* context - a 4B leaves room for 109k tokens, which does not make it
the better choice. Models that cannot fit stay on the list so the omission explains
itself.

The shortlist is curated and will date. `ailocal model search` is the way to anything
not on it.

### Anything else on Hugging Face

Search Hugging Face, then look at what quantisations a repo offers and which of them
leave room for context on your card:

```
$ ailocal model search qwen3 coder
REPO                                                          DOWNLOADS
unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF                      12581343
unsloth/Qwen3-Coder-Next-GGUF                                    191200
lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF              88305

$ ailocal model files unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
12887 MiB available for weights + KV cache

FILE                                                     SIZE  VERDICT
Qwen3-Coder-30B-A3B-Instruct-UD-IQ2_XXS.gguf              9 G  3033 MiB left for context
Qwen3-Coder-30B-A3B-Instruct-Q2_K_L.gguf                 10 G  2081 MiB left for context
Qwen3-Coder-30B-A3B-Instruct-UD-IQ3_XXS.gguf             11 G  634 MiB left for context
Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf             12 G  too large

$ ailocal model install hf:unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q2_K_L.gguf
```

Reading that output:

- **Quantisation** is the suffix. `Q4_K_M` is the usual default; `Q5`/`Q6`/`Q8` are
  larger and closer to the original; `Q3`, `IQ2` and `Q2` trade accuracy for fitting.
  A `UD-` prefix is an Unsloth dynamic quant, generally better at the same size.
- **"MiB left for context"** is the number that decides whether a model is pleasant to
  use. A 30B squeezed in at IQ3 with 634 MiB spare holds only a few thousand tokens,
  which is useless for coding. Prefer a smaller model with several GiB spare.
- Split GGUFs (`-00002-of-00003`) are hidden, because a shard cannot be loaded alone.

`ailocal model install` re-checks all of this against the real remote file before
downloading, so a bad choice costs about two seconds rather than an hour.

### The Ollama registry

Ollama is far faster to download from where it has what you want, and its blobs are
content-addressed so they verify against a published digest. It has no search API, so
browse [ollama.com/library](https://ollama.com/library) and use the name and tag exactly
as shown there:

```
ailocal model install ollama:gemma4:12b
ailocal model install ollama:qwen3:14b
```

Its catch is quant coverage - mostly just the default `Q4_K_M` per size. When you need a
specific quantisation to make something fit, that comes from Hugging Face.

Neither source needs the vendor's CLI installed. For gated or rate-limited Hugging Face
repos, put a token at `$HF_HOME/token` (`~/.ailocal/hf/token` by default).

## Extras

The core binary is what a fresh machine downloads and what runs under systemd all day,
so it stays small. Anything occasional and heavier ships as a separate binary released
from the same tag, installed only if you want it:

```
ailocal extras install eval
```

They are reached through the core CLI as subcommands - `ailocal eval ...` runs
`ailocal-eval` - and `ailocal update` keeps whichever ones you have installed in step
with the core, since a mismatched pair is the one failure a split CLI can create by
itself.

### `eval` - is this model actually any good here?

A coding suite scored by programs rather than by another model, so the number means the
same thing every time it is produced. Where the answer is code, the check is `rustc`:
compiling and passing its tests are separate checks with different weights, so "writes
plausible Rust that does not work" is visibly different from "gets it right".

```
ailocal eval run                                   # whatever is loaded
ailocal eval run --reasoning off --reasoning on    # two arms, side by side
ailocal eval run --reasoning on --reasoning-budget 512   # capped thinking
ailocal eval run --model a --model b               # two models, side by side
ailocal eval compare <run-a> <run-b>
```

```
ARM                                 SCORE   PASS   EMPTY    CUT   TOK/S
gemma4-12b-Q4_K_M (reasoning off)     91%    86%       0      0    23.9
```

### Does turning reasoning off make the model worse?

On gemma4-12b, measured: no. Four arms over the same 14 tasks.

| | score | empty answers |
| --- | --- | --- |
| off | 95% | 0 |
| on, 1024-token answer budget | 45% | 7 |
| on, 4096-token budget | 52% | 6 |
| on, thinking capped at 512 | 82% | 0 |

The headline understates it. Restricted to only the tasks where reasoning actually
produced an answer, it scored *identically* to `off` - 89% against 89%, then 91%
against 91% - with no task going either way. Capping thinking fixes the termination
failure completely but still loses, and loses precisely the two hardest code-generation
tasks.

Across 42 task-arm comparisons, reasoning never beat `off` on a single task. The saved
`reasoning_content` shows why: on the task it lost worst, the model degenerates into a
loop - `Correct: write!(...) -> No.` repeated until the budget is gone - and then emits
truncated code that will not compile.

That is one model on single-turn tasks, so re-measure rather than assume. It is one
command, and `--reasoning-budget` makes the middle setting measurable too.

`EMPTY` and `CUT` are there because reasoning models fail in a way a score alone hides:
chain-of-thought goes to `reasoning_content` and `content` stays empty until it is done,
so a model that thinks past its token budget returns *nothing* rather than something
short. To a harness that looks like a broken model, not a slow one.

Reasoning is a llama-server launch flag, so comparing modes means reloading the model.
The run pauses the model service for its duration and starts it again afterwards,
including if it fails part way.

**It runs code the model writes**, under a timeout and without a sandbox - which is
inherent to checking whether code works, and is what every coding benchmark does, but
is worth knowing before you run it. `--no-exec` compiles without running, and reports
those checks as unjudged rather than failed.

## Downloads

Downloads resume after an interruption, and land in a `.part` file that is only renamed
into place once length and checksum check out - so a killed download can never leave
something that looks like a usable model. Hugging Face publishes no plain checksum, so
those are length-checked only.

## Building

```
cargo build --release
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Licence

MIT.
