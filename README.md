# ailocal

Download, serve and expose local LLMs to your coding harness - Pi, Codex and Claude
Code - on your own GPU, reachable from anywhere.

```
curl -fsSL https://zottiben.github.io/ai-local/install.sh | sh
ailocal setup --model ollama:gemma4:12b
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
NAME                  SIZE  ARCH     KV/TOK   TRAINED   MAX CTX
gemma4-12b-Q4_K_M      6 G  gemma4   11 KiB      256k      256k  (swa)
qwen3-14b              8 G  qwen3   106 KiB       40k     39572
```

The interesting column is the last one. "Does it fit in VRAM" is the wrong question -
weights and KV cache have to fit *together*, and the KV cache is what actually decides
how much context you get. ailocal computes that from the model's own attention geometry
and refuses loads that cannot work, rather than discovering the limit at runtime.

That matters more than it sounds: on AMD there is no graceful VRAM exhaustion. A model
that over-allocates does not get an error - the compositor loses its framebuffer and the
desktop session dies.

## Commands

| | |
| --- | --- |
| `ailocal setup` | check prerequisites, install a model, start services, configure harnesses |
| `ailocal config data-dir [path]` | where weights, caches and datasets live |
| `ailocal model pick` | choose interactively from models ranked for your hardware |
| `ailocal model search <query>` | find GGUF repos on Hugging Face |
| `ailocal model files <owner/repo>` | list quantisations and which of them fit |
| `ailocal model install <ref>` | resumable, checksum-verified download |
| `ailocal model ls` / `rm` | what is on disk, and what fits |
| `ailocal serve <model>` | run it, sized to the VRAM budget |
| `ailocal ps` / `stop` | what is loaded |
| `ailocal gateway run` / `key` / `check` | the authenticated front end |
| `ailocal harness configure pi\|claude-code` | point a harness at it |
| `ailocal service install` / `status` | systemd user units for 24/7 |
| `ailocal update` | install the latest release in place |
| `ailocal extras list` / `install <name>` | optional companions, see below |
| `ailocal eval run` | score a model on a coding suite (needs the `eval` extra) |

## The gateway

One port serves both protocols, because harnesses disagree about which to speak:

- **OpenAI** - `/v1/chat/completions`, `/v1/models`
- **Anthropic** - `/v1/messages`, so `ANTHROPIC_BASE_URL` works for Claude Code
- **llama.cpp router** - `/models`, `/models/load`, so Pi's model picker drives it

Requests for a model that is not resident swap it in; only one fits at a time.
Authentication is a bearer key, which is what every harness already sends, so the
gateway can sit behind a tunnel without anything else in front of it. See
[docs/tunnel.md](docs/tunnel.md).

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
ailocal eval run --model a --model b               # two models, side by side
ailocal eval compare <run-a> <run-b>
```

```
ARM                                 SCORE   PASS   EMPTY    CUT   TOK/S
gemma4-12b-Q4_K_M (reasoning off)     91%    86%       0      0    23.9
```

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
