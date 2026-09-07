# ai-local - project knowledge

A Rust CLI and gateway to download, serve, fine-tune and expose local LLMs to my coding
harnesses (Pi, Codex, Claude Code), locally and remotely via the the tunnel host Cloudflare tunnel.

**Stack:** Rust (CLI, registry, gateway) + a pinned Python sidecar for training only.
Inference is llama.cpp's `llama-server`, Vulkan backend. Models are GGUF from Hugging Face.

**Layout**
- `crates/` - Rust workspace (CLI, registry, gateway)
- `python/` - training sidecar, its own uv-managed venv, never the system interpreter
- `/mnt/kingston/ailocal/` - all weights, adapters, datasets and caches. Not in this repo.

**Commands** (from repo root)
- Build / lint / test: `cargo build`, `cargo clippy -- -D warnings`, `cargo test`
- Plan: `aip status` (the build plan is in ai-planner, not in a markdown file)

The Rust workspace is established by PR0; until then the cargo commands have nothing to
run against.

## Hard rules

### 1. Nothing large is written outside /mnt/kingston
`/` has ~17 GB free and `/home` ~13 GB. A single model download fills either one.
`/mnt/kingston` has 325 GB.

```
HF_HOME=/mnt/kingston/ailocal/hf
```

Any code path that touches the Hugging Face hub must set this explicitly. The library
default is `~/.cache/huggingface`, which is on the 13 GB partition.

### 2. Never use the system Python
It is 3.14.6 and no ML library ships wheels for it. The sidecar gets its own 3.11/3.12
venv under `/mnt/kingston/ailocal/venv`, created and verified by the Rust CLI.

### 3. Size against ~14.8 GiB of VRAM, not 16 GB
The desktop permanently holds ~1255 MiB of the 16368 MiB on the RX 7600 XT. Until the
RAM upgrade is earned, models must fit **entirely** in VRAM with room for KV cache -
there are only 8.5 GB of system RAM, so CPU offload thrashes.

### 4. ROCm never touches the inference path
Inference is Vulkan. gfx1102 ROCm is flaky (upstream segfaults, hipBLASLt reports the
arch unsupported). ROCm is for training only, so a training-side breakage can never take
down the daily driver.

### 5. Root steps go to the user, and Arch is never partially upgraded
`sudo` needs a password, so the agent cannot install packages. Hand the user an exact
command. It always syncs first:

```
sudo pacman -Syu <packages>
```

`pacman -S` into a stale database is how a rolling-release install gets broken.

### 6. Retrieval for facts, fine-tuning for behaviour
Codebase knowledge is a retrieval problem, not a QLoRA problem. An adapter trained on a
repo produces confident wrong API signatures and is stale on the next commit. No
fine-tuning lands before the eval harness (PR11) can prove it helped.
