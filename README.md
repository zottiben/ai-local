# ailocal

Download, serve and expose local LLMs to your coding harness - Pi, Codex and Claude
Code - on your own GPU, reachable from anywhere.

```
curl -fsSL https://zottiben.github.io/ai-local/install.sh | sh
ailocal setup --model ollama:gemma4:12b
```

Linux, x86_64 or aarch64. Needs [llama.cpp](https://github.com/ggml-org/llama.cpp) with
the Vulkan backend (`llama-cpp` + `ggml-vulkan` on Arch).

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
| `ailocal model install ollama:<name>:<tag>` | resumable, checksum-verified download |
| `ailocal model ls` / `rm` | what is on disk, and what fits |
| `ailocal serve <model>` | run it, sized to the VRAM budget |
| `ailocal ps` / `stop` | what is loaded |
| `ailocal gateway run` / `key` / `check` | the authenticated front end |
| `ailocal harness configure pi\|claude-code` | point a harness at it |
| `ailocal service install` / `status` | systemd user units for 24/7 |
| `ailocal update` | install the latest release in place |

## The gateway

One port serves both protocols, because harnesses disagree about which to speak:

- **OpenAI** - `/v1/chat/completions`, `/v1/models`
- **Anthropic** - `/v1/messages`, so `ANTHROPIC_BASE_URL` works for Claude Code
- **llama.cpp router** - `/models`, `/models/load`, so Pi's model picker drives it

Requests for a model that is not resident swap it in; only one fits at a time.
Authentication is a bearer key, which is what every harness already sends, so the
gateway can sit behind a tunnel without anything else in front of it. See
[docs/tunnel.md](docs/tunnel.md).

## Downloads

Models come from the Ollama registry or Hugging Face:

```
ailocal model install ollama:gemma4:12b
ailocal model install hf:unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf
```

Ollama registry blobs are plain content-addressed GGUFs, so they verify against a
published digest and need no Ollama install. Hugging Face has far better quant coverage
but publishes no plain checksum, so those downloads are length-checked only.

Before downloading, ailocal reads the first 8 MiB of the remote file to check the model
can actually run - a 16 GB model that could never load is refused in about two seconds.

## Building

```
cargo build --release
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Licence

MIT.
