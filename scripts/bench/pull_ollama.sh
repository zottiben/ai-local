#!/usr/bin/env bash
# Pull a GGUF from the Ollama registry by manifest digest, verify it, name it sanely.
#
# Ollama registry blobs are plain content-addressed GGUF files, so this needs no ollama
# install. Measured 79-97 MB/s from Adelaide vs 0.016-2.7 MB/s from huggingface.co.
# Limitation: the registry only carries the default Q4_K_M plus a few named tags, so
# low quants (IQ3/Q3) still have to come from HF.
#
# usage: pull_ollama.sh <name> <tag> [outfile]
set -euo pipefail
NAME=${1:?model name, e.g. qwen3}
TAG=${2:?tag, e.g. 14b}
OUT=${3:-/mnt/kingston/ailocal/models/${NAME}-${TAG}.gguf}
REG=https://registry.ollama.ai/v2/library

DIG=$(curl -sf --max-time 20 "${REG}/${NAME}/manifests/${TAG}" | python3 -c "
import sys,json
for l in json.load(sys.stdin).get('layers',[]):
    if 'model' in l.get('mediaType',''):
        print(l['digest']); break
")
[ -n "$DIG" ] || { echo "no model layer for ${NAME}:${TAG}" >&2; exit 1; }

echo "==> ${NAME}:${TAG}  ${DIG:0:23}..."
start=$(date +%s)
curl -L -C - --fail --retry 3 --progress-bar -o "$OUT" "${REG}/${NAME}/blobs/${DIG}"
el=$(( $(date +%s) - start )); [ $el -eq 0 ] && el=1
sz=$(stat -c%s "$OUT")

printf '    %d MiB in %ds (%d MiB/s)\n' $((sz/1048576)) "$el" $((sz/1048576/el))
[ "$(head -c4 "$OUT")" = GGUF ] || { echo "    NOT a GGUF" >&2; exit 1; }
[ "sha256:$(sha256sum "$OUT" | cut -d' ' -f1)" = "$DIG" ] \
  && echo "    sha256 OK" \
  || { echo "    sha256 MISMATCH" >&2; exit 1; }
