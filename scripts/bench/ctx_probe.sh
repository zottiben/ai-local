#!/usr/bin/env bash
# Find the largest context a model can hold WITHOUT starving the desktop compositor.
#
# HISTORY: the first version of this script probed upward until llama-server failed to
# allocate. On amdgpu/Vulkan that does not fail cleanly - it takes the framebuffer away
# from the compositor ("pin failed", "-12") and kills the graphical session. It crashed
# this machine on 2026-09-07 at ctx=65536 / 15835 MiB.
#
# So: never probe to failure. A watchdog samples VRAM every 200 ms while the server
# loads and kills it the instant total usage crosses CEILING_MIB, which is set well
# below free VRAM because the compositor allocates framebuffers on demand.
#
# usage: ctx_probe.sh <model.gguf> [kv_type] [max_ctx]
set -uo pipefail

MODEL=${1:?model path}
KV=${2:-f16}
MAX_CTX=${3:-131072}
PORT=18099
CEILING_MIB=14400          # total board usage, desktop included
VRAM=/sys/class/drm/card1/device/mem_info_vram_used

vram_mib() { echo $(( $(cat "$VRAM") / 1048576 )); }

BASE=$(vram_mib)
echo "  baseline (desktop) = ${BASE} MiB | ceiling = ${CEILING_MIB} MiB"

for CTX in 4096 8192 16384 32768 65536 131072 196608 262144; do
  [ "$CTX" -gt "$MAX_CTX" ] && break

  llama-server -m "$MODEL" -ngl 99 -c "$CTX" --port $PORT \
      --cache-type-k "$KV" --cache-type-v "$KV" \
      --no-webui >/tmp/ctx_probe.log 2>&1 &
  pid=$!

  ok=""; killed=""; peak=$BASE
  for _ in $(seq 1 900); do          # 900 * 0.2s = 180s budget
    sleep 0.2
    kill -0 $pid 2>/dev/null || break
    cur=$(vram_mib); [ "$cur" -gt "$peak" ] && peak=$cur
    if [ "$cur" -ge "$CEILING_MIB" ]; then
      kill -9 $pid 2>/dev/null; killed=1; break
    fi
    curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
  done

  cur=$(vram_mib); [ "$cur" -gt "$peak" ] && peak=$cur
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 4                            # let the driver actually release it

  if [ -n "$killed" ]; then
    printf '  kv=%-5s ctx=%-7s ABORTED at %s MiB (would starve the compositor)\n' "$KV" "$CTX" "$peak"
    break
  elif [ -n "$ok" ]; then
    printf '  kv=%-5s ctx=%-7s OK    peak=%s MiB  (model+kv=%s MiB, %s MiB spare)\n' \
      "$KV" "$CTX" "$peak" "$((peak-BASE))" "$((CEILING_MIB-peak))"
  else
    printf '  kv=%-5s ctx=%-7s FAILED to start (peak=%s MiB)\n' "$KV" "$CTX" "$peak"
    break
  fi
done
