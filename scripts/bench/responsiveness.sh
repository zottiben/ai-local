#!/usr/bin/env bash
# Measure what a running model actually feels like: how fast it reads a prompt, how
# fast it writes an answer, and how much of both an idle spell costs.
#
# WHY THE IDLE PART MATTERS: a server that has been sitting still is not the server
# you benchmarked. On unified memory the operating system compresses the weights out
# from under an idle process, and the next prompt pays to fault all of them back.
# Measured on an M4 Max/64 GB on 2026-09-11 with a 30B at 256k context: 13.4 tok/s on
# the first request after a pause, 97.3 tok/s on the next one. Any measurement that
# only ever asks a warm server will report the second number and miss the complaint.
#
# Every figure comes from llama-server's own `timings` block, not from wall clock
# around curl, so a slow network or a busy client cannot flatter or spoil a run.
#
# usage: responsiveness.sh [--idle MINUTES] [--sizes "N N N"] [--url URL] [--model NAME]
set -uo pipefail

URL=http://127.0.0.1:8080
MODEL=""
IDLE=0
# Prompt sizes in tokens. The largest is about the size of a real coding harness's
# opening prompt here; the smallest isolates generation speed from any prompt at all.
SIZES="100 4096 16384 32768"

while [ $# -gt 0 ]; do
  case "$1" in
    --idle)  IDLE=$2; shift 2 ;;
    --sizes) SIZES=$2; shift 2 ;;
    --url)   URL=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    *) echo "usage: $0 [--idle MINUTES] [--sizes \"N N N\"] [--url URL] [--model NAME]" >&2; exit 2 ;;
  esac
done

if [ -z "$MODEL" ]; then
  MODEL=$(curl -sf "$URL/props" | python3 -c 'import json,sys; print(json.load(sys.stdin)["model_alias"])') \
    || { echo "no llama-server answering at $URL" >&2; exit 1; }
fi
echo "model $MODEL at $URL"

# One request, reported from the server's own timings.
#
# Each prompt starts with a nonce so llama.cpp's prompt cache cannot match a previous
# run: the point is to measure processing, and a cache hit reports a rate for work it
# did not do.
probe() {
  local tokens=$1 label=$2
  python3 - "$URL" "$MODEL" "$tokens" "$label" <<'PY'
import json, random, sys, urllib.request

url, model, tokens, label = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
nonce = f"session {random.randrange(10**12)}."
# Each line runs about twelve tokens. The count the server actually saw is reported
# alongside the rate, so the estimate only has to be close.
filler = "\n".join(f"line {i} holds a value of {i * 7 % 97}." for i in range(tokens // 12))
# A fixed-length answer, so the generation rate is measured over enough tokens to
# mean something. "Reply ok" produces two, and two tokens time nothing but latency.
ask = "Count from 1 to 40, separated by spaces, and write nothing else."
body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": f"{nonce}\n{filler}\n{ask}"}],
    "max_tokens": 120,
    "temperature": 0.1,
}).encode()

req = urllib.request.Request(url + "/v1/chat/completions", body,
                             {"Content-Type": "application/json"})
try:
    t = json.load(urllib.request.urlopen(req, timeout=1800))["timings"]
except Exception as e:                       # a refused or timed-out probe is a result
    print(f"{label:>14}  FAILED: {e}")
    sys.exit(0)

print(f"{label:>14}  prompt {t['prompt_n']:>7} tok at {t['prompt_per_second']:>7.1f} tok/s "
      f"({t['prompt_ms'] / 1000:>6.1f} s)   generated {t['predicted_n']:>3} at "
      f"{t['predicted_per_second']:>6.1f} tok/s")
PY
}

if [ "$IDLE" -gt 0 ]; then
  echo "idling ${IDLE}m so the OS can page the model out..."
  sleep $((IDLE * 60))
  # The first request after the pause is the whole point of the idle run, so it is
  # reported on its own rather than averaged into the warm numbers below.
  probe 100 "after ${IDLE}m idle"
fi

for size in $SIZES; do
  probe "$size" "~${size} tok"
done
