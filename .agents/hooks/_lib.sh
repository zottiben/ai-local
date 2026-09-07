#!/usr/bin/env bash
# Shared helpers for the hook scripts. Prefer jq; fall back to python3.
# Each hook reads the payload once:  HOOK_JSON=$(cat)

json_have_parser() { command -v jq >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1; }

# json_field key [key ...] -> prints the nested string value ("" if absent)
json_field() {
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$HOOK_JSON" | jq -r ".$(IFS=.; printf '%s' "$*") // empty" 2>/dev/null
  elif command -v python3 >/dev/null 2>&1; then
    printf '%s' "$HOOK_JSON" | python3 -c '
import sys, json
try: d = json.load(sys.stdin)
except Exception: sys.exit(0)
for k in sys.argv[1:]:
    if isinstance(d, dict) and k in d: d = d[k]
    else: sys.exit(0)
sys.stdout.write(d if isinstance(d, str) else "")
' "$@"
  fi
}

# emit_context EVENT STRING -> prints the additionalContext JSON on stdout
emit_context() {
  if command -v jq >/dev/null 2>&1; then
    jq -n --arg e "$1" --arg c "$2" '{hookSpecificOutput:{hookEventName:$e,additionalContext:$c}}'
  elif command -v python3 >/dev/null 2>&1; then
    EV="$1" CX="$2" python3 -c 'import os,json;print(json.dumps({"hookSpecificOutput":{"hookEventName":os.environ["EV"],"additionalContext":os.environ["CX"]}}))'
  else
    local esc=${2//\\/\\\\}; esc=${esc//\"/\\\"}
    printf '{"hookSpecificOutput":{"hookEventName":"%s","additionalContext":"%s"}}\n' "$1" "$esc"
  fi
}
