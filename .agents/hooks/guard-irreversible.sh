#!/usr/bin/env bash
# PreToolUse (Bash): block irreversible / costly / destructive commands.
# exit 2 blocks the call and feeds the reason back to the model.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"; . "$DIR/_lib.sh"
HOOK_JSON=$(cat)

json_have_parser || { echo "guard-irreversible: needs jq or python3 to inspect commands — guard INACTIVE, install one" >&2; exit 1; }

cmd=$(json_field tool_input command)
[ -z "$cmd" ] && exit 0

block() { echo "BLOCKED by guard-irreversible hook: $1" >&2; exit 2; }

# Release tags / pushing tags — often triggers a BILLED release build (e.g. EAS).
# Opt-in per repo: delete this check where release tags are routine.
echo "$cmd" | grep -qiE 'git[[:space:]]+tag([[:space:]].*)?v[0-9]|git[[:space:]]+push([[:space:]].*)?(--tags|v[0-9])' \
  && block "pushing a v* tag can trigger a billed release build — get an explicit release request first."

# Force push (rewrites history).
echo "$cmd" | grep -qiE 'git[[:space:]]+push([[:space:]].*)?(--force([[:space:]]|=|$)|-f([[:space:]]|$)|--force-with-lease)' \
  && block "force-push rewrites remote history — confirm with the user first."

# Recursive force-delete.
echo "$cmd" | grep -qiE '\brm[[:space:]]+-[a-z]*r[a-z]*f|\brm[[:space:]]+-[a-z]*f[a-z]*r|\brm[[:space:]]+-rf\b' \
  && block "recursive force-delete — confirm the exact target with the user."

# Reading secret/env files.
echo "$cmd" | grep -qiE '(cat|less|more|head|tail|printenv)[[:space:]]+[^|]*\.env(\.|[[:space:]]|$)' \
  && block "reading secret/.env files — handle secrets out-of-band, not through the model."

exit 0
