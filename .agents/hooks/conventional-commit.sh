#!/usr/bin/env bash
# PreToolUse (Bash): enforce Conventional Commits on `git commit -m`.
# Only validates inline -m/--message commits; editor commits pass through. exit 2 blocks.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"; . "$DIR/_lib.sh"
HOOK_JSON=$(cat)

json_have_parser || { echo "conventional-commit: needs jq or python3 — check INACTIVE, install one" >&2; exit 1; }

cmd=$(json_field tool_input command)
echo "$cmd" | grep -qE '\bgit[[:space:]]+commit\b' || exit 0

# Pull the first -m / --message value (single- or double-quoted).
msg=$(printf '%s' "$cmd" \
  | grep -oE -- "(-m|--message)[[:space:]]+'[^']*'|(-m|--message)[[:space:]]+\"[^\"]*\"" \
  | head -1 | sed -E "s/^(-m|--message)[[:space:]]+.//; s/.$//")
[ -z "$msg" ] && exit 0   # no inline message (editor commit) — let it through

echo "$msg" | grep -qE '^(feat|fix|chore|docs|style|refactor|perf|test|build|ci|revert)(\([a-z0-9_.-]+\))?!?: .+' \
  || { echo "BLOCKED: not Conventional Commits. Use 'type(scope): summary' — type in feat|fix|chore|docs|style|refactor|perf|test|build|ci|revert." >&2; exit 2; }
exit 0
