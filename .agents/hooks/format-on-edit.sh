#!/usr/bin/env bash
# PostToolUse (Write|Edit): format/lint the file that was just written.
# Best-effort — NEVER blocks (always exits 0). Uses the project's own tools if present.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"; . "$DIR/_lib.sh"
HOOK_JSON=$(cat)

# Claude Code reports the edited path in tool_input.file_path. (Codex edits go through
# apply_patch, whose payload has no file_path and whose hooks are still maturing upstream —
# so this best-effort formatter simply no-ops there until that lands.)
file=$(json_field tool_input file_path)
[ -z "$file" ] && exit 0
[ -f "$file" ] || exit 0

# Resolve the project root from the payload's cwd (Codex) or ${CLAUDE_PROJECT_DIR} (Claude).
proj=$(json_field cwd); : "${proj:=${CLAUDE_PROJECT_DIR:-.}}"
cd "$proj" 2>/dev/null || true
have() { command -v "$1" >/dev/null 2>&1; }

case "$file" in
  *.ts|*.tsx|*.js|*.jsx|*.mjs|*.cjs|*.json|*.css|*.scss|*.md|*.html|*.yaml|*.yml)
    [ -x node_modules/.bin/prettier ] && node_modules/.bin/prettier --write "$file" >/dev/null 2>&1
    [ -x node_modules/.bin/eslint ]   && node_modules/.bin/eslint --fix "$file"   >/dev/null 2>&1
    ;;
  *.go)
    have gofmt     && gofmt -w "$file"     >/dev/null 2>&1
    have goimports && goimports -w "$file" >/dev/null 2>&1
    ;;
  *.py)
    if have ruff; then ruff format "$file" >/dev/null 2>&1; elif have black; then black "$file" >/dev/null 2>&1; fi
    ;;
  *.rs)
    have rustfmt && rustfmt "$file" >/dev/null 2>&1
    ;;
esac
exit 0
