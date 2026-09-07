#!/usr/bin/env bash
# SessionStart: inject current git state so the session starts oriented.
# Cross-harness: resolves the project dir from the payload's `cwd` (Codex) or
# ${CLAUDE_PROJECT_DIR} (Claude Code), falling back to the current directory.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"; . "$DIR/_lib.sh"
HOOK_JSON=$(cat 2>/dev/null || true)

proj=$(json_field cwd); : "${proj:=${CLAUDE_PROJECT_DIR:-.}}"
cd "$proj" 2>/dev/null || true
command -v git >/dev/null 2>&1 || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
dirty=$(git status --short 2>/dev/null | wc -l | tr -d ' ')
last=$(git log -1 --oneline 2>/dev/null)

emit_context "SessionStart" "Git: on '${branch}', ${dirty} uncommitted file(s). Last commit: ${last}"
exit 0
