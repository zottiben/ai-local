---
name: gh-cli
description: Operate GitHub via the gh CLI — PRs, issues, CI checks, runs, releases. Use for anything GitHub instead of a heavy GitHub MCP (gh is already authed and far lower context cost).
---

# gh — GitHub CLI recipes

- **PRs:** `gh pr create -t "<conventional title>" -b "<body>"` · `gh pr view` ·
  `gh pr diff` · `gh pr checks --watch`
- **CI:** `gh run list` · `gh run watch <id>` · `gh run view <id> --log-failed`
- **Issues:** `gh issue list` · `gh issue view <n>` · `gh issue create`
- **Releases:** `gh release create …` — a release action; **only on an explicit
  request** (may trigger tag/build workflows that cost money).

Safety: never create tags or releases without an explicit ask. Read PR/issue
context before acting; don't merge without confirmation.
