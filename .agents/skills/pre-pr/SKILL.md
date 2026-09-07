---
name: toolbox-pre-pr
description: Before you push or open a PR, run THIS project's real checks (typecheck/lint/test/build + any project-specific gates) on the current diff, plus native /code-review, so you catch what CI would reject without burning a round-trip. Discovers the checks from the repo — never hardcodes them. Use when a change is ready and you're about to push.
---

# toolbox pre-pr — local gate before you push

Catch what CI would reject, locally, on the current diff. **Native-first**: this
*composes* the project's own checks with native review — it does not reimplement
either.

## 1. Scope to the diff
Diff the current branch against its base (`main`/`master`/`develop`):
`git diff --stat <base>...HEAD` + `git status`. Everything below is scoped to the
changed files/areas.

## 2. Discover the checks (don't hardcode)
Find the commands THIS repo actually uses, in priority order:
- `AGENTS.md` "Commands" section (the curated truth).
- `package.json` scripts · `Makefile`/`Taskfile` · `turbo.json` tasks.
- `.github/workflows/*.yml` — mirror what CI runs so local == CI, including any
  bespoke gate (e.g. a design-conformance script).

Identify: **typecheck · lint · test · build · any project-specific gate.**

## 3. Run them (cheap → expensive, scoped)
Typecheck/lint first, then tests, then build. Scope to changed packages where the
tooling allows (e.g. `turbo run test --filter=<changed>`). On the first failure,
**stop and report it with the actual output** — don't push past red.

## 4. Native review
Run your harness's native review on the diff (`/code-review` in Claude Code,
`/review` in Codex). If it touches auth, data, secrets, or payments, also run a
security pass where available (`/security-review`). Fix the clear findings; flag
the judgment calls.

## 5. Behaviour check (if user-facing)
For UI/behaviour changes, actually exercise the change (e.g. Claude Code's
`/verify` or `/run`) — tests passing isn't the same as the thing actually working.

## 6. Report — then stop
Summarise as a checklist: each check ✅/❌ + the review findings. Then **hand
back**. Do NOT push, tag, or open the PR yourself — respect the project's release
rules (e.g. never push a `v*` tag without an explicit request). The user decides
when to ship.

> Some stacks can't fully run locally (e.g. Unity builds need a licensed editor).
> Run what you can and say plainly what only CI can verify — never fake a green.
