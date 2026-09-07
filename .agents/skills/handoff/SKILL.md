---
name: toolbox-handoff
description: Checkpoint the build before you clear/compact the context or end a session, so a fresh context resumes exactly where you left off. Commits + pushes in-progress work, certifies the repo is green with the project's own gates, and refreshes a committed HANDOFF.md "resume here" doc. Use when the context is getting full, before /clear or /compact, when wrapping up a session, or when the user says "hand off" / "clear context" / "checkpoint" / "pick this up later".
---

# toolbox handoff - checkpoint before clearing context

Make it safe to clear or compact mid-build and pick up seamlessly in a fresh context.
This *complements* the harness's native compaction - it adds the durability compaction
skips. Two things must be true when you finish: the **repo** is the durable source of
truth (committed + pushed + green), and a **resume doc** tells the next context where
things are and what to do first. A fresh context sees only what's in git plus what it's
pointed at - everything else is lost. Run the steps in order.

**Where the resume doc goes.** If `aip` is on PATH, it goes in the ai-planner database
(step 3a) - scoped to this plan and this worktree, so it can't be copied between
worktrees or drift out of sync with the plan. Otherwise fall back to a committed
`HANDOFF.md` (step 3b).

## 1. Commit and push in-progress work
`git status --short`. Uncommitted work vanishes on handoff. If there's any:
- Bring it to a coherent state - never leave the default branch broken; if on the
  default branch, branch first.
- Commit (Conventional Commits; never add an agent co-author) and `git push`. If
  genuinely mid-slice, commit a clearly-labelled WIP and flag it in `HANDOFF.md` (step 3).
Don't open a PR or push a `v*` tag unless the user asked.

## 2. Certify the green baseline
Give the next context a known-good point to trust. Discover and run THIS project's real
gates the way `toolbox-pre-pr` does (AGENTS.md "Commands", `package.json` scripts,
Makefile/Taskfile, CI) - typecheck / lint / test / build. Record the short HEAD sha and
each PASS/FAIL. If anything is red, fix it - or record precisely what's red and why.
Never certify green over a failure.

## 3a. Record the handoff (when `aip` is available)
The plan already holds the slice statuses, decisions and progress, so record only what
it doesn't: the gates you actually ran, what to do first, and what you learned.

```sh
aip handoff write \
  --gate typecheck=pass --gate "test=pass:731 tests" --gate lint=pass \
  --next "PR2 - button variant + Broker Settlements" \
  --notes "…anything the plan doesn't already say…"
```

Record each gate's **real** result - a failure is `--gate lint=fail`, never omitted.
Before this, make sure the plan itself is current, since that's what the next context
reads:

```sh
aip slice set PR1 in_review        # or done / blocked --reason "…"
aip log "…what happened this session…" --slice PR1
aip gotcha add "<title>" "<the API quirk / verification trick>"
aip question add "<the thing only Ben can decide>"
```

A gotcha that's a *durable* project rule belongs in `AGENTS.md` instead (use
`toolbox-capture`); the planner holds volatile, this-build state.

Then skip to step 4 - `aip resume` renders the resume doc from live state, so there is
no file to keep in sync.

## 3b. Refresh HANDOFF.md (fallback, when `aip` is not installed)
Write/update a committed `HANDOFF.md` at the repo root so it matches reality. Keep it
lean - a "you are here + how to resume + gotchas" pointer, not a changelog (git log,
`CHANGELOG.md`, any roadmap hold the full history). Three parts:
- **RESUME HERE** (top): branch + HEAD sha, clean/pushed status, whether a PR exists,
  and the **next 1-3 concrete work items**.
- **Gotchas learned this session** - API quirks, verification tricks, env/dep facts the
  code alone doesn't reveal. The highest-value part. A gotcha that's a *durable* project
  rule belongs in `AGENTS.md` instead (use `toolbox-capture`); `HANDOFF.md` holds only
  volatile, this-build state - trim stale detail rather than appending forever.
- **How to resume** - the one line a fresh context should start with (see step 4).
If the project auto-loads a memory (Claude Code) or a global `AGENTS.md` (Codex), mirror
the one-line RESUME pointer there too, so it surfaces without being asked.

## 4. Confirm, then it's safe to clear
Tell the user briefly: the HEAD sha, that it's pushed + green, that the handoff is
recorded, and how to resume - start a fresh session and say **"`aip resume` and
continue"** (or **"read HANDOFF.md and continue"** on the fallback path). Then it's safe
to `/clear` or `/compact`.

## What resume looks like (next context)
Fresh session: `git pull`, then `aip resume` (or read `HANDOFF.md`) plus `AGENTS.md`,
re-run the gates to confirm the certified-green baseline still holds, then continue the
next item in small, verified, committed increments.

With `aip` installed the session-start hook already names the plan, the slice and the
next item before you're asked, so `aip resume` is a deepening rather than a discovery.
