---
name: toolbox-lint
description: Health-check an AGENTS.md / CLAUDE.md / RULES.md for drift — bloat, duplication, stale references, and process-creep — and suggest trims. Use to keep your knowledge files from rotting into a heavy framework, or before committing a new/edited pack.
---

# toolbox lint — knowledge health-check

Scan a target knowledge file (default: this project's `AGENTS.md` + `CLAUDE.md`,
plus `RULES.md` if present) and report drift. These are the exact failure modes
that make a knowledge base cost more than it's worth.

Check for:

1. **Length / bloat.** Flag a file over ~150 lines, or any single rule over ~15
   lines. Knowledge is terse facts, not essays.
2. **Always-on cost.** Anything imported into *global* config beyond the tiny base
   charter. Only durable universal norms belong always-on.
3. **Duplication.** The same rule in more than one place (`AGENTS` vs `RULES` vs
   `CLAUDE`). One home per fact; the rest should import, not restate.
4. **Stale references.** Verify every path / file / command / test name mentioned
   actually exists in the repo. Flag each one that doesn't.
5. **Process creep.** Mandatory multi-step workflows, approval gates, spawn/
   orchestration doctrine, "delegate rather than doing it yourself," walls of
   MUST/ALWAYS/NEVER. Knowledge files state facts — they don't run a process.
6. **Identity / altitude.** Any "you are X, not Claude," forced announcements, or
   language that suppresses the model's own judgment.
7. **Guessable content.** Rules the model could read straight from the code — cut
   them; they're noise.

Report each finding as `file:line — issue — suggested fix`. End with an overall
verdict (**lean / drifting / bloated**) and the top 3 trims. Don't edit unless asked.
