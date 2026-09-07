---
name: toolbox-slim
description: Debloat a bloated project AGENTS.md - extract procedural, multi-step content into its own skill, condense verbose rules without losing any constraint, and relocate misplaced content to its right home. Use when AGENTS.md has grown long (past ~100 lines), reads like essays, when toolbox-lint flags it as bloated / drifting, or when the user says "AGENTS.md is too big" / "slim this down" / "debloat".
---

# toolbox slim - debloat AGENTS.md by extraction + condensing

AGENTS.md earns its always-loaded cost only as terse, durable facts the model can't read
from the code. When it bloats, the fix is rarely "delete rules" - it's moving each block
to its *right home* and tightening what stays. Lose no real constraint: change form, not
substance. This is the *fix* for what `toolbox-lint` diagnoses.

## 1. Diagnose first
Read the project's `AGENTS.md` (+ `CLAUDE.md` / `RULES.md`) and get its line count. Run
`toolbox-lint` (or apply its checks) to surface the bloat, duplication, stale refs, and
guessable content - act on that diagnosis rather than repeating it.

## 2. Classify every block by its right home
AGENTS.md holds declarative *facts*, not procedures. For each section / rule, decide:
- **Durable fact / constraint / gotcha** -> stays in AGENTS.md (condense in step 4).
- **A repeatable procedure** (numbered how-to, "to add a new X do 1)2)3)", a runbook, a
  multi-step task workflow) -> **extract into a skill** (step 3). Usually the biggest win.
- **Generic stack footgun** already in `starters/rules/*` -> replace with the lean line.
- **UI / design tokens or conventions** -> `design/design-system.md` (if a design layer exists).
- **Constraint shared across projects or read by CI / humans** -> `RULES.md`, then import it.
- **Readable straight from the code**, or **duplicated elsewhere** -> cut it (keep one home).

## 3. Extract procedures into skills
For each procedure to lift out:
- First check it isn't already covered by a native command or an existing skill - if it
  is, just cut it and leave a pointer; never author a duplicate skill (that's the sprawl
  `toolbox-audit` guards against).
- Otherwise author `<skill-name>/SKILL.md` with `name` + `description` frontmatter (so it's
  auto-selected). Move the steps in, then tighten to a recipe.
- Install the identical folder into every harness the repo targets: `.claude/` present ->
  `.claude/skills/<name>/`; `.codex/` present -> `.agents/skills/<name>/` (Codex reads
  skills from `.agents/skills/`, NOT `.codex/`). If both are set up, write it to both.
- Replace the AGENTS.md block with one pointer line: "To <do X>, use the `<skill-name>` skill."
  AGENTS.md is shared, so that single edit reaches both harnesses.

## 4. Condense what stays
Tighten each surviving rule to the template's altitude - *rule / why it matters / exact
snippet / how it's enforced*, no more. Cut essays, repeated MUST/ALWAYS/NEVER walls, and
over-explanation; merge near-duplicate bullets. Preserve every real constraint's substance
- if unsure whether a line carries information, keep it. Match the file's existing format.

## 5. Confirm, apply, verify
Show the plan first: what moves where, what condenses, and the projected line count (aim
for the template's shape, well under ~100 lines). On confirmation, apply. Then verify
nothing was lost: every constraint still has exactly one home, each extracted skill reads
standalone, all pointers/paths resolve, and a re-run of `toolbox-lint` comes back lean.
