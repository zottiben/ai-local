---
name: toolbox-audit
description: Audit the ai-toolbox itself for drift — total footprint, oversized files, duplication, process-creep, native-capability duplication, stale references. Run periodically so the toolkit never becomes the heavy framework it replaced.
---

# toolbox audit — keep the toolkit lean

Run the health-check across the WHOLE toolbox (not one project file), plus a
footprint tally. This is the guard that ai-toolbox never turns into Software Teams.

1. **Footprint.** Tally words/tokens across `templates/`, `starters/`, `skills/`.
   Flag if the *always-on* surface (the base charter + anything wired into global
   config) grows past a tiny budget. Flag any single file over ~150 lines, or any
   skill over ~80 — **except `README.md`**, which is reference docs (never loaded
   into context), so length there is fine.
2. **Duplication.** The same rule/guidance repeated across snippets, templates, or
   skills. One home per fact.
3. **Process creep.** Any skill drifting from recipe → program: mandatory
   multi-step orchestration, approval gates, spawn/handoff doctrine, "delegate
   instead of doing it yourself."
4. **Native duplication.** Any skill/agent reimplementing a tool the harness already
   ships (Claude Code: `/code-review`, `/security-review`, `/verify`, `/run`, `/init`,
   `/simplify`; Codex: `/review`, `/init`). Cut or redirect to the native tool.
5. **Stale references.** Every path / file / skill name referenced actually exists.
6. **Law check.** Re-read the 7 laws in `README.md`; flag anything that violates one.

Report each as `file:line — issue — fix`. End with a verdict (**lean / drifting /
bloated**), the total footprint, and the top 3 trims. Don't edit unless asked.
