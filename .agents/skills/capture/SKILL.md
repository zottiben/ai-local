---
name: toolbox-capture
description: Capture a just-learned project gotcha into the project's AGENTS.md so you never re-teach it. Use when the user corrects you, when a non-obvious footgun surfaces mid-task, or when the user says "remember this for the project" / "add this to AGENTS".
---

# toolbox capture — turn a correction into permanent knowledge

The flywheel: every correction becomes a durable rule, so the same thing is never
taught twice. When the user corrects you, or a non-obvious gotcha surfaces during
a task, persist it.

1. **Distill to one rule.** What to do or avoid, *why* it matters, and how it's
   caught (test / lint / review) if known. Drop the incident story — keep the rule.
2. **Check it's worth saving.** Save only what the model *can't* read from the
   code: conventions, safety/cost constraints, cross-cutting gotchas. Skip
   anything obvious from the source, and skip one-off task detail.
3. **Dedup.** If `AGENTS.md` already covers it, sharpen that line — don't add a
   near-duplicate.
4. **Place it** under the right section — operational "how to run" facts (dev
   server, build, release, seed/reset) go in **Commands**; constraints and gotchas
   go in **Hard rules**; structural notes in **Layout**. Match the file's existing
   format and terseness.
5. **Keep it lean.** One or two sentences plus a snippet only if needed. If
   `AGENTS.md` is growing long, that's a signal to tighten it, not to append
   forever — offer to run `toolbox-lint`.
6. **Confirm** the exact one-line addition with the user before writing.
