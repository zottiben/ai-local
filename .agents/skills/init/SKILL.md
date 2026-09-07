---
name: toolbox-init
description: Bootstrap a repo for the ai-toolbox end to end. Scaffolds the knowledge files (AGENTS.md + a thin CLAUDE.md adapter, optional RULES.md) AND sets up the functional layer — hooks, MCPs, and helper skills — installing only what the detected stack needs. Detects the stack (web, backend, mobile, OR game engine - Unity/Godot/Unreal), auto-fills mechanical facts from real manifests, composes matching starter rule-snippets, interviews you for the non-obvious rules only you know, then recommends and installs the tailored tooling via the `ai-toolbox` CLI. Use when setting up a new or existing project to use this toolkit.
---

# ai-toolbox init — scaffold project knowledge

Produce hand-owned knowledge files for a project. Automate the boilerplate;
interview for the wisdom. **One-shot** — after this runs the files are yours,
with no regeneration dependency. Templates live in `templates/`, snippets in
`starters/rules/`.

## 0. Greenfield or brownfield?

- Real source present (manifests, code) → **brownfield**: harvest it (steps 1–6).
- Empty / planning-stage → **greenfield**: the design guide comes first —

  1. Confirm the *intended* stack + project type (game / site / SaaS) — ask.
  2. Scaffold `design/` from `templates/design/`: always `concept.md` +
     `design-system.md`; add `system-gdd.md` (games) or `product-brief.md`
     (sites/SaaS). Leave them as skeletons for the user (or Claude Design) to
     fill — never invent the creative content.
  3. Write a thin seed `AGENTS.md`: the intended stack + a **Design** section
     pointing at `design/` as the source of truth, with the rule *"greenfield —
     the design guide is the source of truth; build against it, ask when a
     decision isn't covered and record it there. As code lands, graduate
     conventions into Hard rules and design tokens into a conformance check."*
  4. Add the `CLAUDE.md` adapter; skip harvesting (nothing to harvest yet).

  Then hand off so the guide gets filled before real coding begins.

## 1. Detect the stack (read manifests — never guess)

| Signal | Stack |
|---|---|
| `package.json` (+ `bun.lock`/`pnpm-lock`) | Node/Bun/TS; its `scripts` are your commands |
| `go.mod` · `composer.json` · `pyproject.toml`/`requirements.txt` · `Cargo.toml` · `Gemfile` | Go · PHP/Laravel · Python · Rust · Ruby |
| `app.json`/`app.config.*` · `Podfile`/`android/` | Expo/React Native · native mobile |
| `turbo.json` · `nx.json` · workspaces | monorepo — map each app/package |
| `ProjectSettings/ProjectVersion.txt` (+ `Assets/`, `*.asmdef`) | **Unity** |
| `project.godot` · `*.uproject` | **Godot** · **Unreal** |
| `.github/workflows/*.yml` | real CI commands + gates (incl. GameCI) |

## 2. Auto-fill the mechanical facts (don't ask what you can read)

- Name + one-line purpose (from README), stack + versions, monorepo layout.
- **Commands copied verbatim** from `package.json` scripts / `Makefile` / turbo
  tasks / CI workflow: install · lint · test · build · codegen · deploy. Never
  invent a command — if you can't find it, mark it `TODO`.
- Test framework, and generated-file locations to never hand-edit (tygo, protobuf,
  OpenAPI, Unity `.meta`, etc.).

## 3. Compose matching starter snippets

Pull the relevant files from `starters/rules/` for each detected stack (e.g.
`go` + `expo` + `supabase`, or `unity`). Inline the rules that apply; drop the rest.

## 4. Interview for the non-obvious rules (the part only the user knows)

Ask a SHORT, stack-tuned set — a few at a time — and fold answers into "Hard
rules" as *rule / why / how it's enforced*:

- **Operational commands (ask first — the most-forgotten facts):** how do you run
  the dev server? how and where do you build for release? any seed / reset / env
  setup steps? These go in the **Commands** section.
- **Any project:** safety-critical constraints? cost or irreversible actions
  (paid builds, prod deploys, destructive migrations)? authz/multi-tenant rules?
  files never to hand-edit? release/versioning gotchas?
- **Web/SaaS:** row-level security / tenant isolation? migration conventions?
  design-system or lint gates that fail CI?
- **Mobile:** build-cost gating (EAS)? simulator smoke-test expectations? store
  review constraints?
- **Game (Unity/Godot/Unreal):** asmdef / assembly boundaries (what may reference
  what; where editor vs runtime code must live)? asset & `.meta` conventions
  (metas travel with assets; never hand-merge scene/prefab YAML)? EditMode vs
  PlayMode test split? target platforms & build steps (WebGL/Steam/console)?
  frame-budget / GC / object-pooling rules? engine MCP usage (e.g. drive the
  Unity editor via unity-mcp to run tests and inspect the scene)?

## 5. Write the files

- `AGENTS.md` — the filled template (stack · layout · commands · hard rules).
- `CLAUDE.md` — `@AGENTS.md` (+ optional Claude-only notes).
- `RULES.md` — **only** if the user has constraints shared across projects or
  read by humans/CI; otherwise skip it (no empty stubs).

## 6. Health-check

Run `toolbox-lint` on the result and trim what it flags. Keep `AGENTS.md` lean —
durable facts the model can't read from the code, nothing more.

## 7. Bootstrap the functional layer (hooks · MCPs · skills)

The knowledge files are only half the setup. Now wire the *functional* layer,
installing **only what this stack needs** — the mechanics live in the `bin/ai-toolbox`
CLI (this repo), so you just decide and confirm; never hand-copy files.

- **See the tailored recommendation:** `ai-toolbox recommend` (run from the repo)
  detects the stack and proposes a set per group (hooks · MCPs · skills; rules
  were already inlined in step 3). Present it and let the user **trim each group** —
  install only what they approve ("few, not a fleet"; never override their call).
- **Install the approved groups** with the matching subcommands — e.g.
  `ai-toolbox hooks format-on-edit session-context guard-irreversible conventional-commit`,
  `ai-toolbox mcp context7 supabase`, `ai-toolbox skill pre-pr capture`. Each is
  idempotent. (Or `ai-toolbox bootstrap` walks the same groups with a `[Y/n]` prompt.)
  Everything lands once, under `AGENTS.md` / `.mcp.json` / `.agents/`; each harness gets
  pointers. If the repo still has per-harness copies from an older setup, run
  `ai-toolbox migrate` first (`--dry-run` to see the plan).
- **Report the manual follow-ups the CLI can't do:** put the MCP secrets it listed in
  the repo `.env` (or export them), restart your harness, then connect/OAuth (Claude
  Code `/mcp`; Codex `codex mcp login <server>`; Pi trust the project folder, then
  `/mcp`). On Pi, MCP itself needs `ai-toolbox pi-init` once per machine - Pi's core
  ships no MCP client. For Unity, the unity MCP self-configures (no preset).
- **Greenfield:** hooks + helper skills still apply; hold off on stack-specific
  MCPs until the stack actually lands.

> Once per machine (not per repo): `ai-toolbox base-charter` appends the always-on
> charter to each detected harness's global config (`~/.claude/CLAUDE.md`,
> `~/.codex/AGENTS.md`, `~/.pi/agent/AGENTS.md`). Skip if already done.
>
> Prefer to hand the whole thing to the CLI? `ai-toolbox setup` runs the same install as
> a guided walkthrough (harnesses, prerequisites, secrets, per-harness start steps) - it
> just can't author `AGENTS.md` by interview the way this skill does.
