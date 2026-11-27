// node --test crates/ailocal/pi/ailocal.test.ts

import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, describe, it } from "node:test";

import ailocal, {
  GIT_ENV,
  HELD,
  isPush,
  isTest,
  JSON_REASON,
  type Lesson,
  lessonReason,
  lessonsMessage,
  lessonsPath,
  pendingNotice,
  PROMPT_LESSONS,
  PUSH_REASON,
  pythonOnJson,
  readLessons,
  REBASE_TODO_MESSAGE,
  relevantLessons,
  repoId,
  runs,
  Session,
  SETTLE_MESSAGE,
  simpleCommands,
  touches,
  tripped,
} from "./ailocal.ts";

const tests = (command: string) => simpleCommands(command).some(isTest);

describe("isTest", () => {
  it("knows the usual test runners and test scripts", () => {
    for (const command of [
      "cargo test",
      "cargo +stable test --all-targets",
      "time cargo nextest run",
      "RUST_BACKTRACE=1 cargo test",
      "npm test",
      "npm run test:unit",
      "npm --prefix ui test",
      "pnpm test",
      "yarn test-e2e",
      "bun test",
      "npx vitest run",
      "./node_modules/.bin/jest",
      "uv run pytest -q",
      "python3 -m pytest tests/test_json.py",
      "python -m unittest discover",
      "go test ./...",
      "make test",
      "make -C ui check",
      "make ci",
      "deno task test",
      "bundle exec rspec",
      "bin/rails test",
      "php artisan test",
      "./vendor/bin/phpunit",
      "./gradlew test",
      "./test.sh",
      "bash scripts/run-tests.sh",
    ]) {
      assert.ok(tests(command), command);
    }
  });

  // A push after `cargo clippy` alone was measured going out untested.
  it("does not take a linter, a formatter, a build or an install for the tests", () => {
    for (const command of [
      "cargo clippy --all-targets -- -D warnings",
      "cargo fmt --all -- --check",
      "cargo check",
      "cargo build --release",
      "npm run lint",
      "npx prettier --check .",
      "npx tsc --noEmit",
      "npm run build",
      "npm ci",
      "pnpm add -D vitest",
      "deno task lint",
      "make",
      "make build",
      "test -f Cargo.toml",
      "[ -d tests ]",
      "ls tests/",
      "python3 -m json.tool data.json",
      "git status",
    ]) {
      assert.ok(!tests(command), command);
    }
  });
});

describe("isPush and pythonOnJson", () => {
  it("finds a push however git is invoked", () => {
    assert.ok(simpleCommands("git -C repo push origin main").some(isPush));
    assert.ok(simpleCommands("git add -A && git commit -m x && git push").some(isPush));
    assert.ok(!simpleCommands("git commit -m 'git push later'").some(isPush));
  });

  it("finds Python over JSON as the eval scores it", () => {
    assert.ok(pythonOnJson(`python3 -c "import json; print(json.load(open('a.json'))['v'])"`));
    assert.ok(pythonOnJson("cat releases.json | python3 -m json.tool"));
    assert.ok(pythonOnJson("python3 - <<'EOF'\nimport json\nEOF"));
    assert.ok(!pythonOnJson("jq -r .version package.json"));
    assert.ok(!pythonOnJson("python3 build.py"));
    // The project's own tests, not a query.
    assert.ok(!pythonOnJson("python3 -m pytest tests/test_json.py"));
  });

  it("starts every hold with the prefix the eval looks for", () => {
    assert.ok(PUSH_REASON.startsWith(HELD));
    assert.ok(JSON_REASON.startsWith(HELD));
  });
});

describe("Session", () => {
  const edit = (s: Session, id: string) => {
    s.started(id, "edit", { path: "src/lib.rs" });
    s.ended(id, false);
  };
  const bash = (s: Session, id: string, command: string, isError = false) => {
    s.started(id, "bash", { command });
    s.ended(id, isError);
  };

  it("holds a push once per unchecked change", () => {
    const s = new Session();
    edit(s, "1");
    assert.equal(s.holdPush("git push"), PUSH_REASON);
    // Sent again, it goes through: a project with no checks must still be able to push.
    assert.equal(s.holdPush("git push"), undefined);

    edit(s, "2");
    assert.equal(s.holdPush("git push"), PUSH_REASON);
  });

  it("lets a push through after tests that passed, and not after ones that failed", () => {
    const s = new Session();
    edit(s, "1");
    bash(s, "2", "cargo test", true);
    assert.equal(s.holdPush("git push"), PUSH_REASON);

    edit(s, "3");
    bash(s, "4", "cargo test 2>&1 | tail -20");
    assert.equal(s.holdPush("git push"), undefined);
  });

  it("holds a push after a linter alone", () => {
    const s = new Session();
    edit(s, "1");
    bash(s, "2", "cargo fmt && cargo clippy -- -D warnings");
    assert.equal(s.holdPush("git push"), PUSH_REASON);
  });

  it("lets a push through when the same command tests first", () => {
    const s = new Session();
    edit(s, "1");
    assert.equal(s.holdPush("cargo test && git push"), undefined);
    assert.equal(s.holdPush("git push; cargo test"), PUSH_REASON);
  });

  it("holds nothing when the agent changed nothing", () => {
    const s = new Session();
    assert.equal(s.holdPush("git push"), undefined);
    assert.equal(s.nudgeOnce(), false);
  });

  it("does not count a change that failed", () => {
    const s = new Session();
    s.started("1", "edit", {});
    s.ended("1", true);
    assert.equal(s.holdPush("git push"), undefined);
  });

  it("holds each Python-on-JSON command once", () => {
    const s = new Session();
    const command = "python3 -c 'import json'";
    assert.equal(s.holdJson(command), JSON_REASON);
    assert.equal(s.holdJson(command), undefined);
    assert.equal(s.holdJson(`${command} # again`), JSON_REASON);
  });

  it("nudges once per prompt, only with unchecked changes", () => {
    const s = new Session();
    edit(s, "1");
    assert.equal(s.nudgeOnce(), true);
    // A model that answers the nudge with more edits and no check is not asked again.
    edit(s, "2");
    assert.equal(s.nudgeOnce(), false);

    s.newPrompt();
    bash(s, "3", "cargo test");
    assert.equal(s.nudgeOnce(), false);

    edit(s, "4");
    assert.equal(s.nudgeOnce(), true);
  });

  it("does not ask again for changes it already asked about", () => {
    const s = new Session();
    edit(s, "1");
    assert.equal(s.nudgeOnce(), true);

    // The project has no tests, and the next prompt is a question.
    s.newPrompt();
    assert.equal(s.nudgeOnce(), false);
  });
});

type Handler = (event: unknown, ctx: unknown) => unknown;

/** The extension as Pi loads it, on a machine with or without jq, in a repository. */
function load(jq: boolean, origin = "") {
  const handlers = new Map<string, Handler>();
  const entries: { type: string; data: unknown }[] = [];
  const notes: string[] = [];
  const pi = {
    on: (name: string, handler: Handler) => handlers.set(name, handler),
    exec: async (program: string) =>
      program === "git" ? { code: origin ? 0 : 1, stdout: origin + "\n" } : { code: jq ? 0 : 1 },
    appendEntry: (type: string, data: unknown) => entries.push({ type, data }),
  };
  ailocal(pi as never);
  const ui = { notify: (message: string) => notes.push(message) };
  const emit = (name: string, event: object, provider = "llama.cpp") =>
    handlers.get(name)?.(event, { model: { provider }, cwd: process.cwd(), ui });
  return Object.assign(emit, { entries, notes });
}

describe("the extension", () => {

  function editedFile(emit: ReturnType<typeof load>) {
    emit("tool_execution_start", { toolCallId: "e", toolName: "edit", args: {} });
    emit("tool_execution_end", { toolCallId: "e", toolName: "edit", isError: false });
  }

  it("leaves a frontier model alone", async () => {
    const emit = load(true);
    editedFile(emit);
    const push = { toolName: "bash", toolCallId: "p", input: { command: "git push" } };
    assert.equal(await emit("tool_call", push, "anthropic"), undefined);
    const settle = { entries: [], outcome: "completed" };
    assert.equal(await emit("agent_before_settle", settle, "anthropic"), undefined);
  });

  it("runs a local model's commands with git's editors set, and a frontier model's as sent", async () => {
    const emit = load(true);
    const local = { toolName: "bash", toolCallId: "l", input: { command: "git rebase --continue" } };
    assert.equal(await emit("tool_call", local), undefined);
    assert.equal(local.input.command, GIT_ENV + "git rebase --continue");

    const frontier = { toolName: "bash", toolCallId: "f", input: { command: "git rebase --continue" } };
    assert.equal(await emit("tool_call", frontier, "anthropic"), undefined);
    assert.equal(frontier.input.command, "git rebase --continue");
  });

  it("holds a local model's unchecked push", async () => {
    const emit = load(true);
    editedFile(emit);
    const push = { toolName: "bash", toolCallId: "p", input: { command: "git push" } };
    assert.deepEqual(await emit("tool_call", push), { block: true, reason: PUSH_REASON });
  });

  it("holds Python on JSON only where jq is installed", async () => {
    const call = { toolName: "bash", toolCallId: "j", input: { command: "python3 -m json.tool a.json" } };
    assert.equal(await load(false)("tool_call", call), undefined);
    assert.deepEqual(await load(true)("tool_call", call), { block: true, reason: JSON_REASON });
  });

  it("asks for the checks once before a prompt with unchecked changes ends", async () => {
    const emit = load(true);
    await emit("before_agent_start", { messageType: "user", prompt: "Fix it" });
    editedFile(emit);
    assert.equal(await emit("agent_before_settle", { entries: [], outcome: "error" }), undefined);

    const kept = { type: "custom", customType: "other" };
    const result = (await emit("agent_before_settle", { entries: [kept], outcome: "completed" })) as {
      entries: { content?: string }[];
      continue: boolean;
    };
    assert.equal(result.continue, true);
    assert.equal(result.entries[0], kept);
    assert.equal(result.entries[1].content, SETTLE_MESSAGE);
    assert.equal(await emit("agent_before_settle", { entries: [], outcome: "completed" }), undefined);
  });
});

describe("lessons", () => {
  const root = mkdtempSync(join(tmpdir(), "ailocal-lessons-"));
  after(() => rmSync(root, { recursive: true, force: true }));

  const lesson = (id: string, text: string, more: Partial<Lesson> = {}): Lesson => ({ id, text, ...more });
  const template = lesson(
    "pr-template",
    "Before writing a PR body, read the pull request template in .github and use its headings.",
    { commands: ["gh pr create", "gh pr edit"] },
  );
  const suppress = lesson("no-disable", "Fix a lint error at its cause rather than disabling the rule.", {
    adds: ["eslint-disable"],
  });
  const changelog = lesson("changelog", "Leave CHANGELOG.md alone: the release job writes it.", {
    files: ["CHANGELOG.md"],
  });

  function store(name: string, lessons: object[], version = 1): string {
    const path = join(root, name);
    writeFileSync(path, JSON.stringify({ version, lessons }));
    return path;
  }

  it("are kept where ailocal keeps them, unless the eval says otherwise", () => {
    assert.equal(lessonsPath({ HOME: "/h" }), "/h/.config/ailocal/lessons.json");
    assert.equal(lessonsPath({ HOME: "/h", XDG_CONFIG_HOME: "/x" }), "/x/ailocal/lessons.json");
    assert.equal(lessonsPath({ HOME: "/h", XDG_CONFIG_HOME: "/x", AILOCAL_LESSONS: "/e.json" }), "/e.json");
  });

  it("are read only when active, and not at all from a newer ailocal", () => {
    assert.deepEqual(readLessons(join(root, "missing.json")), { active: [], pending: 0 });
    const path = store("states.json", [
      { ...template, state: "active" },
      { ...suppress, state: "pending" },
      { ...changelog, state: "rejected" },
    ]);
    const read = readLessons(path);
    assert.deepEqual(
      read.active.map((l) => l.id),
      ["pr-template"],
    );
    assert.equal(read.pending, 1);
    assert.throws(() => readLessons(store("newer.json", [], 2)), /newer ailocal/);
  });

  it("waiting for review are counted in words that agree", () => {
    assert.match(pendingNotice(1), /: 1 lesson learned .* waits for you: ailocal lessons review$/);
    assert.match(pendingNotice(3), /: 3 lessons learned .* wait for you: /);
  });

  it("name a repository as lessons::repo_id does", () => {
    const cases = JSON.parse(readFileSync(new URL("./remotes.json", import.meta.url), "utf8")) as [
      string,
      string | null,
    ][];
    for (const [remote, id] of cases) assert.equal(repoId(remote), id ?? undefined, JSON.stringify(remote));
  });

  it("go along with a prompt that shares words with them, the closest first", () => {
    const all = [template, suppress, changelog];
    assert.deepEqual(relevantLessons(all, "hello?"), []);
    assert.deepEqual(relevantLessons(all, "Fix it, then commit and push."), []);
    assert.deepEqual(
      relevantLessons(all, "Open a pull request for this branch").map((l) => l.id),
      ["pr-template"],
    );
    // One word in common is chance, and a trigger's words are for the moment it names.
    assert.deepEqual(relevantLessons(all, "Open a PR for this branch"), []);
    assert.deepEqual(
      relevantLessons(all, "The lint errors in the changelog script are back").map((l) => l.id),
      ["no-disable"],
    );
    assert.deepEqual(
      relevantLessons(all, "The lint errors in CHANGELOG.md are back").map((l) => l.id),
      ["changelog", "no-disable"],
    );
    const many = Array.from({ length: 6 }, (_, i) => lesson(`l${i}`, "Squash fixups before pushing."));
    assert.deepEqual(relevantLessons(many, "squash these"), []);
    assert.deepEqual(
      relevantLessons(many, "squash these fixups").map((l) => l.id),
      ["l5", "l4", "l3"],
    );
    assert.equal(PROMPT_LESSONS, 3);
  });

  it("trip on a command however it is invoked, and not on its neighbours", () => {
    const words = (command: string) => simpleCommands(command)[0];
    assert.ok(runs(words("gh pr create --fill"), "gh pr create"));
    assert.ok(runs(words("GH_TOKEN=x /opt/homebrew/bin/gh pr create -t T -b B"), "gh pr create"));
    assert.ok(runs(words("git -C ../other push origin main"), "git push"));
    assert.ok(runs(words("npx eslint ."), "eslint"));
    assert.ok(!runs(words("gh pr view 12"), "gh pr create"));
    assert.ok(!runs(words("git commit -m 'gh pr create'"), "gh pr create"));
    assert.ok(!runs(words("echo gh pr create"), "gh pr create"));

    const bash = (command: string) => tripped([template], "bash", { command }, root, () => "");
    assert.deepEqual(bash("git push -u origin fix && gh pr create --fill"), [template]);
    assert.deepEqual(bash("gh pr list"), []);
  });

  it("trip on a file wherever it sits, by name, directory or extension", () => {
    assert.ok(touches("CHANGELOG.md", "CHANGELOG.md"));
    assert.ok(touches("packages/ui/CHANGELOG.md", "CHANGELOG.md"));
    assert.ok(touches("database/migrations/2024_01_01_x.php", "database/migrations/"));
    assert.ok(touches("src/Button.tsx", ".tsx"));
    assert.ok(touches(".github/pull_request_template.md", "./.github"));
    assert.ok(!touches("MYCHANGELOG.md", "CHANGELOG.md"));
    assert.ok(!touches("src/Button.tsx.orig", ".tsx"));
    assert.ok(!touches("src/migrations.ts", "migrations"));

    const edit = { path: join(root, "docs", "CHANGELOG.md"), edits: [{ oldText: "a", newText: "b" }] };
    assert.deepEqual(tripped([changelog], "edit", edit, root, () => ""), [changelog]);
    assert.deepEqual(tripped([changelog], "read", edit, root, () => ""), []);
  });

  it("trip on text an edit or a write adds, not on text that was already there", () => {
    const edit = (oldText: string, newText: string) =>
      tripped([suppress], "edit", { path: "a.ts", edits: [{ oldText, newText }] }, root, () => "");
    assert.deepEqual(edit("const a = 1;", "// eslint-disable-next-line\nconst a = 1;"), [suppress]);
    assert.deepEqual(edit("// eslint-disable-next-line\nlet a = 1;", "// eslint-disable-next-line\nconst a = 1;"), []);
    assert.deepEqual(edit("// eslint-disable-next-line\nlet a", "let a"), []);

    let reads = 0;
    const write = (before: string, content: string) =>
      tripped([suppress, { ...suppress, id: "again" }], "write", { path: "a.ts", content }, root, () => {
        reads++;
        return before;
      });
    assert.equal(write("", "/* eslint-disable */\n").length, 2);
    assert.equal(reads, 1);
    assert.deepEqual(write("/* eslint-disable */\nold\n", "/* eslint-disable */\nnew\n"), []);
  });

  it("hold a call once per lesson in each prompt", () => {
    const s = new Session();
    assert.deepEqual(s.holdLessons([template]), [template]);
    assert.deepEqual(s.holdLessons([template, suppress]), [suppress]);
    assert.deepEqual(s.holdLessons([template, suppress]), []);
    s.newPrompt();
    assert.deepEqual(s.holdLessons([template]), [template]);
  });

  it("say what they are and how to go on", () => {
    assert.ok(lessonReason([template]).startsWith(HELD));
    assert.match(lessonReason([template]), /a lesson .* applies to it:\n- Before writing a PR body/);
    assert.match(lessonReason([template, suppress]), /lessons .* apply to it:\n- Before.*\n- Fix.*\nIf the call already follows them/);
    assert.equal(
      lessonsMessage([template, suppress]).split("\n").slice(1).join("\n"),
      `- ${template.text}\n- ${suppress.text}`,
    );
  });

  describe("in the extension", () => {
    const saved = process.env.AILOCAL_LESSONS;
    after(() => {
      if (saved === undefined) delete process.env.AILOCAL_LESSONS;
      else process.env.AILOCAL_LESSONS = saved;
    });

    const elsewhere = { ...suppress, id: "elsewhere", repo: "github.com/other/repo" };
    const here = { ...changelog, repo: "github.com/positivegroup/nodifi-data" };
    const seed = () => {
      process.env.AILOCAL_LESSONS = store(
        "extension.json",
        [template, elsewhere, here].map((l) => ({ ...l, state: "active" })),
      );
    };
    const prompt = (text: string) => ({ type: "before_agent_start", messageType: "user", prompt: text });
    const pr = { toolName: "bash", toolCallId: "pr", input: { command: "gh pr create --fill" } };

    it("go along with a local model's prompt, for this repository and every one", async () => {
      seed();
      const emit = load(true, "git@github.com:positivegroup/nodifi-data.git");
      const result = (await emit("before_agent_start", prompt("Open a pull request, then fix the lint in CHANGELOG.md"))) as {
        message: { customType: string; content: string; display: boolean; details: { lessons: string[] } };
      };
      assert.deepEqual(result.message.details.lessons.sort(), ["changelog", "pr-template"]);
      assert.equal(result.message.customType, "ailocal-lessons");
      assert.equal(result.message.display, true);
      assert.deepEqual(
        emit.entries.map((e) => e.data),
        result.message.details.lessons.map((id) => ({ id, how: "prompt" })),
      );
      assert.equal(await emit("before_agent_start", prompt("hello?")), undefined);
    });

    it("hold a local model's call once per prompt, and never a frontier model's", async () => {
      seed();
      const emit = load(true);
      assert.equal(await emit("before_agent_start", prompt("Open a pull request"), "anthropic"), undefined);
      assert.equal(await emit("tool_call", pr, "anthropic"), undefined);

      await emit("before_agent_start", prompt("Ship it"));
      assert.deepEqual(await emit("tool_call", pr), { block: true, reason: lessonReason([template]) });
      assert.deepEqual(emit.entries.at(-1)?.data, { id: "pr-template", how: "held" });
      assert.equal(await emit("tool_call", { ...pr, input: { command: "gh pr create --fill" } }), undefined);

      // Outside the repository the scoped lesson is for, it holds nothing.
      const edit = { toolName: "edit", toolCallId: "e", input: { path: "CHANGELOG.md", edits: [] } };
      assert.equal(await emit("tool_call", edit), undefined);

      await emit("before_agent_start", prompt("Ship it again"));
      assert.equal(((await emit("tool_call", pr)) as { block: boolean }).block, true);
    });

    it("say once that a store cannot be read, and give no lessons meanwhile", async () => {
      process.env.AILOCAL_LESSONS = join(root, "broken.json");
      writeFileSync(process.env.AILOCAL_LESSONS, "{");
      const emit = load(true);
      assert.equal(await emit("before_agent_start", prompt("Open a PR")), undefined);
      assert.equal(await emit("before_agent_start", prompt("Open a PR")), undefined);
      assert.equal(emit.notes.length, 1);
      assert.match(emit.notes[0], /^ailocal: no lessons until .*broken\.json can be read/);
      assert.equal(await emit("tool_call", pr), undefined);
    });
  });
});

// Real git through bash, started the way Pi starts a command: detached, with no
// terminal and nothing on stdin. Without GIT_ENV the first of these sits in vim.
describe("GIT_ENV", () => {
  const root = mkdtempSync(join(tmpdir(), "ailocal-git-"));
  after(() => rmSync(root, { recursive: true, force: true }));

  const env = {
    ...process.env,
    GIT_CONFIG_GLOBAL: "/dev/null",
    GIT_CONFIG_NOSYSTEM: "1",
    GIT_AUTHOR_NAME: "t",
    GIT_AUTHOR_EMAIL: "t@example.invalid",
    GIT_COMMITTER_NAME: "t",
    GIT_COMMITTER_EMAIL: "t@example.invalid",
  };
  const git = (dir: string, ...args: string[]) =>
    execFileSync("git", args, { cwd: dir, env, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();

  /** A repository on `feature`, stopped by a conflict partway through a rebase onto `main`. */
  function conflicted(name: string): string {
    const dir = join(root, name);
    git(root, "init", "-q", "-b", "main", dir);
    writeFileSync(join(dir, "notes.txt"), "one\n");
    git(dir, "add", ".");
    git(dir, "commit", "-qm", "base");
    git(dir, "checkout", "-qb", "feature");
    writeFileSync(join(dir, "notes.txt"), "feature\n");
    git(dir, "commit", "-qam", "feature change");
    git(dir, "checkout", "-q", "main");
    writeFileSync(join(dir, "notes.txt"), "main\n");
    git(dir, "commit", "-qam", "main change");
    git(dir, "checkout", "-q", "feature");
    assert.throws(() => git(dir, "rebase", "main"));
    return dir;
  }

  function run(dir: string, command: string): Promise<{ code: number | null; out: string }> {
    return new Promise((resolve) => {
      const child = spawn("bash", ["-c", GIT_ENV + command], {
        cwd: dir,
        detached: true,
        env,
        stdio: ["ignore", "pipe", "pipe"],
      });
      let out = "";
      child.stdout.on("data", (data) => (out += data));
      child.stderr.on("data", (data) => (out += data));
      const timer = setTimeout(() => child.pid && process.kill(-child.pid, "SIGKILL"), 10_000);
      child.on("close", (code) => {
        clearTimeout(timer);
        resolve({ code, out });
      });
    });
  }

  it("lets a rebase continue with the message git proposes", async () => {
    const dir = conflicted("continue");
    writeFileSync(join(dir, "notes.txt"), "both\n");
    const { code, out } = await run(dir, "git add -A && git rebase --continue");
    assert.equal(code, 0, out);
    assert.equal(git(dir, "log", "--format=%s"), "feature change\nmain change\nbase");
  });

  it("makes a commit with no message fail at once", async () => {
    const dir = conflicted("commit");
    git(dir, "rebase", "--abort");
    const { code, out } = await run(dir, "git commit --allow-empty");
    assert.equal(code, 1);
    assert.match(out, /empty commit message/);
  });

  it("stops rebase -i before it begins and says how to go on", async () => {
    const dir = conflicted("interactive");
    git(dir, "rebase", "--abort");
    const { code, out } = await run(dir, "git rebase -i HEAD~1");
    assert.equal(code, 1);
    assert.ok(out.startsWith(REBASE_TODO_MESSAGE), out);
    assert.ok(!REBASE_TODO_MESSAGE.startsWith(HELD));
    assert.equal(existsSync(join(dir, ".git", "rebase-merge")), false);
  });

  it("lets through both ways on that it names", async () => {
    const dir = conflicted("ways-on");
    git(dir, "rebase", "--abort");
    assert.equal((await run(dir, "GIT_SEQUENCE_EDITOR=true git rebase -i --autosquash HEAD~1")).code, 0);

    const example = /(GIT_SEQUENCE_EDITOR="[^"]+") git rebase -i <base>/.exec(REBASE_TODO_MESSAGE)?.[1];
    assert.ok(example);
    git(dir, "commit", "-q", "--allow-empty", "-m", "second");
    const { code, out } = await run(dir, `${example} git rebase -i HEAD~2`);
    assert.equal(code, 0, out);
    assert.equal(git(dir, "log", "--format=%s"), "feature change\nbase");
  });
});
