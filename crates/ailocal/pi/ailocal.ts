// Written by ailocal. `ailocal harness unconfigure pi` removes it.
//
// Holds a local model to the checks a frontier model runs without being told. Pi loads
// every extension for every model, so nothing here acts unless the session's model is
// served by the ailocal gateway (provider `llama.cpp`): a Claude or GPT session is
// untouched.
//
// It nudges and never walls. Each guard fires once per occasion and says how to go on,
// because a pattern that misreads a project must not stop real work:
// - a push with files changed since the tests last passed is held once per change;
// - Python over JSON is held once per command while jq is installed;
// - a prompt that ends with changes no test has seen gets one more turn to run them.
//
// Its commands also run with git's editors set, so git never waits for a person to
// type into one.

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

/** The provider ailocal registers in Pi. */
export const PROVIDER = "llama.cpp";

/**
 * How every held call's result starts. Pi reports a held call as an ordinary failed
 * one, so this is how `ailocal eval agent` tells a command that never ran from one
 * that ran and failed.
 */
export const HELD = "ailocal: ";

export const PUSH_REASON =
  HELD +
  "this push was held once, because files changed since the tests last passed. Run " +
  "this project's formatter, linter and tests, as AGENTS.md or the CI workflow names " +
  "them, fix what they report, then push. If the project has no tests, push again and " +
  "it will go through.";

export const JSON_REASON =
  HELD +
  "this command was held because it runs Python over JSON, and jq is installed. Every " +
  "such command is held the first time, however it is written, so use jq instead - for " +
  "example `jq -r '.version' package.json`. If only Python will do, send this exact " +
  "command again and it will run.";

export const SETTLE_MESSAGE =
  "ailocal: files changed since the tests last passed. Run this project's formatter, " +
  "linter and tests now, as AGENTS.md or the CI workflow names them, and fix what they " +
  "report. If you already committed or pushed, commit and push the fixes too. If the " +
  "project has no tests, say so and stop.";

/**
 * What the editor of a rebase's todo list prints. It does not start with `HELD`: the
 * command ran, and failed.
 */
export const REBASE_TODO_MESSAGE =
  "There is no editor here for the rebase todo list, so ailocal stopped the rebase " +
  "before it began. To keep the list as it is, as with --autosquash, run the same " +
  "command with GIT_SEQUENCE_EDITOR=true in front. To change it, pass a command that " +
  "edits the file: GIT_SEQUENCE_EDITOR=\"sed -i.bak '2,$ s/^pick/fixup/'\" git rebase " +
  "-i <base> folds every commit after the first into it.";

/**
 * Run ahead of each of a local model's shell commands.
 *
 * Pi's shell has no terminal, so an editor that git opens waits for keys that never
 * come: a `git rebase --continue` sat in vim for nearly 12 minutes until the person
 * stopped it. `GIT_EDITOR=true` keeps the message git proposes, which is what
 * `--continue` and `--amend` want, and a bare `git commit` fails at once. The todo list
 * of `rebase -i` gets an editor that fails and says how to go on instead, because
 * keeping the list would report a squash that never happened.
 *
 * Pi runs a copy of the call's arguments, so the model never sees this line and cannot
 * learn to type it.
 */
export const GIT_ENV =
  `export GIT_EDITOR=true AILOCAL_TODO=${shellQuote(REBASE_TODO_MESSAGE)} ` +
  `GIT_SEQUENCE_EDITOR='printf "%s\\n" "$AILOCAL_TODO" >&2; false'\n`;

export default function ailocal(pi: ExtensionAPI): void {
  const session = new Session();
  let jq: Promise<boolean> | undefined;
  const hasJq = () =>
    (jq ??= pi.exec("sh", ["-c", "command -v jq"]).then(
      (result) => result.code === 0,
      () => false,
    ));

  pi.on("session_start", () => session.reset());
  pi.on("before_agent_start", (event) => {
    if (event.messageType === "user") session.newPrompt();
  });

  // Tracked whatever the model, so switching to a local one mid-session starts from
  // what actually happened. Pi reports a held call here too, as a failure.
  pi.on("tool_execution_start", (event) => {
    session.started(event.toolCallId, event.toolName, event.args);
  });
  pi.on("tool_execution_end", (event) => {
    session.ended(event.toolCallId, event.isError);
  });

  pi.on("tool_call", async (event, ctx) => {
    if (!isLocal(ctx) || event.toolName !== "bash") return;
    const input = event.input as { command?: unknown };
    const command = String(input.command ?? "");

    let reason = session.holdPush(command);
    if (reason === undefined && pythonOnJson(command) && (await hasJq())) {
      reason = session.holdJson(command);
    }
    if (reason !== undefined) return { block: true, reason };
    input.command = GIT_ENV + command;
  });

  pi.on("agent_before_settle", (event, ctx) => {
    // Pi does not continue after an error, so a nudge then would be spent on nothing.
    if (!isLocal(ctx) || event.outcome !== "completed" || !session.nudgeOnce()) return;
    return {
      entries: [
        ...event.entries,
        { type: "custom_message", customType: "ailocal-checks", content: SETTLE_MESSAGE, display: true },
      ],
      continue: true,
    };
  });
}

function isLocal(ctx: ExtensionContext): boolean {
  return ctx.model?.provider === PROVIDER;
}

/** What one session has done, and which nudges it has already had. */
export class Session {
  /** Successful file-tool changes so far. */
  private edits = 0;
  /** `edits` as it was when the tests last passed. */
  private testedAt = 0;
  /** `edits` as it was when a push was last held, so each change holds one push. */
  private pushHeldAt = -1;
  /** `edits` as it was at the last settle nudge, so each change earns one. */
  private nudgedAt = -1;
  /** Whether this prompt has had its nudge, so a model that only edits cannot loop. */
  private nudged = false;
  private heldJson = new Set<string>();
  private running = new Map<string, { tool: string; command: string }>();

  reset(): void {
    this.edits = 0;
    this.testedAt = 0;
    this.pushHeldAt = -1;
    this.nudgedAt = -1;
    this.nudged = false;
    this.heldJson.clear();
    this.running.clear();
  }

  /** A new prompt from the person, which may earn one more settle nudge. */
  newPrompt(): void {
    this.nudged = false;
  }

  started(id: string, tool: string, args: unknown): void {
    const command = tool === "bash" ? String((args as { command?: unknown })?.command ?? "") : "";
    this.running.set(id, { tool, command });
  }

  ended(id: string, isError: boolean): void {
    const call = this.running.get(id);
    this.running.delete(id);
    if (!call || isError) return;
    // The same file tools the eval counts. A `sed -i` is missed, which only means one
    // nudge fewer.
    if (call.tool === "edit" || call.tool === "write") this.edits += 1;
    else if (call.tool === "bash" && simpleCommands(call.command).some(isTest)) {
      this.testedAt = this.edits;
    }
  }

  /** The reason to hold this command once, if it pushes changes no test has seen. */
  holdPush(command: string): string | undefined {
    const commands = simpleCommands(command);
    const push = commands.findIndex(isPush);
    if (push < 0 || this.edits <= this.testedAt || this.pushHeldAt === this.edits) return undefined;
    // `cargo test && git push`: the push only happens after the tests.
    if (commands.slice(0, push).some(isTest)) return undefined;
    this.pushHeldAt = this.edits;
    return PUSH_REASON;
  }

  /** The reason to hold this Python-on-JSON command, the first time it is sent. */
  holdJson(command: string): string | undefined {
    if (this.heldJson.has(command)) return undefined;
    this.heldJson.add(command);
    return JSON_REASON;
  }

  /**
   * Whether to ask for the checks as this prompt ends: only for changes no test has
   * seen, at most once a prompt, and once per change, so a project without tests is
   * not asked again after every prompt.
   */
  nudgeOnce(): boolean {
    if (this.nudged || this.edits <= this.testedAt || this.edits === this.nudgedAt) return false;
    this.nudged = true;
    this.nudgedAt = this.edits;
    return true;
  }
}

/** `git push`, `git -C dir push`. */
export function isPush(words: string[]): boolean {
  return subcommand(words, "git", ["-C", "-c"]) === "push";
}

/**
 * A script or target name that runs the tests: `test`, `test:unit`, `test-e2e`, and the
 * `check`, `ci` and `verify` that usually include them.
 *
 * Only tests count. A linter or a build that passes says nothing about behaviour, and a
 * push after `cargo clippy` alone was measured going out untested.
 */
const TEST_SCRIPT = /^(t|tests?|spec|check|ci|verify)([:_-].*)?$/i;

/** A script file named for the tests: `./test.sh`, `scripts/run-tests.sh`, `bin/test`. */
const TEST_FILE = /^(run[-_]?)?tests?([-_.].*)?$/i;

const SHELLS = new Set(["sh", "bash", "zsh"]);

/** Test runners, whatever their arguments. */
const TEST_RUNNERS = new Set([
  "vitest", "jest", "mocha", "ava", "pytest", "tox", "nox", "unittest", "nose2",
  "phpunit", "pest", "rspec", "ctest", "bats",
]);

/** Programs that run the tests when their subcommand is one of these. */
const TEST_SUBCOMMANDS: Record<string, string[]> = {
  cargo: ["test", "nextest"],
  go: ["test"],
  deno: ["test"],
  dotnet: ["test"],
  mix: ["test"],
  swift: ["test"],
  xcodebuild: ["test"],
  rails: ["test", "spec"],
  rake: ["test", "spec"],
  gradle: ["test", "check"],
  gradlew: ["test", "check"],
  mvn: ["test", "verify"],
  mvnw: ["test", "verify"],
  playwright: ["test"],
};

/** Package managers, which run a project script by name. */
const PACKAGE_MANAGERS = new Set(["npm", "pnpm", "yarn", "bun", "composer"]);

/** Subcommands of a package manager that are its own, not a script: `npm ci` installs. */
const PACKAGE_MANAGER_BUILTINS = new Set(["ci", "install", "i", "add", "remove", "update", "exec", "dlx", "x"]);

/** Programs whose targets are named like scripts: `make test`, `just check`, `make ci`. */
const TASK_RUNNERS = new Set(["make", "just", "task"]);

/** Words that run the command after them: `npx vitest`, `time cargo test`. */
const WRAPPERS = new Set(["npx", "bunx", "pnpx", "time", "env", "nice"]);

/** Pairs of words that do the same: `uv run pytest`, `bundle exec rspec`. */
const WRAPPER_PAIRS = new Set([
  "pnpm exec", "pnpm dlx", "yarn dlx", "yarn exec", "npm exec",
  "uv run", "poetry run", "pipenv run", "bundle exec",
]);

/** Options that take a value before a runner's subcommand: `npm --prefix ui test`. */
const TAKES_VALUE = ["-C", "--cwd", "--prefix", "--filter", "--dir", "--manifest-path"];

/** Whether one simple command runs the tests. */
export function isTest(words: string[]): boolean {
  let rest = words.filter((w) => !/^[A-Za-z_][A-Za-z0-9_]*=/.test(w));
  for (;;) {
    const skip = WRAPPER_PAIRS.has(rest.slice(0, 2).join(" ")) ? 2 : WRAPPERS.has(rest[0] ?? "") ? 1 : 0;
    if (skip === 0) break;
    rest = rest.slice(skip);
    while (rest[0]?.startsWith("-")) rest = rest.slice(1);
  }
  if (isPython(rest[0] ?? "") && rest[1] === "-m") rest = rest.slice(2);

  // A bare `test` is the shell's own builtin, so a script counts as a path or when a
  // shell is handed it: `./test.sh`, `bash scripts/run-tests.sh`.
  const first = rest[0] ?? "";
  const file = SHELLS.has(first) ? rest.slice(1).find((w) => !w.startsWith("-")) : first.includes("/") ? first : undefined;
  if (file !== undefined && TEST_FILE.test(file.split("/").pop() ?? "")) return true;

  const program = first.split("/").pop() ?? "";
  if (TEST_RUNNERS.has(program)) return true;
  if (program === "php" && rest[1] === "artisan") return rest[2] === "test";

  const sub = subcommand(rest, rest[0] ?? "", TAKES_VALUE);
  if (sub === undefined) return false;
  if (TEST_SUBCOMMANDS[program]?.includes(sub)) return true;
  if (TASK_RUNNERS.has(program)) return TEST_SCRIPT.test(sub);
  const named = rest[rest.indexOf(sub) + 1] ?? "";
  if (program === "deno") return sub === "task" && TEST_SCRIPT.test(named);
  if (!PACKAGE_MANAGERS.has(program) || PACKAGE_MANAGER_BUILTINS.has(sub)) return false;
  return TEST_SCRIPT.test(sub === "run" || sub === "run-script" ? named : sub);
}

/**
 * Whether this shell command runs Python over JSON, by the same test the eval scores -
 * except a test run, which is the project's own code and not a query.
 */
export function pythonOnJson(command: string): boolean {
  const commands = simpleCommands(command);
  if (commands.some(isTest)) return false;
  const invokesPython = commands.some((words) => words.some(isPython));
  return invokesPython && command.toLowerCase().includes("json");
}

/** `python`, `python3`, `python3.12`, or any of those as a path. */
function isPython(word: string): boolean {
  const name = word.split("/").pop() ?? word;
  return /^python[0-9.]*$/.test(name);
}

/**
 * The first word after `program` that is not an option, skipping the argument of any
 * option in `takesValue`.
 */
function subcommand(words: string[], program: string, takesValue: string[]): string | undefined {
  const start = words.indexOf(program);
  if (start < 0) return undefined;
  for (let i = start + 1; i < words.length; i++) {
    const word = words[i];
    if (takesValue.includes(word)) i++;
    else if (!word.startsWith("-") && !word.startsWith("+")) return word;
  }
  return undefined;
}

/** `word` as one shell word, whatever it holds. */
function shellQuote(word: string): string {
  return `'${word.replaceAll("'", "'\\''")}'`;
}

/**
 * Split a shell command into its simple commands, each a list of words.
 *
 * Enough of the shell to read what a model typed: quotes group words and hide the
 * separators inside them, and `;`, `&`, `|`, newlines and parentheses end a command.
 */
export function simpleCommands(command: string): string[][] {
  const commands: string[][] = [];
  let words: string[] = [];
  let word = "";
  let quote: string | undefined;
  const endWord = () => {
    if (word !== "") words.push(word);
    word = "";
  };
  const endCommand = () => {
    endWord();
    if (words.length > 0) commands.push(words);
    words = [];
  };

  for (let i = 0; i < command.length; i++) {
    const c = command[i];
    if (quote !== undefined) {
      if (c === quote) quote = undefined;
      else if (quote === '"' && c === "\\") word += command[++i] ?? "";
      else word += c;
    } else if (c === "'" || c === '"') quote = c;
    else if (c === "\\") word += command[++i] ?? "";
    else if (";&|\n()".includes(c)) endCommand();
    else if (/\s/.test(c)) endWord();
    else word += c;
  }
  endCommand();
  return commands;
}
