import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { describeApplied, describeProbe, describeStep, describeTrouble } from "./format";
import { Setup } from "./Setup";
import { Ticks } from "./Ticks";
import type { GatewayCheck, GatewayView, HarnessView, Meta, Services } from "./types";

const services = vi.fn();
const gateway = vi.fn();
const harnesses = vi.fn();
const config = vi.fn();
const installServices = vi.fn();
const serviceAction = vi.fn();
const checkGateway = vi.fn();
const configureHarness = vi.fn();
const unconfigureHarness = vi.fn();

vi.mock("./api", () => ({
  api: {
    services: () => services(),
    gateway: () => gateway(),
    harnesses: () => harnesses(),
    config: () => config(),
    installServices: (...a: unknown[]) => installServices(...a),
    removeServices: () => Promise.resolve([]),
    serviceAction: (...a: unknown[]) => serviceAction(...a),
    checkGateway: () => checkGateway(),
    configureHarness: (...a: unknown[]) => configureHarness(...a),
    unconfigureHarness: (...a: unknown[]) => unconfigureHarness(...a),
    saveSettings: () => Promise.resolve({}),
  },
}));

const meta: Meta = {
  version: "0.6.3",
  config_path: "/Users/benzotti/.config/ailocal/config.toml",
  harnesses: ["pi", "claude-code"],
  log_hint: "/Users/benzotti/Library/Logs/ailocal/io.github.zottiben.ailocal.gateway.log",
};

const stopped: Services = {
  linger: true,
  units: [
    { name: "io.github.zottiben.ailocal.gateway", active: false, enabled: false },
    { name: "io.github.zottiben.ailocal.model", active: false, enabled: false },
  ],
};

const answering: GatewayView = {
  url: "http://127.0.0.1:8081",
  port: 8081,
  state: "answering",
  trouble: null,
  key: "ail_a84bde1122334455",
};

const pi: HarnessView = {
  name: "pi",
  configured: false,
  path: "/Users/benzotti/.pi/agent/models.json",
  plan: {
    harness: "pi",
    url: "http://127.0.0.1:8081",
    model: "gemma4-12b",
    context: 131_072,
    why: "the configured default_model",
    others: ["qwen3-coder-30b"],
    without_thinking: ["qwen3-coder-30b"],
    trouble: null,
    already_configured: false,
  },
};

async function setup(over: { services?: Services; gateway?: GatewayView; harnesses?: HarnessView[] } = {}) {
  services.mockResolvedValue(over.services ?? stopped);
  gateway.mockResolvedValue(over.gateway ?? answering);
  harnesses.mockResolvedValue(over.harnesses ?? [pi]);
  config.mockResolvedValue({
    data_dir: "/Users/benzotti/.ailocal",
    cache_type: "q8_0",
    reasoning: "auto",
    reasoning_budget: -1,
    max_context: 131_072,
    default_model: "gemma4-12b",
    gateway_host: "127.0.0.1",
    gateway_port: 8081,
  });
  render(<Setup generation={0} meta={meta} />);
  await screen.findByText("Gateway");
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("Setup", () => {
  it("offers to install the services when none are running", async () => {
    installServices.mockResolvedValue({ units: [], unit_dir: "", exe: "", chose_default: null, gateway_only: false, answering: true, linger: true });
    await setup();

    fireEvent.click(screen.getByRole("button", { name: "Install and start" }));
    expect(installServices).toHaveBeenCalled();
  });

  it("starts the one unit whose button was pressed", async () => {
    serviceAction.mockResolvedValue(stopped);
    await setup();

    // Both units offer a Start, so the handler has to carry which row it came from.
    const starts = screen.getAllByRole("button", { name: "Start" });
    expect(starts).toHaveLength(2);
    fireEvent.click(starts[1] as HTMLElement);
    expect(serviceAction).toHaveBeenCalledWith("io.github.zottiben.ailocal.model", "start");
  });

  it("says a headless reboot leaves the gateway down when lingering is off", async () => {
    await setup({ services: { ...stopped, linger: false } });
    // Rule 5: enabling it needs root, so the command is handed over, not run.
    expect(screen.getByText(/sudo loginctl enable-linger/)).toBeTruthy();
  });

  it("shows the key to copy, because it is loopback only and already on this disk", async () => {
    await setup();
    expect(screen.getByText("ail_a84bde1122334455")).toBeTruthy();
  });

  it("reports which assertion of the gateway check failed", async () => {
    const check: GatewayCheck = {
      base: "http://127.0.0.1:8081",
      reachable: true,
      probes: [
        { label: "health, unauthenticated", expected: 200, got: 200 },
        // The one that matters: 200 here is a gateway open to anything on the port.
        { label: "models rejects no credential", expected: 401, got: 200 },
      ],
    };
    checkGateway.mockResolvedValue(check);
    await setup();

    fireEvent.click(screen.getByRole("button", { name: "Check it" }));
    expect(await screen.findByText("1 of 2 failed")).toBeTruthy();
    expect(screen.getByText("got 200, expected 401")).toBeTruthy();
  });

  it("names the model a harness would get before it is configured", async () => {
    await setup();
    expect(screen.getByText("gemma4-12b")).toBeTruthy();
    expect(screen.getByText(/the configured default_model/)).toBeTruthy();
    // Said here because the alternative is finding out from `/effort` refusing.
    expect(screen.getByText(/has no thinking mode/)).toBeTruthy();
  });

  it("sends the chosen model, and nothing when the choice was left alone", async () => {
    configureHarness.mockResolvedValue({ ...pi.plan, unchanged: false, path: null, backup: null });
    await setup();

    fireEvent.click(screen.getByRole("button", { name: "Configure" }));
    expect(configureHarness).toHaveBeenCalledWith("pi", undefined);

    // The button reads "Writing…" until the write lands, so wait for it rather than
    // clicking a second time into an action the page is already busy with.
    fireEvent.change(screen.getByLabelText("Model for pi"), { target: { value: "qwen3-coder-30b" } });
    fireEvent.click(await screen.findByRole("button", { name: "Configure" }));
    expect(configureHarness).toHaveBeenLastCalledWith("pi", "qwen3-coder-30b");
  });

  it("says why a harness cannot be pointed at anything rather than offering a button", async () => {
    await setup({
      harnesses: [{ name: "pi", configured: false, path: null, plan: { error: "no installed model can run here" } }],
    });
    expect(screen.getByText("no installed model can run here")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Configure" })).toBeNull();
  });
});

describe("Ticks", () => {
  it("sets a command as code instead of printing the backticks", () => {
    const { container } = render(<Ticks>{"Or run `ailocal gateway run` in a terminal."}</Ticks>);
    // A backtick is not markup in a browser, so it reached the window as a character.
    expect(container.textContent).toBe("Or run ailocal gateway run in a terminal.");
    expect(container.querySelector("code")?.textContent).toBe("ailocal gateway run");
  });

  it("leaves a stray backtick as typed rather than swallowing the rest", () => {
    const { container } = render(<Ticks>{"a ` b"}</Ticks>);
    expect(container.textContent).toBe("a ` b");
    expect(container.querySelector("code")).toBeNull();
  });
});

describe("the setup wording", () => {
  it("names the cause of a dead gateway, and the fix", () => {
    expect(describeTrouble(null)).toBeNull();
    expect(describeTrouble("not_running")?.what).toContain("not running");
    // The free port is carried so the window can offer it rather than say "pick one".
    expect(describeTrouble({ port_taken: { free: 8082 } })?.fix).toContain("8082");
    // The unit bakes the port into ExecStart, which a hand-edited config leaves stale.
    expect(describeTrouble({ wrong_port: { expected: 8081 } })?.fix).toContain("Reinstall");
  });

  it("tells a failed probe from a request that never finished", () => {
    expect(describeProbe({ label: "x", expected: 200, got: 200 })).toEqual({ ok: true, said: "200" });
    expect(describeProbe({ label: "x", expected: 401, got: 200 }).ok).toBe(false);
    const broke = describeProbe({ label: "x", expected: 200, got: { error: "connection refused" } });
    expect(broke).toEqual({ ok: false, said: "connection refused" });
  });

  it("hands over the install command for llama.cpp rather than offering a button", () => {
    // Rule 6: it needs root, so a button could only fail at the password prompt.
    const step = describeStep("install_llama_server", meta, "llama-server not found - not on PATH");
    expect(step?.commands).toContain("brew install llama.cpp");
    expect(step?.why).toContain("not on PATH");
  });

  it("says what a saved setting means for what is already running", () => {
    expect(
      describeApplied({ changed: [], path: null, gateway_restarted: null, reload_model: null, repoint_harnesses: false }),
    ).toEqual(["Nothing changed - those are the settings already."]);

    const moved = describeApplied({
      changed: ["gateway_port"],
      path: "/Users/benzotti/.config/ailocal/config.toml",
      gateway_restarted: { error: "launchctl: no such unit" },
      reload_model: null,
      repoint_harnesses: true,
    });
    // A restart that failed must not read as a save that worked.
    expect(moved[1]).toContain("could not be restarted");
    // Harnesses hold the URL in their own config and do not follow a port change.
    expect(moved[2]).toContain("Harnesses keep the old URL");
  });
});
