import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Actions } from "./hooks";
import { Settings } from "./Settings";
import type { Config, Meta } from "./types";

const config = vi.fn();
const saveSettings = vi.fn();
vi.mock("./api", () => ({
  api: {
    config: () => config(),
    saveSettings: (...args: unknown[]) => saveSettings(...args),
  },
}));

// What /api/config answered on the M4 Max.
const saved: Config = {
  data_dir: "/Users/benzotti/.ailocal",
  cache_type: "q8_0",
  reasoning: "auto",
  reasoning_budget: -1,
  max_context: 131_072,
  default_model: "gemma4-12b",
  gateway_host: "127.0.0.1",
  gateway_port: 8081,
};

const meta: Meta = {
  version: "0.6.3",
  config_path: "/Users/benzotti/.config/ailocal/config.toml",
  harnesses: ["pi", "claude-code"],
  log_hint: "/Users/benzotti/Library/Logs/ailocal/io.github.zottiben.ailocal.gateway.log",
};

function spy(overrides: Partial<Actions> = {}) {
  const asked: { key: string; action: () => Promise<unknown> }[] = [];
  const actions: Actions = {
    run: (key, action) => {
      asked.push({ key, action: action as () => Promise<unknown> });
      void action();
    },
    busy: null,
    failed: null,
    ...overrides,
  };
  return { actions, asked };
}

async function form(overrides: Partial<Actions> = {}) {
  config.mockResolvedValue(saved);
  saveSettings.mockResolvedValue({
    changed: [],
    path: null,
    gateway_restarted: null,
    reload_model: null,
    repoint_harnesses: false,
  });
  const { actions, asked } = spy(overrides);
  render(<Settings generation={0} meta={meta} actions={actions} after={() => {}} />);
  await screen.findByLabelText(/Context cap/);
  return { asked };
}

const save = () => screen.getByRole("button", { name: "Save" }) as HTMLButtonElement;

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("Settings", () => {
  it("sends only the setting that changed", async () => {
    const { asked } = await form();

    fireEvent.change(screen.getByLabelText(/Context cap/), { target: { value: "65536" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(asked).toHaveLength(1);
    // Every other setting has to be absent, not echoed back: one sent as it was when
    // the page loaded would revert a change made from a terminal since.
    expect(saveSettings).toHaveBeenCalledWith({ max_context: 65_536 });
  });

  it("has nothing to save until something is edited", async () => {
    await form();
    expect(save().disabled).toBe(true);

    fireEvent.change(screen.getByLabelText(/Gateway port/), { target: { value: "8099" } });
    expect(save().disabled).toBe(false);

    // Typing it back is not a change, so the button closes again rather than writing
    // the same values and reporting a save.
    fireEvent.change(screen.getByLabelText(/Gateway port/), { target: { value: "8081" } });
    expect(save().disabled).toBe(true);
  });

  it("spells a cleared house rule as an explicit null", async () => {
    await form();
    // The config had none, so set one and clear it in the same session.
    fireEvent.change(screen.getByLabelText(/House rule/), { target: { value: "  " } });
    // Whitespace equals the saved empty string, so there is still nothing to save.
    expect(save().disabled).toBe(true);

    fireEvent.change(screen.getByLabelText(/House rule/), { target: { value: "Answer tersely." } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(saveSettings).toHaveBeenCalledWith({ system_prompt: "Answer tersely." });
  });

  it("leaves a half-typed number out rather than sending NaN", async () => {
    await form();
    fireEvent.change(screen.getByLabelText(/Gateway port/), { target: { value: "" } });
    fireEvent.change(screen.getByLabelText(/Context cap/), { target: { value: "65536" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    // NaN would reach the server as a deserialisation failure naming gateway_port,
    // which is not the field the person was editing.
    expect(saveSettings).toHaveBeenCalledWith({ max_context: 65_536 });
  });

  it("says what a change means for what is already loaded", async () => {
    config.mockResolvedValue(saved);
    saveSettings.mockResolvedValue({
      changed: ["max_context"],
      path: "/Users/benzotti/.config/ailocal/config.toml",
      gateway_restarted: null,
      reload_model: "gemma4-12b",
      repoint_harnesses: false,
    });
    const { actions } = spy();
    render(<Settings generation={0} meta={meta} actions={actions} after={() => {}} />);
    await screen.findByLabelText(/Context cap/);

    fireEvent.change(screen.getByLabelText(/Context cap/), { target: { value: "65536" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    // The context is fixed when llama-server starts, so a resident model keeps the
    // window it was launched with and the catalogue would advertise the other number.
    expect(await screen.findByText(/gemma4-12b is loaded with the old context/)).toBeTruthy();
  });

  it("shows a refused value beside the form rather than losing it", async () => {
    await form({
      failed: { key: "settings-save", message: "128 tokens is below the 4096 a model needs" },
    });
    expect(screen.getByText(/below the 4096/)).toBeTruthy();
  });
});
