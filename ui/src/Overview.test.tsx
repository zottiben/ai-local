import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { MachinePanel, NextStepPanel } from "./Overview";
import type { Meta, Overview } from "./types";

const meta: Meta = {
  version: "0.6.3",
  config_path: "/Users/benzotti/.config/ailocal/config.toml",
  harnesses: ["pi", "claude-code"],
  log_hint: "/Users/benzotti/Library/Logs/ailocal/io.github.zottiben.ailocal.gateway.log",
};

// What /api/overview answered on the M4 Max with no services installed.
const overview: Overview = {
  models: [
    { id: "gemma4-12b", context_window: 131_072, thinking: "toggle" },
    { id: "qwen3-coder-30b", context_window: 131_072, thinking: "none" },
  ],
  default_model: "gemma4-12b",
  loaded: null,
  gateway_url: "http://127.0.0.1:8081",
  gateway: "not_answering",
  advertised: null,
  services: [
    ["io.github.zottiben.ailocal.gateway", "not_installed"],
    ["io.github.zottiben.ailocal.model", "not_installed"],
  ],
  harnesses: ["pi"],
  next: "start_gateway",
};

// Testing Library only cleans up by itself when vitest's globals are on, and they
// are not, so a table from one test would still be on screen in the next.
afterEach(() => {
  cleanup();
});

describe("Overview", () => {
  it("reports each line ailocal status prints", () => {
    render(<MachinePanel overview={overview} />);
    expect(screen.getByText("2 runnable, default gemma4-12b")).toBeTruthy();
    expect(screen.getByText("nothing resident (the gateway loads on demand)")).toBeTruthy();
    expect(screen.getByText("Service: gateway")).toBeTruthy();
    expect(screen.getByText("Service: model")).toBeTruthy();
    // The gateway was not asked, so there is no advertised line at all.
    expect(screen.queryByText("Advertised")).toBeNull();
  });

  it("shows why the gateway could not be asked, causes and all", () => {
    render(
      <MachinePanel
        overview={{
          ...overview,
          gateway: "answering",
          advertised: { error: "asking http://127.0.0.1:8081/v1/models: connection refused" },
        }}
      />,
    );
    expect(
      screen.getByText(
        "could not ask the gateway: asking http://127.0.0.1:8081/v1/models: connection refused",
      ),
    ).toBeTruthy();
  });

  it("leads with the next step, and says so when there is none", () => {
    render(<NextStepPanel overview={overview} meta={meta} />);
    expect(screen.getByText("Start the gateway")).toBeTruthy();
    expect(screen.getByText("ailocal service install")).toBeTruthy();
    cleanup();

    render(<NextStepPanel overview={{ ...overview, next: "ready" }} meta={meta} />);
    expect(screen.getByText("Everything is in place")).toBeTruthy();
  });
});
