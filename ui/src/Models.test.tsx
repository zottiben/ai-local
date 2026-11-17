import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { CatalogueList, InstalledTable, fitLabel } from "./Models";
import type { Candidate, Installed } from "./types";

// What /api/models answered on the M4 Max, plus one file that cannot run.
const gemma: Installed = {
  name: "gemma4-12b",
  path: "/Users/benzotti/.ailocal/models/gemma4-12b.gguf",
  size_mib: 7039,
  arch: "gemma4",
  sliding_window: true,
  kv_bytes_per_token: 10880,
  trained_context: 262_144,
  max_context: 131_072,
  thinking: "toggle",
};
const coder: Installed = {
  ...gemma,
  name: "qwen3-coder-30b",
  arch: "qwen3moe",
  sliding_window: false,
  size_mib: 17_800,
  thinking: "none",
};
const broken: Installed = {
  ...gemma,
  name: "half-download",
  arch: null,
  sliding_window: false,
  trained_context: null,
  max_context: null,
};

const candidate = (overrides: Partial<Candidate>): Candidate => ({
  entry: {
    reference: "qwen3.8:27b",
    summary: "latest Qwen, coding and long agentic tasks",
    note: "dense, so slower per token than the 35B mixture-of-experts",
    tier: 5,
  },
  size_mib: 16_031,
  headroom_mib: 23_281,
  context: 131_072,
  fits: true,
  ...overrides,
});

// Testing Library only cleans up by itself when vitest's globals are on, and they
// are not, so a table from one test would still be on screen in the next.
afterEach(() => {
  cleanup();
});

describe("InstalledTable", () => {
  it("marks the default and the loaded model, and says which cannot run", () => {
    render(
      <InstalledTable
        models={[gemma, coder, broken]}
        defaultModel="gemma4-12b"
        loadedModel="qwen3-coder-30b"
      />,
    );

    const [, gemmaRow, coderRow, brokenRow] = screen.getAllByRole("row");
    expect(within(gemmaRow!).getByText("default")).toBeTruthy();
    expect(within(gemmaRow!).queryByText("loaded")).toBeNull();
    expect(within(gemmaRow!).getByText("128k")).toBeTruthy();
    expect(within(gemmaRow!).getByText("on/off")).toBeTruthy();

    expect(within(coderRow!).getByText("loaded")).toBeTruthy();
    expect(within(coderRow!).getByText("no thinking mode")).toBeTruthy();

    expect(within(brokenRow!).getByText("cannot run here")).toBeTruthy();
    expect(within(brokenRow!).getByText("not a readable GGUF")).toBeTruthy();

    // gemma4 is sliding-window, so the footnote explaining (swa) is there.
    expect(screen.getByText(/sliding-window attention: most layers/)).toBeTruthy();
  });

  it("offers the picker when nothing is installed", () => {
    render(<InstalledTable models={[]} defaultModel={null} loadedModel={null} />);
    expect(screen.getByText("ailocal model pick")).toBeTruthy();
  });
});

describe("the catalogue", () => {
  it("says what the picker says", () => {
    expect(fitLabel(candidate({}))).toBe("128k context");
    // Not measured yet: the spare room is all there is to show.
    expect(fitLabel(candidate({ context: null }))).toBe("22.7 GiB spare");
    // The library drew the line; 1023 MiB of headroom is under it.
    expect(fitLabel(candidate({ headroom_mib: 1023, fits: false }))).toBe("will not fit");
  });

  it("dims what will not fit rather than hiding it", () => {
    render(
      <CatalogueList
        offer={{
          available_mib: 39_312,
          downloaded: [],
          unusable: 0,
          remote: [
            candidate({}),
            candidate({
              entry: { reference: "huge:405b", summary: "too big", note: "", tier: 5 },
              fits: false,
              headroom_mib: -200_000,
            }),
          ],
        }}
      />,
    );

    const [fits, unfit] = screen.getAllByRole("listitem");
    expect(fits!.className).toBe("");
    expect(within(fits!).getByText(/dense, so slower/)).toBeTruthy();
    expect(unfit!.className).toBe("unfit");
    expect(within(unfit!).getByText("will not fit")).toBeTruthy();
  });
});
