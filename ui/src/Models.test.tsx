import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Actions } from "./hooks";
import { CatalogueList, InstalledTable, SearchHuggingFace, fitLabel, quantFit } from "./Models";
import type { Candidate, Choice, Installed } from "./types";

const install = vi.fn();
const serve = vi.fn();
const setDefault = vi.fn();
const search = vi.fn();
const files = vi.fn();
vi.mock("./api", () => ({
  api: {
    install: (...args: unknown[]) => install(...args),
    serve: (...args: unknown[]) => serve(...args),
    setDefault: (...args: unknown[]) => setDefault(...args),
    search: (...args: unknown[]) => search(...args),
    files: (...args: unknown[]) => files(...args),
  },
}));

/**
 * Buttons with the request behind them, but without a server: what matters here is
 * which action a click asks for, with what, and that a disabled one asks for nothing.
 */
function spy(overrides: Partial<Actions> = {}) {
  const asked: string[] = [];
  const actions: Actions = {
    run: (key, action) => {
      asked.push(key);
      void action();
    },
    busy: null,
    failed: null,
    ...overrides,
  };
  return { actions, asked };
}

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
  for (const mock of [install, serve, setDefault]) mock.mockClear();
  // Reset rather than clear: these are given per-test answers, and a queued `*Once`
  // that nothing consumed would otherwise be handed to the next test.
  for (const mock of [search, files]) mock.mockReset();
});

describe("InstalledTable", () => {
  it("marks the default and the loaded model, and says which cannot run", () => {
    render(
      <InstalledTable
        models={[gemma, coder, broken]}
        defaultModel="gemma4-12b"
        loadedModel="qwen3-coder-30b"
        actions={spy().actions}
        loading={false}
        onRemove={() => {}}
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

  it("points at the catalogue when nothing is installed", () => {
    render(
      <InstalledTable
        models={[]}
        defaultModel={null}
        loadedModel={null}
        actions={spy().actions}
        loading={false}
        onRemove={() => {}}
      />,
    );
    expect(screen.getByText(/The catalogue below offers/)).toBeTruthy();
  });

  it("offers only the actions that mean something for each row", () => {
    const { actions, asked } = spy();
    const removing: string[] = [];
    render(
      <InstalledTable
        models={[gemma, coder, broken]}
        defaultModel="gemma4-12b"
        loadedModel="qwen3-coder-30b"
        actions={actions}
        loading={false}
        onRemove={(name) => removing.push(name)}
      />,
    );
    const [, gemmaRow, coderRow, brokenRow] = screen.getAllByRole("row");

    // The default needs no "Make default"; the loaded one needs no "Load".
    expect(within(gemmaRow!).queryByText("Make default")).toBeNull();
    expect(within(coderRow!).queryByText("Load")).toBeNull();

    fireEvent.click(within(gemmaRow!).getByText("Load"));
    fireEvent.click(within(coderRow!).getByText("Make default"));
    expect(asked).toEqual(["serve-gemma4-12b", "default-qwen3-coder-30b"]);
    expect(serve).toHaveBeenCalledWith("gemma4-12b");
    expect(setDefault).toHaveBeenCalledWith("qwen3-coder-30b");

    // The library already said this one cannot hold a usable context here.
    const load = within(brokenRow!).getByText("Load") as HTMLButtonElement;
    expect(load.disabled).toBe(true);
    fireEvent.click(load);
    expect(asked).toHaveLength(2);
    expect(serve).toHaveBeenCalledTimes(1);

    fireEvent.click(within(gemmaRow!).getByText("Delete"));
    expect(removing).toEqual(["gemma4-12b"]);
  });

  it("holds Load shut while another model is coming up, as the server would refuse it", () => {
    const { actions, asked } = spy();
    render(
      <InstalledTable
        models={[gemma]}
        defaultModel={null}
        loadedModel={null}
        actions={actions}
        loading={true}
        onRemove={() => {}}
      />,
    );

    const load = screen.getByText("Load") as HTMLButtonElement;
    expect(load.disabled).toBe(true);
    expect(load.title).toBe("Another model is loading");
    fireEvent.click(load);
    expect(asked).toEqual([]);

    // A load in flight is no reason to stop a model being deleted or made default.
    expect((screen.getByText("Make default") as HTMLButtonElement).disabled).toBe(false);
    expect((screen.getByText("Delete") as HTMLButtonElement).disabled).toBe(false);
  });

  it("shows a failed row action under the table, against the row that failed", () => {
    const { actions } = spy({
      failed: { key: "serve-gemma4-12b", message: "llama-server did not come up" },
    });
    render(
      <InstalledTable
        models={[gemma]}
        defaultModel={null}
        loadedModel={null}
        actions={actions}
        loading={false}
        onRemove={() => {}}
      />,
    );
    expect(screen.getByText("llama-server did not come up")).toBeTruthy();

    // A failure from somewhere else on the page is not this table's to report.
    cleanup();
    const other = spy({ failed: { key: "install-form", message: "elsewhere" } });
    render(
      <InstalledTable
        models={[gemma]}
        defaultModel={null}
        loadedModel={null}
        actions={other.actions}
        loading={false}
        onRemove={() => {}}
      />,
    );
    expect(screen.queryByText("elsewhere")).toBeNull();
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
        actions={spy().actions}
        busy={false}
      />,
    );

    const [fits, unfit] = screen.getAllByRole("listitem");
    expect(fits!.className).toBe("");
    expect(within(fits!).getByText(/dense, so slower/)).toBeTruthy();
    expect(unfit!.className).toBe("unfit");
    expect(within(unfit!).getByText("will not fit")).toBeTruthy();
  });

  it("holds Install shut while one is running, as the server takes one at a time", () => {
    render(
      <CatalogueList
        offer={{ available_mib: 39_312, downloaded: [], unusable: 0, remote: [candidate({})] }}
        actions={spy().actions}
        busy={true}
      />,
    );

    const button = screen.getByLabelText("Install qwen3.8:27b") as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    expect(button.title).toBe("Another install is running");
    fireEvent.click(button);
    expect(install).not.toHaveBeenCalled();
  });

  it("installs a catalogue entry, forcing only the one that will not fit", () => {
    const huge = candidate({
      entry: { reference: "huge:405b", summary: "too big", note: "", tier: 5 },
      fits: false,
    });
    render(
      <CatalogueList
        offer={{ available_mib: 39_312, downloaded: [], unusable: 0, remote: [candidate({}), huge] }}
        actions={spy().actions}
        busy={false}
      />,
    );

    fireEvent.click(screen.getByLabelText("Install qwen3.8:27b"));
    // The reference the registry takes, not the catalogue's bare name.
    expect(install).toHaveBeenLastCalledWith("ollama:qwen3.8:27b", false);

    // One that will not fit is still downloadable, as it is from the picker - forced,
    // because the check would otherwise refuse it.
    fireEvent.click(screen.getByLabelText("Install huge:405b"));
    expect(install).toHaveBeenLastCalledWith("ollama:huge:405b", true);
  });
});

describe("quantFit", () => {
  function choice(over: Partial<Choice> = {}): Choice {
    return { path: "q4.gguf", size_mib: 8000, headroom_mib: 2000, fits: true, ...over };
  }

  it("says how much is spare, how much is over, and the gap between them", () => {
    expect(quantFit(choice())).toBe("2.0 GiB spare");
    expect(quantFit(choice({ headroom_mib: -3000, fits: false }))).toBe("2.9 GiB too large");
    // Smaller than the budget and still unusable: the weights fit and the KV cache
    // has nowhere to go. A bare "too large" would be a lie about a 500 MiB gap.
    expect(quantFit(choice({ headroom_mib: 500, fits: false }))).toBe("no room for a context");
  });
});

describe("SearchHuggingFace", () => {
  const repos = [
    { id: "unsloth/gemma-3-12b-it-GGUF", downloads: 86_188 },
    { id: "bartowski/gemma-3-12b-GGUF", downloads: 12_000 },
  ];
  const quants: Choice[] = [
    { path: "gemma-3-12b-it-BF16.gguf", size_mib: 22_449, headroom_mib: 16_863, fits: true },
    { path: "UD-Q2_K_XL/gemma-3-12b.gguf", size_mib: 4000, headroom_mib: 35_312, fits: true },
  ];

  it("searches on submit and lists what came back", async () => {
    search.mockResolvedValue(repos);
    render(<SearchHuggingFace actions={spy().actions} busy={false} />);

    // Nothing is asked for until there is something to ask.
    const button = screen.getByRole("button", { name: "Search" });
    expect((button as HTMLButtonElement).disabled).toBe(true);

    fireEvent.change(screen.getByLabelText("Search Hugging Face"), {
      target: { value: "  gemma 12b  " },
    });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    // Trimmed, because a trailing space is a typo rather than part of the query.
    expect(search).toHaveBeenCalledWith("gemma 12b");
    expect(await screen.findByText("unsloth/gemma-3-12b-it-GGUF")).toBeTruthy();
    expect(screen.getByText("86.2K downloads")).toBeTruthy();
  });

  it("lists a repository's quants on demand and installs the one that was clicked", async () => {
    search.mockResolvedValue(repos);
    files.mockResolvedValue(quants);
    const { actions } = spy();
    render(<SearchHuggingFace actions={actions} busy={false} />);

    fireEvent.change(screen.getByLabelText("Search Hugging Face"), {
      target: { value: "gemma" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("unsloth/gemma-3-12b-it-GGUF");

    // The quants are not fetched until one repository is actually opened: a search
    // answers with twenty, and listing all of them would be twenty slow requests.
    expect(files).not.toHaveBeenCalled();
    fireEvent.click(screen.getByLabelText("Quants in unsloth/gemma-3-12b-it-GGUF"));
    expect(files).toHaveBeenCalledWith("unsloth/gemma-3-12b-it-GGUF");

    await screen.findByText("gemma-3-12b-it-BF16.gguf");
    fireEvent.click(screen.getByLabelText("Install gemma-3-12b-it-BF16.gguf"));
    expect(install).toHaveBeenLastCalledWith(
      "hf:unsloth/gemma-3-12b-it-GGUF/gemma-3-12b-it-BF16.gguf",
      false,
    );

    // A quant in a subdirectory keeps its directory in the reference. The repository
    // is the first two segments, so this is still the same repository.
    fireEvent.click(screen.getByLabelText("Install UD-Q2_K_XL/gemma-3-12b.gguf"));
    expect(install).toHaveBeenLastCalledWith(
      "hf:unsloth/gemma-3-12b-it-GGUF/UD-Q2_K_XL/gemma-3-12b.gguf",
      false,
    );
  });

  it("shows the server's own words when a repository cannot be read", async () => {
    search.mockResolvedValue(repos);
    files.mockRejectedValue(new Error("no repository nope/nope you can read - check the name"));
    render(<SearchHuggingFace actions={spy().actions} busy={false} />);

    fireEvent.change(screen.getByLabelText("Search Hugging Face"), {
      target: { value: "gemma" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("unsloth/gemma-3-12b-it-GGUF");
    fireEvent.click(screen.getByLabelText("Quants in unsloth/gemma-3-12b-it-GGUF"));

    expect(
      await screen.findByText("no repository nope/nope you can read - check the name"),
    ).toBeTruthy();
  });

  it("drops a quant list that a newer search has already replaced", async () => {
    // The race that can actually happen. A second search cannot start while the first
    // is in flight, because the button is disabled and says so - but nothing stops a
    // search while a quant list is still loading, and Hugging Face is slow and
    // variable enough from here that the slow listing then lands into a page showing
    // somebody else's results.
    search.mockResolvedValueOnce(repos).mockResolvedValueOnce([
      { id: "second/answer-GGUF", downloads: 1 },
    ]);
    let finishListing: (value: unknown) => void = () => {};
    files.mockImplementationOnce(() => new Promise((resolve) => (finishListing = resolve)));

    render(<SearchHuggingFace actions={spy().actions} busy={false} />);
    const box = screen.getByLabelText("Search Hugging Face");

    fireEvent.change(box, { target: { value: "first" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("unsloth/gemma-3-12b-it-GGUF");
    fireEvent.click(screen.getByLabelText("Quants in unsloth/gemma-3-12b-it-GGUF"));

    fireEvent.change(box, { target: { value: "second" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("second/answer-GGUF");

    finishListing(quants);
    await waitFor(() =>
      expect(screen.queryByText("gemma-3-12b-it-BF16.gguf")).toBeNull(),
    );
  });

  it("says it is searching, and will not start a second one meanwhile", async () => {
    let finish: (value: unknown) => void = () => {};
    search.mockImplementationOnce(() => new Promise((resolve) => (finish = resolve)));
    render(<SearchHuggingFace actions={spy().actions} busy={false} />);

    fireEvent.change(screen.getByLabelText("Search Hugging Face"), {
      target: { value: "gemma" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    const button = await screen.findByRole("button", { name: "Searching" });
    expect((button as HTMLButtonElement).disabled).toBe(true);

    finish(repos);
    await screen.findByText("unsloth/gemma-3-12b-it-GGUF");
    expect((screen.getByRole("button", { name: "Search" }) as HTMLButtonElement).disabled).toBe(
      false,
    );
  });

  it("says why a shard-only repository looks empty", async () => {
    search.mockResolvedValue([repos[0]]);
    files.mockResolvedValue([]);
    render(<SearchHuggingFace actions={spy().actions} busy={false} />);

    fireEvent.change(screen.getByLabelText("Search Hugging Face"), {
      target: { value: "deepseek" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByText("unsloth/gemma-3-12b-it-GGUF");
    fireEvent.click(screen.getByLabelText("Quants in unsloth/gemma-3-12b-it-GGUF"));

    expect(await screen.findByText(/No single-file GGUF here/)).toBeTruthy();
  });
});
