import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Actions } from "./hooks";
import { Jobs } from "./Jobs";
import type { Check, Job, JobState } from "./types";

const install = vi.fn();
const cancel = vi.fn();
const dismiss = vi.fn();
vi.mock("./api", () => ({
  api: {
    install: (...args: unknown[]) => install(...args),
    cancel: (...args: unknown[]) => cancel(...args),
    dismiss: (...args: unknown[]) => dismiss(...args),
  },
}));

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

const job = (state: JobState, extra: Partial<Job> = {}): Job => ({
  id: 1,
  kind: "install",
  target: "ollama:qwen3.5:4b",
  force: false,
  check: null,
  state,
  ...extra,
});

/** devstral-small-2:24b, whose vision encoder is packed into the weights. */
const packedEncoder: Check = {
  header: {
    read: { arch: "mistral3", trained_context: 131_072, encoder_tensors: 222, tensor_count: 585 },
  },
  fit: { fits: 131_072 },
  weights_mib: 14_336,
  available_mib: 39_312,
  blocked: true,
};

afterEach(() => {
  cleanup();
  for (const mock of [install, cancel, dismiss]) mock.mockClear();
});

describe("the job list", () => {
  it("stays off the page when there is nothing running", () => {
    render(<Jobs jobs={[]} actions={spy().actions} />);
    expect(screen.queryByLabelText("Jobs")).toBeNull();
  });

  it("measures a download against the whole file, and offers to stop it", () => {
    const { actions, asked } = spy();
    render(
      <Jobs
        jobs={[
          job({
            phase: "downloading",
            written: 142_786_443,
            total: 2_648_593_216,
            resumed_from: 0,
          }),
        ]}
        actions={actions}
      />,
    );

    const bar = screen.getByRole("progressbar") as HTMLProgressElement;
    expect(bar.value).toBe(142_786_443);
    expect(bar.max).toBe(2_648_593_216);
    expect(screen.getByText("136 MiB of 2.5 GiB")).toBeTruthy();

    fireEvent.click(screen.getByText("Stop"));
    expect(asked).toEqual(["job-1"]);
    expect(cancel).toHaveBeenCalledWith(1);
  });

  it("leaves the bar indeterminate when the registry gave no size", () => {
    render(
      <Jobs
        jobs={[job({ phase: "downloading", written: 1 << 20, total: null, resumed_from: 0 })]}
        actions={spy().actions}
      />,
    );
    // No value attribute at all: a percentage it cannot know would be a lie.
    expect(screen.getByRole("progressbar").hasAttribute("value")).toBe(false);
    expect(screen.getByText("1.0 MiB")).toBeTruthy();
  });

  it("gives a refusal its reason and the override beside it", () => {
    const { actions } = spy();
    render(
      <Jobs
        jobs={[job({ phase: "refused" }, { target: "ollama:devstral-small-2:24b", check: packedEncoder })]}
        actions={actions}
      />,
    );

    expect(screen.getByText(/222 of its 585 tensors/)).toBeTruthy();
    fireEvent.click(screen.getByText("Download anyway"));
    // A fresh job, forced, so the header is read again rather than taken from this one.
    expect(install).toHaveBeenCalledWith("ollama:devstral-small-2:24b", true);
  });

  it("offers a dismiss only once there is nothing left to wait for", () => {
    render(
      <Jobs
        jobs={[
          job({ phase: "verifying", total: 2_648_593_216 }),
          job({ phase: "cancelled", written: 142_786_443, total: 2_648_593_216 }, { id: 2 }),
        ]}
        actions={spy().actions}
      />,
    );

    const [running, finished] = screen.getAllByRole("listitem");
    expect(within(running!).queryByLabelText(/Dismiss/)).toBeNull();
    expect(within(running!).getByText("Stop")).toBeTruthy();

    expect(within(finished!).queryByText("Stop")).toBeNull();
    fireEvent.click(within(finished!).getByLabelText("Dismiss ollama:qwen3.5:4b"));
    expect(dismiss).toHaveBeenCalledWith(2);
  });

  it("shows a failure against the job it came from, never a job whose id it prefixes", () => {
    const { actions } = spy({ failed: { key: "job-1", message: "already finished" } });
    render(
      <Jobs
        jobs={[
          job({ phase: "failed", error: "registry said 404" }),
          job({ phase: "verifying", total: 2_648_593_216 }, { id: 10 }),
        ]}
        actions={actions}
      />,
    );

    const [first, tenth] = screen.getAllByRole("listitem");
    expect(within(first!).getByText("already finished")).toBeTruthy();
    expect(within(first!).getByText("registry said 404")).toBeTruthy();
    expect(within(tenth!).queryByText("already finished")).toBeNull();
  });
});
