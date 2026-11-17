import { describe, expect, it } from "vitest";

import {
  describeCheck,
  describeJob,
  describeRemoved,
  describeStep,
  formatBytes,
  formatCount,
  formatSize,
  serviceShortName,
  tilde,
} from "./format";
import type { Check, Job, JobState } from "./types";

const meta = {
  log_hint: "journalctl --user -u ailocal-gateway.service -n 50",
  harnesses: ["pi", "claude-code"],
};

describe("format", () => {
  it("prints a context the way model ls does", () => {
    // 131072 is the cap every model here launches at.
    expect(formatCount(131_072)).toBe("128k");
    expect(formatCount(262_144)).toBe("256k");
    // A size that is not a whole number of k is printed whole, not rounded.
    expect(formatCount(4000)).toBe("4000");
  });

  it("never rounds a small file down to 0 GiB", () => {
    // gemma4-12b is 7039 MiB on disk.
    expect(formatSize(7039)).toBe("6.9 GiB");
    expect(formatSize(400)).toBe("400 MiB");
  });

  it("names the services the same on launchd and systemd", () => {
    expect(serviceShortName("io.github.zottiben.ailocal.gateway")).toBe("gateway");
    expect(serviceShortName("io.github.zottiben.ailocal.model")).toBe("model");
    expect(serviceShortName("ailocal-gateway.service")).toBe("gateway");
    expect(serviceShortName("ailocal-model.service")).toBe("model");
  });

  it("gives every step the command ailocal status prints", () => {
    expect(describeStep("install_model", meta)?.commands).toEqual(["ailocal model pick"]);
    expect(describeStep("start_gateway", meta)?.commands).toEqual(["ailocal service install"]);
    expect(describeStep("configure_harness", meta)?.commands).toEqual([
      "ailocal harness configure pi",
      "ailocal harness configure claude-code",
    ]);
    expect(describeStep("ready", meta)).toBeNull();
  });

  it("points a broken gateway at its log, which is a command on Linux", () => {
    const step = describeStep("fix_gateway", meta);
    expect(step?.commands).toEqual(["ailocal gateway check"]);
    expect(step?.log).toBe(meta.log_hint);
  });

  it("keeps a decimal until a download is worth whole megabytes", () => {
    // qwen3.5:4b's blob, and the point a cancel left it at.
    expect(formatBytes(2_648_593_216)).toBe("2.5 GiB");
    expect(formatBytes(142_786_443)).toBe("136 MiB");
    // The first second of a download, which "0 MiB" would make look broken.
    expect(formatBytes(1 << 20)).toBe("1.0 MiB");
  });

  it("shortens a home directory and nothing else", () => {
    expect(tilde("/Users/benzotti/.ailocal")).toBe("~/.ailocal");
    expect(tilde("/home/ben/.config/ailocal/config.toml")).toBe("~/.config/ailocal/config.toml");
    expect(tilde("/mnt/kingston/ailocal")).toBe("/mnt/kingston/ailocal");
  });
});

const job = (state: JobState, check: Check | null = null): Job => ({
  id: 1,
  kind: "install",
  target: "ollama:qwen3.5:4b",
  force: false,
  check,
  state,
});

describe("a job's phase", () => {
  it("says what a download is doing, and what a stopped one left behind", () => {
    expect(
      describeJob(job({ phase: "downloading", written: 1 << 20, total: null, resumed_from: 0 })),
    ).toBe("Downloading");
    // Measured: the cancelled job left this much, and the next install carried on.
    expect(
      describeJob(
        job({
          phase: "downloading",
          written: 200 << 20,
          total: 2_648_593_216,
          resumed_from: 142_786_443,
        }),
      ),
    ).toBe("Resumed from 136 MiB");
    expect(
      describeJob(job({ phase: "cancelled", written: 142_786_443, total: 2_648_593_216 })),
    ).toBe("Stopped at 136 MiB of 2.5 GiB - installing it again carries on from there");
  });

  it("names what landed, and tells an install from a file already there", () => {
    expect(
      describeJob(
        job({
          phase: "installed",
          name: "qwen3.5-4b",
          path: "/m/qwen3.5-4b.gguf",
          outcome: { downloaded: { bytes: 2_648_593_216, resumed_from: 142_786_443 } },
        }),
      ),
    ).toBe("Installed as qwen3.5-4b (2.5 GiB)");
    expect(
      describeJob(
        job({
          phase: "installed",
          name: "qwen3.5-4b",
          path: "/m/qwen3.5-4b.gguf",
          outcome: "already_present",
        }),
      ),
    ).toBe("Already on disk as qwen3.5-4b");
  });

  it("says where a model came up, and what loading it will evict", () => {
    expect(
      describeJob(job({ phase: "starting", context: 40_960, evicting: "gemma4-12b" })),
    ).toBe("Loading at 40k context, evicting gemma4-12b");
    expect(describeJob(job({ phase: "starting", context: 40_960, evicting: null }))).toBe(
      "Loading at 40k context",
    );
    expect(
      describeJob(
        job({
          phase: "served",
          pid: 77_023,
          model: "qwen3-0.6b",
          path: "/m/qwen3-0.6b.gguf",
          host: "127.0.0.1",
          port: 8080,
          context: 40_960,
          cache_type: "q8_0",
          desktop_mib: 1,
          cache_ram_mib: 2975,
          reasoning: "auto",
          log: "/tmp/ailocal/server.log",
        }),
      ),
    ).toBe("Up at 127.0.0.1:8080 with 40k context");
  });
});

describe("a check", () => {
  it("gives the encoder refusal the command's own reason", () => {
    // devstral-small-2:24b: 222 of its 585 tensors are a packed vision encoder.
    const verdict = describeCheck({
      header: {
        read: {
          arch: "mistral3",
          trained_context: 131_072,
          encoder_tensors: 222,
          tensor_count: 585,
        },
      },
      fit: { fits: 131_072 },
      weights_mib: 14_336,
      available_mib: 39_312,
      blocked: true,
    });
    expect(verdict.blocked).toBe(true);
    expect(verdict.summary).toBe("arch mistral3, trained to 128k");
    expect(verdict.detail).toContain("222 of its 585 tensors");
    expect(verdict.detail).toContain("mmproj");
  });

  it("says nothing is wrong with one that fits", () => {
    // qwen3.5:4b, as the server measured it.
    const verdict = describeCheck({
      header: {
        read: { arch: "qwen35", trained_context: 262_144, encoder_tensors: 0, tensor_count: 426 },
      },
      fit: { fits: 131_072 },
      weights_mib: 2525,
      available_mib: 39_312,
      blocked: false,
    });
    expect(verdict).toEqual({ blocked: false, summary: "arch qwen35, trained to 256k" });
  });

  it("reports a head it could not read, and weights that are too large", () => {
    const unreadable = describeCheck({
      header: { unreadable: "reading x: connection reset" },
      fit: "unknown",
      weights_mib: 1801,
      available_mib: 39_312,
      blocked: true,
    });
    expect(unreadable.detail).toContain("reading x: connection reset");

    const huge = describeCheck({
      header: {
        read: { arch: "qwen3moe", trained_context: 262_144, encoder_tensors: 0, tensor_count: 579 },
      },
      fit: "weights_too_large",
      weights_mib: 45_000,
      available_mib: 39_312,
      blocked: true,
    });
    expect(huge.detail).toBe("Will not fit: 45000 MiB of weights against 39312 MiB available.");
  });
});

describe("what a delete also did", () => {
  it("lists it in the words model rm prints", () => {
    expect(
      describeRemoved(
        { stopped: true, default: "restarted", pi: true, claude_code_pinned: true },
        "gemma4-12b",
      ),
    ).toEqual([
      "stopped gemma4-12b",
      "cleared default_model",
      "restarted the model service on it",
      "removed pi's entry for it",
      "claude-code is still pointed at gemma4-12b - run `ailocal harness configure claude-code` to move it",
    ]);
  });

  it("says nothing about what it did not touch", () => {
    expect(
      describeRemoved(
        { stopped: false, default: null, pi: false, claude_code_pinned: false },
        "gemma4-12b",
      ),
    ).toEqual([]);
  });

  it("reports a step that failed after the weights were already gone", () => {
    expect(
      describeRemoved(
        {
          stopped: false,
          default: { error: "writing config.toml: permission denied" },
          pi: { error: "reading models.json: not valid JSON" },
          claude_code_pinned: false,
        },
        "gemma4-12b",
      ),
    ).toEqual([
      "could not clear default_model: writing config.toml: permission denied",
      "could not update pi's models.json: reading models.json: not valid JSON",
    ]);
  });

  it("says when the model service was left pointing at what is gone", () => {
    const lines = describeRemoved(
      {
        stopped: false,
        default: { failed: "launchctl exited 5" },
        pi: false,
        claude_code_pinned: false,
      },
      "gemma4-12b",
    );
    expect(lines[0]).toBe("cleared default_model");
    expect(lines[1]).toContain("launchctl exited 5");
    expect(lines[1]).toContain("ailocal service install");
  });
});
