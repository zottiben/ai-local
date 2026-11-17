import { describe, expect, it } from "vitest";

import { describeStep, formatCount, formatSize, serviceShortName, tilde } from "./format";

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

  it("shortens a home directory and nothing else", () => {
    expect(tilde("/Users/benzotti/.ailocal")).toBe("~/.ailocal");
    expect(tilde("/home/ben/.config/ailocal/config.toml")).toBe("~/.config/ailocal/config.toml");
    expect(tilde("/mnt/kingston/ailocal")).toBe("/mnt/kingston/ailocal");
  });
});
