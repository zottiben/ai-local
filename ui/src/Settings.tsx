import { type ReactNode, useEffect, useState } from "react";

import { api } from "./api";
import { describeApplied, tilde } from "./format";
import { type Actions, useResource } from "./hooks";
import { Ticks } from "./Ticks";
import type { Applied, Config, Meta, SettingsPatch } from "./types";

/** What `ailocal config show` lists, as inputs. The data directory is not among them. */
interface Props {
  generation: number;
  meta: Meta;
  actions: Actions;
  after: () => void;
}

const KEY = "settings-save";

/**
 * The settings, editable.
 *
 * Only the fields that differ are sent (`setup::Settings` is a patch), so a setting
 * changed in a terminal while this form was open survives instead of being reverted to
 * whatever the page loaded.
 *
 * The data directory is deliberately absent: moving it means moving tens of gigabytes
 * of weights, which is not something a text field should start.
 */
export function Settings({ generation, meta, actions, after }: Props) {
  const config = useResource(() => api.config(), [generation]);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [applied, setApplied] = useState<Applied | null>(null);

  // Adopt each fresh read, except while there are unsaved edits - re-reading over
  // someone's half-typed port number is worse than showing a stale one.
  useEffect(() => {
    if (config.data && draft === null) setDraft(toDraft(config.data));
  }, [config.data, draft]);

  if (config.error) {
    return (
      <section className="panel">
        <h2>Settings</h2>
        <p className="bad">{config.error.message}</p>
      </section>
    );
  }
  if (!config.data || !draft) {
    return (
      <section className="panel">
        <h2>Settings</h2>
        <p className="muted">Reading the config…</p>
      </section>
    );
  }

  const saved = toDraft(config.data);
  const patch = diff(saved, draft);
  const dirty = Object.keys(patch).length > 0;
  const edit = (change: Partial<Draft>) => setDraft({ ...draft, ...change });

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Settings</h2>
        {/* Where Save writes, which is the question a form raises. The data
            directory is on the overview, beside the other things that do not change. */}
        <code className="small">{tilde(meta.config_path)}</code>
      </div>

      <form
        className="settings-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (!dirty) return;
          actions.run(KEY, () =>
            api.saveSettings(patch).then((result) => {
              setApplied(result);
              setDraft(null);
              after();
            }),
          );
        }}
      >
        <Field
          label="KV cache"
          hint="q8_0 roughly halves the cache at no measurable cost in throughput."
        >
          <select value={draft.cache_type} onChange={(e) => edit({ cache_type: e.target.value })}>
            <option value="f16">f16</option>
            <option value="q8_0">q8_0</option>
            <option value="q4_0">q4_0</option>
          </select>
        </Field>

        <Field
          label="Reasoning"
          hint="A default, not a lock: a harness can still ask per request. `auto` follows what the model's own template offers."
        >
          <select value={draft.reasoning} onChange={(e) => edit({ reasoning: e.target.value })}>
            <option value="auto">auto</option>
            <option value="on">on</option>
            <option value="off">off</option>
          </select>
        </Field>

        <Field
          label="Reasoning budget"
          hint="A cap on the thinking, in tokens. -1 leaves it uncapped; 0 ends thinking at once."
        >
          <input
            type="number"
            value={draft.reasoning_budget}
            min={-1}
            onChange={(e) => edit({ reasoning_budget: e.target.value })}
          />
        </Field>

        <Field
          label="Context cap"
          hint="The most a launch will ask for. A model is still held to what fits on this machine."
        >
          <input
            type="number"
            value={draft.max_context}
            min={4096}
            step={4096}
            onChange={(e) => edit({ max_context: e.target.value })}
          />
        </Field>

        <Field label="Gateway port" hint="Harnesses keep the old one, so configure them again after.">
          <input
            type="number"
            value={draft.gateway_port}
            min={1}
            max={65535}
            onChange={(e) => edit({ gateway_port: e.target.value })}
          />
        </Field>

        <Field
          label="Agent rules"
          hint="Built-in rules for every request that offers tools: run the project's checks before a commit or push, and use jq on JSON. In Pi, an extension also holds a local model's untested push once. `ailocal config agent-rules` prints them."
        >
          <select value={draft.agent_rules} onChange={(e) => edit({ agent_rules: e.target.value })}>
            <option value="on">on</option>
            <option value="off">off</option>
          </select>
        </Field>

        <Field label="House rule" hint="Added after the agent rules, to every request the gateway forwards.">
          <textarea
            rows={3}
            value={draft.system_prompt}
            placeholder="none"
            onChange={(e) => edit({ system_prompt: e.target.value })}
          />
        </Field>

        <div className="controls">
          <button type="submit" className="primary" disabled={!dirty || actions.busy !== null}>
            {actions.busy === KEY ? "Saving…" : "Save"}
          </button>
          {dirty && (
            <button type="button" className="ghost" onClick={() => setDraft(null)}>
              Discard
            </button>
          )}
          {dirty && <span className="muted small">{Object.keys(patch).join(", ")}</span>}
        </div>
      </form>

      {actions.failed?.key === KEY && <p className="bad small">{actions.failed.message}</p>}
      {applied && !dirty && (
        <div className="note small">
          {describeApplied(applied).map((line) => (
            <p key={line}>{line}</p>
          ))}
        </div>
      )}
    </section>
  );
}

function Field({ label, hint, children }: { label: string; hint: string; children: ReactNode }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      <span className="muted small">
        <Ticks>{hint}</Ticks>
      </span>
    </label>
  );
}

/** Every input is a string, so the comparison that decides what to send is one too. */
interface Draft {
  cache_type: string;
  reasoning: string;
  reasoning_budget: string;
  max_context: string;
  gateway_port: string;
  agent_rules: string;
  system_prompt: string;
}

function toDraft(config: Config): Draft {
  return {
    cache_type: config.cache_type,
    reasoning: config.reasoning,
    reasoning_budget: String(config.reasoning_budget),
    max_context: String(config.max_context),
    gateway_port: String(config.gateway_port),
    agent_rules: config.agent_rules === false ? "off" : "on",
    system_prompt: config.system_prompt ?? "",
  };
}

/**
 * Only what the form actually changed.
 *
 * A number that will not parse is left out rather than sent as NaN, which would reach
 * the server as a deserialisation failure naming a field the person did not touch.
 */
export function diff(saved: Draft, draft: Draft): SettingsPatch {
  const patch: SettingsPatch = {};
  if (draft.cache_type !== saved.cache_type) patch.cache_type = draft.cache_type;
  if (draft.reasoning !== saved.reasoning) patch.reasoning = draft.reasoning;
  if (draft.agent_rules !== saved.agent_rules) patch.agent_rules = draft.agent_rules === "on";

  for (const key of ["reasoning_budget", "max_context", "gateway_port"] as const) {
    if (draft[key] === saved[key]) continue;
    const value = Number(draft[key]);
    if (draft[key].trim() !== "" && Number.isInteger(value)) patch[key] = value;
  }

  // Compared trimmed, because the server trims too: typing two spaces into an empty
  // box is not a change, and offering to save it would be offering to save nothing.
  const prompt = draft.system_prompt.trim() === "" ? null : draft.system_prompt;
  const was = saved.system_prompt.trim() === "" ? null : saved.system_prompt;
  if (prompt !== was) {
    // Emptied means cleared, which the patch spells as an explicit null: leaving the
    // field out would mean "keep whatever is there".
    patch.system_prompt = prompt;
  }
  return patch;
}
