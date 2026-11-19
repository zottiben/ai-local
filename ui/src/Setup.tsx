import { useState } from "react";

import { api } from "./api";
import { describeProbe, describeTrouble, formatCount, serviceShortName, tilde } from "./format";
import { Ticks } from "./Ticks";
import { type Actions, useActions, useResource } from "./hooks";
import { Settings } from "./Settings";
import type { GatewayCheck, HarnessView, Meta } from "./types";

interface Props {
  generation: number;
  meta: Meta;
}

/**
 * What `ailocal service install`, `gateway check` and `harness configure` do, in a
 * window.
 *
 * Every panel reads its own state rather than taking it from the overview, because
 * these are the things that change while you are looking at them: a service starts, a
 * port frees up, a harness is pointed somewhere else from a terminal.
 */
export function Setup({ generation, meta }: Props) {
  const actions = useActions();
  // Bumped by anything that changes the machine here, so the panels re-read without
  // waiting for the server's own event to come back around.
  const [changed, setChanged] = useState(0);
  const after = () => setChanged((n) => n + 1);
  const at = generation + changed;

  return (
    <>
      <ServicesPanel generation={at} actions={actions} after={after} />
      <GatewayPanel generation={at} actions={actions} after={after} />
      <HarnessesPanel generation={at} actions={actions} after={after} />
      <Settings generation={at} meta={meta} actions={actions} after={after} />
    </>
  );
}

interface PanelProps {
  generation: number;
  actions: Actions;
  after: () => void;
}

function Failure({ actions, keys }: { actions: Actions; keys: string[] }) {
  if (!actions.failed || !keys.includes(actions.failed.key)) return null;
  return <p className="bad small">{actions.failed.message}</p>;
}

function ServicesPanel({ generation, actions, after }: PanelProps) {
  const services = useResource(() => api.services(), [generation]);
  const data = services.data;
  const installed = data?.units.some((u) => u.enabled || u.active) ?? false;
  const keys = ["svc-install", "svc-remove", ...(data?.units.flatMap(unitKeys) ?? [])];

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Services</h2>
        <span className="muted small">so the gateway is up without a terminal open</span>
      </div>

      {services.error ? (
        <p className="bad">{services.error.message}</p>
      ) : !data ? (
        <p className="muted">Asking the service manager…</p>
      ) : (
        <>
          <div className="status">
            {data.units.map((unit) => (
              <div className="status-row" key={unit.name}>
                <span className={`mark ${unit.active ? "ok" : "none"}`}>
                  {unit.active ? "running" : "stopped"}
                </span>
                <span className="status-label" title={unit.name}>
                  {serviceShortName(unit.name)}
                </span>
                <span className="status-value row-actions">
                  <span className="muted small">
                    {unit.enabled ? "starts with the session" : "not enabled"}
                  </span>
                  <button
                    type="button"
                    className="ghost small"
                    disabled={actions.busy !== null}
                    onClick={() =>
                      actions.run(unitKey(unit.name, unit.active ? "stop" : "start"), () =>
                        api.serviceAction(unit.name, unit.active ? "stop" : "start").then(after),
                      )
                    }
                  >
                    {unit.active ? "Stop" : "Start"}
                  </button>
                </span>
              </div>
            ))}
          </div>

          {/* Rule 5: without lingering the units only run after a login, so a headless
              reboot leaves the gateway down. Enabling it needs root, so it is handed
              over as a command rather than offered as a button. */}
          {!data.linger && (
            <p className="note small">
              These only start once you have logged in. For the gateway to come up at boot and
              survive a logout, run <code>sudo loginctl enable-linger $USER</code>.
            </p>
          )}

          <div className="controls">
            <button
              type="button"
              className="primary"
              disabled={actions.busy !== null}
              onClick={() =>
                actions.run("svc-install", () => api.installServices().then(after))
              }
            >
              {actions.busy === "svc-install"
                ? "Installing…"
                : installed
                  ? "Rewrite and restart"
                  : "Install and start"}
            </button>
            {installed && (
              <button
                type="button"
                className="ghost"
                disabled={actions.busy !== null}
                onClick={() => actions.run("svc-remove", () => api.removeServices().then(after))}
              >
                {actions.busy === "svc-remove" ? "Removing…" : "Remove"}
              </button>
            )}
          </div>
          <Failure actions={actions} keys={keys} />
        </>
      )}
    </section>
  );
}

const unitKey = (name: string, action: string) => `unit:${action}:${name}`;
const unitKeys = (unit: { name: string }) => [unitKey(unit.name, "start"), unitKey(unit.name, "stop")];

function GatewayPanel({ generation, actions, after }: PanelProps) {
  const gateway = useResource(() => api.gateway(), [generation]);
  const [check, setCheck] = useState<GatewayCheck | null>(null);
  const [copied, setCopied] = useState(false);
  const data = gateway.data;
  const trouble = describeTrouble(data?.trouble ?? null);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Gateway</h2>
        {data && <code className="small">{data.url}</code>}
      </div>

      {gateway.error ? (
        <p className="bad">{gateway.error.message}</p>
      ) : !data ? (
        <p className="muted">Asking the gateway…</p>
      ) : (
        <>
          <div className="status">
            <div className="status-row">
              <span className={`mark ${data.state === "answering" ? "ok" : "miss"}`}>
                {data.state === "answering" ? "ok" : "missing"}
              </span>
              <span className="status-label">Answering</span>
              <span className="status-value">
                {trouble ? (
                  <>
                    {trouble.what}{" "}
                    <span className="muted">
                      <Ticks>{trouble.fix}</Ticks>
                    </span>
                  </>
                ) : (
                  "every harness on this machine reaches it here"
                )}
              </span>
            </div>
            <div className="status-row">
              <span className="mark none">key</span>
              <span className="status-label">Credential</span>
              <span className="status-value row-actions">
                {/* Shown rather than hidden: it is loopback-only and already on this
                    machine's disk, so a reveal control would guard nothing. */}
                <code className="small">{data.key ?? "not made yet"}</code>
                {data.key && (
                  <button
                    type="button"
                    className="ghost small"
                    onClick={() => {
                      void navigator.clipboard.writeText(data.key ?? "");
                      setCopied(true);
                    }}
                  >
                    {copied ? "Copied" : "Copy"}
                  </button>
                )}
              </span>
            </div>
          </div>

          <div className="controls">
            <button
              type="button"
              className="primary"
              disabled={actions.busy !== null}
              onClick={() =>
                actions.run("gw-check", () =>
                  api.checkGateway().then((result) => {
                    setCheck(result);
                    after();
                  }),
                )
              }
            >
              {actions.busy === "gw-check" ? "Checking…" : "Check it"}
            </button>
          </div>
          <Failure actions={actions} keys={["gw-check"]} />

          {check && <CheckResult check={check} />}
        </>
      )}
    </section>
  );
}

/**
 * What `ailocal gateway check` found.
 *
 * The assertion that matters is the second: a `/v1/models` answering 200 with no
 * credential is a gateway open to anything that can reach the port.
 */
function CheckResult({ check }: { check: GatewayCheck }) {
  if (!check.reachable) {
    return (
      <p className="bad small">
        Nothing is listening on <code>{check.base}</code>, so there was nothing to check.
      </p>
    );
  }
  const failures = check.probes.filter((probe) => !describeProbe(probe).ok).length;
  return (
    <>
      <h3>{failures === 0 ? "All checks passed" : `${failures} of ${check.probes.length} failed`}</h3>
      <div className="status">
        {check.probes.map((probe) => {
          const said = describeProbe(probe);
          return (
            <div className="status-row" key={probe.label}>
              <span className={`mark ${said.ok ? "ok" : "miss"}`}>{said.ok ? "ok" : "FAIL"}</span>
              <span className="status-label">{probe.label}</span>
              <span className="status-value">{said.said}</span>
            </div>
          );
        })}
      </div>
    </>
  );
}

function HarnessesPanel({ generation, actions, after }: PanelProps) {
  const harnesses = useResource(() => api.harnesses(), [generation]);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Harnesses</h2>
        <span className="muted small">what runs against the local model</span>
      </div>

      {harnesses.error ? (
        <p className="bad">{harnesses.error.message}</p>
      ) : !harnesses.data ? (
        <p className="muted">Reading what is configured…</p>
      ) : (
        harnesses.data.map((harness) => (
          <Harness key={harness.name} harness={harness} actions={actions} after={after} />
        ))
      )}
    </section>
  );
}

function Harness({
  harness,
  actions,
  after,
}: {
  harness: HarnessView;
  actions: Actions;
  after: () => void;
}) {
  const [model, setModel] = useState("");
  const plan = "error" in harness.plan ? null : harness.plan;
  const configure = `harness-set:${harness.name}`;
  const clear = `harness-clear:${harness.name}`;

  return (
    <div className="harness">
      <div className="panel-head">
        <h3>{harness.name}</h3>
        <span className={`mark ${harness.configured ? "ok" : "none"}`}>
          {harness.configured ? "configured" : "not configured"}
        </span>
      </div>

      {plan === null ? (
        <p className="bad small">{(harness.plan as { error: string }).error}</p>
      ) : (
        <>
          <p className="small">
            {/* Which model it gets is the thing that used to be decided invisibly, so
                it is said before the button rather than after. */}
            Would use <strong>{plan.model}</strong> at {formatCount(plan.context)} context
            <span className="muted"> - {plan.why}</span>
          </p>
          {plan.without_thinking.length > 0 && (
            <p className="muted small">
              {plan.without_thinking.join(", ")} has no thinking mode, so <code>/effort</code> can
              only ever offer <code>off</code> for it.
            </p>
          )}

          <div className="controls">
            {/* Pi keeps the whole catalogue and chooses per session; Claude Code takes
                exactly one model, so for it this choice is the whole configuration. */}
            <select
              value={model}
              aria-label={`Model for ${harness.name}`}
              onChange={(event) => setModel(event.target.value)}
            >
              <option value="">{plan.model} (chosen for you)</option>
              {plan.others.map((other) => (
                <option key={other} value={other}>
                  {other}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="primary"
              disabled={actions.busy !== null}
              onClick={() =>
                actions.run(configure, () =>
                  api.configureHarness(harness.name, model || undefined).then(after),
                )
              }
            >
              {actions.busy === configure
                ? "Writing…"
                : harness.configured
                  ? "Update"
                  : "Configure"}
            </button>
            {harness.configured && (
              <button
                type="button"
                className="ghost"
                disabled={actions.busy !== null}
                onClick={() =>
                  actions.run(clear, () => api.unconfigureHarness(harness.name).then(after))
                }
              >
                Unconfigure
              </button>
            )}
          </div>
          <Failure actions={actions} keys={[configure, clear]} />

          {/* Claude Code's file is sourced by hand on purpose: writing it into
              ~/.claude/settings.json would redirect every session on this machine,
              including the ones meant for the real Anthropic API. */}
          {harness.name === "claude-code" && harness.configured && harness.path && (
            <p className="note small">
              Source it in a shell, then run claude there:{" "}
              <code>source {tilde(harness.path)} &amp;&amp; claude</code>
            </p>
          )}
        </>
      )}
    </div>
  );
}
