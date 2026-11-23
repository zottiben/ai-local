import type { ReactNode } from "react";

import { api } from "./api";
import {
  describeStep,
  formatCount,
  gatewayLabel,
  serviceLabel,
  serviceShortName,
  tilde,
} from "./format";
import { useActions, useResource } from "./hooks";
import { type View, navigate } from "./router";
import type { Config, Meta, Overview as OverviewData } from "./types";

interface Props {
  generation: number;
  meta: Meta;
}

/** What `ailocal status` reports, and what `ailocal config show` reads. */
export function Overview({ generation, meta }: Props) {
  const overview = useResource(() => api.overview(), [generation]);
  const config = useResource(() => api.config(), [generation]);

  if (overview.error) {
    return (
      <section className="panel">
        <h2>This machine could not be read</h2>
        <p className="bad">{overview.error.message}</p>
      </section>
    );
  }
  if (!overview.data) {
    return <p className="muted pad">Reading this machine…</p>;
  }

  return (
    <>
      <NextStepPanel overview={overview.data} meta={meta} />
      <MachinePanel overview={overview.data} />
      <SettingsPanel config={config.data} error={config.error} meta={meta} />
    </>
  );
}

export function NextStepPanel({ overview, meta }: { overview: OverviewData; meta: Meta }) {
  const step = describeStep(overview.next, meta, overview.llama_server);
  const actions = useActions();
  const act = nextAction(overview.next);
  if (!step) {
    return (
      <section className="panel ready">
        <h2>Everything is in place</h2>
        <p className="muted">
          A model can run, the gateway is answering and a harness is pointed at it.
        </p>
      </section>
    );
  }
  return (
    <section className="panel highlight" aria-label="Next step">
      <h3>Next step</h3>
      <h2>{step.title}</h2>
      <p className="muted">{step.why}</p>
      {/* The button does it; the commands stay because the terminal is still how
          this gets scripted, and because a step needing root has no button (rule 6). */}
      {act && (
        <div className="controls">
          <button
            type="button"
            className="primary"
            disabled={actions.busy !== null}
            onClick={() => {
              if (act.go) return navigate(act.go);
              actions.run("next-step", () => act.run());
            }}
          >
            {actions.busy === "next-step" ? `${act.label}…` : act.label}
          </button>
        </div>
      )}
      {actions.failed && <p className="bad small">{actions.failed.message}</p>}
      <ul className="commands">
        {step.commands.map((command) => (
          <li key={command}>
            <code>{command}</code>
          </li>
        ))}
      </ul>
      {step.log && (
        <p className="small">
          <span className="muted">The gateway's log: </span>
          <code>{step.log}</code>
        </p>
      )}
    </section>
  );
}

type Mark = "ok" | "miss" | "none";

/** One line of `ailocal status`, with its `ok` / `MISS` / `-` mark. */
function Row({ mark, label, children }: { mark: Mark; label: string; children: ReactNode }) {
  return (
    <div className="status-row">
      <span className={`mark ${mark}`}>{{ ok: "ok", miss: "missing", none: "-" }[mark]}</span>
      <span className="status-label">{label}</span>
      <span className="status-value">{children}</span>
    </div>
  );
}

export function MachinePanel({ overview }: { overview: OverviewData }) {
  const { advertised, loaded } = overview;
  return (
    <section className="panel">
      <h2 className="panel-title">This machine</h2>
      <div className="status">
        <Row mark={overview.models.length > 0 ? "ok" : "miss"} label="Models">
          {overview.models.length === 0
            ? "none installed that can run here"
            : `${overview.models.length} runnable, default ${overview.default_model ?? "not set"}`}
        </Row>

        <Row mark={loaded ? "ok" : "none"} label="Loaded">
          {loaded ? (
            <>
              {loaded.model} at {formatCount(loaded.context)} context, reasoning {loaded.reasoning}
            </>
          ) : (
            "nothing resident (the gateway loads on demand)"
          )}
        </Row>

        <Row mark={overview.gateway === "answering" ? "ok" : "miss"} label="Gateway">
          <code>{overview.gateway_url}</code> {gatewayLabel(overview.gateway)}
        </Row>

        {/* What the gateway offers a harness, not what is on disk. The two disagree when
            a model is installed but judged not to fit, which looks from inside a harness
            like the model vanishing from its picker. */}
        {advertised !== null && (
          <Row
            mark={"models" in advertised && advertised.models.length > 0 ? "ok" : "miss"}
            label="Advertised"
          >
            {"error" in advertised ? (
              <span className="bad">could not ask the gateway: {advertised.error}</span>
            ) : advertised.models.length === 0 ? (
              "none - the gateway is serving but offers no model. The Models view shows why."
            ) : (
              advertised.models.join(", ")
            )}
          </Row>
        )}

        {overview.services.map(([name, state]) => (
          <Row
            key={name}
            mark={state === "running" ? "ok" : "miss"}
            label={`Service: ${serviceShortName(name)}`}
          >
            <span title={name}>{serviceLabel(state)}</span>
          </Row>
        ))}

        <Row mark={overview.harnesses.length > 0 ? "ok" : "miss"} label="Harnesses">
          {overview.harnesses.length === 0 ? "none configured" : overview.harnesses.join(", ")}
        </Row>
      </div>
    </section>
  );
}

function SettingsPanel({
  config,
  error,
  meta,
}: {
  config: Config | undefined;
  error: Error | undefined;
  meta: Meta;
}) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Settings</h2>
        <code className="small" title={meta.config_path}>
          {tilde(meta.config_path)}
        </code>
      </div>
      {error ? (
        <p className="bad">{error.message}</p>
      ) : !config ? (
        <p className="muted">Reading the config…</p>
      ) : (
        <dl className="settings">
          <dt>Data directory</dt>
          <dd>
            <code>{tilde(config.data_dir)}</code>
          </dd>
          {config.models_dir && (
            <>
              <dt>Models directory</dt>
              <dd>
                <code>{tilde(config.models_dir)}</code>
              </dd>
            </>
          )}
          <dt>Default model</dt>
          <dd>{config.default_model ?? <span className="muted">not set</span>}</dd>
          <dt>KV cache</dt>
          <dd>
            <code>{config.cache_type}</code>
          </dd>
          <dt>Context cap</dt>
          <dd>{formatCount(config.max_context)}</dd>
          <dt>Reasoning</dt>
          <dd>
            {config.reasoning}
            <span className="muted">
              {config.reasoning_budget < 0
                ? ", no budget"
                : config.reasoning_budget === 0
                  ? ", budget 0 (thinking ends at once)"
                  : `, budget ${config.reasoning_budget} tokens`}
            </span>
          </dd>
          <dt>Gateway</dt>
          <dd>
            <code>
              {config.gateway_host}:{config.gateway_port}
            </code>
          </dd>
          <dt>Agent rules</dt>
          <dd>{config.agent_rules === false ? "off" : "on"}</dd>
          {config.system_prompt && (
            <>
              <dt>House rule</dt>
              <dd className="prompt">{config.system_prompt}</dd>
            </>
          )}
        </dl>
      )}
    </section>
  );
}

/**
 * What the next step's button does: go somewhere, or do it here.
 *
 * Installing llama.cpp has neither, because it needs root - rule 6 says that is handed
 * to a person as an exact command, and a button that could only fail at a password
 * prompt would be worse than none.
 */
function nextAction(
  next: OverviewData["next"],
): { label: string; go?: View; run: () => Promise<unknown> } | null {
  switch (next) {
    case "install_llama_server":
      return null;
    case "install_model":
      return { label: "Choose a model", go: "models", run: () => Promise.resolve() };
    case "start_gateway":
      return { label: "Install and start it", run: () => api.installServices() };
    case "fix_gateway":
      return { label: "Check the gateway", go: "setup", run: () => Promise.resolve() };
    case "configure_harness":
      return { label: "Point a harness at it", go: "setup", run: () => Promise.resolve() };
    case "ready":
      return null;
  }
}
