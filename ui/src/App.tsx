import { ApiError, api, token } from "./api";
import { useResource } from "./hooks";
import { type Connection, useLive } from "./live";
import { Models } from "./Models";
import { Overview } from "./Overview";
import { Setup } from "./Setup";
import { PATHS, type View, navigate, useView } from "./router";

const TABS: { view: View; label: string }[] = [
  { view: "overview", label: "Overview" },
  { view: "models", label: "Models" },
  { view: "setup", label: "Setup" },
];

export function App() {
  // No token at all, so every request would be refused. Make none of them.
  if (!token) return <Unauthorized />;
  return <Console />;
}

function Console() {
  const view = useView();
  const { generation, connection, jobs } = useLive();
  const meta = useResource(() => api.meta(), []);

  if (meta.error instanceof ApiError && meta.error.status === 401) {
    return <Unauthorized />;
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-name">ai-local</span>
          {meta.data && <span className="muted small">{meta.data.version}</span>}
        </div>
        <nav className="tabs" aria-label="Views">
          {TABS.map((tab) => (
            <a
              key={tab.view}
              href={PATHS[tab.view]}
              className={tab.view === view ? "tab on" : "tab"}
              aria-current={tab.view === view ? "page" : undefined}
              onClick={(event) => {
                // A modified click opens a new tab, which is what it is for.
                if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) return;
                event.preventDefault();
                navigate(tab.view);
              }}
            >
              {tab.label}
            </a>
          ))}
        </nav>
        <LiveIndicator connection={connection} />
      </header>

      <main className="content">
        {/* The views wait for /meta, which is also what says whether the token is
            good. Starting them at once would send five more requests to be refused. */}
        {meta.error ? (
          <section className="panel">
            <h2>The console could not start</h2>
            <p className="bad">{meta.error.message}</p>
          </section>
        ) : !meta.data ? (
          <p className="muted pad">Connecting…</p>
        ) : view === "models" ? (
          <Models generation={generation} jobs={jobs} />
        ) : view === "setup" ? (
          <Setup generation={generation} meta={meta.data} />
        ) : (
          <Overview generation={generation} meta={meta.data} />
        )}
      </main>
    </div>
  );
}

const CONNECTION_LABEL: Record<Connection, string> = {
  connecting: "connecting",
  live: "live",
  reconnecting: "reconnecting",
};

function LiveIndicator({ connection }: { connection: Connection }) {
  return (
    <span
      className={`connection ${connection}`}
      title={
        connection === "live"
          ? "Changes made from a terminal show up here as they happen"
          : "Not following changes right now - what is shown may be out of date"
      }
    >
      <span className="dot" aria-hidden="true" />
      {CONNECTION_LABEL[connection]}
    </span>
  );
}

/**
 * The token is minted per run and never stored, so a bookmarked link or a tab left open
 * across a restart arrives here. Say what to do rather than showing empty panels.
 */
function Unauthorized() {
  return (
    <main className="fatal">
      <h1>This link is not for this console</h1>
      <p>
        Every run of <code>ailocal ui</code> makes a new link, and only that link is let in. A
        tab from an earlier run, or one opened without the link, lands here.
      </p>
      <p className="muted">
        Open the link <code>ailocal ui</code> printed in the terminal, or run it again.
      </p>
    </main>
  );
}
