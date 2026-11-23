// The model that is resident, and what it is doing.
//
// What `ailocal ps` reports plus the figure `ailocal serve` prints once a model is up,
// and then the server's own output - because the question a loading model raises is
// "is it progressing", and only the log answers that.

import { useEffect, useLayoutEffect, useRef, useState } from "react";

import { api } from "./api";
import { formatCount, formatSampling, formatSize } from "./format";
import { type Actions, useResource } from "./hooks";
import type { Instance, LogView } from "./types";

/**
 * How often the log and the memory figure are read.
 *
 * Both are cheap - a seek and a read of what is new, and a `footprint` probe measured
 * at 72 ms on the M4 Max - and this only runs while the panel is on screen.
 */
const POLL_MS = 2000;

/** How much of the tail to keep. llama-server writes a few hundred lines as it loads. */
const KEEP = 64 * 1024;

export function Loaded({ instance, actions }: { instance: Instance; actions: Actions }) {
  const tick = useTick(POLL_MS);
  const memory = useResource(() => api.memory(), [instance.pid, tick]);
  const log = useLogTail(tick);
  const failure = actions.failed?.key === "stop" ? actions.failed.message : null;

  return (
    <section className="panel" aria-label="Loaded model">
      <div className="panel-head">
        <h2>
          {instance.model} <span className="badge ok">loaded</span>
        </h2>
        <button
          disabled={actions.busy !== null}
          onClick={() => actions.run("stop", () => api.stop())}
        >
          Stop
        </button>
      </div>

      <dl className="settings">
        <dt>Serving</dt>
        <dd>
          <code>
            {instance.host}:{instance.port}
          </code>{" "}
          <span className="muted">pid {instance.pid}</span>
        </dd>
        <dt>Context</dt>
        <dd>
          {formatCount(instance.context)} with <code>{instance.cache_type}</code> KV
        </dd>
        <dt>Reasoning</dt>
        <dd>{instance.reasoning}</dd>
        <dt>Sampling</dt>
        <dd>
          {instance.sampling ? (
            <>
              {formatSampling(instance.sampling)}{" "}
              <span className="muted">(from the publisher)</span>
            </>
          ) : (
            <span className="muted">the model file's, else llama.cpp's defaults</span>
          )}
        </dd>
        {memory.data && (
          <>
            <dt>{memory.data.label === "vram" ? "VRAM" : "Memory"}</dt>
            <dd>
              {formatSize(memory.data.mib)}
              {instance.cache_ram_mib !== null && (
                <span className="muted">
                  {" "}
                  (prompt cache up to {formatSize(instance.cache_ram_mib)} in host memory)
                </span>
              )}
            </dd>
          </>
        )}
      </dl>

      {failure && <p className="bad small">{failure}</p>}
      <LogTail log={log.view} text={log.text} error={log.error} />
    </section>
  );
}

function LogTail({
  log,
  text,
  error,
}: {
  log: LogView | null;
  text: string;
  error: string | null;
}) {
  const box = useRef<HTMLPreElement>(null);
  // Follow the end, unless the reader has scrolled up to look at something.
  const following = useRef(true);

  useLayoutEffect(() => {
    const el = box.current;
    if (el && following.current) el.scrollTop = el.scrollHeight;
  }, [text]);

  if (error) return <p className="bad small">{error}</p>;
  if (log === null) return <p className="muted small">Reading its log…</p>;

  if (log.kind === "elsewhere") {
    return (
      <p className="muted small">
        This model is run by a service, so its output goes where the service's does:{" "}
        <code>{log.service_log}</code>
      </p>
    );
  }
  if (log.kind === "idle" || text === "") {
    return <p className="muted small">Nothing in its log yet.</p>;
  }

  return (
    <>
      <h3>Log</h3>
      <pre
        className="log"
        ref={box}
        tabIndex={0}
        aria-label="The loaded model's log"
        onScroll={(event) => {
          const el = event.currentTarget;
          following.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
        }}
      >
        {text}
      </pre>
      <p className="muted small">
        <code>{log.path}</code>
      </p>
    </>
  );
}

function useTick(ms: number): number {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setTick((n) => n + 1), ms);
    return () => clearInterval(timer);
  }, [ms]);
  return tick;
}

interface Tail {
  view: LogView | null;
  text: string;
  error: string | null;
}

/**
 * The loaded model's log, kept across polls.
 *
 * The cursor and the text live in refs so a poll continues where the last one ended
 * rather than re-reading the window every two seconds. A `from` other than the cursor
 * means the file was truncated by a new start, or the page fell more than a window
 * behind, so what it holds is not continuous with what arrived and goes.
 */
function useLogTail(tick: number): Tail {
  const [view, setView] = useState<LogView | null>(null);
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const cursor = useRef<number | undefined>(undefined);
  const held = useRef("");

  useEffect(() => {
    let live = true;
    api
      .log(cursor.current)
      .then((next) => {
        if (!live) return;
        if (next.kind === "file") {
          if (next.from !== cursor.current) held.current = "";
          held.current = (held.current + next.text).slice(-KEEP);
          cursor.current = next.to;
        } else {
          held.current = "";
          cursor.current = undefined;
        }
        setText(held.current);
        setView(next);
        setError(null);
      })
      .catch((err: unknown) => {
        if (live) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      live = false;
    };
  }, [tick]);

  return { view, text, error };
}
