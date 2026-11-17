// Following the machine while other processes change it (D4).
//
// A model installed from a terminal, the gateway restarted by launchd, a model loaded on
// demand: none of these go through this page. The server notices and announces that
// something changed, with no payload, and every view fetches again.

import { useEffect, useRef, useState } from "react";

import { token } from "./api";

export type Connection = "connecting" | "live" | "reconnecting";

interface Live {
  /** Bumped on every announced change. Use it as a dependency to fetch again. */
  generation: number;
  connection: Connection;
}

export function useLive(): Live {
  const [generation, setGeneration] = useState(0);
  const [connection, setConnection] = useState<Connection>("connecting");
  // Tells the first connect from a reconnect, so the indicator says which is happening.
  const opened = useRef(false);

  useEffect(() => {
    // EventSource cannot set headers, which is why the token has a query form.
    const source = new EventSource(`/api/events?t=${encodeURIComponent(token)}`);

    source.onopen = () => {
      // Anything could have happened while the stream was down, so coming back is
      // itself a reason to fetch again.
      if (opened.current) setGeneration((n) => n + 1);
      opened.current = true;
      setConnection("live");
    };

    source.addEventListener("changed", () => setGeneration((n) => n + 1));

    // EventSource retries by itself, so this reports rather than repairs. Claiming
    // "live" over a dead stream is the one thing this must never do.
    source.onerror = () => setConnection(opened.current ? "reconnecting" : "connecting");

    return () => source.close();
  }, []);

  return { generation, connection };
}
