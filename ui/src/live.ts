// Following the machine while other processes change it (D4).
//
// A model installed from a terminal, the gateway restarted by launchd, a model loaded on
// demand: none of these go through this page. The server notices and announces that
// something changed, with no payload, and every view fetches again.

import { useEffect, useRef, useState } from "react";

import { token } from "./api";
import type { Job } from "./types";

export type Connection = "connecting" | "live" | "reconnecting";

interface Live {
  /** Bumped on every announced change. Use it as a dependency to fetch again. */
  generation: number;
  connection: Connection;
  /**
   * The console's own jobs, in full.
   *
   * The only thing on this stream with a payload: jobs live in the server process and
   * nowhere else, so there is nothing to refetch them from, and a download's progress
   * is the one thing here that moves four times a second.
   */
  jobs: Job[];
}

export function useLive(): Live {
  const [generation, setGeneration] = useState(0);
  const [connection, setConnection] = useState<Connection>("connecting");
  const [jobs, setJobs] = useState<Job[]>([]);
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

    source.addEventListener("jobs", (event) => {
      // Whatever the server sent is what the list is now, including empty. Merging
      // would keep a job the server has already forgotten.
      try {
        setJobs(JSON.parse((event as MessageEvent<string>).data) as Job[]);
      } catch {
        // A frame that does not parse is one update lost, and the next one carries the
        // whole list again. Nothing to repair, and nothing worth blanking the list for.
      }
    });

    // EventSource retries by itself, so this reports rather than repairs. Claiming
    // "live" over a dead stream is the one thing this must never do.
    source.onerror = () => setConnection(opened.current ? "reconnecting" : "connecting");

    return () => source.close();
  }, []);

  return { generation, connection, jobs };
}
