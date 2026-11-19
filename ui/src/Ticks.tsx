import { Fragment } from "react";

/**
 * A sentence that names a command inline, with the command set as code.
 *
 * The strings in `format.ts` mark a command the way the CLI's own messages do, in
 * backticks - and a backtick is not markup in a browser, so it reached the window as a
 * literal character beside "run ailocal gateway run in a terminal". Splitting on the
 * pairs here keeps one spelling for both surfaces rather than two copies of each
 * sentence, and an unpaired backtick stays as typed instead of swallowing the rest.
 */
export function Ticks({ children }: { children: string }) {
  const parts = children.split("`");
  // An odd count means every backtick was paired; an even one leaves a stray.
  if (parts.length % 2 === 0) return <>{children}</>;
  return (
    <>
      {parts.map((part, i) => (
        <Fragment key={i}>{i % 2 === 1 ? <code>{part}</code> : part}</Fragment>
      ))}
    </>
  );
}
