When you work on code:
- Finish every change by running the project's own checks - its formatter, its linter and its tests, as AGENTS.md, the CI workflow or the build files name them - and fix what they report.
- Run them again before every commit or push, on the files as they are then, even after a one-line or non-code edit.
- Check that the change does what was asked: run the exact case from the request, and cover a change in behaviour with a test.
- Read and query JSON with jq, not python or node.
- Only say a check passed if you ran it and saw it pass.
