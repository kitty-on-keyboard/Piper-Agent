# LM_Pipe v2

A local coding agent for Apple Silicon: a VS Code extension plus one native sidecar
that loads a Qwen3 model in-process via MLX and drives a tool-using ReAct loop.

Built to the spec in `docs/BUILD_SPEC.md`. The implementation lands via a single
reviewable pull request rather than a pile of unreviewed commits on `main` — which is
the v1 failure this repo exists to avoid.
