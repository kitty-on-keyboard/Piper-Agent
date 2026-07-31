# LM_Pipe

A local coding agent for Apple Silicon: Qwen3 in-process via MLX, driven from the editor
sidebar. One sidecar process owns the model and the loop; the extension speaks the
private `lmp/*` protocol to it over stdio.

## Install

From the repo root, build the sidecar first — the package refuses to build without it:

```bash
cmake --build build --target lmp_sidecar
```

Then, from `extension/`:

```bash
npm install && npm run install-local
```

That compiles the TypeScript, stages the binary into `extension/bin/`, produces
`lm-pipe.vsix`, and installs it into Cursor (or VS Code / VSCodium, whichever it finds).
**Reload the editor window afterwards** — a newly installed extension does not activate
in an already-open window.

To install by hand instead: `npm run package`, then Extensions view → `...` →
"Install from VSIX".

## Prerequisites

- **Apple Silicon.** Not a portability gap to be fixed; a settled scope decision.
- **The MLX runtime.** `lmp_sidecar` resolves `libmlx.dylib` through an absolute rpath
  into the venv it was built against
  (`/Users/dev/.venvs/lmp-mlx/lib/python3.12/site-packages/mlx/lib`). The package does
  **not** bundle it: `libmlx.dylib` plus `mlx.metallib` are ~180 MB, and bundling them is
  a deliberate decision nobody has needed to make yet. If that venv moves or is deleted,
  the sidecar dies at startup and the extension surfaces the dyld error verbatim.
- **A model.** Set `lmPipe.modelDir` to a Qwen3 MLX checkpoint directory. There is no
  default model and no fallback — an unset `modelDir` refuses loudly (S7.5).

## Use

1. Open the folder you want the agent to work in. The first workspace folder becomes the
   sandbox root, and the agent cannot write outside it.
2. Command palette → **LM_Pipe: Start a run**. Enter a mission: the one place the
   deliverable is named.
3. Watch the **LM_Pipe** sidebar (activity bar): streaming answer, thinking behind a
   disclosure, a tool timeline, verification state, and approval cards.

### Approvals

A tool call that scores above `lmPipe.hitl.autoApproveBelowRisk` (0.35) raises an
approval card with the command preview, its risk score, and the capability chips the
classifier set. **The run blocks until you answer it.** Above
`lmPipe.hitl.rejectAboveRisk` (0.85) the call is rejected without ever reaching you.

Everything that is not an explicit approval denies: closing the window, cancelling the
run, or shutting the sidecar down all deny the pending call rather than letting it
through.

### Cancelling

**LM_Pipe: Cancel the current run** is deliverable mid-generation — the sidecar's reader
thread sets the cancel token as soon as the message is framed, without waiting for the
model to finish the token it is on.

## Known limits

- `lmPipe.sandboxTier` 2 (container) **refuses**. Unattended runs require T2 (S7.2), so
  in practice runs are attended-only today.
- `lmp/edit` writes files directly rather than through the editor's edit API, so agent
  edits have **no undo and no diff review**.
- Settings other than `modelDir`, `workspaceRoot` and `mission` are sent but not yet read
  by the sidecar, which uses its own defaults for sampling, iteration caps and mode.
