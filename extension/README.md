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

Ordinary fully-parsed commands follow risk routing; with `lmPipe.autoApproveExec` on,
low-risk builds and tests run without a card. Irreversible capabilities (destroying
data, writing outside the workspace, privilege escalation, history rewrite) and opaque
commands (`PartiallyParsed` / `Unparseable`, e.g. `bash unknown.sh`) always raise a
card — those are property overrides, not score bumps, and a remembered allowlist entry
cannot skip them. **The run blocks until you answer.** Risk above the internal reject
ceiling is refused without a card.

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
- Agent edits go through `lmp/edit` → VS Code `WorkspaceEdit` when the client advertises
  `applies_edits` (the extension does). **Undo and dirty-buffer / diff review work** on
  that path. Headless clients that leave `applies_edits` false still write directly.
- Most `RunSettings` fields (sampling, budgets, mode, allowlists, MCP servers, prompts)
  are applied by the sidecar at `lmp/start`. Container tier remains refused as above.

## Product scope

Mac-local **one-model** Qwen/MLX agent: a single in-process checkpoint, no subagents, no
second inference server. Speculative decode stays opt-in (`LMP_SPECULATIVE=1`).
