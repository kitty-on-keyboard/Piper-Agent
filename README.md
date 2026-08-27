# Piper Agent

A local coding agent for Apple Silicon: a VS Code / Cursor extension plus one native
sidecar that loads a **single** Qwen3 model in-process via [MLX](https://github.com/ml-explore/mlx)
and drives a tool-using loop against the workspace you open.

**Scope:** Mac-local Qwen/MLX, one model loaded, no subagents, no second inference server.

License: [Apache-2.0](LICENSE). Credits: [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
**Model weights are not in this repo.** Follow the checkpoint's own license.

## Install

Apple Silicon only.

**From a release (easiest):** download `lm-pipe.vsix` from
[Releases](https://github.com/kitty-on-keyboard/Piper-Agent/releases). In the editor:
Extensions → `...` → Install from VSIX. **Reload the window**, then set `lmPipe.modelDir`.

**From source.** First build of MLX from source is slow.

```bash
cmake --preset dev && cmake --build --preset dev --target lmp_sidecar -j8
cd extension && npm install && npm run install-local
```

That compiles the sidecar (MLX linked statically, `mlx.metallib` staged next to it),
packages `lm-pipe.vsix`, and installs it into every VS Code-family editor it finds
(Cursor, VS Code, VSCodium). Override with `LMP_EDITOR_CLI` to target one.

**Reload the editor window** — a newly installed extension does not activate in an
already-open window.

To package a VSIX locally: `cd extension && npm run package`, then Extensions → `...` →
Install from VSIX.

## Use

1. Download one of the [tested checkpoints](#tested-checkpoints) (MLX 4-bit folder with
   `config.json`, `tokenizer.json`, and `*.safetensors`).
2. Open the folder you want the agent to work in. The first workspace folder is the
   sandbox root; the agent cannot write outside it.
3. Command palette → **LM_Pipe: Choose the model directory** (or set `lmPipe.modelDir`
   in settings). There is no default; an unset path refuses loudly.
4. Open the **Piper Agent** sidebar and send a mission, or Command palette →
   **LM_Pipe: Start a run**.
5. Approvals: ordinary builds and tests can auto-run if `lmPipe.autoApproveExec` is on.
   Destructive or opaque commands always raise a card. The run blocks until you answer.
6. The composer **Stop** button (or Shift+Escape / command **Stop run**) stops
   mid-generation.

Modes: **plan** (reads only), **debug** (edits, never deletes), **agent** (full tools).

## Tested checkpoints

Qwen3 MLX 4-bit folders. Typical source: Hugging Face `lmstudio-community`.

| Checkpoint | Kind | Notes |
|---|---|---|
| `Qwen3.6-35B-A3B-MLX-4bit` | MoE | Primary agent model. No thinking-level control. |
| `Qwen3.8-27B-MLX-4bit` | Dense | Thinking levels (`low` / `medium` / `xhigh`) work. |
| `Qwen3.8-27B-MTP-4bit` | MTP draft head | Not a standalone model. Pair with the dense 27B for speculative decode (sidebar checkbox). Ignored on A3B. |

Other families refuse at load. Other Qwen3 MLX checkpoints may work; these three are the
ones actually run.

## Layout

```
src/platform/   L0  arenas, event log, SPSC channel, clock, fs
src/model/      L1  tokenizer, vocab, KV cache, sampler, grammar mask, MLX backends
src/tools/      L2  registry, schemas, structured results, sandbox, capability classifier
src/context/    L3  event store, tiering, compaction, prompt assembly
src/loop/       L4  the loop: classifier, repeat cache, HITL gate, operator check
src/surface/    L5  JSON-RPC protocol, extension, webview UI, settings
```

## Hard constraints

Apple Silicon only. One in-process model, no inference server, no subagents. C++20,
`-Werror`, assertions on. **The agent never touches git** — it edits the workspace;
staging, committing and branching are yours.

Developers: `cmake --preset dev && cmake --build --preset dev -j8 && ctest --preset gate`.
See [CONTRIBUTING.md](CONTRIBUTING.md). Measurements live in [docs/PHASES.md](docs/PHASES.md).
