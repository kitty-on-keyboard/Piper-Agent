# Piper Agent

A local coding agent for Apple Silicon: one native sidecar that loads a **single** Qwen3
model in-process via [MLX](https://github.com/ml-explore/mlx) and drives a tool-using
loop against a workspace. A parent LLM writes a slice brief; `piper packet` emits the
task; local Piper writes the code. `piper ui` is how you watch that happen.

**Scope:** Mac-local Qwen/MLX, one model loaded in the sidecar, no subagents.
An optional flag-off **tiny gate** helper (`LMP_TINY_GATE`, default off) may run a
second small Qwen3 process for T1 Choice — see [docs/TINY_GATE.md](docs/TINY_GATE.md).

License: [Apache-2.0](LICENSE). Credits: [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
**Model weights are not in this repo.** Follow the checkpoint's own license.

## Install

Apple Silicon only. First build of MLX from source is slow.

```bash
cmake --preset dev && cmake --build --preset dev --target lmp_sidecar -j8
cmake --build --preset dev --target install-piper-link   # or: ./scripts/install_piper_link.sh
export LMP_QWEN_DIR=/path/to/Qwen3-MLX-4bit
piper init
```

`install-piper-link` points `/opt/homebrew/bin/piper` at this checkout's parent
harness, `scripts/piper_worker.py`. The built worker stays at `build/src/surface/piper`.
One MLX process at a time.

## Parent loop

Write `prompt.md` (horizon, this slice, EDIT / CREATE / DO NOT TOUCH, the check).
Do not hand-write `task.json`.

```bash
piper packet --id slice-001 --cwd "$PWD" --prompt-file prompt.md --check "pytest tests/test_slice.py"
piper dispatch --task "$PWD/.piper/slices/slice-001/task.json"
piper ui --cwd "$PWD"
```

`piper packet` writes `.piper/slices/<id>/task.json`, copies the brief beside it, bakes
`model_dir` from `--model-dir` or `LMP_QWEN_DIR`, and records `.piper/active.json`.
`piper dispatch` stays attached and prints the review card. `piper ui` follows the
active slice.

Keep weights warm across slices with `piper worker serve`, then `piper worker run --task`
the same packet path.

## Editor sidebar

The VS Code / Cursor extension is optional and not the orchestration path. It loads the
workspace into the local model, which is the wrong shape for a long task. If you still
want it: `cd extension && npm install && npm run install-local`, then reload the window.
Do not run the sidebar and the CLI worker at the same time.

Re-run `install-piper-link` after a build or a checkout move. Direct worker invocation
is `lmp_sidecar --worker --task <path>` or `--serve`. Parent commands on that binary
exec the harness. Override the binary with `LMP_SIDECAR`.

## Use

1. Download one of the [tested checkpoints](#tested-checkpoints) and set `LMP_QWEN_DIR`.
2. From the workspace, write `prompt.md` and run the parent loop above.
3. Open `piper ui` to watch the active slice. The page follows `.piper/active.json`.

The worker edits inside `cwd` and cannot write outside it. Modes on a packet are
**plan** (reads only), **debug** (edits, never deletes), and **agent** (full tools).
Default is agent.

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
See [CONTRIBUTING.md](CONTRIBUTING.md). Capability and log-compaction corpora live under
`tests/testdata/`; engines that were measured historically ship under `src/`.
