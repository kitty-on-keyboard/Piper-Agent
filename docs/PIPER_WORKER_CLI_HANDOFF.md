# Piper Worker CLI — Cursor handoff

**Goal.** Ship a first-class **headless worker** beside the existing IDE agent so
cloud orchestrators (Cursor Grok, Grok Bot, etc.) can dispatch a scoped task to
local Piper (especially A3B), block until it finishes, read a structured result,
review the work, then assign the next task.

**Product shape (OSS).** Keep **one repo**. Two fronts, one core:

| Front | Audience | Artifact |
|---|---|---|
| VS Code / Cursor extension | Humans | `lm-pipe.vsix` (already ships) |
| CLI worker | Orchestrators / scripts | `piper` (or `lmp`) on `PATH` |

Do **not** fork a second agent. The sidecar + loop stay shared. The CLI is a new
thin driver over the same JSON-RPC / `lmp_sidecar` path `scripts/agent_eval.py`
already uses headlessly. README already markets “extension + one native
sidecar”; the CLI is the missing third install surface, not a new product.

---

## Why

Bakeoff signal (Aider Python mini-6, same Qwen A3B): Piper 6/6 ~4m vs Cline 4/6
~12m. Piper is the better *local coding harness*. Cloud models stay better at
planning / review. Compose them:

```
Cursor Grok / Bot  --task packet-->  piper worker (A3B local)
                   <--result.json--
                   review diff + tests
                   --next packet-->
```

A3B is the intended worker: MoE speed, good enough for scoped edits, cheap, private.

---

## Non-goals (v1)

- Subagents inside Piper (RAM; out of scope).
- Cloud model providers inside Piper.
- Persistent multi-turn chat UX in the CLI (IDE keeps that).
- Replacing the VSIX; both ship forever.
- Full MCP server (nice later; v1 is process + files).

---

## MVP behavior

### Install

```bash
# after building sidecar (existing preset)
cmake --preset dev && cmake --build --preset dev --target lmp_sidecar -j8

# new: install CLI shim onto PATH (pipx / brew formula / `cargo` later OK;
# simplest v1: python entry or small shell wrapper next to extension packaging)
piper --help
```

Document in README under Install: **VSIX for humans, CLI for agents.**

### Dispatch contract

Orchestrator writes a task directory (or a single packet file):

```
task_dir/
  task.json          # required — see schema below
  prompt.md          # optional long mission text (else inline in task.json)
```

```json
{
  "id": "2026-09-07T23-00-fix",
  "cwd": "/absolute/path/to/workspace",
  "mode": "agent",
  "model_dir": "/Users/dev/Desktop/Models/Qwen3.6-35B-A3B-MLX-4bit",
  "prompt": "Implement X. Do not touch Y. Run the existing tests.",
  "auto_approve_exec": true,
  "timeout_s": 900,
  "result_path": "/tmp/piper-results/2026-09-07T23-00-fix/result.json",
  "commit_think": true,
  "shadow_compact": true
}
```

### Command

```bash
piper worker run --task /path/to/task.json
# synonym
piper run --task /path/to/task.json
```

- Loads sidecar (or reuses a long-lived worker daemon in v1.1 — see below).
- Runs one mission against `cwd` (sandbox = workspace root, same as IDE).
- Blocks until finish / fail / timeout.
- Writes `result.json` (and optionally streams ndjson events to stdout).
- Exit codes:
  - `0` — completed with `status: "ok"` (agent finished; does **not** mean tests passed)
  - `1` — agent error / crash / protocol failure
  - `2` — timeout
  - `3` — invalid task packet / missing model

Orchestrator treats process exit as the “done” signal, then reads `result.json`.

### Result schema (`result.json`)

```json
{
  "task_id": "…",
  "status": "ok | error | timeout",
  "message": "Short human summary for the orchestrator (what changed, what failed).",
  "cwd": "…",
  "model_dir": "…",
  "wall_seconds": 0,
  "turns": 0,
  "generated_tokens": 0,
  "files_touched": ["rel/path.py"],
  "diff_stat": {"insertions": 0, "deletions": 0, "files": 0},
  "git_diff_path": "optional path to unified diff file",
  "test": {
    "ran": false,
    "exit_code": null,
    "command": null,
    "output_tail": null
  },
  "log_path": "…/events.ndjson",
  "error": null
}
```

`message` is mandatory and meant for the cloud agent to read without grepping logs.

### Event stream (optional but useful)

`--jsonl` prints progress to stdout (tool calls, finish) so the orchestrator can
tail; completion is still the process exit + `result.json`.

---

## Implementation sketch (for Cursor Grok)

Reuse, don’t reinvent:

1. **Core path already exists:** `scripts/agent_eval.py` → `drive_sidecar` /
   `build_start_request` talking to `build/src/surface/lmp_sidecar` over JSON-RPC.
2. **New thin module:** e.g. `scripts/piper_worker.py` (or `extension`-adjacent
   `cli/piper`) that:
   - parses `task.json`
   - sets `LMP_QWEN_DIR`, `LMP_COMMIT_THINK`, `LMP_SHADOW_COMPACT`, auto-approve env
     (mirror whatever the extension uses for `lmPipe.autoApproveExec`)
   - spawns / drives sidecar for one mission
   - collects file touch list (from tool events) + optional `git diff --stat`
   - writes `result.json`
3. **Wrapper on PATH:** `piper` → that module. Package with the VSIX build *or*
   a separate `pip install -e .` / homebrew formula later; v1 can be
   `ln -s …/scripts/piper_worker.py /usr/local/bin/piper` documented for Sean.
4. **Approvals in worker mode:** default auto-approve ordinary exec; keep the
   same hard blocks the sidecar already has for destructive ops (or fail the
   task with a clear `error` asking the orchestrator to escalate). Never hang
   on a UI card — there is no UI.
5. **Detach / SIGPIPE:** bakeoff taught us Cursor Shell reaps jobs. Worker should
   ignore SIGHUP when stdin is null / stdout is a file, same idea as
   `detach_from_launch_session` (helpers must exist — fix if still incomplete on
   the branch you land on).

### v1.1 (same PR series OK if cheap)

Long-lived `piper worker serve` that keeps A3B resident and accepts task packets
on a Unix socket / localhost port — kills cold-load cost between orchestrator
turns. v1 one-shot load is fine for first demo.

---

## Acceptance tests

1. **Smoke:** `piper worker run` on a tiny workspace (“add `hello()` to `a.py`”)
   exits 0, `result.json` has `status=ok`, `files_touched` includes `a.py`,
   `message` non-empty.
2. **Timeout:** `timeout_s=5` on a hard task → exit 2, `status=timeout`.
3. **Orchestrator loop (manual):** Cursor Grok script:
   - write task 1 → run piper → read result → `git diff` / pytest
   - write task 2 → run piper again
   Prove the human (or Grok) only orchestrates.
4. **No IDE required:** kill the editor; CLI still works with built sidecar.
5. **README:** Install section documents VSIX **and** CLI; one-paragraph
   “Use Piper as a local worker for cloud agents.”

---

## Docs / messaging (OSS)

- Position as **one agent, two interfaces** — not “Piper IDE” vs “Piper CLI.”
- Emphasize: cloud plans, local executes (privacy, cost, Apple Silicon speed).
- Keep Apache-2.0; model weights still not in repo.

---

## Suggested PR split

1. `piper worker run` MVP + result schema + README blurb + smoke test.
2. (Follow-up) `piper worker serve` keep-warm daemon.
3. (Follow-up) richer `test` block (caller-supplied check command in task.json).

---

## Constraints (Sean’s machine)

- 48 GB M5 Pro: **one** heavy model at a time. Worker assumes it owns the GPU;
  orchestrator must not start a second Piper/Cline load in parallel.
- Default worker model: A3B path via `model_dir` / `LMP_QWEN_DIR`.
- Optimal flags on by default: commit_think + shadow_compact (product defaults).

---

## Done means

Sean (or Cursor Grok) can run:

```bash
piper worker run --task ./examples/worker_task.json && cat ./result.json
```

…and wire that into a two-step “edit then verify” loop from Cursor without
opening the Piper sidebar.
