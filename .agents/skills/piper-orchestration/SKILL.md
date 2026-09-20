---
name: piper-orchestration
description: Use this when orchestrating Piper / local Piper worker / cloud directs local writes. Guides cloud agents on delegating implementation to local Apple Silicon MLX models via bounded task packets.
---

# Piper Local Worker Orchestration

Piper is a headless local coding worker running on Apple Silicon via MLX (typically `Qwen3.6-35B-A3B-MLX-4bit`). It executes edits, tools, and shell commands inside `cwd` with zero cloud output token cost.

## 1. Division of Labor
- **Cloud Orchestrator (Long-Horizon Brain):** Decomposes goals, specifies file-level bounds, writes acceptance criteria, reviews diffs/outcomes, troubleshoots failures, decides when the horizon is complete.
- **Piper Worker (Local MLX Hands):** Writes code, modifies files, runs local tools/tests inside `cwd`.
- **Core Loop:** `Cloud directs → Piper writes → Cloud verifies → Repeat`.
- **Concurrency Rule:** Never run IDE Piper and CLI Piper worker simultaneously (one MLX process per system to avoid unified memory contention).

---

## 2. Packet Design (The Quality Lever)
Local models require fat direction and thin scope: burn cloud input tokens on explicit briefing so Piper burns free local tokens on implementation.

- **Scope:** 1–3 files per slice. Never dispatch open-ended multi-module tasks.
- **Explicitness:** Explicitly enumerate `EDIT`, `CREATE`, and `DO NOT TOUCH` files.
- **Acceptance:** Include an automated `check` command. A passing check promotes incomplete loop stops (e.g., hitting `max_iterations` without an explicit finish call) to `status: "ok"`.
- **Stop Condition:** Give explicit instructions on when to stop if stuck.

### `task.json` Template
Place in a slice directory (e.g. `.piper/slices/slice-001/task.json`):

```json
{
  "id": "feature-x/slice-001",
  "cwd": "/absolute/path/to/workspace",
  "mode": "agent",
  "model_dir": "/Users/dev/Desktop/Models/Qwen3.6-35B-A3B-MLX-4bit",
  "prompt": "See prompt.md or inline string",
  "auto_approve_exec": true,
  "auto_approve_writes": true,
  "auto_approve_irreversible": true,
  "timeout_s": 600,
  "max_iterations": 30,
  "check": "pytest tests/test_slice.py",
  "check_timeout_s": 60.0,
  "result_path": "/absolute/path/to/slice-001/result.json"
}
```

*Note: If `prompt` is omitted in `task.json`, Piper automatically loads a sibling `prompt.md`.*

### `prompt.md` Template
```markdown
## Horizon Context
Brief 2–4 sentence summary of the overarching goal and architecture.

## This Slice Only
Single focused outcome for this turn. Do not start subsequent steps.

## Files
- EDIT: `src/path/to/file.py` — specific function/class to modify.
- CREATE: `tests/path/to/test_file.py` — test cases to implement.
- DO NOT TOUCH: `src/core/`, lockfiles, configuration files.

## Constraints
- No external dependencies without explicit instruction.
- Preserve existing function signatures and docstrings.
- No drive-by refactorings outside the named edits.

## Done When
1. Code modifications satisfy the slice requirement.
2. Verification passes: `pytest tests/path/to/test_file.py`.
3. Final response summarizes changes and verification result.

## If Stuck
Stop after 3 failed attempts at the same error. Explain the blocker in your final answer rather than hallucinating alternatives or modifying out-of-scope files.
```

---

## 3. Dispatch Contract

### Launching
- **Attached Mode (Preferred):** Cloud parent waits on process exit.
  ```bash
  piper run --task /abs/path/to/task.json --auto-approve-irreversible
  # Equivalent: piper worker run --task /abs/path/to/task.json
  # Pass --auto-approve-all for unattended exec + writes + irreversible tools
  ```
- **Exit Codes:**
  - `0`: Completed successfully (`status: "ok"`).
  - `1`: Error (worker failure or denied irreversible tool).
  - `2`: Execution timed out (`timeout_s` exceeded).
  - `3`: Invalid packet or detached launch without wake URL.
- **Detached Mode (`nohup`, `screen`, background):**
  - **MANDATORY:** Must pass `--orch-webhook <URL>`, task field `"orch_webhook": "<URL>"`, or env `LMP_ORCH_WEBHOOK`.
  - Detached runs without a wake URL exit immediately with code `3`. Never run silent background workers.
  - Parent owns the wake URL. Do not ask humans to copy webhooks. Do not poll directories as primary coordination.

### Wake Events (Parent must handle all 4)
One short JSON POST per event:
1. `ask`: Paused on user question or irreversible tool. Parent writes `answer.json` containing `{"text": "..."}` or `allow`/`deny`. Do NOT restart process.
2. `done`: Run completed with `status: "ok"`. Parent reads `result.json` and git diff, then dispatches next slice or marks horizon complete.
3. `stalled`: Loop stopped without completion (`stalled`, `max_turns`, or failed check). **Do NOT treat as done.** Inspect state, narrow scope, or intervene.
4. `died`: Process exited with no `result.json`. Alert operator; do NOT relaunch blindly.

---

## 4. Cheap Cloud Review Rubric

Cloud orchestrator reviews outcomes at a high level—do not reread every line of code unless anomalies appear:

| Check | Pass Signal | Action on Failure |
|---|---|---|
| **Status** | `status == "ok"` | Check `error` field or run acceptance check manually. |
| **Touched Files** | `files_touched ⊆ expected` | Revert unexpected modifications (`git checkout -- <file>`), tighten `DO NOT TOUCH`. |
| **Diff Stat** | Proportional to slice (e.g. +50/-10) | Reject drive-by rewrites or massive churn. |
| **Acceptance** | `test.exit_code == 0` | If failed, review `test.output_tail` and dispatch targeted fix slice. |
| **Summary** | `result.message` confirms goal | Review log if ambiguous. |

*Record a one-line progress log per slice to maintain long-horizon memory:*
`slice-001 | pass | added validator and unit test (+42 -2)`

---

## 5. Failure Playbook

- **Timeout (`exit 2` or `status: "timeout"`):** Split into smaller 1-file slices or increase `timeout_s`.
- **Sidecar Crash / No Result (`died`):** Kill stale processes (`pkill -9 lmp_sidecar`), remove any lockfiles, retry packet once.
- **Model Thrashing (Repeated failed tool loops):** Switch `mode: "plan"` for read-only diagnosis, then dispatch a pinpoint edit slice.
- **Model Ceiling (A3B struggles with complex abstraction):** Orchestrator writes the complex logic directly, then hands unit tests and boilerplate back to Piper.
- **Irreversible Tool Gate Paused:** Ensure `"auto_approve_irreversible": true` or `--auto-approve-irreversible` was passed if unattended execution was intended.

---

## 6. Anti-Patterns
1. **Giant Unbounded Packets:** Asking Piper to "build the auth service and frontend" in one task.
2. **Silent Detach:** Running background jobs without an active `--orch-webhook`.
3. **Polling Instead of Wake:** Spinning on filesystem checks instead of waiting on process exit or webhook events.
4. **Parallel MLX:** Launching multiple CLI workers or running IDE Piper alongside CLI worker on Apple Silicon.
5. **Micro-Reviewing Every Token:** Wasting cloud tokens reading entire source files when diffs and automated checks are green.
6. **Trusting Status Without Check:** Relying on the model claiming "done" without a deterministic `check` command.

---

## 7. First Slice Smoke Checklist
Before executing an ambitious multi-slice plan, verify the setup with a 2-minute smoke test:
1. [ ] Confirm model directory exists (`/Users/dev/Desktop/Models/Qwen3.6-35B-A3B-MLX-4bit` or `$LMP_QWEN_DIR`).
2. [ ] Verify no competing `lmp_sidecar` or IDE Piper processes are running (`pgrep lmp_sidecar`).
3. [ ] Dispatch minimal slice: create a dummy test file and run `piper run --task task.json`.
4. [ ] Confirm exit code `0`, `result.json` written with `status: "ok"`, and `files_touched` recorded.
5. [ ] Clean up smoke test artifact and proceed to Horizon Slice 1.
