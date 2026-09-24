---
name: piper-orchestration
description: Use this when orchestrating Piper / local Piper worker / cloud directs local writes. Guides cloud agents on delegating implementation to local Apple Silicon MLX models via bounded task packets.
---

# Piper Local Worker Orchestration

Piper is a headless local coding worker running on Apple Silicon via MLX (typically `Qwen3.6-35B-A3B-MLX-4bit`). It executes edits, tools, and shell commands inside `cwd` with zero cloud output token cost.

## 1. Division of Labor
- **Cloud Orchestrator (Long-Horizon Brain):** Decomposes goals, writes `prompt.md`, reviews the dispatch card, troubleshoots failures, decides when the horizon is complete.
- **Piper Worker (Local MLX Hands):** Writes code, modifies files, runs local tools/tests inside `cwd`.
- **Harness:** Owns `task.json` shape, `answer.json`, and the progress-log line. Parents call the helpers. They do not hand-author those files.
- **Core Loop:** `Cloud directs → Piper writes → Cloud verifies → Repeat`.
- **Concurrency Rule:** Never run IDE Piper and CLI Piper worker simultaneously (one MLX process per system to avoid unified memory contention).

---

## 2. Slice Brief (The Quality Lever)
Local models require fat direction and thin scope: burn cloud input tokens on the brief so Piper burns free local tokens on implementation.

- **Scope:** 1–3 files per slice. Never dispatch open-ended multi-module tasks.
- **Explicitness:** Enumerate `EDIT`, `CREATE`, and `DO NOT TOUCH` in `prompt.md`.
- **Acceptance:** Put an automated check in the brief and pass it to `piper packet --check`. A passing check promotes an incomplete loop stop (for example `max_turns` without an explicit finish) to `status: "ok"`.
- **Stop Condition:** Say when to stop if stuck.

### `prompt.md` template
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
Stop after 3 failed attempts at the same error. Explain the blocker in your final answer rather than inventing alternatives or editing out-of-scope files.
```

---

## 3. Deterministic Parent Loop

```bash
# 1. Optional: names that exist in cwd/.mcp.json
piper mcp-list --cwd /abs/workspace

# 2. Emit the slice. Do not hand-write task.json. Do not pass --out.
#    Writes <cwd>/.piper/slices/<id>/task.json, copies prompt.md, bakes model_dir.
piper packet --id slice-001 --cwd /abs/workspace \
  --prompt-file prompt.md --check "pytest tests/test_slice.py"
# piper packet ... --trust-mcp godoer

# 3. Attached run. Prints the review card. Does not detach.
piper dispatch --task /abs/workspace/.piper/slices/slice-001/task.json --auto-approve-irreversible

# 4. Watch. Optional; the dashboard follows .piper/active.json.
piper ui --cwd /abs/workspace

# 5. Record the slice. Do not paste the log line.
piper progress --id slice-001 pass --note "validator + test"
```

- **Unattended irreversible tools:** `--auto-approve-irreversible` or `--auto-approve-all` on `piper dispatch`.
- **Ask:** `piper answer allow`, `piper answer deny`, or `piper answer --text "..."`. Do not freehand `answer.json`. Do not restart the process.
- **Already finished:** `piper review --task …`. Status without cat/jq: `piper status --dir …` / `piper await --dir …`.
- **Lower-level attached run:** `piper run --task …` (same as `piper worker run`). Keep weights warm with `piper worker serve`, then `piper worker run`.
- **Exit codes:** `0` ok, `1` worker error, `2` timeout, `3` invalid packet or detached launch with no wake URL.

The emitter writes `<cwd>/.piper/slices/<id>/task.json` unless `--out` is set. It writes `id`, `cwd`, `prompt`, `model_dir` (from `--model-dir` or `LMP_QWEN_DIR`; missing model exits 3), auto-approve flags, `timeout_s` (default 600), and `result_path`. It copies `--prompt-file` to `prompt.md` beside the packet and records `.piper/active.json`. Optional `--check`, `--trust-mcp`. Turn budget defaults to **30**, or **60** when `trust_mcp` is set.

### Wake
Stay attached (`piper dispatch` / `piper run`). Detach only with a wake URL: `--orch-webhook`, task field `orch_webhook`, `LMP_ORCH_WEBHOOK`, or `.piper/orch_webhook` written by `piper_ui`. Resolve it with `piper wake-url`. Do not copy a URL from the panel. Do not poll as the primary wake.

| kind | parent does |
| --- | --- |
| `ask` | `piper answer`. Do not relaunch. |
| `done` | Read the review card. Next slice or stop. |
| `stalled` | Not success. Narrow the brief or stop. |
| `died` | No `result.json`. Tell the user. Do not relaunch blindly. |

---

## 4. Review Card

Use the card from `piper dispatch` or `piper review`. Do not re-read every line when the card is green.

| Check | Pass signal | On failure |
|---|---|---|
| **Status** | `status == "ok"` | `"stalled"` is not done. Read `error` or the card. |
| **Touched files** | `files_touched` ⊆ the brief | Revert the surprise, tighten DO NOT TOUCH, re-slice. |
| **Diff** | Proportional to the slice | Reject a drive-by rewrite. |
| **Acceptance** | `check` exit code 0 | New slice aimed at `test.output_tail`. |
| **Summary** | `result.message` matches the goal | If the card is ambiguous, read the log. |

---

## 5. Failure Playbook

- **Timeout (`exit 2` or `status: "timeout"`):** Smaller slice, or a higher `timeout_s` on the next `piper packet`.
- **`died` / no result:** Kill a stale `lmp_sidecar`, clear the lock, retry the same packet once.
- **Thrash:** `mode` is not on the emitter. For a read-only diagnosis, say so in `prompt.md` and keep the slice to one file. Then a pinpoint edit slice.
- **Model ceiling:** Orchestrator writes the hard logic, then hands tests and boilerplate back to Piper.
- **Irreversible gate paused:** Re-dispatch with `--auto-approve-irreversible`, or `piper answer allow`.

---

## 6. Anti-Patterns
1. **Giant unbounded slices.** "Build the auth service and the frontend" in one task.
2. **Hand-written `task.json`, `answer.json`, or progress lines.**
3. **Silent detach.** Background with no wake URL.
4. **Polling instead of staying attached or receiving the wake.**
5. **Parallel MLX.** IDE Piper and the CLI worker at the same time.
6. **Re-reading every token** when the review card and `check` are green.
7. **Trusting a model "done"** with no `--check`.

---

## 7. First Slice Smoke
1. Model dir exists (`$LMP_QWEN_DIR` or the usual Qwen MLX folder).
2. No competing `lmp_sidecar` (`pgrep lmp_sidecar`).
3. `prompt.md` for a dummy file, then `piper packet`, then `piper dispatch`.
4. Exit `0` and a review card with `status: "ok"`.
5. Delete the smoke artifact. `piper progress`. Start horizon slice 1.
