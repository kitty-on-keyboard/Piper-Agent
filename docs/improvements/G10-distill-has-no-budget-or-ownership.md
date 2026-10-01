# G10: Distill has no budget or ownership contract: a caller timeout discards finished diagnoses while the daemon keeps running abandoned work and starts requests whose client already left

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| VERIFIED (reproduced) | both | reliability | medium | M |

> **Verified.** The problem and fix below are the skeptic's corrected versions. They replace the finder's original claim where the two differ.

Merged from these finder reports:

- distill-piper#5 All-or-nothing report; daemon runs abandoned and dead queued requests
- distill-godoer#3 No budget contract: 180s wall vs 300s per incident
- service-design#4 Cancel analysis jobs on client disconnect
- distill-godoer#1 Auto-distill can load a second model (process-group termination on timeout)

## Problem

`piper distill` has no deadline in its contract, and its report is all-or-nothing, so a caller timeout throws away diagnoses that already finished.

**What happens today**
- piper_worker.py cmd_distill runs each incident in turn as a full agent-loop `run` mission with timeout_s=300 (:1245, :1292).
- That limit is soft: the agent loop checks wall clock only at turn boundaries (agent.cpp:2833). The code's own comments measure ~16 tok/s and ~60 s per turn on the 27B.
- The report is written once, after the loop (:1359-1361). `_forward_task_to_daemon` blocks with no timeout (:1150-1155). The temp dir is removed only on the happy path (:1281/:1309).
- Godoer's auto_distill gives the whole batch 180 s (distiller.py:166). On expiry `subprocess.run` SIGKILLs only the direct child, and TimeoutExpired returns status `error` with 0 diagnoses, reading nothing (:233-239).

**Reproduced through the real Godoer and Piper code** (fake daemon mirroring the real serial serve loop):
- 2 of 3 missions finished, yet auto_distill returned `error` with 0 diagnoses.
- 1 temp dir leaked.
- The in-flight mission ran to completion for nobody. The daemon's CancelToken is local (sidecar.cpp:1972) and is cancelled only at :2086/:2091; the non-jsonl sink is a no-op (:1931). SIGPIPE is ignored (:2413), so the daemon survives the dead client.
- Only the in-flight item is orphaned, because the Python client sends one request at a time.

**A harm the candidate missed:** the report is not tied to the incidents it describes, and nothing invalidates it.
- After a timeout, and on the daemon-offline skip path, the previous run's distilled_diagnosis.json stays on disk.
- play_digest.report_digest (`godoer play --report`) then prints it beside the new run's errors as if it were current. Reproduced on both paths.

**Corrections to the candidate**
1. The MCP godot_run reply is not "nothing". from_run still returns the full deterministic run report plus "Piper distill timed out after 180.0s"; what is lost is every diagnosis.
2. Queued requests from dead clients do run (reproduced; MSG_PEEK at pickup detects them), but only for dispatch/run through the C++ forwarder, since is_daemon_alive treats a busy daemon as alive. The Python distill client cannot queue: its 1 s ping (piper_worker.py:1118) reports a busy daemon as offline. Running a slice twice also requires the orchestrator to ignore the documented no-replay contract (socket_reader.cpp:140-142, 203-205).
3. The proposed "skip a queued run that names no orch_webhook" test would be decided on incomplete data. forward_to_daemon does not forward the CLI `--orch-webhook`, and the daemon's load_packet sees only task.json, the daemon's own env and `.piper/orch_webhook`.
4. Cold path: the SIGKILL orphans the mlx_lm grandchild while /tmp/piper_distill.lock is released (reproduced), so a second cold model load can overlap it.

## Why it matters

Multi-incident red runs, where a card helps most, currently deliver nothing, and they burn the one model on work nobody reads, which delays the user's next dispatch. The queued-orphan behaviour reaches dispatch itself. G01's single bounded turn per incident shrinks the window, which is why this ranks below it.

## Fix to ship

Ship it Python-only, in both repos. The point is to make the deadline and the report the contract, with no C++ changes.

**PIPER: scripts/piper_worker.py, cmd_distill**
1. Add `--deadline-s` (default 600) and compute a monotonic deadline.
2. Before any model work, write the report with the existing `write_result()` (:435, tmp + os.replace):
   - `status: running`
   - `source_sha256`: sha256 of the incidents.json bytes
   - `deadline_s`, `incidents_count`
   - one row per incident (`id`, `kind`, `file`, `line`, `error`) with `status: pending`
3. Walk the incidents in file order, which is Godoer's root_causes order:
   - If remaining < floor (~30 s; derive it from max_tokens once G01 lands), mark the item `not_attempted_budget`.
   - Otherwise mark it `running` and rewrite the report.
   - Send the task with `timeout_s = min(300, remaining)`.
   - `_forward_task_to_daemon` takes `timeout = remaining + grace` and returns None on socket.timeout; mark that `timeout`.
   - Map result.json to `diagnosed`, `timeout` or `error`.
   - Remove the item dir in a per-item `finally`, then rewrite the report.
4. A SIGTERM/SIGINT handler raises. The outer `finally` then:
   - marks the in-flight item `interrupted` and pending items `not_attempted_interrupted`;
   - sets the final status: `diagnosed`, `partial`, `failed` or `skipped_daemon_offline`;
   - rewrites the report and releases the lock.
5. Exit 0 whenever a report exists.
6. Cold path: same statuses, with `subprocess.run(timeout=remaining)`.
7. Add self_test cases using a fake unix-socket daemon:
   - deadline reached gives `partial`;
   - SIGTERM mid-item gives `interrupted` and no temp dir;
   - a stale report is replaced at start.

**GODOER**
1. `telemetry/distiller.py`, `auto_distill`:
   - Hash the incidents.json bytes it writes.
   - Pass `--deadline-s max(30, timeout - 15)`.
   - Launch with `subprocess.Popen(start_new_session=True)` and `communicate(timeout)`.
   - On TimeoutExpired, call the existing `godoer.runner.stop_process_group(p)` (SIGTERM, then SIGKILL on the group). This also reaps a cold mlx grandchild.
   - Always read out_file, but only accept it if `source_sha256` matches.
   - Return `ok` (all diagnosed), `partial` (new documented status) or `error`.
   - Drop the eager `Path(model_dir).name` default.
2. `format_distillation_card`: add a `partial k/N` header and one line per non-diagnosed item (e.g. "[INC-003] not attempted: budget").
3. `play_digest.report_digest`: show the distilled card only if its `source_sha256` matches the current `.godoer/incidents.json`; otherwise label it stale. This fixes the stale card on the timeout and offline paths.
4. `tests/test_telemetry_distiller.py`, using a real-subprocess fake piper:
   - partial report on timeout;
   - sha mismatch is ignored;
   - deadline passed is below timeout.

**Validated in scratch** (`proto_distill.py` + `proto_godoer.py` against a fake daemon whose wall clock is checked only at turn boundaries):
- 8 s deadline under a 10 s caller timeout: returned at 8.1 s, not killed, 2 diagnosed + 1 `not_attempted_budget`, 0 leaked temp dirs, no daemon work past the deadline.
- SIGTERM at 6 s: 1 diagnosed, 1 interrupted, 1 not attempted, 0 leaks.

**Defer as separate items**
1. A hard per-token deadline inside G01's bounded analysis job. This removes the one-turn overrun; don't add it to dispatch runs.
2. `forward_to_daemon` should forward the resolved orch_webhook (existing wake-contract bug).
3. Only after (2): optionally skip a queued run whose client is already gone (MSG_PEEK at pickup), writing a cancelled result.json.

Drop `--max-incidents` and the StdinReader-on-client_fd design.

## Evidence (file:line)

- piper:scripts/piper_worker.py:1245 sequential loop; :1292 timeout_s 300 per incident; :1359-1361 the report is written once, after the loop; :1281 mkdtemp, with :1309 rmtree only if the caller survives; :1150-1155 recv loop with no timeout
- godoer:godoer/telemetry/distiller.py:166 timeout=180.0; :233-239 TimeoutExpired returns status 'error' with no diagnoses
- piper:src/surface/sidecar.cpp:1972 the CancelToken is local to execute_task_packet, cancelled only at :2086 and :2091 (irreversible ask timeout/cancel); :1930-1931 the non-jsonl sink is a no-op; :2356-2364 a single final write to client_fd
- piper:src/surface/sidecar.cpp:2533-2580 the request line is read and run with no client-liveness check
- piper:src/surface/transport.hpp:62-90 StdinReader(SpscChannel&, CancelToken&) with start(int fd) works on any fd; src/surface/transport.cpp:208-215 deliver() cancels on a framed lmp/cancel; :260-271 the reader thread closes its channel on EOF
- piper:src/model/backend.hpp:27-33 CancelToken is an atomic flag
- piper:src/surface/socket_reader.cpp:140-142 'Once connected, an uncertain submission is not permission to replay'; :203-205 'task may have run. Inspect its result and workspace before retrying'; PIPER.md:80 a detached run requires a wake URL before start
- godoer:godoer/mcp_server.py:181-193 from_run waits for auto_distill before replying
- Finder repro (scratchpad repro_h4.py): the caller was killed after 10 s, no report was written, 1 temp dir leaked, and the third distill task ran after the caller died

## Skeptic verdict

real=True, reproduced=True, keep=True, root-cause fix=True, already handled=False, violates constraint=False

**Cited lines checked.** I opened every cited reference and all match:
- piper_worker.py: 1150-1155, 1245, 1281, 1292, 1309, 1359-1361
- distiller.py: 166, 233-239
- mcp_server.py: 181-193
- sidecar.cpp: 1930-1931, 1972, 2086, 2091, 2356-2364, 2533-2580
- transport.hpp: 62-90
- transport.cpp: 208-215, 260-271
- backend.hpp: 27-33
- socket_reader.cpp: 140-142, 203-205
- PIPER.md: 80

**Godoer tests.** `python -m pytest tests/test_telemetry_distiller.py` (run with the scratchpad venv's pytest): 4 pass, 1 fails here. The failure is environment-specific: distiller.py:264 evaluates `Path(model_dir).name` eagerly, and `model_dir` is None when the default model paths and LMP_QWEN_DIR are missing. Daemon mode doesn't need a local model dir, so on such a machine every successful distill reports `error`. That line sits in the block this fix rewrites.

**Repro scripts** (all in <scratch, not kept>/g10/):
- `repro_godoer_timeout.py`: real auto_distill calling the real piper distill against fake_daemon.py; also shows the stale card.
- `repro_queued.py`: queued dead-client run, plus the MSG_PEEK skip.
- `repro_cold_orphan.py`: orphaned mlx_lm grandchild while the lock is free.
- `verify_fix.py`: runs the prototype fix.

**Why I keep it.** It is not subsumed by G01, G02 or G13.
- At ~16 tok/s and ~60 s per turn (from the code's own comments), a 165 s deadline gives roughly the first, highest-priority diagnosis without G01, and most of them with G01. Without G01 an item can still overrun by one turn.
- An async design (G13) still needs bounded daemon time (one model shared with dispatch) and partial durability.
- Raising Godoer's timeout instead would be the band-aid.

**Cut from the proposed fix**
- Item 3a (cancel an attached analyze on EOF): no caller in either flow uses it, since Godoer always passes `--out`. The mechanism is also wrong:
  - A StdinReader on client_fd pushes client bytes into RunInbox, which treats them as steering and approvals.
  - StdinReader's EOF path only closes the channel; it does not cancel.
  - run_loop calls `cancel.reset()` right before `agent.run` (sidecar.cpp:994), so a one-shot cancel during setup is erased.
  - The existing cancel method is `lmp/cancel`.
  - If this is ever needed, use a liveness/deadline watcher that keeps re-asserting the cancel.
- Item 3b belongs to G02/G13. Item 3c is the status quo.
- Item 3d (skip queued dead-client runs) is a rare, timing-dependent change to dispatch semantics. It first needs the webhook-forwarding bug fixed: daemon-forwarded runs started with `--orch-webhook` never post done/stalled/ask.
- `--max-incidents` is redundant once there is a deadline and the incidents are already in priority order.
- `--max-tokens` belongs to G01. "Return the card at once" is G13's async delivery.
- Do not turn timeout_s into a hard mid-turn cancel for dispatch runs.

**Related but out of scope**
- A busy daemon makes both the Python and Godoer pings report "offline". With GODOER_AUTO_DISTILL=1 that cold-loads a second model, which breaks the one-model rule.
- PIPER.md:118 documents `piper distill --input` and a stdin mode; neither exists (the code requires `--incidents`).

**Housekeeping.** No repo was modified. Running the cold path touched the harness's hardcoded /tmp/piper_distill.lock (a zero-byte file that likely predates me); I left it in place. PID 6108 (fake_daemon.py on w.sock) is another agent's process, not mine.

**Field note.** `local_model_can_do_it` is N/A: the fix is deterministic plumbing, and the model's per-incident task does not change.

Back to the [index](README.md).
