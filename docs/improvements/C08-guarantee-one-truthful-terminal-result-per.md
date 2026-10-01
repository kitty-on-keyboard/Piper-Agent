# C08: Guarantee one truthful terminal result per run: a single lifecycle owner, visible running/died states, one cancel path for signals and deadlines

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| UNVERIFIED | piper | reliability | high | L |

> **Not yet verified.** Open every cited line below and confirm the problem still holds before you implement. Drop or reshape the task if it does not.

Merged from these finder reports:

- Give each run an on-disk owner so 'died' and 'running' are visible states
- Turn SIGTERM/SIGINT into a truthful result.json instead of 'died' plus an orphaned worker
- Enforce timeout_s inside a turn and give the parent a hard backstop

## Problem

A run only produces a result when the loop exits on its own between turns. Nothing handles the other ways a run can end.
(1) Nothing on disk separates 'running' from 'never started' from 'crashed'. collect_run_status only knows ask, result and idle. During a live run, and after an OOM kill, `piper status` prints 'state: idle', exits 0, and suggests 'next: piper dispatch', which starts a second model if followed. `piper await` spins until its own 600 s deadline and exits 2, which the docs define as a timeout. piper_ui shows the slice as running forever. Non-daemon runs take no lock, so two cold loads can overlap.
(2) The worker installs no SIGTERM/SIGINT handler. Any kill leaves no result.json, and edits that already landed go unreported: the DIED card has no files or diff. With a wake URL, the C++ supervisor waits on the worker but never forwards signals, while Python forwards signals only to the supervisor. The supervisor dies, the worker keeps running with the model loaded, and Python POSTs died. Later the orphan writes result.json and POSTs done or stalled. Python and the supervisor can both POST died for the same death. Signal exits come back as a negative status, so dispatch exits 247. In serve mode, the signal handler calls _exit mid-slice, and the daemon never notices a client hang-up. Under Claude Code, kills are routine: cold load plus the 600 s default timeout plus the check runs longer than a foreground Bash call is allowed to.
(3) timeout_s and the stall detector are checked only before each step. A long turn overruns by its full length, and a turn that never returns is never stopped; the documented restore hang ran about 900 s until SIGKILL. Dispatch's proc.wait() and forward_to_daemon have no deadline. A wedged daemon still answers ping, so later dispatches queue behind it.

## Why it matters

Priorities 1 and 5. Crashes, kills and hangs (the known OS-kill failure mode, plus routine tool-call timeouts) currently show up as idle, timeout, exit 247 or contradictory wakes. Edits that landed go unreported, and the misleading state invites a relaunch that puts two models on a 48 GB machine. After the fix, every run ends in exactly one labelled outcome (ok, stalled, timeout, cancelled or died) with files_touched, and timeout_s actually bounds a slice's wall-clock.

## Proposed fix

(1) Give each run exactly one lifecycle owner, not two layers. At launch the owner writes `<result_dir>/run.json` {task_id, run_id, pid, started_at} and holds an flock on `<result_dir>/run.lock` for the life of the run. When no daemon is alive, it also holds a global `~/.piper/model.lock` so two cold loads cannot overlap. If the worker exits without writing result.json, the owner writes one with status `died`: the exit or signal name, the tail of worker.stderr.log (C04), files_touched from the durable events.jsonl, and the diff (C03). The owner is the only process that POSTs died, and signal exits map to EXIT_ERROR. Recommended placement: make the C++ supervisor unconditional for worker runs (in serve mode, the forwarding client). Python then only spawns, waits and renders, and the duplicate Python and C++ died paths go away.
(2) Use one cancel path in the worker.
- SIGTERM, SIGINT and SIGHUP handlers only set an atomic flag. A watcher thread trips the active slice's CancelToken, and trips it again after run_loop's reset.
- Agent::run arms a deadline at start + wall_clock_seconds that trips the same token with deadline_hit_. termination_reason becomes wall_clock, which maps to timeout, exit 2.
- No token or tool progress for stall_seconds cancels with stalled_no_turn.
- execute_task_packet then writes result.json (cancelled or timeout) with files_touched and diff, and records the check as not run with a reason.
- The supervisor forwards signals to its child; a second signal escalates to SIGKILL.
- In serve mode, the in-flight slice's result is finished before exit, and the slice is cancelled when the client's socket reports POLLHUP.
(3) Add a backstop for runs that ignore cancel. The owner waits timeout_s plus a grace period, then sends SIGTERM, then SIGKILL, and writes status timeout if no result exists. forward_to_daemon gets the same deadline and kills a wedged daemon through its pid file.
(4) collect_run_status is shared by status, await, review and piper_ui. Precedence: result, then ask (only while the lock is held), then running (lock held), then died (run.json present, lock free, no result), then idle. await keeps waiting while the run is running and reports its own deadline as a distinct state, not exit 2.

## Evidence (file:line)

- scripts/piper_worker.py:2088-2150: collect_run_status only knows ask, result and idle; 2178-2196: the idle card says 'next: piper dispatch' and exits 0; 2243-2251: await's own deadline prints 'timed out still idle' and returns EXIT_TIMEOUT
- scripts/piper_worker.py:2573-2602: Popen(cmd); _forward_sig signals only the direct child; proc.wait() has no timeout; died is POSTed only when a webhook is set; `return ret` turns sys.exit(-9) into exit 247
- src/surface/sidecar.cpp:2701-2726: the attached+webhook supervisor calls waitpid(child) without forwarding signals, and POSTs died itself, duplicating the Python POST
- src/surface/sidecar.cpp:2413: the worker run path only ignores SIGPIPE. 2371-2376: handle_daemon_sig does `listener->stop(); ::_exit(128 + sig);`. 1972: the slice's CancelToken is local. 994: run_loop resets it. 2363: client_fd is never polled for hang-up
- src/loop/agent.cpp:2817-2836: wall-clock and stall are checked only before step(); 2864: last_progress is set after step(). src/loop/turn.hpp:265-269: a legitimate turn can take about 700 s
- src/surface/worker.hpp:10-11 claims the sidecar owns 'the watchdog', but no watchdog exists in src/
- docs/RESTORE_PREFILL_DECODE_HANG.md:7-13: prefill_done, then silence until the harness SIGKILLed it at about 900 s
- src/surface/socket_reader.cpp:83-88: a busy ping still counts as alive; 152-199: the forward read loop has no deadline
- src/model/mlx_backend.cpp:634, 944, 1178: cancel is checked per speculative block, per prefill chunk and per decode token, so a deadline cancel takes effect quickly
- scripts/piper_worker.py:980-990: the DIED card has no files or diff, although sidecar.cpp:1935-1941 writes events.jsonl live. scripts/piper_ui.py:146-151: shows 'running' whenever live.jsonl exists without a result
- Repros (scratchpad r2b and the timeout repro): live.jsonl and events.jsonl with no result gave status 'idle' with exit 0, and `await --timeout-s 1` exited 2. A fake sidecar sleeping 6 s with timeout_s=1 made dispatch return after 6.13 s with 'verdict: DIED' and exit 1

## How the finder suggested verifying it

Python side, here, using fake C++-path workers (non-.py executables):
- A worker that holds run.lock and stays alive: `piper status` reports running, and a second dispatch is refused.
- A worker that writes a write event to events.jsonl and then kill -9s itself: dispatch exits 1 (not 247), status reports died with exit 1, and await returns promptly.
- A worker that ignores timeout_s: dispatch returns within timeout_s plus grace, with exit 2.
C++ side, on the Mac:
- SIGKILL the worker child: the owner writes a died result.json with files_touched.
- With a wake URL, `kill -TERM` the parent mid-run: exactly one cancelled/stalled event, result.json with files_touched, and no lmp_sidecar left running.
- A ScriptedBackend whose generate blocks until cancel: the run ends at the deadline with termination_reason wall_clock.

Back to the [index](README.md).
