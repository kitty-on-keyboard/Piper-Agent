# G02: A busy daemon reads as offline: answer status from a control thread and replace the three hand-written probes with one tri-state client

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| VERIFIED (reproduced) | both | bug | high | M |

> **Verified.** The problem and fix below are the skeptic's corrected versions. They replace the finder's original claim where the two differ.

Merged from these finder reports:

- service-design#3 Daemon answers status while busy; one liveness check everywhere (control thread, tri-state probe, one client)
- distill-piper#1 Busy daemon is misread as offline
- distill-godoer#1 Auto-distill can load a second model (busy-as-offline half)
- service-design#4 Cancel analysis jobs on client disconnect (request-read deadline)
- ideas-godoer#1 One shared analyze path (daemon answers busy while executing)

## Problem

The core claim is true, and I reproduced it with the real code: socket_reader.cpp compiled here, piper_worker.py and Godoer's distiller.py run unchanged. The serve loop at sidecar.cpp:2526-2588 runs on one thread, so it only answers ping between missions.

Measured against a busy daemon (scratchpad/g02/repro_busy.py, driver.cpp):
- piper _is_daemon_alive returns False after 1.00 s.
- Godoer is_piper_daemon_alive returns False after 1.00 s.
- The C++ is_daemon_alive returns True after 2.00 s.
- cmd_distill returns 0 with skipped_daemon_offline and the advice 'Start the warm daemon with: piper worker serve'.
- Following that advice fails: DaemonListener::start says 'is another daemon running?'.

The re-entrant path is real. worker.cpp builds the trust_mcp env. The MCP child inherits HOME, so it probes the same ~/.piper/worker.sock. from_run (mcp_server.py:181-193) appends '[...Piper keep-warm daemon offline... Run piper worker serve...]' to the godot_run reply. Each failing run costs 1 s and leaves one dead connection in the backlog.

The busy-to-cold misread is real. With GODOER_AUTO_DISTILL=1 and a stub mlx_lm, auto_distill cold-loaded a model while the daemon was mid-mission (repro_reentrant.py).

Corrections:
1. Default config does not reach the cold path from the MCP re-entrant path. spawn_env (src/mcp/spawn_env.cpp) passes only allowlisted variables, so GODOER_AUTO_DISTILL reaches the godoer child only if .mcp.json sets it. The default MCP path yields false advice only; a cold load takes a shell or CLI env, force=True or --allow-cold.
2. On Linux listen(5), 6 dead probes fit. The 7th Python connect fails with EAGAIN, and the C++ blocking connect then hangs until the mission ends. The candidate said 'filled after 7'.
3. The macOS path is unverified: a full backlog gives ECONNREFUSED, and socket_reader.cpp:60-66 then unlinks the live daemon's socket and pid file. It fits XNU sonewconn behaviour. Note that each failed `piper worker serve` run on that advice adds another dead probe.
4. The silent-client stall is real, and it also hits an IDLE daemon. Python reads it as offline, C++ as busy.
5. An extra finding not in the candidate: a queued run whose client disconnected still executes later ('ghost run', reproduced). This matters for an abandoned dispatch.
6. The fix overstates what exists today:
   - No 'analyze' daemon method exists (G01).
   - gen_protocol.py emits only C++ and TS.

## Why it matters

Crashes happen while slices run, which is exactly when the daemon is busy. In that window the user's crash analysis reports something false, and with cold loading allowed it threatens the one-model rule. The re-entrant Piper to Godoer to Piper path makes this routine for Godoer slices, which combines the user's two main uses.

## Fix to ship

Ship the core of the candidate and cut the analyze FIFO, the deferral and the codegen.

PIPER C++

1. src/surface/socket_reader.{hpp,cpp}
   - Replace is_daemon_alive with `enum class DaemonState{Absent,Idle,Busy}; DaemonProbe probe_daemon(path, int timeout_ms)`. It sends `ping`, and the new daemon answers ping and status with the same rich payload: {proto, pid, state, job{id, method, started_s}, queue}.
   - Probe rules:
     - Connected, then no reply before the timeout: Busy. This also covers old binaries and a hung daemon.
     - Refused or missing socket: Busy if the daemon lock is held, otherwise Absent.
   - Staleness: the daemon holds flock on ~/.piper/worker.lock for its whole life. A client unlinks the socket and pid file only when it can take that lock. This replaces the unconditional unlinks at :60-66 and :70-73 (share the lock with G03 if it lands).
   - forward_to_daemon:
     - On a {kind:queued} line, print 'piper: queued behind <job> (position N)' to stderr.
     - On {status:busy}, return a clear busy exit. Never fall back to cold.

2. sidecar.cpp serve block, :2504-2597
   - Resolve all paths before starting the threads.
   - Control thread:
     - poll() over the listen fd and the pending client fds. Each client has a 2 s deadline to send its request line.
     - ping/status: answered from a mutex-guarded snapshot.
     - stop: cancel the current job, then exit.
     - run: push {fd, req} onto a bounded queue (4). Reply queued when a job is active. Reply busy when the queue is full, or when the request has if_idle and the daemon is busy.
     - The control thread never calls getenv or setenv.
   - Model loop:
     - Pop a job. Drop it if its client has hung up (recv MSG_PEEK returns 0), which stops ghost runs.
     - Set the snapshot, then call execute_task_packet with a CancelToken& owned by the job. Change its signature to take that token instead of the local token at :1972.
     - Close the fd and clear the snapshot.
     - The control thread flips the token on stop, and on hang-up of a running job's client, matching cold-path kill semantics.
   - Measure idle time from the end of the last job, not from the last accept.

3. tests/surface/test_worker.cpp
   - Replace the case at :1730 with probe_daemon cases:
     - backlog only: Busy
     - refused, lock free: Absent, socket unlinked
     - refused, lock held: Busy, socket kept
   - Add serve tests:
     - ping answered within 100 ms during a held mission
     - a silent client cannot stall status
     - a queued job whose client closed is never executed
     - if_idle while busy returns busy

PIPER PY (scripts/piper_worker.py)
- Replace _is_daemon_alive (:1118-1136) with daemon_state(sock), which maps states the same way as the C++ probe.
- cmd_distill:
  - Absent: today's offline and cold logic. Cold only when the lock is free.
  - Busy: write {status:'skipped_daemon_busy', job, incidents_path, message:'Piper is running job X; incidents saved; rerun piper distill --incidents P when idle'}. Exit 0. Never cold, even with --allow-cold.
  - Idle: forward with if_idle:true. A busy reply leads to skipped_daemon_busy.
- _forward_task_to_daemon parses the first reply line for busy.

GODOER
- Delete is_piper_daemon_alive from distiller.py:136-157 and from the telemetry/__init__.py exports.
- auto_distill always runs `piper distill`, on the failure path only, so green runs are untouched. It passes --allow-cold only for force or the env flag, and branches on the report status:
  - diagnosed: the card
  - skipped_daemon_busy: '[N incident(s) in .godoer/incidents.json; Piper busy with job X; model diagnosis skipped]'
  - skipped_daemon_offline: today's text
- Fix the eager Path(model_dir) at :264.
- tests/test_telemetry_distiller.py: replace the probe patches with fake per-status piper reports. Add a busy case asserting no cold path and an honest summary.

## Evidence (file:line)

- piper:src/surface/sidecar.cpp:2526-2588 single-threaded serve loop. The request line is read with blocking 1-byte reads and no deadline (:2533-2538); ping is answered only after accept (:2543-2548); run executes inline (:2580)
- piper:src/surface/socket_reader.cpp:83-89 the C++ probe returns true on poll timeout, with the comment 'busy, not absent. Returning false would let worker_main load a second model'; :60-66 any connect failure unlinks the socket and pid file; :249 listen(listen_fd_, 5)
- piper:scripts/piper_worker.py:1118-1136 _is_daemon_alive: settimeout(1.0), and any exception returns False; :1186-1200 offline writes skipped_daemon_offline with 'Start the warm daemon with: piper worker serve'; :1203-1241 offline plus allow_cold goes cold
- godoer:godoer/telemetry/distiller.py:136-157 is_piper_daemon_alive is a copy with the same 1 s / False behavior; :203-215 the offline summary says 'Run piper worker serve'; :204 allow_cold = force or GODOER_AUTO_DISTILL=='1'
- piper:src/surface/sidecar.cpp:2510-2514 serve refuses with 'is another daemon running?' while a daemon answers
- piper:src/surface/worker.cpp:548-553 'Godoer workers that pass trust_mcp: [godoer] still get godoer' (the re-entrant path); godoer:godoer/mcp_server.py:181-193 from_run appends the distill summary to the MCP reply
- piper:src/surface/socket_reader.cpp:105-208 forward_to_daemon connects, writes and blocks on read, with no queued/busy signal and no read deadline
- Reproduced here (scratchpad/merge): a listening socket that never accepts makes piper _is_daemon_alive and godoer is_piper_daemon_alive each return False after 1.00 s; cmd_distill then returns 0 with status skipped_daemon_offline
- Finder repros, not re-run: timed-out pings queue and are answered after the mission, and the Linux listen queue filled after 7 probes. The macOS ECONNREFUSED-then-unlink path is unverified

## Skeptic verdict

real=True, reproduced=True, keep=True, root-cause fix=True, already handled=False, violates constraint=False

How I checked:
- Compiled the real src/surface/socket_reader.cpp with g++ -std=c++20 into a driver that replays the serve loop from sidecar.cpp:2526-2588, with the mission replaced by a sleep and SIGPIPE ignored as worker_main does at :2413.
- Ran the real piper_worker.py and Godoer distiller.py against it.
- Godoer tests: pytest is not installed system-wide, so I used a scratchpad venv. Result: 4 passed, 1 failed. test_auto_distill_daemon_online_invokes_distill fails in this container for an unrelated reason: distiller.py:264 evaluates `Path(model_dir).name` eagerly, and find_local_model() returns None here.

History:
- The C++ 'busy, not absent' rule landed on 2026-09-16 in PR #150, with the test busy_daemon_is_not_mistaken_for_absent_worker at test_worker.cpp:1730.
- The Python probe arrived on 2026-09-30 in c69086d without that rule, and Godoer copied it. The two Python copies drifted from a rule the C++ side already learned. Not already handled: nothing answers status while busy.

Arguments against, and where they land:
1. In the re-entrant case, a busy daemon can never diagnose anyway; one model runs one job. The harm is the false advice, the 1-2 s per probe, and dead backlog entries. This is mostly true. But the false advice tells agents and humans to run `piper worker serve`, which adds probes. With the opt-in cold path it loads a second model (reproduced). On macOS the refused-connect unlink chain can lead to a second daemon and model.
2. A two-line fix exists: make the Python probes treat connect-plus-silence as busy, like the C++ probe. That is a band-aid. It leaves the backlog accumulation, the silent-client stall, ghost runs, a stop that cannot interrupt, and busy indistinguishable from hung.
3. The root cause is that status shares the model thread. Extension mode already keeps a reader thread separate from the model (StdinReader, surface/transport.cpp:261). CancelToken is already atomic (model/backend.hpp:27-37). So a control thread fits existing patterns, and it keeps one model on one thread. No hard constraint is violated, and no measured engine under src/ is rewritten.

What should be cut from the candidate:
1. Cut step 2b and the deferral half of step 4: the in-daemon analyze FIFO and deferred_daemon_busy.
   - It depends on G01, and that method does not exist.
   - In the re-entrant case the deferred card arrives after the slice. Its agent is the same model and never sees the card.
   - Each queued incident costs about a minute of 1500-token generation on 27B. That delays the next dispatch.
   - Successive jobs overwrite the same out path, so stale analyses replace each other.
2. Cut the gen_protocol.py Python emitter. One Python client plus contract tests is enough for 4 methods.

Lessons from my Python reference (ref_control.py):
- My first version read requests serially with a deadline. One silent client delayed every probe up to that deadline, and the thread crashed on EPIPE. The control thread must multiplex pending clients with per-client deadlines.
- The multiplexed version measured:
  - Status during a job: Busy in about 1 ms.
  - 20 probes plus a silent client: all Busy within 10 ms, no backlog left.
  - A queued run got {kind:queued, position:1}.
  - The abandoned queued run was not executed.
- The control thread must never call getenv. execute_task_packet calls ::setenv on the model thread (sidecar.cpp:1910-1911).

Impact is high because the trigger is routine: every failing godot_run in a warm-daemon Godoer slice. The trimmed fix is effort M.

Back to the [index](README.md).
