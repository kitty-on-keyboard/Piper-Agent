# G03: Enforce one model where the weights live: delete the mlx_lm cold path, make cold mean starting the one daemon, and hold a kernel lock for the model's lifetime

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| VERIFIED (reproduced) | both | safety | high | M |

> **Verified.** The problem and fix below are the skeptic's corrected versions. They replace the finder's original claim where the two differ.

Merged from these finder reports:

- distill-piper#2 Cold path runs a second inference stack and orphans its loader
- distill-godoer#1 Auto-distill can load a second model (orphaned cold loader, GODOER_AUTO_DISTILL mapping)
- service-design#3 Daemon answers status while busy (kernel model lock)
- distill-godoer#6 Godoer duplicates Piper's model/python discovery (mlx_lm and python discovery)

## Problem

Every cited file:line checks out, and the candidate's reproduction holds when rerun independently. Four corrections and additions.

(1) The orphaned loader is not the most likely way to get two models. The more likely path is a busy daemon. The C++ serve loop handles one connection at a time, so both 1 s ping probes (Godoer is_piper_daemon_alive, distiller.py:136, and Piper _is_daemon_alive, piper_worker.py:1118) report a live daemon in the middle of a slice as offline. With GODOER_AUTO_DISTILL=1 (or force), the cold path then loads mlx_lm next to the daemon's resident weights. Reproduced through Godoer's real auto_distill: the probe returned False in 1.0 s, a cold LOAD started at 3.5 s while the daemon still held its model until 12 s, and the result was status=ok.

(2) On lock contention the cold path returns rc 0 without writing --out. Godoer never clears .godoer/distilled_diagnosis.json, so the previous crash's card is shown as status ok for the new crash (reproduced).

(3) The IDE stdio sidecar (extension/src/client.ts:87 leads to lmp/load_model) and the LMP_USE_CPP_WORKER=0 path also load weights with no check at all. All product loads funnel through surface::load_model (session.cpp:25), so that is the right enforcement point.

(4) The DaemonListener::start probe-then-unconditional-unlink (socket_reader.cpp:218/226) is real by reading, and is_daemon_alive itself unlinks on connect failure (:62-64, :72). The macOS ECONNREFUSED-on-full-backlog trigger cannot be reproduced on Linux.

Also: find_local_model IS used by auto_distill (:217). distiller.py:264 `diag_data.get("model", Path(model_dir).name)` evaluates Path(None) eagerly, so every successful distill becomes status "error" when no model dir is found. That is why tests/test_telemetry_distiller.py::test_auto_distill_daemon_online_invokes_distill fails in this container (4 passed, 1 failed). Step 3 of the fix, which deletes find_local_model, must fix that line.

## Why it matters

This breaks Piper's hard one-model constraint through the very mechanism documented as the protection, and Godoer's own timeout makes the orphan case easy to hit. On a 48 GB Mac the result is a crashed machine, not a slow one. Cold diagnoses also disagree with warm ones, because they run on a different stack.

## Fix to ship

Ship in order 1 → (G01) → 2 → 3.

1. Piper C++, the kernel model lock.
- New src/platform/model_lock.{hpp,cpp} with a process-wide ModelSlot singleton.
  - It opens $LMP_MODEL_LOCK or ~/.piper/model.lock with O_RDWR|O_CREAT|O_CLOEXEC.
  - It takes flock(LOCK_EX|LOCK_NB) and writes {pid, role: daemon|ide|run, socket, model_dir, since}.
  - acquire() is idempotent within the process, because a second fd self-conflicts. The kernel frees the lock on any death.
- surface::load_model (session.cpp) acquires after the tokenizer loads and before session.backend.reset() / any weight read. When busy it returns ModelLoad{false, "busy: pid N (role) holds the model"}.
  - The IDE sees this as model_status failed and the sidecar stays alive.
  - worker run writes it to result.json.
  - unload_model releases unless the daemon pinned the lock.
- DaemonListener::start acquires and pins first. If busy, it refuses without touching the path. If acquired, any existing socket is stale by definition: unlink, then bind.
- Delete the unlink-on-failure branches in is_daemon_alive (socket_reader.cpp:62-64, 72).
- worker_main's run path decides residency from the holder record:
  - held by the daemon: forward and queue (busy is not absent);
  - held by another role: fail fast naming the holder;
  - free: load.
- test_worker.cpp:
  - a forked child holds the lock, and both `worker run --no-daemon` and a second serve refuse before load;
  - the lock fd has FD_CLOEXEC;
  - SIGKILL of the holder frees the lock.

2. Piper Python, cmd_distill.
- Delete 1202-1238 and 1309-1327: the mlx_lm path, the /Users/dev literals and /tmp/piper_distill.lock.
- Read the holder from the lock:
  - held by the daemon: submit over the socket via G01's method and wait, bounded.
  - held by another role: write {status: busy, holder} to --out and return EXIT_ERROR.
  - free with --allow-cold: Popen([resolved_sidecar(), '--worker', '--serve', '--idle-timeout', '300'], start_new_session=True, stdin=DEVNULL, stderr=~/.piper/worker.log), wait for a ping with a deadline, then submit. Weights load once.
  - free without opt-in: skipped_daemon_offline.
- Resolve model_dir via resolve_emit_model_dir (--model-dir / LMP_QWEN_DIR) only for a cold start. Never ask a resident daemon to swap checkpoints.
- Fix PIPER.md:120-126 and SKILL.md:15/82/136 (both the .cursor and .agents copies): describe model.lock, and remove the nonexistent --input/stdin.

3. Godoer (distiller.py, telemetry/__init__.py, tests).
- Delete _DEFAULT_MODELS, _DEFAULT_MLX_PYTHONS, find_mlx_python, find_local_model and is_piper_daemon_alive. Piper owns residency; GODOER_AUTO_DISTILL=1 or force only adds --allow-cold.
- Change :264 to `diag_data.get('model') or ''`.
- Unlink out_file before launching.
- Report ok only when the report status is 'diagnosed'. Surface busy/skipped with the holder message.
- Launch with start_new_session=True and use runner.stop_process_group on timeout. The spawned daemon is setsid'd, so it survives on purpose.
- Add tests: stale-out file, busy holder, and no model dir.

## Evidence (file:line)

- piper:scripts/piper_worker.py:1202-1209 a hardcoded /Users/dev 27B model with a 35B-A3B fallback; :1211-1225 probes sys.executable, '/Users/dev/.local/share/godoer-venv/bin/python' and python3 for mlx_lm
- piper:scripts/piper_worker.py:1311-1319 inside `for inc in incidents:`, a fresh `python -c load(...); generate(...)` per incident via subprocess.run, with no timeout and no process group
- piper:scripts/piper_worker.py:1227-1238 flock on /tmp/piper_distill.lock, taken only when the daemon is offline and held by the parent; on contention it returns EXIT_OK without writing --out
- godoer:godoer/telemetry/distiller.py:166 timeout=180.0 and :233 subprocess.run(cmd, ..., timeout=timeout), which kills only the direct child; :22-29, :40-42 and :85-101 export find_mlx_python and _DEFAULT_MLX_PYTHONS, which auto_distill never uses
- piper:src/surface/session.cpp:45-48 'the peak is two checkpoints at once: 38 GB on a 48 GB host ... takes the machine down ... never run two MLX processes at once'
- piper: a grep of src/ for flock/lockf finds nothing; src/surface/socket_reader.cpp:217-226 start() probes, then unlink()s the socket path before bind
- piper:PIPER.md:125-126 '--allow-cold only for explicit single-shot runs', and a lock 'ensuring only one generation process can execute on the system at a time'; .cursor/skills/piper-orchestration/SKILL.md:128 anti-pattern 5 is 'Parallel MLX. IDE Piper and the CLI worker at the same time'
- piper:src/surface/sidecar.cpp:2404,2435,2475-2476,2508 serve already takes --idle-timeout (default 3600 s)
- Reproduced here (scratchpad repro_cold.py, with a fake mlx_lm whose load sleeps 8 s): 'first distill killed by caller timeout', then '1 orphaned mlx_lm grandchild still running', then a second distill with rc 0. The log shows LOAD start pid=11689 at t=872.2 and LOAD start pid=11719 at t=874.8, before 11689 finished at t=880.2

## Skeptic verdict

real=True, reproduced=True, keep=True, root-cause fix=True, already handled=False, violates constraint=False

EVIDENCE. Every reference was checked.
- piper_worker.py: /Users/dev literals at 1202/1204/1213. The per-incident `subprocess.run([mlx_python,'-c',...])` at 1319 runs inside the loop at 1245, with no timeout and no process group.
- /tmp/piper_distill.lock flock (1227-1238) is held by the parent only, and contention returns EXIT_OK without writing --out.
- Godoer timeout=180 (:166) and subprocess.run (:233).
- session.cpp:45-48. There is no flock/lockf anywhere in src/.
- socket_reader.cpp:217-226.
- PIPER.md:126 makes the false 'only one generation process' claim. SKILL.md:128 is anti-pattern 5.
- --idle-timeout plumbing at sidecar.cpp:2404/2435/2475/2508.
- Callers: godoer cli.py:1833, mcp_server.py:187 and watch.py:637 call auto_distill synchronously on every failed run, so watch mode re-fires quickly after a timeout.

Repros (scratchpad/g03; a scratch wrapper pw.py exec's the real harness with only the lock path redirected):
- expA_orphan.py: the caller timeout kills only piper_worker.py. The loader (pid 16012) is reparented to ppid=1 and keeps holding weights from 0.1 to 10.1 s. A second distill exits rc 0 and LOADs from 3.4 to 13.4 s, so the two overlap.
- Three incidents produce three LOADs in three pids.
- expC_busy.py: busy daemon leads to a second model, reproduced through Godoer's real code.
- expD_contention.py: stale card returned as status ok.
- expE_lock.py: flock facts the design depends on. With O_CLOEXEC, a holder that dies by SIGKILL frees the lock. With an inheritable fd plus a tool-spawned background child (Piper's exec tool forks and execlps without closefrom, sandbox.cpp:660), the lock outlives the holder and names a dead pid. A second open() of the lock file in the same process gets EWOULDBLOCK, so one process-wide handle is required.
- proto/run_proto.py: a lock-holding serve stub plus a socket-only distill client. After a killed caller and a follow-up 3-incident distill, there was 1 LOAD and 1 serve, and a second serve refused with 'busy: {pid, role: daemon}'.

DESIGN CRITIQUE. The fix survives.
- The kernel lock at the weight-load site is the root-cause enforcement of the hard one-model rule. It also closes the IDE+dispatch anti-pattern and G02's misprobe hole, even if those probes stay imperfect.
- Deleting mlx_lm removes a second inference stack. Spawning `serve` keeps the explicit opt-in and gives exactly one generation path.

Gaps in the candidate as written:
- It must specify O_CLOEXEC and a re-entrant process-wide handle.
- Busy/contention must never be rc 0 with no output.
- Residency must be decided by the lock, not by ping. Otherwise a busy daemon still reads as offline: no second load happens, but the result is a spurious busy/skip.
- 'Exit with busy' is wrong for the IDE. It should fail the load via ModelLoad.error and keep the sidecar alive.
- Step 2 must land after G01. Today's warm path runs a full agent loop with auto_approve_exec, writes and irreversible all true in .godoer/, and routing cold through it would widen that.
- distill should not send a model_dir that makes the daemon swap checkpoints.

Not a constraint violation: Apple Silicon, flock and C++20 are all fine, and no measured engine is touched. local_model_can_do_it=true only because the change is deterministic infrastructure and needs no new model capability.

Back to the [index](README.md).
