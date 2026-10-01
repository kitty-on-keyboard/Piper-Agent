# C13: Make keep-warm the default launch path and carry every per-run option with the run

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| UNVERIFIED | piper | latency | high | M |

> **Not yet verified.** Open every cited line below and confirm the problem still holds before you implement. Drop or reshape the task if it does not.

Merged from these finder reports:

- Make keep-warm the default dispatch path, and make the daemon carry every per-run option

## Problem

Nothing ever starts the keep-warm daemon. `piper dispatch` runs `lmp_sidecar --worker --task`, which forwards only to a daemon that is already alive. Otherwise it cold-loads the ~19 GB checkpoint, respawns every trusted MCP server (Godoer), prefills the full static prefix, and exits, so every slice pays the load again. To avoid that, the orchestrator has to manage `piper serve` itself, which execs in the foreground and has no stop or status command. Two concurrent cold dispatches have no lock between them, which the code itself describes as the 38 GB-on-48 GB case that takes the machine down. The daemon path also has a bug that must be fixed before it becomes the default. The forward request carries only task, jsonl and auto-approve. `--orch-webhook` is applied, and the detach gate runs, only after the forward branch. The daemon then re-resolves the webhook and LMP_DRAFT_DIR from its own environment. So in serve mode a CLI `--orch-webhook` or a client-side LMP_ORCH_WEBHOOK is silently dropped, and a detached parent gets no done/stalled/ask wake.

## Why it matters

Priority 4: removes the model load (tens of seconds to about a minute) and the MCP respawn from every slice after the first. It is also the precondition for C18's cross-slice prefix reuse. Priority 1: closes a silent no-wake, where the CLI or client-env webhook is dropped in serve mode. Priority 5: a single serialized model holder stops two concurrent dispatches from loading two models.

## Proposed fix

This stays within the one-in-process-model constraint: the daemon is the same lmp_sidecar process that hosts both the model and the loop.
(1) Auto-start the daemon in the harness run/dispatch path when none is alive and the caller has not opted out (`--cold` or LMP_NO_DAEMON=1):
- Take an flock on ~/.piper/serve.lock, then re-check that no daemon is alive.
- Spawn `lmp_sidecar --worker --serve --idle-timeout N` with setsid, stdin from /dev/null and output to ~/.piper/worker.log. Use a shorter default idle timeout (900-1800 s) for auto-started daemons.
- Poll ping until it answers. listen is immediate and the model loads lazily on the first run.
- Then use the existing forward path.
- Add `piper worker stop|status`; the stop socket method already exists.
(2) Make per-run state travel with the run:
- The client resolves the webhook (CLI, then task, then client env, then file) and LMP_DRAFT_DIR, and sends both in the forward request (or bakes them into task.json at packet time).
- The daemon applies the request's values and never consults its own getenv for per-run options.
- In worker_main, apply cli_orch_webhook and run the detach gate before the forward branch.
(3) Optionally, stream turn lines back over the socket so an attached parent sees progress.
C08's model.lock covers --cold runs.

## Evidence (file:line)

- scripts/piper_worker.py:1063-1086: cmd_dispatch calls main(run_argv) without checking for or starting a daemon
- scripts/piper_worker.py:2499-2507: `serve` does os.execv in the foreground; build_parser (2256-2330) has no stop or status subcommand
- scripts/piper_worker.py:2569-2584: spawns `lmp_sidecar --worker --task …` and waits
- src/surface/sidecar.cpp:2604-2611: forwards only if a daemon is already alive; otherwise falls through to a cold Session plus execute_task_packet (2729-2731)
- src/surface/session.hpp:9-11: the '~19 GB reload' is what 'makes a follow-up cost a prefill instead of a minute'; sidecar.cpp:838: the load owns the thread 'for tens of seconds'
- src/surface/session.cpp:44-47: two checkpoints at once are '38 GB on a 48 GB host … takes the machine down'; the only flock in the harness is distill's (scripts/piper_worker.py:1233)
- src/surface/session.cpp:176-180: the registry and MCP connections are reused only inside one process
- src/surface/socket_reader.cpp:126-132: the forward request carries only method, task, jsonl and the auto-approve flags
- src/surface/sidecar.cpp:2641-2653: cli_orch_webhook and the detach gate are applied only after the forward branch (noted by the detach finder); the daemon handler at 2557-2575 applies only the auto-approve flags
- src/surface/worker.cpp:410-419: the daemon resolves the webhook with its own getenv; 479-482: LMP_DRAFT_DIR is read by whichever process builds lmp/start
- docs/ORCHESTRATOR_WORKER_VISION.md:101-105: keep-warm is the intended orchestrator loop ('Still one task at a time on 48GB')

## How the finder suggested verifying it

Python half: add a self-test with a fake sidecar that implements the socket protocol (ping and run). Under concurrent invocations, dispatch with no daemon must spawn exactly one serve process (flock). The forward request must contain orch_webhook taken from --orch-webhook or LMP_ORCH_WEBHOOK. On the Mac: run two consecutive dispatches; the second must show no model_status=loading event and ModelLoad elapsed_ms 0. With --orch-webhook pointed at a local listener, the done POST must arrive in serve mode.

Back to the [index](README.md).
