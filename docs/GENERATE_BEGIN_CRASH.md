# Sidecar silent death at `generate_begin` (dense 27B)

Status: **detached generate_begin vanish is explained and fixed** (2026-09-07 noon). Do not resume mini-6 / full polyglot until you choose to.

## What it looked like

- Events reached `phase: generate_begin`
- Process vanished: no `generation`, no `run_end`, no waitpid on the harness
- stderr often only `dense concat: …` on older binaries
- Foreground agent-Shell smokes **passed**; `nohup … &` (mini-6, bakeoff) **died** at the same boundary

## Actual cause

Not 38 GB jetsam and not an uncaught C++ throw (`MlxBackend::generate` already catches).

Cursor’s agent Shell reaps the **Python job PID** when a `nohup … &` command returns. The sidecar is in its own session, so prefill finishes (~16–21 GB). The first token notify after `generate_begin` writes to a closed stdout → **SIGPIPE**. That is the vanish.

Foreground Python lives because it is the blocking tool, so stdout stays open.

## Fix

1. Harness `detach_from_launch_session()`: double-fork when stdin is `/dev/null` **or** stdout is a regular file (`>> log`). Cursor nohup keeps piped stdin, so the file check is what fires. `LMP_DAEMONIZE=0` if a parent script `wait`s on Python.
2. Sidecar ignores `SIGHUP`/`SIGPIPE`.
3. Prefill-chunk `mx::clear_cache()` kept (memory headroom; not the detached trigger).

## Acceptance runs (2026-09-07)

| Case | Prompt tokens | Result |
|------|---------------|--------|
| Detached `nohup` grade-school from Cursor Shell | ~5332 | solved, 98s, waitpid 0 |
| Detached fat no-tools | 15443 | 2× `generation`, completed, waitpid 0, ttft 43.6s (prefill) / 0.65s reuse |

Tiny ~5k TTFT stays in the ~13s prefill band, not the old always-on cache-cap 5–6s tax.

## Resume bakeoff (when you want it)

```bash
cd /Users/dev/Desktop/seans_projects_local/piper-bench
rm -f .run.lock logs/BAKEOFF_PAUSED.txt
# sidecar must be the crash-fix binary in LM_Pipe_2/build/src/surface/lmp_sidecar
nohup bash run_mini6_27b_mtp.sh >> logs/mini6-27b.chain.log 2>&1 &
```
