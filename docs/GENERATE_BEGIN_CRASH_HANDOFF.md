# Dense 27B silent death at `generate_begin` — handoff

**Status (2026-09-07 ~12:00 MT):** Detached grade-school and fat ~15k both complete `generation` events. Bakeoff/mini-6 stay paused until you decide to resume.

Companion: `docs/GENERATE_BEGIN_CRASH.md`.

## Root cause

The sidecar was not jetsam’d at 38 GB and was not dying inside chunked prefill.

`nohup python3 run_aider_python.py … &` launched from Cursor’s agent Shell stays in that Shell’s job table. When the Shell tool returns, zsh/Cursor reaps the **Python harness PID**. Prefill keeps running (sidecar `start_new_session=True`). The first `sink.on_token` write after `generate_begin` then hits a closed pipe: historically **SIGPIPE**, which looks like a silent vanish at that phase. Foreground `python3 run_aider_python.py` lives because it *is* the still-running tool.

Proof:

- Detached stderr reached `decode_speculative: seeded` / `first_step_done`; no jetsam lines; peak ~20 GB.
- With `SIGPIPE` ignored, an orphan sidecar **finished** decode after Python was already gone.
- Double-fork so the survivor is `ppid=1` (not the job PID): exact `nohup` grade-school **solved** in 98s; fat 15443-token prompt **2× `generation`**, `waitpid 0`.

`LMP_DAEMONIZE=0` when a wrapper script must `wait` on Python (mini-6). Cursor `nohup` does **not** point stdin at `/dev/null` (stdin is already a pipe), so detach also fires when **stdout is a regular file** (`>> log`).

## What changed

- `scripts/agent_eval.py`: `detach_from_launch_session()`, waitpid/signal log, Darwin task policy on the harness.
- `piper-bench/run_aider_python.py`: call detach at `main` start.
- `piper-bench/run_mini6_27b_mtp.sh`: `LMP_DAEMONIZE=0` + self-detach when stdin is `/dev/null`.
- Sidecar: ignore `SIGHUP`/`SIGPIPE`; per-chunk `mx::clear_cache()`; mem logs; optional host QoS/IOPM.

## Acceptance (done)

| Run | How | Result |
|-----|-----|--------|
| `nohup python3 run_aider_python.py --arm piper --run 1 --seed 7 --tries 1 --names grade-school` from Cursor Shell | self-detach, no caffeinate | **solved** 98s, waitpid 0, 6 turns |
| `nohup python3 logs/fat_repro.py` | 15443-token no-tools | **generation×2**, completed, waitpid 0, ttft 43.6s (prefill, not the old 5–6s cache-cap tax) |

Do not resume `run_mini6_27b_mtp.sh` or full polyglot until you choose to.

## Rebuild

Crash-fix binary lives in `/private/tmp/LM_Pipe_2-crash` (`fix/generate-begin-27b-kill`). Copy into this tree’s `build/src/surface/lmp_sidecar` (what bakeoff launches) and `extension/bin/`.
