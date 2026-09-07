# Sidecar silent death at `generate_begin` (dense 27B)

Status: **fixed on `fix/generate-begin-27b-kill`** (2026-09-07). Bakeoff stays paused until the operator resumes it. Do not start `run_optimal_27b_seed7.sh` from this note.

## Signal

Historical deaths left **no DiagnosticReports entry** and no C++ exception. That is the signature of **SIGKILL / jetsam**, not `std::terminate`. `MlxBackend::generate` already catches `std::exception`.

This session did not catch a live SIGKILL: after #56/#57, a 20.7k-token dense-27B prefill reached `generation` with `sidecar waitpid: returncode=0`. At the last 2048-token chunk, MLX **active+cache was ~38 GB** — the historical kill band — while peak active stayed ~22 GB. macOS `log show` jetsam and `~/Library/Logs/DiagnosticReports/` were empty after these runs.

`scripts/agent_eval.py` `drive_sidecar` now logs `returncode` and, when negative, the signal name. The next OS kill will print `sidecar waitpid: ... signal=9 (SIGKILL)`.

## Cause

Cold first turn is `ReuseMode::Extend` from an empty ledger (`plan_turn_reuse`: empty is a prefix). The Reset-path `mx::clear_cache()` therefore **never runs** on the prefill that dies.

Each prefill chunk (`LMP_PREFILL_CHUNK` default 2048) leaves dead activations in MLX's allocator cache. They stacked ~2.5 GB per chunk. At ~20k tokens that is ~38 GB (active+cache) on a 48 GB machine. `set_memory_ceiling` used to set the MLX memory valve at the device-recommended working set (~37.4 GiB here), which is **on the jetsam side** of the cliff.

MTP / shadow-compact / commit_think are not required. A3B at ~8.9k also lived; this is shared prefill-cache pressure, worse on dense 27B because KV+activations are larger.

## Fix

1. `mx::clear_cache()` at the end of each **prefill chunk** after KV is synced and last-chunk logits are copied. Not in the decode loop. Not the reverted always-on 4.75 GB cache cap (that one cost 5–6 s TTFT).
2. MLX **memory** limit = recommended working set minus 4 GiB (~33.4 GiB here) so an over-large allocation throws into `generate()`'s catch instead of SIGKILL. **Cache** limit stays at the recommended size.

After the fix, a 20.7k-token chunk ends at ~17.8 GB active and **cache ≈ 0**.

## Acceptance (2026-09-07, crash-branch sidecar)

| Case | Prompt tokens | MTP | Result |
|------|---------------|-----|--------|
| Tiny smoke | 5084 | on | **PASS** — cold ttft 12.7 s (was 12.9 s before reclaim; not the 5–6 s ceiling tax), turn2 0.58 s |
| Affine-cipher try 1 | 6114 → 7966 | on | **PASS** — 4 `generation` events; `waitpid` 0; tools ran (read/list/write/shell). Stopped at `max_turns=4` (not scored) |
| Fat pad | 20812 | on | **PASS** — `generation`; cache 0 after each chunk; peak 22.6 GB |
| Fat pad | 20811 | off | **PASS** (pre-fix even, but cache stacked to 38 GB) |
| Isolation: draft loaded, `LMP_SPECULATIVE=0`, shadow/commit off | 4834 | head loaded | **PASS** — ttft 12.7 s |
| A3B control ~8.9k | 8938 | n/a | **PASS** |

## Resume bakeoff (operator)

```bash
cd /Users/dev/Desktop/seans_projects_local/piper-bench
rm -f .run.lock BAKEOFF_PAUSED.txt 2>/dev/null
# point PIPE at a sidecar built from this branch, then:
# nohup bash run_optimal_27b_seed7.sh >> logs/optimal-27b-seed7.nohup.outer.txt 2>&1 </dev/null &
```
