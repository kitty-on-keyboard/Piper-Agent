# KV compaction baseline (M0) — A3B

Quiet machine, one MLX process. Flag **off** (`LMP_SHADOW_COMPACT` unset).
Harness: `lmp_diag compact 8` (ContextStore + `MlxBackend::generate`, not Agent).
Target: `/Users/dev/Desktop/Models/Qwen3.6-35B-A3B-MLX-4bit`.

Recorded 2026-09-06. Binary: `./build/tests/model/lmp_diag`.
Compact policy in the agent is still **75 / 35**; this driver rewrites the prompt
with `compact_oldest` / `supersede_stale_copies_of_path` to show the same KV tax.

## Command

```bash
export LMP_QWEN_DIR=/Users/dev/Desktop/Models/Qwen3.6-35B-A3B-MLX-4bit
unset LMP_DRAFT_DIR LMP_SPECULATIVE LMP_SHADOW_COMPACT
./build/tests/model/lmp_diag compact 8
```

Fixture: 12 fat `read_file` turns (~1510-token prompt), greedy, `max_new=8`.

## Table

| Case | prompt tokens | `prefill_reused_tokens` | TTFT |
|---|---|---|---|
| Turn 1 (cold) | 1510 | **0** | 869 ms |
| Turn 2, no compact (stable prefix) | 1536 | **1505** | **284 ms** |
| First turn after `compact_oldest` (8 spans, 5 recent) | 1517 | **0** | 890 ms |
| First turn after collapse-only (1 stale copy superseded, no drop) | 1492 | **0** | 882 ms |

Turn-to-turn reuse without compact matches `test_kv_reuse_realmodel`: thousands-scale
on a larger prefix, here **1505 / 1536**. After compact or collapse the live ledger
is the *old* prompt, `plan_turn_reuse` Resets, reuse is **0**, TTFT returns to the
cold-prefill band.

This fixture is short (~1.5k tokens), so the tax is hundreds of milliseconds, not
the ~43 s median TTFT measured on long real runs. The *shape* is the same: compact
or collapse ⇒ full re-prefill. Shadow-swap (`docs/KV_SHADOW_SWAP.md`) is the
flag-on path that prefills `[0, checkpoint_at)` before the next `generate`.

## M3 — Agent + flag (`test_kv_compact_realmodel`)

Same A3B, 12-turn fat context, 85%-of-prompt budget so `compact_to_budget` fires
between turn 1 and turn 2. Greedy. `max_new_tokens=16`.

| Arm | turn-2 `prefill_reused_tokens` | turn-2 TTFT | sampled text vs flag-off |
|---|---|---|---|
| `shadow_compact=false` | **0** | 2801 ms | — |
| `shadow_compact=true` | **5024** | **37 ms** | **identical** |

Agent prompts include the tools JSON, so they are larger than the diag fixture.
Flag on warms `[0, checkpoint_at)` after compact; generate only prefills the live
tail.

## Not changed

- `kCompactAtPercent = 75` / `kCompactToPercent = 35`
- `test_spec_cache`
- Dual GPU cache
