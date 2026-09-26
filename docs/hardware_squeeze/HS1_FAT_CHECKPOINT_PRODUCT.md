# HS1 — honest fat `checkpoint_at` (plan B, product path)

**Status:** Product wire for the HS1 KEEP  
**Upstream KEEP:** #227 `194de7e` — tip `plan_turn_reuse` already Extend/Restore at fat P
(reuse_frac 1.0) on the honest `lmp_diag reuse` harness  
**Rejected:** plan C / `LMP_SUFFIX_PREFILL` (force-suffix / T1 algebra flag)

## What landed

Measure KEEP proved the **algebra**. Live agent turns must present the same **honest fat
stable prefix** that harness used: `task.checkpoint_at` (and the backend ledger snapshot)
= end of system + tools + persona + workspace + uncompacted shared history — not a
~11-token chat-header stump.

`Agent::step` sets that boundary via `ContextStore::stable_message_count` →
`offsets[stable]` after `ChatTemplate::render_with_offsets` (see `src/loop/agent.cpp`).
Existing `plan_turn_reuse` then Extend|Restore naturally. **No second cache. Never force
Extend past an id mismatch.**

## Flag

| Flag | Status |
|------|--------|
| `LMP_SUFFIX_PREFILL` | **Do not implement / do not ship** |
| `LMP_FORCE_KV_RESET` | Optional A0 contrast only; not required for B |

Honest fat checkpointing is the **default** agent path (no enable switch).

## Footguns (expect Reset — do not fight)

| Cause | Why |
|-------|-----|
| Compact without shadow warm ids | Live ledger no longer matches next prompt |
| `reasoning_brief` / tools guidance / persona change | Rewrites token 0 of the system prefix |
| Mid-prefix rewrite (collapse applied outside compact, etc.) | Id mismatch inside claimed P |
| Model swap / `reset_cache` | Fresh ledger |
| Any id mismatch at or before checkpoint | `plan_turn_reuse` → Reset; forcing Extend would be wrong-answer risk |

## Gate regression

```bash
ctest --preset gate -R 'hs1_agent_fat_history|hs1_checkpoint_at_chat_header|hs1_fat_prefix' --output-on-failure
```

- `hs1_agent_fat_history_plans_reuse_far_past_chat_header` — agent-shaped fat history →
  planned reused ≫ 11 and Restore at honest `checkpoint_at`
- `hs1_agent_header_only_checkpoint_is_the_false_friend` — stump@11 on the same prompts
- `tests/model/test_kv_reuse.cpp` — algebra fat-P Restore + header false-friend

## HS1-LIVE prove (Mac — document only; waits on Research/Sean GO)

**HS2 QuantizedKV is INFRA HOLD (2026-09-25)** — not a Mac wall for HS1. Mac is **free
for HS1-LIVE** when Research/Sean GO. Do **not** wait on QuantizedKV remounts / B1
MTLCompiler deaths.

- **Model:** Qwen3.8-27B-MLX-4bit only  
- **Shape:** ≥4 agent generates, multi-tool edit/Godoer-class, **medium** think — **not**
  bowling ThinkCap  
- **Tip:** post-#227 main (or this B branch)  
- **Pass:** turns ≥2 median `reused_tokens / stable_prefix_tokens` ≥ **0.95** (journal
  `kv_reuse`), and reused **≫ 11**; mode Extend|Restore when prefix should be stable; no
  Reset storm; mission OK  
- **Optional A0:** `LMP_FORCE_KV_RESET=1` if present — TTFT ≥5% worse than B path  
- **Kill:** still ~11 reuse after ≤1 B fix, or forced-Extend wrong-answer bugs

## Cross-links

- [`HS1_27B_RESULTS.md`](./HS1_27B_RESULTS.md) — KEEP numbers + harness  
- [`HS1_27B_SUFFIX_ONLY_PREFILL_2026-09-25.md`](./HS1_27B_SUFFIX_ONLY_PREFILL_2026-09-25.md) —
  living 1-pager  
- [`KV_SHADOW_SWAP.md`](../KV_SHADOW_SWAP.md) — shadow warm after compact  
