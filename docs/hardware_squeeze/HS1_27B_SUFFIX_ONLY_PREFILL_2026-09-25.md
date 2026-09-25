# HS1 — Fat-prefix suffix-only prefill (27B retarget)

**Date:** 2026-09-25 (America/Denver)  
**Status:** UNPAUSE · retargeted from A3B → dense **Qwen3.8-27B-MLX-4bit** (Sean via Benchbot)  
**Owner:** Research (this 1-pager) → AE (harness / optional fix) → Benchbot (Mac wall)  
**Supersedes:** `HS1_SUFFIX_ONLY_PREFILL_PLAN_2026-09-24.md` (A3B). Old piper-bench staging docs stay deleted — this research NOTE is the living page.  
**Tree:** LM_Pipe_2 / Piper-Agent · branch resume `hs1-suffix-prefill` or fresh from tip if that branch is stale  
**One heavy only:** `Qwen3.8-27B-MLX-4bit` — **no A3B dual-load**. GJF PR #226 Mac micro waits behind this wall. Pulse/Jev parked.

## Claim (unchanged)

Short suffix, fat live prefix, over and over: turn2 should reuse ~P shared tokens (suffix-only prefill / cache-append), not the ~11-token “reuse_on” failure mode from 2026-09-03.

## Hypothesis

Two-`generate()` live-KV run, shared prefix length **P**, suffix **S=128**: tip `Extend`/`Restore` such that `prefill_reused_tokens ≈ P` (small chat-template delta OK) and turn2 TTFT beats cold full prefill of `P+S` by **≥5%** at **P∈{2048,8192}** (primary). P=32768 stretch only if RAM allows on 48GB with 27B resident.

If tip already meets bars on an honest harness → **KEEP docs only** (no product change); Research opens HS2 QuantizedKV next. If reuse ≪ P → **fix** planner / ledger / cache-append behind `LMP_SUFFIX_PREFILL` default **0**.

## Arms

| Arm | Meaning |
|-----|---------|
| **B0** | Current tip, live KV, honest `lmp_diag reuse` (or `suffix`) |
| **T1** | Only if B0 fails: minimal fix; `LMP_SUFFIX_PREFILL=1` to enforce suffix-only append after planned shared prefix. Default stays **0**. |

No MLX fork · no GEMM · no middle-drop pages · no MoE · no GJF changes in this branch · no second heavy model.

## Bars (same)

Per P ∈ {512, 2048, 8192, optional 32768}, S=128, max_new=32, runs≥3:

- turn1 / turn2 **TTFT ms**
- turn2 **`prefill_reused_tokens`**, **reuse_frac = reused / P_shared**
- turn2 prefill tok/s · decode tok/s (no meaningful decode collapse)
- `kv_reuse.mode` / `reason` if journaled

**Pass:** at P=2048 **and** P=8192, median reuse_frac **≥ 0.95** **and** turn2 TTFT **≥5%** better than cold full prefill of P+S (same binary, `reset_cache` between cold runs).

## Kill

- After honest harness + ≤1 bounded fix: reuse_frac still ≪ 0.95 at 2k/8k **or** TTFT win &lt;5% with no path without MLX fork / dual KV  
- Reset storms, greedy mismatch vs B0, decode collapse  
- Needs second heavy model or Seatbelt off  

Then KILL HS1 codepath; keep harness; Research pivots (QuantizedKV or GJF micro).

## Benchbot Mac (27B)

```bash
# Abort if another MLX holder is live. One model only.
# Tip env name (confirmed): LMP_QWEN_DIR  — see tests/model/diag_common.hpp
export LMP_QWEN_DIR=/Users/dev/Desktop/Models/Qwen3.8-27B-MLX-4bit
cd /Users/dev/Desktop/seans_projects_local/LM_Pipe_2/build/tests/model
# Binary: ./lmp_diag   (cmake --preset dev && cmake --build --preset dev --target lmp_diag -j8)

# B0 live KV (AE hands these; AE does not burn Mac wall):
./lmp_diag reuse 3 2048 128 32
./lmp_diag reuse 3 8192 128 32
# Optional stretch if RAM OK:
# ./lmp_diag reuse 3 32768 128 32

# Cold contrast at same P+S (full KV reset each run):
./lmp_diag reuse --cold 3 2048 128 32
./lmp_diag reuse --cold 3 8192 128 32
```

Raw logs: `piper-bench/results/hardware_squeeze/hs1-27b-*.txt` when that tree is live (else Research mirror). One-row verdict → `docs/hardware_squeeze/HS1_27B_RESULTS.md`. **GJF PR #226 stays a separate queue.**

## AE slice

1. Resume/refresh **`hs1-suffix-prefill`** for **27B** (not A3B).  
2. Honest `lmp_diag reuse`/`suffix`: two `generate()`, **byte-identical shared prefix of P tokens**, then suffix S; print reused + TTFT. Must not accidentally share ~11 tokens.  
3. Hand B0 commands to Benchbot (AE does **not** burn Mac wall).  
4. If B0 fails: minimal fix + `LMP_SUFFIX_PREFILL` default off + unit test proving shared-prefix reuse count.  
5. Draft PR only; flag off in product.  
6. **Out of scope:** QuantizedKV, GJF (#226 stays separate), MTP Phase B, Pulse, warm LSP docs resurrection.

## Queue note

Mac is one-heavy: **HS1 27B wall first**. GJF short-arg micro (`LMP_GRAMMAR_JUMP_FORWARD=1`, PR #226) stays queued until HS1 finishes or Sean reorders.

## Coordination

- Benchbot: burn when AE says harness green; 27B only.  
- AE: ping Research + Benchbot when harness green + enable/README.  
- Research: on PASS → HS2 QuantizedKV 1-pager; on KILL → reopen GJF micro or QuantKV with new NOTE.
