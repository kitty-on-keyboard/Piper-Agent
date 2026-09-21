# Pulse force-only binary — KEEP seed

**Date:** 2026-09-21 (America/Denver)  
**Verdict:** **KEEP-seed** (flag-off). Not product-default. Not a 4-way reopen (4-way stays KILL).  
**Evidence:** `/workspace/research/pulse/force-binary/NOTE.md` — best `p_min=0.55`: acc 26/34 (76.5%), ff 6/34, force R 9/10, stall R 10/10.

## KEEP-seed wording (locked)

Ship behind **`LMP_PULSE=0` default**:
1. **Stage-0** (features, no LLM): stall if `consec≥3` or `streak≥4`; else compact if `prompt_tok≥16000` or `reread_max≥3`.
2. **Else** binary Pulse (2-way letter mask force vs nudge, structured features, strip hindsight): emit **force_tool** iff `p_force ≥ 0.55`, else **nudge**.
3. Never ask the model for stall/compact.

`p_min=0.55` is the seed default (best joint acc/ff/forceR). Document `0.50` as recall-max alt (force R 10/10, ff 9).

## AE brief (flag-off stub)

- Seam: same T1 / #150 adjacency as #193 archaeology; **replace** 4-way questionnaire with binary force/nudge only.
- Wire Stage-0 + `p_min` in harness (deterministic), not only in `lmp_diag`.
- Kill switch: `LMP_PULSE` default **unset/0**; when `1`, run KEEP-seed policy above.
- No merge-as-default. Open PR as `pulse-force-binary` (or reopen #193 narrowed) — Research scores live agent A/B later with Benchbot (bowling seed7): keep if stall≤0 or wall↓≥20% with success≥5/6 and ToolError not up; else revert behavior keep metrics.
- Latency budget secondary (~377ms probe); measure live Extend later.

## Explicit non-goals
- Do not revive 4-way enum.
- Do not train on N=34.
- Do not turn flag on without Mac live A/B KEEP.

## Close / next
Research: KEEP-seed confirmed. AE implements flag-off stub. Benchbot: idle on Pulse until AE tip + Sean green-lights live A/B.

## Amend — A3B transfer (2026-09-21)

Source: `/workspace/research/pulse/force-binary-a3b/` (same frozen 34; 27B-native label caveat).

| | 27B @0.55 | A3B @0.55 | A3B @0.50 |
|--|----------:|----------:|----------:|
| Acc | 76.5% | 70.6% | 73.5% |
| FF | 6 | 4 | 6 |
| Force R | 9/10 | 6/10 FAIL | 9/10 PASS |

**Research amend:** do **not** blind-copy `p_min=0.55` to A3B. Seed defaults:
- **27B (dense):** `p_min=0.55`
- **A3B (MoE):** `p_min=0.50` if that path ever gets Pulse

Primary prove-it remains **27B**; A3B is transfer-only / deprioritized. AE stub: `p_min` keyed by model family (or env override), defaulting as above when `LMP_PULSE=1`.
