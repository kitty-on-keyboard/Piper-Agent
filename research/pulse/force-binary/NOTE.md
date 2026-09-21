# Pulse force-only binary NOTE

**Date:** 2026-09-21 (America/Denver)  
**Spec:** `/workspace/research/pulse/FORCE_BINARY_EXPERIMENT.md`  
**Model:** Qwen3.8-27B-MLX-4bit · probe `lmp_diag pulse_binary`  
**Labels:** frozen N=34 · `LMP_PULSE` default off

## Design
Stage-0 feature stall/compact (same as B). Live **2-way** A/B = force vs nudge (shuffled). Residual: force iff `p_force ≥ p_min` else nudge.

## Headline

| Mode | Acc | FF | Force R | Stall R | Dist |
|------|----:|---:|--------:|--------:|------|
| Heuristic-only | 61.8% (21/34) | 0 | 0% | 100% | {'nudge': 19, 'stall': 10, 'compact': 5} |
| Raw Pulse argmax (no Stage-0) | 29.4% (10/34) | 23 | 100% | 0% | {'force_tool': 33, 'nudge': 1} |
| **Best hybrid p_min=0.55** | **76.5%** (26/34) | **6** | **90%** (9/10) | **100%** (10/10) | {'force_tool': 15, 'stall': 10, 'compact': 5, 'nudge': 4} |
| Iter B best (4-way hybrid) | 67.6% (23/34) | 7 | 60% (6/10) | 100% | force13/nudge6/stall10/compact5 |

Mean raw p_force=0.593, p_nudge=0.407.  
Latency p50/p95 ≈ 377/379 ms (shorter than 4-way ~517ms).

## Grid

| p_min | Acc | FF | Force R | Stall R | Gate |
|------:|----:|---:|--------:|--------:|:----:|
| 0.50 | 73.5% (25/34) | 9/34 | 100% (10/10) | 100% (10/10) | PASS |
| 0.55 | 76.5% (26/34) | 6/34 | 90% (9/10) | 100% (10/10) | PASS |
| 0.60 | 79.4% (27/34) | 1/34 | 70% (7/10) | 100% (10/10) | fail |
| 0.65 | 67.6% (23/34) | 0/34 | 20% (2/10) | 100% (10/10) | fail |
| 0.70 | 67.6% (23/34) | 0/34 | 20% (2/10) | 100% (10/10) | fail |

### Confusion (best p_min=0.55)

gold\\pred | force_tool | nudge | stall | compact
---|---|---|---|---
force_tool | 9 | 1 | 0 | 0
nudge | 4 | 2 | 0 | 0
stall | 0 | 0 | 10 | 0
compact | 2 | 1 | 0 | 5

## Success gate
| Gate | Result |
|------|--------|
| Acc≥61.8% OR (acc≥55%∧force≥8/10) | PASS |
| FF≤10 | PASS (6) |
| Force R≥8/10 | PASS (9/10) |
| Stall R≥8/10 | PASS (10/10) |
| Any cell clears all | YES |

## Verdict
**PASS** — force-only binary clears the gate. Candidate KEEP-seed for flag-off product wire (Stage-0 + binary Pulse). Not a reopen of 4-way.

## Artifacts
- `piper-bench/results/pulse-phase1/force-binary/` (preds, scored_best, grid, NOTE)
