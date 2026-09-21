# Pulse

**Status:** v1 T1 wire behind `LMP_PULSE` (default **off**). Product stays off until Mac labeled set KEEP (Benchbot / Research).

## What it is

Pulse is a schema-only Choice gate: the already-loaded MLX model answers a tiny closed questionnaire under an enum mask; option-token logprobs → **P**; the harness branches on `(choice, P)`. No free text, no TypeSafe / Jev cloud client, no second model.

## Enable (Benchbot prove)

```bash
export LMP_PULSE=1
# optional confidence floor (default 0.55)
export LMP_PULSE_P_MIN=0.55
```

Unset or `LMP_PULSE=0` → baseline #150 / stall heuristics only.

## T1 trigger (v1 only)

After `degenerate_text` / TextOnly (text-instead-of-tool) where #150 would nudge:

`next ∈ {force_tool, nudge, stall, compact}`

- Argmax if `P[choice] ≥ p_min`, else **fallback** to existing nudge/stall path.
- Does not change the #150 detector; Pulse only selects among harness actions.
- Journals `pulse` events: `when`, `questions`, `choice`, `p`, `p_vec`, `latency_ms`, `kv_reuse`, …

## Kill bars (Mac labeled set)

| Gate | Kill if |
|------|---------|
| Quality | No lift vs #150/heuristic on labeled set |
| Harm | False-stall or false-force_tool exceeds agreed cap |
| Cost | Gate latency eats claimed wall savings |
| KV | Pulse causes Reset storms / TTFT regressions |
| Shape | Escape from enum mask / free text |

Revert / leave flag off if kill bars fail. Hang fix #192 is already merged — do not block on it.

## Not in v1

T2 expensive-tool / T3 orch / Godoer mutate gates, MTP / ThinkCap default changes, generative-path replacement.
