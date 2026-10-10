# Pulse

**Status:** v1 T1 wire behind `LMP_PULSE` (default **off**). Iteration A prompt/decode (letter codes + shuffle + hindsight strip). Product stays off until Mac labeled set KEEP.

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

After `degenerate_text` / TextOnly where #150 would nudge.

### Iteration A decode shape

- **Neutral** forced prefix (no “text-instead-of-tool” hostility).
- Options presented as **letter codes** `A`/`B`/`C`/`D` with one-line defs.
- **Mask over single-letter token ids** (`encoding=letter`). Journal maps letter → enum (`force_tool|nudge|stall|compact`).
- **Shuffle** of which enum sits under A–D, seeded by turn/inert/prompt size (label id in offline micro).
- **Structured features only** (`consec=… why=…`); hindsight (`next=`, `outcome=`, …) stripped from any probe summary.
- Argmax if `P[choice] ≥ p_min` (default 0.55), else fallback to #150 / stall heuristics.

## Kill bars (Mac labeled set)

| Gate | Kill if |
|------|---------|
| Quality | No lift vs #150/heuristic on labeled set |
| Harm | False-stall or false-force_tool exceeds agreed cap |
| Cost | Gate latency eats claimed wall savings |
| KV | Pulse causes Reset storms / TTFT regressions |
| Shape | Escape from enum mask / free text |

Iteration A success gate (before B): see `research/pulse/ITERATION_A_PROMPT.md`.

## Not in v1

T2 expensive-tool / T3 orch / Godoer mutate gates, MTP / ThinkCap default changes, generative-path replacement.
