# Second tiny Qwen as typed gate — logistics

**Date:** 2026-09-22  
**Status:** Design note (not KEEP). Same-weights Pulse = NO-KEEP; this is a *new* hypothesis.

## Why not “minimal MTP-style”

Piper already loads a second artifact via `LMP_DRAFT_DIR`, but that is an **MTP draft head** on the dense target (`load_mtp`), not a second causal LM. `MlxBackend::load` is **one full model per process**.

So a tiny gate model is **not** a one-line sibling of MTP. It needs one of:
1. **Second process** (small `lmp_gate` sidecar / mlx_lm server) + RPC from the agent loop — least invasive to MLX residency rules.
2. **Second `MlxBackend` in-process** — requires lifting/relaxing the one-load invariant and careful Metal memory accounting.
3. **Out-of-band batch** (offline only) — fine for labeling, not live.

Recommendation for v0: **(1) second process**, Qwen3-family tokenizer, reuse Pulse journal schema + Stage-0 hybrid.

## RAM (Sean 48GB M-series)

Observed peaks (Pulse micros): 27B ~16GB active; A3B ~19–20GB. MTP head on disk ~253MB (not a full second 27B).

Tiny candidates (MLX 4-bit, public):
| Model | Weights (order) | Role |
|-------|----------------:|------|
| Qwen3-0.6B-4bit | ~0.3GB | Cheapest co-resident gate |
| Qwen3-1.7B / 4B-4bit | ~1–3GB | More headroom for soft Choice |

Context must stay **tiny** (structured features + letter defs, ~100–200 tok) — matches Sean’s “context doesn’t grow” constraint. Do **not** feed full agent transcripts into the gate.

Headroom next to 27B+MTP: comfortable for 0.6B; 4B still plausible if no other heavy resident.

## Code / harness delta (honest)

**Reuse from Pulse archaeology (low cost):**
- Stage-0 feature rules (stall/compact)
- Letter-mask Choice + p_min policy
- Frozen N=34 labels + scoring scripts
- Journal `pulse` / `gate` events

**New work (real cost):**
- Gate process lifecycle (start with Piper, health, kill)
- Prompt builder → gate → parse enum (same as Pulse host text)
- Wire T1 seam to call gate instead of same-weights `pulse_decode`
- Tokenizer: prefer **same Qwen3 vocab** so A/B letter ids stay stable; verify encode(`A`) etc.
- Godoer: separate enum sets (e.g. continue / retarget / bake / stop) — same transport, different questionnaire

**Not free:** dual-Metal scheduling, failure modes when gate dies, cold-start TTFT of the small model.

## How it speeds Piper / Godoer development

| Seam | Today | With tiny gate | Speed win |
|------|-------|----------------|-----------|
| Piper T1 stuck text | #150 nudge / stall heuristics | force vs nudge from *unbiased* small model | Fewer thrash turns on hard tasks *if* live prove passes |
| Piper compact | token thresholds | optional confirm compact vs continue | Less over-compact / under-compact |
| Godoer turn | generative “what next” | closed mechanic/scene verbs | Faster iteration loops; fewer invalid engine commands |
| Eval | human / full agent runs | offline gate accuracy on labeled seams | Cheaper harness tuning before full Aider/Godoer plays |

Win condition is **wall/solve on live tasks**, not offline accuracy alone (lesson from force-binary bowling NO-KEEP).

## Prove plan (bounded)

1. Download Qwen3-0.6B-MLX-4bit; smoke letter-mask Choice latency.
2. Offline: same N=34 host text → tiny model; compare to 27B same-weights binary + heuristic.
3. If offline clears force-recall/ff bars: live bowling seed7 A/B (gate process on vs off), same keep bar as before.
4. Only then: Godoer one enum questionnaire (narrow).

## Non-goals
- Revive same-weights 4-way Pulse
- Fine-tune on N=34 before a live miss analysis
- Large-context gate prompts
