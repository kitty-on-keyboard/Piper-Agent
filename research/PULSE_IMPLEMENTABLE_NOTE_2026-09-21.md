> **Superseded (2026-09-21):** 4-way T1 questionnaire is KILL. Flag-off KEEP-seed is Stage-0 + binary force/nudge (`docs/PULSE.md`, `research/pulse/PULSE_FORCE_BINARY_KEEP_SEED_2026-09-21.md`).

# Pulse — implementable design NOTE (Research = design lead)

**Date:** 2026-09-21 (America/Denver)  
**Owners:** Research (design) → AE (code only after **KEEP seed**) · Benchbot (Mac prove)  
**Status:** Design complete enough to prove. **No AE code until Sean/Research KEEP-seed.**  
**Priority:** Hang fix **PR #192** (MTP Restore prefill hang) **ahead of Pulse implement.**  
**Parents:** `JEV_PIPER_GODOER_BRIEF_2026-09-21.md`, `PULSE_LOCAL_SYSTEM_ONE_NOTE_2026-09-21.md`

---

## 0. One-liner

**Pulse** = between turns / before expensive tools or Godot mutations, the **already-loaded** heavy MLX model answers a tiny **schema-only** Choice/Noul questionnaire under existing grammar+enum masks; **option-token logprobs → P**; harness branches on (choice, P). No free text, no TypeSafe cloud, no second heavy model.

---

## 1. Name

| Option | Decision |
|--------|----------|
| **Pulse** | **Canonical** for research + future flag `LMP_PULSE` |
| Gatelet / Ballot | Aliases only if product branding wants softer name later |
| Noul / Snap | Reject (vendor-colored / KV-snapshot clash) |

---

## 2. Non-goals (explicit)

1. Not a TypeSafe / Jev API client.  
2. Not a replacement for 27B/A3B code edit, CoT, or MTP draft.  
3. Not “can’t hallucinate” = truth — **closed set ≠ correct judgment**.  
4. Not a second local model / not SSD / not MTP-on-MoE / not second PLD.  
5. Not free-text rationale generation.  
6. Not blocking #192.

---

## 3. Seam map

| Layer | Path / mechanism | Pulse use |
|-------|------------------|-----------|
| Mask / schema | `TurnGrammar`, parsephony `ToolCallGuard`, enum museum (#149) | Decode only option token-ids |
| Logits | MLX last-step logits (same stack MTP verify reads) | Softmax over option id set → P |
| Loop hooks | `src/loop/agent.cpp` after `Outcome::TextOnly` / `degenerate_text` emit / before high-risk tools | Insert Pulse micro-turn |
| Journal | `events.jsonl` | `pulse` event: `when`, `questions[]`, `choice`, `p`, `latency_ms`, `kv_reuse` |
| Prefix | `plan_turn_reuse` / shadow | Pulse suffix must **Extend** when possible |
| Godoer | Piper tools that mutate Godot via Godoer CLI (control plane) | Choice allow/deny/mode **before** mutate tools fire |
| Config | New `LMP_PULSE=0` default **off**; optional `LMP_PULSE_P_MIN` | Kill switch |

---

## 4. Wire sketch (agent loop)

```
on_turn_end(turn) / before_dispatch(tool):
  if !LMP_PULSE: return
  if not pulse_trigger(turn, tool): return   # see triggers
  q = build_questionnaire(context)           # closed options only
  mask = enum_mask(option_token_ids(q))
  # Prefer: single forward with forced prefix + mask over options
  # Alt: K one-token masked steps sharing KV prefix
  (choice, P, ms) = pulse_decode(model, prefix, mask)
  emit pulse{...}
  apply_policy(choice, P)                    # see policies
```

### 4.1 Triggers (v1 — keep tiny)

| Trigger | When | Questionnaire (example) |
|---------|------|-------------------------|
| **T1 Degenerate pressure** | After `degenerate_text` or TextOnly with #150 nudge budget remaining | `next ∈ {force_tool, nudge, stall, compact}` |
| **T2 Expensive tool** | Before `shell` / `write`/`replace` / MCP mutate / Godoer mutate | `risk ∈ {allow, deny, require_confirm}` or Godoer `mode ∈ {inspect, edit, play}` |
| **T3 Orch handoff** (later) | Before worker wake | `admit ∈ {yes, no, defer}` |

v1 prove-it uses **T1 only**. T2/T3 after KEEP.

### 4.2 Policy (v1)

- Argmax if `P[choice] ≥ p_min` (default 0.55, tunable).  
- Else fall back to **current heuristic** (#150 / stall rules) — Pulse must not hard-fail open.  
- Never invent tools; Pulse only **selects among harness actions**.

### 4.3 Godoer control plane (v1.1 sketch)

Before Piper issues a Godoer command that mutates scene/assets/engine:

- Pulse Choice: `{allow, deny}` or `{inspect_only, allow_edit}` with short state = last user goal + command summary.  
- Deny → return ToolError-class harness message (typed), do not shell out.  
- Prove after T1 KEEP; do not block T1.

---

## 5. Smallest prove-it (API-free)

### Phase 0 — Offline labels (Research, box)
1. Mine journals: bowling + siblings + ThinkCap/reasoning-ab events.  
2. Build **N≥30** moments with gold `next ∈ {force_tool, nudge, stall, compact}` (human or strict rule sheet documented in NOTE companion).  
3. Export `pulse_labels.jsonl` under `/workspace/research/pulse/`.

### Phase 1 — Mac micro (Benchbot, one heavy — prefer **27B** per Sean focus)
1. Tip with #192 landed if available (otherwise note hang risk).  
2. For each label: build Pulse prompt + mask; one constrained decode; log choice/P/ms.  
3. Metrics vs gold **and** vs frozen heuristic baseline on same labels: accuracy, false-stall, false-force_tool, p50/p95 ms, Reset count.  
4. **No product loop wire yet** — harness probe / `lmp_diag`-style OK.

### Kill → no AE
- Acc ≤ heuristic, **or** false-stall/force harm above threshold, **or** p95 gate ≥ ~200–300ms without quality lift (order-of-magnitude; tune on first measure — do not invent), **or** systematic Reset/TTFT tax, **or** mask escape (free text).

### KEEP seed (Sean + Research)
- Clear lift on joint score vs heuristic **and** latency acceptable **and** no mask escape.  
- Then Research hands AE the brief in §7.

---

## 6. Kill bars (summary)

| Gate | Kill if |
|------|---------|
| Quality | No lift vs #150/heuristic on labeled set |
| Harm | False-stall or false-force_tool exceeds agreed cap |
| Cost | Gate latency eats the wall savings it claims |
| KV | Pulse causes Reset storms / TTFT regressions |
| Shape | Escape from enum mask / free text |
| Priority | Attempting implement before #192 verify on 27B |

---

## 7. AE build brief (DORMANT — paste only after KEEP seed)

```
Title: Pulse v1 — T1 degenerate gate behind LMP_PULSE (default off)

Do:
- Add Pulse micro-decode helper: forced prefix + enum mask over option token ids;
  return {choice, p_vec, latency_ms} from last-step logits (no free text).
- Hook T1 only: after degenerate_text / TextOnly path where #150 would nudge;
  apply_policy with fallback to existing nudge/stall.
- Journal `pulse` events; unit tests: mask permits only options; escape fails.
- Flag LMP_PULSE=0 default; docs one pager.

Do not:
- TypeSafe client, second model, T2/T3 Godoer yet, change #150 detector,
  MTP/ThinkCap defaults, block on #192.

Prove ref: /workspace/research/PULSE_IMPLEMENTABLE_NOTE_2026-09-21.md
Kill: revert flag path if Mac labeled set fails §5 kill bars.
```

---

## 8. Coordination

| Who | Now | After KEEP |
|-----|-----|------------|
| Research | Labels + kill scoring; design lead | Hand §7 to AE |
| Benchbot | Wait #192; then Phase 1 Mac micro | A/B in loop if wired |
| AE | **Idle on Pulse** | §7 only |
| Sean | KEEP-seed or kill after Phase 1 numbers | Product default later |

---

## 9. Coworker pitch

“Jev’s useful idea is typed probabilistic gates in software. Pulse does that on-device with the model we already run — grammar-closed choices + logprobs — and never pretends closed sets equal truth. Hang fix first; prove on labeled journals before any merge.”
