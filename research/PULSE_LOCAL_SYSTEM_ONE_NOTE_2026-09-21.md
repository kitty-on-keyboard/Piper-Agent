# Pulse — local System-One-shaped gates for Piper / Godoer

**Date:** 2026-09-21 (America/Denver)  
**Status:** Research design only. **No AE until KEEP.** Hang fix **#192** stays ahead of any implement.  
**Parent:** Jev brief `/workspace/research/JEV_PIPER_GODOER_BRIEF_2026-09-21.md`  
**Working name (Benchbot):** Pulse — force a tiny Choice/Noul questionnaire via existing grammar + enum masks; **option-token logprobs → P**; same heavy MLX model; no free text; no cloud; no second heavy model.

---

## 1. Name options

| Name | Vibe | Note |
|------|------|------|
| **Pulse** | Between-turn heartbeat gate | Benchbot default — recommend keep |
| **Noul** | Steal TypeSafe’s yes/no type name | Too vendor-colored |
| **Gatelet** | Tiny harness gate | Clear, bland |
| **Snap** | Fast typed snap decision | Easy to confuse with snapshot/KV |
| **Ballot** | Multi-choice vote | OK for Godoer allowlists |

**Recommendation:** ship research under **Pulse**; rename later if product wants softer branding.

---

## 2. What we copy from Jev (gist only)

| Copy | How (local) |
|------|-------------|
| Typed closed answers | Grammar / enum mask: only option token-ids legal |
| Probabilities | Softmax / normalized logprobs over option tokens (one forward or last-logit peek) |
| Fast software-shaped gates | Run **between** agent turns or **before** expensive tools — not as the coder |
| Parallel multi-question | Pack K Choice/Noul items in one constrained decode (or K tiny masked steps sharing prefix KV) |

## 3. What we refuse to copy

| Refuse | Why |
|--------|-----|
| TypeSafe / any cloud API | Local-first; code + Godot state stay on Mac |
| Replace 27B / A3B generative path | Pulse is a **gate**, not the editor / CoT / MTP drafter |
| “Can’t hallucinate” = truth | Schema ≠ correctness — same honesty as Jev brief |
| Second heavy model | One-heavy doctrine; Pulse uses the **already loaded** weights |
| Vendor 193× KPI theater | Kill bars are Piper wall / false-gate rate, not TypeSafe Pareto charts |
| Free-text “explain why” | That reintroduces strings and wall; Pulse answers are enums only |

---

## 4. Seam map (existing Piper)

| Need | Seam (scan / tip) | Pulse use |
|------|-------------------|-----------|
| Closed token sets | `TurnGrammar` + parsephony `ToolCallGuard` / enum masks (`grammar.hpp`, PR #149 museum) | Mask decode to `{opt_0…opt_n}` only |
| Forced structure | Prefill / response prefix (“Answer with exactly one of: …”) + mask | No prose escape |
| Logits / logprobs | MLX generate path / speculative verify logits (same stack as MTP accept) | P(option) = normalized mass on option ids |
| When to fire | `agent.cpp` loop — after TextOnly / `degenerate_text` / before shell·write·MCP·Godoer mutate | Insert **Pulse turn** (no tools in mask) |
| Journal | `events.jsonl` — new `pulse` event: questions, chosen, P, latency_ms | Bakeoff + kill bar |
| KV | Prefer **Extend** on shared prefix; Pulse prompt should be suffix-stable | Avoid Reset storms (E6: bowling already Restore-healthy) |
| Godoer | Action allowlists as Choice sets | Before engine mutation |

**Not required:** new architecture, RLCD, cloud keys, second PLD.

**Risk:** Using the generative model as its own gate can be **self-consistent but wrong** (same failure mode as LLM-as-judge). Kill bars must use **labeled** turns / human or deterministic oracle, not “agrees with itself.”

---

## 5. Smallest prove-it (Mac OK, **API-free**)

### Offline / CPU-first (preferred start)
1. From frozen journals (`piper-bench-seed7` + ThinkCap / reasoning-ab events), build **20–40** labeled moments: after degenerate / TextOnly / pre-shell — gold ∈ `{force_tool, nudge, stall, compact}`.
2. Script a **logit-only** probe if dumpable; else go Mac tiny.

### Mac micro (same loaded 27B or A3B — one heavy)
1. Load product model once.
2. For each labeled state: build Pulse prompt + enum mask over 4 options; one constrained step (or greedy among options); record argmax + P vector + ms.
3. Metrics: accuracy vs gold, ECE/Brier optional, false-stall rate, false-force_tool rate, p50/p95 latency, TTFT of gate only.
4. Compare to **current heuristic** (#150 nudge rules / stall detector) — Pulse must beat heuristic on a joint score (e.g. accuracy↑ or same accuracy with fewer harmful false-stalls).

**Godoer twin (optional second):** 15 allow/deny labels on recorded actions; same mask machinery.

---

## 6. Kill bars

**KILL Pulse (keep heuristics only)** if any:
- Accuracy ≤ heuristic on the labeled set (no lift), **or**
- False-stall or false-deny rate exceeds agreed harm threshold, **or**
- Gate latency ≥ meaningful fraction of a normal tool turn (e.g. competes with decode wall we’re trying to save) without quality lift, **or**
- Forces frequent KV Reset / busts prefix (TTFT tax cancels savings), **or**
- Model escapes mask / emits free text under Pulse grammar (shape failure).

**KEEP → AE stub** only if: clear lift vs heuristic on labels **and** latency acceptable **and** journaled `pulse` events are stable. Then behind `LMP_PULSE=0` default off.

---

## 7. Priority vs live 27B work

1. **#192** MTP hang after large Restore — land / verify first  
2. ThinkCap = already **KILL** (leave 8192)  
3. QuantizedKV long-context = still valid when memory binds  
4. **Pulse** = research spike after #192; do not block hang fix  

---

## 8. One-line pitch for coworkers

“Jev’s useful idea is typed probabilistic gates in software. **Pulse** does that on-device with the model we already run — grammar-closed choices + logprobs — and never pretends that closed sets equal truth.”


---
**Superseded for implement detail by:** [`PULSE_IMPLEMENTABLE_NOTE_2026-09-21.md`](./PULSE_IMPLEMENTABLE_NOTE_2026-09-21.md)
