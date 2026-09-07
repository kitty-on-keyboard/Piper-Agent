# KV reuse + compaction shadow-swap — handoff for Grok (Cursor)

You are improving Piper's context×KV story in `LM_Pipe_2`. Be conservative: **do not change compaction quality policy** until shadow-swap (or another measured path) hides the re-prefill tax. Wrong KV reuse is silent fluent wrongness (S5.10).

## 0. Non-negotiables

- Repo: `/Users/dev/Desktop/seans_projects_local/LM_Pipe_2`
- Branch: `fix/greenfield-write-caps` (dirty tree — **do not clean/stash/reset**)
- Compaction today is **summarize, not silent drop** (S8.3). Keep journal + `context_rehydrate` / `context_recall`.
- Reuse must be planned with `render_with_offsets` / `plan_turn_reuse` — never assume message[0:k] tokens equal a prefix of the full render (generation prompt makes that false).
- Measured fact: after compaction, live KV is invalid → **full re-prefill**, historically ~**zero reuse**, median TTFT ~**43–44s**. Collapsing duplicate reads *outside* compaction was even worse (~59s). Collapses are already queued until `compact_to_budget`.
- Related doc: `piper-bench/HARDWARE_SQUEEZE.md` (shadow-swap is the named next step).
- Separate: `docs/COMMIT_THINK_BLOCK_HANDOFF.md` — do not mix.
- 48GB M5 Pro; one model at a time for realmodel checks.
- `test_spec_cache` currently fails all-pos / final-row identity on A3B and 27B — **out of scope** unless you prove it blocks shadow-swap; file a note, do not silently "fix" tolerances to green.

## 1. What exists today (read before changing)

### Compaction policy (`src/loop/agent.hpp` / `agent.cpp`)

- `kCompactAtPercent = 75`, `kCompactToPercent = 35`
- `compact_to_budget()`: if `prompt_tokens() > 75%` budget → apply pending collapses first → while above 35% and recent > min, `compact_oldest(keep = n-1)`
- `compact_oldest` builds an anchor summary span (tool/path/first line; longer for human steers), journals via sink, erases dropped recent turns
- One `compaction` event per pass (not per dropped turn)

### Prompt layout for KV (`ContextStore::render` in `src/context/context.cpp`)

Stable-first:

1. System (persona, tools, workspace, cross-session recall) — must stay byte-stable across compact  
2. Mission (first user)  
3. Compacted spans (rehydrate hint on **first span only**, not system — landed 2026-09-01 so mission KV survives)  
4. Recent verbatim turns  
5. Live state / pins as implemented  

### KV reuse

- Backend: `MlxBackend` ledger + checkpoint (`plan_turn_reuse`, Extend/Restore/Reset)
- Realmodel proof: `test_kv_reuse_realmodel` (A3B: thousands of reused tokens when prefix stable)
- Diag: `lmp_diag reuse` — earlier sweep reused only 11 tokens because the fixture shared almost no prefix; do not cite as “reuse broken”

## 2. Product goal

**Speed:** hide or eliminate the post-compaction full re-prefill wall without corrupting decode.  
**Quality:** keep extractive anchors + rehydrate; do not switch to aggressive lossy summarization as the first move.  
**Observability:** every turn logs `prefill_reused_tokens` meaningfully; compaction events stay attributable.

Primary deliverable: **shadow-swap** as described in HARDWARE_SQUEEZE:

> Warm a compact KV during tool idle, swap at 75/35. Live cache stays append-only until then.

Secondary (only after shadow-swap GO or explicit NO-GO): tune `kCompactToPercent` with bakeoff/agent evidence (code already says 35% is modelled, not measured — raise first if quality drops).

## 3. Shadow-swap design sketch (investigate → implement)

Hypothesis to verify, not decree:

1. At compaction time (or when high water approaches), build the **post-compact prompt** tokens.  
2. On a **shadow** cache (second ledger/checkpoint or offline prefill into a spare cache), prefill that prompt while the agent is in tool idle / waiting on shell — without blocking the live decode path incorrectly.  
3. When compact commits, **swap** shadow → live (or Restore to shadow checkpoint) so the next turn’s `plan_turn_reuse` sees a matching prefix and reused ≫ 0.  
4. If shadow is stale (steer arrived, tool mutated files that rewrite prompt, mode change), discard shadow and fall back to today’s full re-prefill.

**Correctness bar:** byte-identical next-turn prompt to what render() would produce; reuse plan agrees; `test_kv_reuse_realmodel`-style identity on a fixture that forces compaction.

**Concurrency bar:** do not race live generate with shadow prefill on the same MLX model without a clear lock / queue. Prefer “after tool result, before next generate” single-threaded first.

## 4. Milestones

### M0 — Measurement pass (no behavior change)

Document current numbers on A3B (or 27B if already loaded) for:

- turn-to-turn reuse when no compact  
- first turn after compact (expect ~0 reuse, high TTFT)  
- collapse-only vs compact (already measured historically — confirm still true)

Write results under `piper-bench/results/kv_compaction_baseline.md`.

### M1 — Shadow lifecycle design note

One short design in-repo (comment or docs subsection): when shadow is built, invalidated, swapped; threading model; failure fallback.

GO only if it cannot reuse against a wrong prefix (S5.10).

### M2 — Implement shadow-swap behind a flag

Env e.g. `LMP_SHADOW_COMPACT=0` default off.  
On: attempt shadow; on any doubt, fall back to old path and emit `shadow_compact_fallback`.

### M3 — Proof

- Synthetic: force compact (tiny budget) with flag on → next turn `prefill_reused_tokens` ≫ 0 and TTFT much lower than baseline  
- Identity: same sampled tokens as flag-off path on a fixed seed / greedy smoke if feasible  
- Flag off: zero behavior change  

### M4 — Optional policy tweak

Only if M3 green and quality eval says 35% hurts: try 75/50 or 80/40 with before/after on a small agent suite — **constants first**, not smarter NLP summaries.

## 5. Explicit non-goals (v1)

- LLM-generated compaction summaries  
- Middle-drop KV holes (HARDWARE_SQUEEZE gates this later; hybrid Qwen can go fluent-wrong)  
- Forking MLX for a faster matmul  
- Fixing `test_spec_cache` all-pos numerics unless blocking M2  
- commit_think_block  

## 6. Key code pointers

| Topic | Where |
|---|---|
| 75/35 + compact_to_budget | `src/loop/agent.hpp` (~376), `agent.cpp` `compact_to_budget` |
| compact_oldest + spans | `src/context/context.cpp` |
| render / rehydrate hint | `src/context/context.cpp` render path |
| pending collapses | `agent.cpp` collapse_duplicate_read / apply_pending_collapses |
| reuse plan + checkpoint_at | `agent.cpp` prompt assembly ~1034+, `mlx_backend.cpp` generate/reuse |
| realmodel reuse test | `tests/model/test_kv_reuse_realmodel.cpp` |
| squeeze notes | `piper-bench/HARDWARE_SQUEEZE.md` |

## 7. Report back

1. M0 baseline table  
2. Shadow design (invalidate rules)  
3. Flag + what shipped  
4. M3 numbers: reused tokens + TTFT vs baseline  
5. What you did not change (compact percents, spec_cache, commit_think)

## 8. Why this lines up

Sean felt KV could be used better; compaction is already thoughtful on **quality** but pays a brutal **speed** tax (full re-prefill). Shadow-swap attacks the tax without throwing away extractive compaction. commit_think is a separate decode win.
