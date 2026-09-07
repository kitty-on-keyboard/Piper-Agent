# commit_think + follow-ups — next test handoff for Grok (Cursor)

Run this after the ping smokes. Do **not** treat the 26-byte ping as proof of the token win. Do **not** claim a shadow-swap win on a run that never compacted.

## 0. Non-negotiables

- Repo: `/Users/dev/Desktop/seans_projects_local/LM_Pipe_2` (dirty tree — do not clean/stash/reset).
- One model at a time on the M5 Pro 48GB. Prefer **A3B** for Parts A–B; optional **27B** after A3B is green.
- Models:
  - A3B: `/Users/dev/Desktop/Models/Qwen3.6-35B-A3B-MLX-4bit`
  - Dense: `/Users/dev/Desktop/Models/Qwen3.8-27B-MLX-4bit`
- Flags for Parts A–B: `LMP_COMMIT_THINK=1`, `LMP_SHADOW_COMPACT=0`.
- Log a token ledger every run: turns, decode, think, text, tool, `prefill_reused_tokens`, first TTFT, tok/s, wall, whether `write_file` / `commit_think_block` fired, harvest sha vs file sha.
- Related prior results: ping smokes (exact bytes, ~42 tool tokens, A3B turn-1 commit). Shadow compact proof remains the earlier compact run (≈5024 reused / 37 ms TTFT) — not these pings.

## 1. What we already know

| Finding | Implication |
|---|---|
| A3B commit_think works (exact sha, no write_file) | Feature path is real |
| Tool call ≈ 42 tokens for path + `block_id: 0` | Tiny commit is the win mechanism |
| 26-byte ping | Cannot show meaningful token savings |
| 27B sometimes commits with no fence | Needs hard refusal / schema gating — **agent/tool**, not matmul inference |
| 27B Reset + ~8GB reclaim under pressure | Separate runtime/memory issue — **out of scope** for this handoff unless it blocks A/B |
| Pings never hit 75% context | Shadow on/off within noise — expected |

## 2. Part A — fat fence token win (required)

### Goal
Prove decode-once-in-think + tiny commit beats retyping into `write_file`.

### Setup
- Fresh workspace dir (e.g. under `/tmp` or a bench scratch folder).
- A3B, `LMP_COMMIT_THINK=1`, `LMP_SHADOW_COMPACT=0`.
- Prompt (adapt to Piper’s mission style; keep intent):

> Implement `big_ping.py` as a single self-contained Python module of roughly 80–120 lines (real functions, docstrings, a `main`, not filler comments). Draft the **entire** file inside `<think>` as **one** closed ` ```python ` fence. After think, call **only** `commit_think_block` with `path=big_ping.py` and `block_id=0`. Do **not** call `write_file`. Do not paste the file body into any tool argument.

### Pass
1. `big_ping.py` exists; byte sha256 == harvested block 0 sha.
2. `commit_think_block` ran; `write_file` did **not**.
3. Tool args contain only path + numeric `block_id` (0) — **no** source body in the tool call / tool tokens.
4. Fence size ≥ ~2KB (or ≥ ~500 tokens of think that are the fence). Report:
   - fence bytes / approx fence tokens
   - tool tokens (expect ~40–50)
   - estimated savings ≈ fence tokens − tool tokens (vs a counterfactual write_file paste)

### Fail
- Falls back to `write_file` with full body
- Commit args include the source text
- Sha mismatch / partial fence / multiple ambiguous fences written wrong

### Optional control
Same prompt but forbid `commit_think_block` and require `write_file` — compare tool+decode tokens. Only if cheap; not required if Part A already shows tiny tool tokens vs large fence.

## 3. Part B — empty harvest hard-fail (required)

### Goal
Close the 27B footgun: commit with no closed fence must not touch disk.

### Setup
A3B (and 27B if time). Flag on. Prompt that finishes think **without** a markdown fence (plan-only / prose-only), then somehow issues `commit_think_block` on `block_id: 0` (steer or a second user line: “now commit block 0 to `should_not_exist.txt`”).

### Pass
1. ToolError (or equivalent) with a clear reason (no block / empty harvest).
2. Target path **absent** or unchanged.
3. Event/log distinguishes this from sha mismatch.

### Nice-to-have (implement if small)
Omit `commit_think_block` from the advertised tool list when harvest count is 0 for this turn, so the model cannot sample it. Still keep server-side refusal as defense in depth.

## 4. Part C — shadow compact (optional, separate run)

Do **not** mix into A/B.

### Goal
Reconfirm shadow on the path it was built for: **after** `compact_to_budget`.

### Setup
- Force context near/over **75%** of budget (shrink `context_budget_tokens` and/or pad history with large reads).
- Compare next turn **shadow off vs on** (same seed/prompt if possible).

### Pass
- Off: post-compact `prefill_reused_tokens` ≈ 0, high TTFT.
- On: reused ≫ 0, TTFT collapses (order-of-magnitude like prior 2801→37 ms / thousands reused).
- Prefer greedy identity check if already wired.
- Emit/confirm `shadow_compact` (or document absence of fallback).

### Non-pass
TTFT differences on short prompts with no compaction event.

## 5. Out of scope (do not digress)

- Fixing dense 27B memory Reset / 8GB reclaim (file a note only if it blocks A/B).
- Full Aider bakeoff.
- LoRA studio / MTP dual-backend.
- Changing 75/35 compaction percents.

## 6. Report back to Sean

1. Part A table: fence bytes, think/tool/decode, sha match, write_file used?  
2. Part B: error string + disk proof.  
3. Part C if run: reused + TTFT on/off + compaction event present.  
4. Any code fixes you landed (empty-harvest refusal / schema gating) with flag defaults unchanged unless Sean asked.

## 7. Why this test

Ping proved correctness. This proves **value** (fat fence) and **safety** (empty harvest), and keeps shadow measured only when compact actually fires.
