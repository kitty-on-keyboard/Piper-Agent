# Compaction floor: a fat tool result sticks

Cursor Grok. Piper only. Do not edit Godoer. Do not stash or reset the dirty
`fix/greenfield-write-caps` tree. Godoer is slicing and compressing `godot_guide`
on its own side. This patch still has to hold for any oversized tool result.

## Symptom

A real Cube_world session on 27B (2026-08-29, journal
`/Users/dev/Library/Logs/LM_Pipe/events.1.jsonl`, run
`r-18d04e814eedb1b8-25f22ccb`, workspace
`/Users/dev/Desktop/seans_projects_local/Games/Cube_world`) rendered a prompt of
51092 tokens after compaction had already run.

Same run, the compaction event:

- `tokens_before` 51308
- `tokens_after` 50965
- `budget_tokens` 46521
- `turns_dropped` 2
- `recent_turns` 4

The next generate prefills all 51092 (`prefill_reused_tokens` 0) and spends
`ttft_ms` 504120. The count is the real tokenizer length of
`ChatTemplate::render_with_offsets` (`Agent` emits `prompt` with
`tokens=task.prompt.size()`). It is not a heuristic and not the typed mission.

Requested context budget was 96000, clamped to applied 46521 because KV would
not fit beside the 27B weights. The rendered prompt then sat above that applied
budget and generated anyway.

## What sticks

Hypothesis, verify and discard if wrong. `Agent::compact_to_budget` in
`src/loop/agent.cpp` trims oldest recent turns while
`recent().size() > kMinRecentTurns` (`kMinRecentTurns = 4` in
`src/loop/agent.hpp`) and tokens are still above the low-water mark. When the
recent window hits 4 it stops, even if tokens are still over the applied budget.

The prefix is never in that loop: system, tool schemas, and the mission. On this
run the first render (`messages=2`, `compactions=0`) was already 29979 tokens.
A later `godot_guide` result (summary about 59k characters) landed inside the
protected window. Compaction dropped two older turns, hit the floor, and left
the fat result in place. That is the stick. A big initial prompt does not get
trimmed, and a fat observation inside the last 4 turns cannot leave until four
newer turns bury it.

The 4-turn floor exists so a run can still see its last observations. That is
the right preference. It is the wrong hard stop when the render does not fit KV.

## Invariant

Do not call generate while the rendered prompt token count is above the applied
context budget (the clamped budget compaction already uses, not the requested
96000).

Recency stays a preference. Fitting the applied budget is the requirement.

Normal path stays as it is: start compacting at `kCompactAtPercent` (75), trim
toward `kCompactToPercent` (35), keep dropping whole oldest turns while more
than 4 recent turns remain. Do not change that hysteresis. The new behavior is
only the case the current loop returns from with `tokens` still above budget
and `recent_turns == 4` (or a single kept tool body larger than the remaining
budget).

## What to do when the floor is not enough

If after the existing turn-drop loop the rendered prompt is still over the
applied budget, shrink tool payloads inside the protected window. Do not drop
the turn shells if shrinking the body is enough. The model still needs to know
which tool it called and on what.

For each oversized tool result, keep the tool name, the path or topic, and a
short stub that the body was truncated and must be fetched again. Drop the
body. Prefer shrinking the oldest fat result first, then the largest, until
`prompt_tokens()` is at or under the applied budget. A single result that alone
exceeds the remaining budget has to be cut even if it is the newest one.

Do not summarize by calling the model. This has to be a local truncate. A
generate-to-compact on an already-over-budget prompt is the failure mode.

If shrinking every tool body in the window still cannot fit (prefix alone over
budget), still do not generate over the applied budget. Emit a distinct event
and fail the turn closed rather than prefilling past KV. That prefix-over-budget
case is real (first render was already ~30k of a 46k applied budget) but the
schema diet is out of scope here. Do not start ripping tool schemas in this
patch.

Shrinking a payload rewrites history the same way dropping a turn does. Set
whatever flag shadow-compact already uses so the next generate does not Extend
a KV prefix that no longer matches (`kv_invalidated_by_compact_` today).

## Events

Keep the existing `compaction` event. Add a field, or a second event, that says
the floor was hit and a body was shrunk: tokens before, tokens after, applied
budget, how many results were truncated, and that recent turn count did not
drop. A silent truncate will look like a lost tool result in the next bakeoff.

## Done when

- A fixture whose last 4 turns contain one tool result larger than the applied
  budget ends with `prompt_tokens()` at or under that budget before generate,
  and the tool name plus a refetch stub are still in the render.
- The existing 75/35 path is unchanged when dropping whole turns gets under the
  low-water mark. Do not compact when already under the high-water mark.
- The Cube_world shape (over budget, recent turns already 4, one fat tool body)
  does not generate at 51k.
- Shadow-compact does not reuse a prefix from before the shrink.
- No Godoer edits. No tool-schema diet in this patch.

