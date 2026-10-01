# C16: Keep the failing check's diagnostics through compaction and make context_rehydrate work

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| UNVERIFIED | piper | success-rate | medium | M |

> **Not yet verified.** Open every cited line below and confirm the problem still holds before you implement. Drop or reshape the task if it does not.

Merged from these finder reports:

- Keep the failing check's diagnostics through compaction (one span per trim, anchors from observed bytes, pinned red reason)
- Set first_event_seq on every turn record so context_rehydrate actually works

## Problem

When a slice runs long enough to compact, it loses the output of the failing acceptance check, and the advertised recovery path is broken.
(1) The compaction anchor is first_line(observation). For failed shell results and the operator-check marker, that first line is a header the harness wrote ('[exit N]' or '(operator check `cmd`: FAIL. Output:'), so the compacted line carries no diagnostic.
(2) The live-state pin is just '- FAIL cmd'.
(3) The floor pass shrinks the oldest observation first and does not exempt the check marker. Its stub says 'fetch again', but no tool can fetch an operator check.
(4) compact_to_budget calls compact_oldest(size-1) in a loop. Every dropped turn becomes its own span headed 'turns 1-1', and the whole prompt is re-tokenized once per dropped turn. Harness [Note:] records become a blank '- said:'.
(5) The main TurnRecord, the operator_check marker and harness notes never set first_event_seq. As a result, span headers print 'events 0-N'. The hint tells the model to copy that range into context_rehydrate, which rejects first_event==0 as Malformed. Store::rehydrate filters on first_event > 0, so summarized turns can never be rehydrated. Every test sets first_event_seq by hand, so the Agent path is never exercised.

## Why it matters

Priority 2, on long slices. Worker slices of 30-60 turns at about 2.3k tokens per turn reach the compaction mark. After a trim, all the model knows is 'FAIL cmd', and the advertised recovery (context_rehydrate) returns Malformed or nothing. So it re-runs the check, guesses, or stalls. This restores the one piece of feedback the model iterates on, at a cost of about 100 tokens of live state. It also revives a recovery path that already exists, and removes the per-turn re-tokenizes during compaction.

## Proposed fix

(a) compact_to_budget decides how many of the oldest turns to drop in a single pass (estimate per-record tokens once, or binary-search the keep count). It then calls compact_oldest once per compaction event. Number spans absolutely, starting from turns_recorded() - recent().size().
(b) compact_oldest takes the anchor from the observed bytes and skips harness prefixes: '[exit N]', '[killed…]', '[terminated…]' and '(operator check …: X. Output:'. For error observations, use log_triage::analyze(...).primary_diagnostics[0] (path:line: message) when present, otherwise the first non-header line. This calls log_triage's existing API and does not change the engine. Records with no tool, user text or assistant text are labelled as harness notes or dropped from the span.
(c) In render_live_state, when last_check_ failed, add up to 2 primary-diagnostic lines (about 300 chars) under '# Operator check'.
(d) Exempt the newest operator_check record from floor stubbing.
(e) Set first_event_seq on every TurnRecord the Agent writes. For the primary turn, capture the next event seq before the turn's first emit. Do the same for the operator_check marker and harness notes. The span header then stops printing 'events 0-N', context_rehydrate accepts the printed range, and Store::rehydrate finds the rows. No PCC engine change is needed. Add Agent-level tests that use a ContextJournal and force a compaction.

## Evidence (file:line)

- src/context/context.cpp:57-59: the span header 'Earlier in this run (turns 1-' is computed per call and uses recent_.front().first_event_seq
- src/context/context.cpp:71-73: a record with no tool becomes '- said: ' + first_line(assistant_text), which is empty for harness notes. 94-95: the anchor is first_line(observation, 200)
- src/loop/agent.cpp:2681-2689: the operator-check marker observation starts '(operator check `cmd`: FAIL. Output:'. src/tools/registry.cpp:2159: a failed shell summary starts '[exit N]'
- src/context/context.cpp:333-338: live state pins only '- FAIL <command>'
- src/loop/agent.cpp:2536-2541: compact_oldest(size-1) runs in a loop with a full re-tokenize per dropped turn, so each dropped turn becomes one span
- src/loop/agent.cpp:108-130: floor stubbing does not exempt the operator_check marker; the stub says 'Body dropped; fetch again.'
- src/loop/agent.hpp:403: kMinRecentTurns = 4. 447-448: trims down to 35%. 427: the prompt grows about 2.3k tokens per turn
- src/loop/agent.cpp:2933-2952: the TurnRecord sets only last_event_seq. 3112-3113: only extra calls in a batch set first_event_seq. 2681-2690: the operator_check marker sets no seq
- src/context/context.cpp:245-251: the hint says to copy the printed event range into context_rehydrate. src/tools/context_tools.cpp:202: first == 0 returns Malformed. src/surface/context_journal.cpp:175, 201: rows are stored with first_event = 0. src/pcc/store.cpp:303-304: rehydrate uses WHERE first_event > 0
- tests/surface/test_context_journal.cpp:52 and tests/loop/test_loop.cpp:27, 892 set first_event_seq by hand
- Python replica (scratchpad sim.py): produces 4 spans, all 'turns 1-1', a check anchor ending at 'Output:', and a blank '- said:'

## How the finder suggested verifying it

Tests on the Mac, in tests/loop/test_loop.cpp:
- Build a store with a failed shell turn, an operator_check marker and a [Note:] record, then compact. The span must contain the diagnostic line, not just '[exit' or 'Output:', and no bare '- said:'.
- Run compact_to_budget over 20 turns. Expect one span per compaction event.
- render_live_state with a failed check must include the primary diagnostic.
- Run an Agent with a ContextJournal and a small budget. The span header must not contain 'events 0-', and pcc::rehydrate over the printed range must return a non-empty result.
Then run `agent_eval.py run --include-heavy --only long_context` and failing_test_median with a small budget, and compare how often the check is re-run after compaction.

Back to the [index](README.md).
