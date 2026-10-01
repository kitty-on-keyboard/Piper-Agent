# G12: Deterministic run digest on STALLED/FAIL cards from events.jsonl and loop_metrics

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| UNVERIFIED | piper | feature | medium | S |

> **Not yet verified.** Open every cited line below and confirm the problem still holds before you implement. Drop or reshape the task if it does not.

Merged from these finder reports:

- ideas-orchestration#5 Deterministic run digest on STALLED/FAIL cards

## Problem

STALLED (max_turns, no_progress) is a common outcome for a local worker, and PIPER.md tells the parent to 'read what landed'. The card shows status, a slice of the model's own narration, files, diff and test.

It omits result.error and loop_metrics, although write_result already puts loop_metrics in result.json. It also doesn't say:
- whether the in-loop verify_contract ever passed;
- which nudges fired;
- which tool kept failing;
- which checklist items stayed open.

All of this is already recorded as structured events (verification, nudged, tool_result, checklist, run_end). Today the only way for the cloud to learn it is to read an event log that covers every turn and often runs to megabytes.

## Why it matters

4-6 grounded lines replace reading the whole event log for the outcome a local worker produces most often after PASS. That is enough to choose between narrowing the brief, raising max_iterations, or doing the edit in the cloud. It is pure Python and deterministic, with no model call and no cost on green.

## Proposed fix

The hook point is format_review_card, shared by review and dispatch. Only when the verdict is not PASS, read result.log_path once and fold:
- run_end: termination_reason, iterations, unfinished_items;
- verification events: 'check ran N times, passed M, last FAIL', plus a same-fingerprint streak when G04's triage fingerprint exists;
- nudged counts, by why;
- tool_result failures grouped by (tool, error_code), top 3;
- the open items of the last checklist event;
- the last 3 tool calls.

Always print result.error. Add 4-6 digest lines to the card and a digest object to review/dispatch --json.

If a local-model postmortem is ever wanted, it should take this digest as its grounded input.

## Evidence (file:line)

- piper:scripts/piper_worker.py:1006-1018 card lines are task/status/exit/message/files/diff/test/git_diff/verdict, with no error and no loop_metrics
- piper:src/surface/worker.cpp:1117-1126 write_result emits loop_metrics {degenerate_text_count, text_only_turns, tool_error_count, nudged_by_why}
- piper:src/surface/sidecar.cpp:2270-2275 a stall sets error to 'agent did not complete (<reason>)'
- piper:src/loop/agent.cpp:2673 'verification' event {contract, why, ran, passed}; :3385 'nudged' {why, ...}; :3462 'run_end' {termination_reason, ...}
- piper:PIPER.md:171 on stalled: 'read what landed. Do not treat it as success.'

## How the finder suggested verifying it

Here: piper_worker.py self-test scenarios with synthetic events.jsonl fixtures, e.g. stalled at max_turns with repeated no_match tool errors and 4 no_progress nudges. Assert the digest lines and the --json digest object, and assert that the card is byte-identical on a PASS result.

Back to the [index](README.md).
