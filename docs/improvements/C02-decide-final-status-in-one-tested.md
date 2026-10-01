# C02: Decide final status in one tested function so a green check promotes max_turns and never promotes a crash

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| VERIFIED (reproduced) | piper | false-success | high | M |

> **Verified.** The problem and fix below are the skeptic's corrected versions. They replace the finder's original claim where the two differ.

Merged from these finder reports:

- Fix the inverted green-check promotion by deciding final status in one tested function
- Let a green check promote max_turns/stalled runs, as documented
- Fix green-check promotion: it never fires for max_turns/stalled but does fire for backend_error/cancelled
- Promote a green check after max_turns/stalled (status 'stalled' never matches the promotion test)
- Fix the max_turns path: a green check never promotes a 'stalled' run, and the documented max_iterations knob can't be set

## Problem

Confirmed core: two places decide the final status, and they disagree. sidecar.cpp:2270-2275 labels max_turns, stalled and stalled_no_turn as status "stalled". run_check's promotion predicate, is_incomplete_agent_stop (worker.cpp:454-460), only fires on status=="error" plus the "agent did not complete" prefix. Result: max_turns or stalled with a green check stays stalled, exit 1, wake "stalled", card verdict STALLED. That contradicts everything that describes the contract: PIPER.md:59, :99 and :216-218, AGENT_WAKE.md:59 (copied into user projects by `piper init`), SKILL.md:24 (the skill the cloud follows), the comments at worker.hpp:49-53 and :130-132, and the code's own comments at sidecar.cpp:2324 and :2337-2338. In the other direction, the catch-all branch (sidecar.cpp:2280-2284) is promoted to ok, exit 0, wake done, verdict PASS, with the error cleared. P5's motivating case was lt-004 (green check plus max_turns), which still fails. test_worker.cpp:1619-1664 builds the old status:"error" shape, so the gate stays green.

Where the claim is wrong or overstated:
(a) Of the catch-all reasons, only backend_error and ended-with-completed=false can actually happen in worker mode. backend_error comes from prompt over context budget, prompt over the model's max sequence, or an MLX generation error.
  - cancelled is diverted: an ask Timeout is forced to timeout_awaiting_user, and an ask Cancelled sets stopped_on_unanswered_ask, which hits the irreversible-denial branch.
  - loop_exit and no_run_end cannot happen: Agent::run always sets a reason, and the zero-iteration, no-reason case is caught by the start-failure branch.
(b) The message becomes "check passed" only when the model left no finish summary and no narration. Otherwise the narration's first and last lines are kept.
(c) The "check already green before the run" false success is not specific to backend_error. The same weak check gives PASS to a model that calls finish without doing anything, and would give PASS to max_turns under the proposed fix. PIPER.md's own example check `ctest -R test_validator` exits 0 with "No tests were found!!!" (I verified this with ctest 3.28). So surfacing the termination reason is required, not optional.
(d) The P2-before-P5 ordering cannot be checked: git history is squashed into 23c6867. It does not affect the finding.
(e) `piper packet` really has no --max-iterations flag. The C++ load_packet parses the field; the Python one does not.

## Why it matters

Priorities 1 and 2. The status is wrong in both directions on common paths. max_turns (default budget 30) is how a local model usually ends a long slice without calling finish. Those slices pass their check yet come back STALLED, and the cloud re-slices or writes the code itself, costing thousands of output tokens plus a slice of wall-clock. In the other direction, a backend crash or cancel on a workspace whose check was already green reports PASS with no work done.

## Fix to ship

1. Add a pure function to worker.hpp/.cpp: `Finalized finalize_run(const RunFacts&, const TestBlock&)`.
   - RunFacts = {termination_reason, completed, started, irreversible_unanswered, irreversible_detail, timeout_s}.
   - Finalized = {status, error, exit_code, wake_kind, promoted_by_check}.
   - Ordered policy:
     - Not started: error, exit 1.
     - timeout_awaiting_user or wall_clock: timeout, exit 2, never promoted.
     - Unanswered irreversible ask: error, exit 1, never promoted. Put this before the stalled branch; today it comes after.
     - completed or plan_ready: ok. A red check demotes it to error ("check command failed"), exit 1.
     - Promotable allowlist {max_turns, stalled, ended && !completed}: a green check gives ok, exit 0, promoted_by_check=true. Otherwise max_turns and stalled give status "stalled" and ended gives "error", exit 1.
     - Everything else (backend_error, cancelled, loop_exit, no_run_end, stalled_no_turn): error or stalled, exit 1, never promoted.
     - wake_kind is "done" exactly when status is ok.
2. Make run_check only run the command and fill result.test. Delete is_incomplete_agent_stop.
3. In sidecar.cpp execute_task_packet, collect RunFacts, run the check, call finalize_run once, then compose the message from the final status. Take exit_code and the hook kind from the result. Delete the blocks at 2254-2285 and 2324-2330.
4. Add `termination_reason` and `promoted_by_check` to RunResult and write_result (turns is already written). In piper_worker.py format_review_card, print one `loop:` line when the stop was not ended/plan_ready or the run was promoted, e.g. "loop: max_turns after 30 turns; ok because check passed". It costs a few tokens and tells the cloud that a PASS rests on the check alone.
5. Replace test_worker.cpp:1619-1689 with a table test driven by RunFacts: every reason x {no check, green, red} x {irreversible flag}, asserting status, exit, wake and promoted.
6. Docs: PIPER.md:171 and AGENT_WAKE.md:153 should say stalled means "and the check did not pass".
7. Separate small item: add `piper packet --max-iterations N` to emit `max_iterations`. The C++ load_packet already validates it, and PIPER.md:60 promises this knob.

## Evidence (file:line)

- src/surface/sidecar.cpp:2270-2275: is_stalled_termination sets status 'stalled', error 'agent did not complete (<reason>)', exit kExitError
- src/surface/sidecar.cpp:2280-2284: every other incomplete reason (backend_error, cancelled, loop_exit, no_run_end) gets status 'error' plus 'agent did not complete (<reason>)'
- src/surface/worker.cpp:454-460: is_incomplete_agent_stop requires status == 'error' and an error starting with 'agent did not complete'
- src/surface/worker.cpp:1064-1079: a green check promotes only through that predicate, clears error and sets the message to 'check passed'; a red check demotes ok to error
- src/surface/sidecar.cpp:2323-2330: exit code is re-derived from status; 2339-2340: webhook kind comes from status
- tests/surface/test_worker.cpp:1619-1664: fixtures build status 'error' with '(max_turns)' and '(stalled)', a shape execute_task_packet no longer emits
- PIPER.md:59 and :99, src/surface/worker.hpp:50-53 and SKILL.md:24 promise that a green check after max_turns gives ok/done; docs/AGENT_LOOP_WINS_BRANCH_PLAN.md:318 and :321 show P2 and P5 landed as separate changes
- scripts/piper_worker.py:959-968: review_verdict maps status stalled to STALLED whatever test.exit_code is
- src/surface/worker.cpp:1104-1127: write_result writes no termination_reason or completed field
- PIPER.md:60 says 'Raise it … no rebuild', but `piper packet` (scripts/piper_worker.py:2330-2354, 685-733) has no max_iterations flag, even though load_packet parses the field
- Repro (scratchpad repro_check_table.cpp, linked against the real worker.cpp with a mach-o stub): after a green check, max_turns, stalled and stalled_no_turn stay 'stalled'; ended, backend_error, cancelled and loop_exit go from error to ok; wall_clock and timeout_awaiting_user stay timeout

## Skeptic verdict

real=True, reproduced=True, keep=True, root-cause fix=True, already handled=False, violates constraint=False

Reproduction (C++): I compiled the real src/surface/worker.cpp on Linux with my own mach-o/dyld.h stub. The sidecar status blocks were copied verbatim with sed, not retyped (sidecar.cpp:2254-2285, 2309-2321, 2323-2330, 2339-2340), and run for every termination reason against no check, a green check and a red check. Harness: <scratch, not kept>/c02v/harness.cpp.
- After a green check: max_turns, stalled and stalled_no_turn stay stalled (exit 1, wake stalled).
- backend_error, cancelled (no ask flag), loop_exit, no_run_end and ended-incomplete become ok (exit 0, wake done).
- wall_clock and timeout_awaiting_user stay timeout. A start failure and an irreversible denial stay error.

Reproduction (Python): c02v/card_demo.py feeds those result shapes to the real `piper review` and `piper status`.
- max_turns plus green check gives verdict STALLED and status exit 1, even though the card shows test exit=0.
- backend_error at turn 1 plus green check gives verdict PASS with files (none) and diff +0 -0.

Why it slipped through: the decision is inline in execute_task_packet, which needs a loaded model, so no gate test covers it. test_worker (label gate) only checks hand-built status strings that production no longer emits. Nothing else handles this: review_verdict and collect_run_status map stalled to STALLED/stalled whatever test.exit_code says. The Python fallback run_mission (piper_worker.py:1642-1663) is a third copy of this logic and never runs the check at all. It only runs with LMP_USE_CPP_WORKER=0 or a .py sidecar (self-test), so it is out of scope.

Arguments against shipping, and why they fail:
- A band-aid (also accept "stalled" in is_incomplete_agent_stop) would keep the error-string coupling and the backend_error promotion. Don't ship that.
- Promotion itself creates a new false-PASS path when the check is weak. agent.cpp:2641-2654 records the design decision that the operator's check is authoritative and is never required to fail first, so I do not propose a baseline red-first gate. The answer is to promote but say so on the card.
- stalled_no_turn is the hang detector. In worker mode it cannot fire under default settings (stall_seconds 1200 is longer than the default 600s timeout_s), so treat it like a timeout rather than a promotable stop.
- Leave the "timed-out check" semantics (C10) out of this change.
- No hard constraint is touched. The changes are harness-side only and none of the measured engines.

Back to the [index](README.md).
