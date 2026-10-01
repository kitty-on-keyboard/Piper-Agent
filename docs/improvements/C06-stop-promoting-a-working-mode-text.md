# C06: Stop promoting a working-mode text summary into a blocking ask_user

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| VERIFIED (reproduced) | piper | bug | medium | S |

> **Verified.** The problem and fix below are the skeptic's corrected versions. They replace the finder's original claim where the two differ.

Merged from these finder reports:

- Stop turning a working-mode text summary into a blocking ask_user

## Problem

The core claim is correct. The cited lines match the code (agent.cpp:2866-2893, turn.cpp:71-73, sidecar.cpp:2111-2212 and 2259-2262, worker.cpp:454-460 and 1064-1075).

What happens in Agent and Debug mode:
- Any text-only turn (not cut for looping) whose text contains a '?' anywhere and has 2 or more lines starting like '1. ', '2) ', 'a. ' or 'Option N:' is rewritten into an ask_user call. The run halts with awaiting_user and the event records promoted_from_text=1.
- Packets run in Agent mode by default: worker.hpp:39 sets mode="agent", and emit_task_packet writes no mode. So this is the default orchestration path.
- The C++ worker then writes awaiting_user.json and polls answer.json until timeout_s, counted from when the run started. Every 5 s it re-prints the whole question (the entire summary) to stderr.
- When the wait expires the result is status=timeout, exit 2. run_check never promotes a timeout. `piper dispatch` waits on the subprocess (Popen + wait) and never answers, so the card says FAIL.

Corrections to the claim:
1. The fix rationale says agent.cpp:3371-3376 tells the model to call finish, or ask_user if blocked. It does not. That working-mode nudge only says "act now" or "call `finish`". ask_user stays reachable through its own tool description and through finish's description ("to raise something you cannot decide alone, call `ask_user`").
2. "An attached dispatch cannot answer" is only strictly true for a foreground run. If the cloud runs dispatch in the background and follows PIPER.md, it can `piper answer`. The cost is then one extra round trip: reading the whole summary as if it were a question, then answering it. A foreground run stalls to timeout_s.
3. The scope is wider than closing summaries. Mid-run narration also triggers it. Think-budget overflow turns are TextOnly and not excluded; only cut_for_looping is. So self-questions like '1. Where is parse() defined?', announced intent containing a Swift `Foo?`, or a blocker message ('1. … 2. … Should Bar.swift be in scope?') can halt a run mid-work. That blocker message is exactly what the vision doc's prompt template asks for ("If stuck … Explain blockers in the final message").

The rest checks out. The motivating measurement was Plan-mode only (the comment, plus the plan-mode run cited in turn.hpp:176-178). For Agent mode, only the no-'?' summary is tested.

## Why it matters

Priorities 1, 4 and 5. Ordinary summary phrasing turns a finished slice into a 600 s stall and a FAIL/timeout card, and the cloud then spends tokens investigating or re-briefing. Even after C05 bounds the stall, each promotion still costs an extra round trip.

## Fix to ship

1. In src/loop/agent.cpp, Agent::run (lines 2876-2882), delete is_question_candidate and gate the promotion on the mode policy:

`const int enum_lines = policy_.conversational && turn.outcome == Outcome::TextOnly && !turn.cut_for_looping ? enumerated_choice_lines(turn.assistant_text) : 0;`

   - Plan mode behaves exactly as today, because conversational is true there.
   - Agent and Debug lose the '?' path.
   - Rewrite the comment to point at ModePolicy::conversational: in a working mode, text is the final answer, and a working run asks with the ask_user tool.

2. Update the doc comment on enumerated_choice_lines in src/loop/turn.hpp:174-179 to say the promotion applies only in conversational modes.

3. Add tests to tests/loop/test_agent_step.cpp:
   - (a) Agent mode: the text turn "Done:\n1. Added X\n2. Added tests\n\nWould you like me to add more edge cases?" followed by a `finish` call. Expect no ask_user TurnRecord, termination_reason=="ended", and iterations==2. This pins the intended working-mode ending, not just the absence of awaiting_user.
   - (b) Debug mode: "1. Made `x` a `Foo?`\n2. Rebuilt". Expect no ask_user record.
   - Keep an_enumerated_text_turn_in_plan_mode_asks_instead_of_nudging and the existing no-'?' Agent test unchanged. No other test depends on Agent-mode promotion; every other awaiting_user test uses an explicit ask_user call.

4. Do not add ask_user to the working-mode text-only nudge at agent.cpp:3371-3376. The candidate assumed it was already there, but asks should stay explicit, and an ask stalls an attached dispatch until C05 lands.

Effort: about 6 lines of source plus about 40 lines of tests.

## Evidence (file:line)

- src/loop/agent.cpp:2866-2875: the justification comment, 'Measured on a plan-mode run …'
- src/loop/agent.cpp:2876-2878: is_question_candidate is true in Plan mode or when assistant_text contains '?' anywhere
- src/loop/agent.cpp:2883-2893: when enum_lines >= 2, tool_name becomes ask_user, halted_ is set, halt_reason_ is 'awaiting_user', and the event records promoted_from_text=1
- src/loop/turn.cpp:71-73: the enumerated-choice regex counts every '1. ' or '2) ' summary line as a choice
- tests/loop/test_agent_step.cpp:3914-3944: only an Agent-mode summary without a '?' is tested
- src/surface/sidecar.cpp:2111-2212: the awaiting loop runs until timeout_s; 2259-2262 maps timeout_awaiting_user to status timeout, exit 2. src/surface/worker.cpp:454-460 and 1064-1075 never promote a timeout
- Repro (scratchpad enum_promo.py, a mirror of turn.cpp:71-73 and agent.cpp:2876-2883): a typical closer gives enum_lines=2 and is promoted; a summary saying 'Changed regex to `^a?b$`' is promoted; a summary with no '?' is not

## Skeptic verdict

real=True, reproduced=True, keep=True, root-cause fix=True, already handled=False, violates constraint=False

EVIDENCE (scratch files are in <scratch, not kept>/c06v/)

- ecl_test.cpp compiles the repo's exact enumerated_choice_lines body (extracted with sed from /home/user/Piper-Agent/src/loop/turn.cpp) with g++ -std=c++20, using real std::regex.
  - A typical closer ending "Would you like me to add more edge cases?" gives enum=2 and is promoted in Agent and Debug mode.
  - '?' that appears only inside code (`user?.profile`, `^a?b$`) is also promoted.
  - The existing test's summary without '?' is not promoted.
- midrun.cpp: think-spill self-questions, announced intent containing `Delegate?`, and a blocker message are all promoted.
- timeout_check.cpp links the real /home/user/Piper-Agent/src/surface/worker.cpp.
  - run_check with a green check on {status=timeout, error='timeout awaiting user answer (600s)'} leaves status=timeout.
  - The same green check on {error='agent did not complete (ended)'} returns ok.
  - So without the promotion, the same slice would come back PASS.
- The agent loop itself cannot be linked here (agent.cpp includes mlx_backend and calls mlx_memory_report). That last hop is traced, not executed. step() returns TextOnly directly when the grammar has no tool call (agent.cpp:1500-1502), and nothing in between synthesizes a call or halts on a green check.

NOT HANDLED ELSEWHERE
- scripts/agent_eval.py:708-718 auto-replies once to awaiting_user ("(unattended run) … decide yourself"). That covers the eval harness and the Python fallback worker (run_mission → ae.drive_sidecar, only when LMP_USE_CPP_WORKER=0).
- The default C++ worker has no such handling.
- Side effect: evals hide this failure, which explains why it was never measured.

DESIGN
- The '?' gate contradicts the project's own documented contract:
  - ModePolicy::conversational (turn.hpp:156-159): in a working mode a text-only turn is the model's final answer and ends `ended`.
  - The working-mode brief: "Answering in text without a tool call ends the run as your final answer."
  - Pinned tests: a_text_only_turn_in_agent_mode_ends_the_run_as_ended and a_text_only_turn_in_debug_mode_ends_the_run_as_ended.
- The '?' gate was itself a patch. The test an_enumerated_summary_in_agent_mode_without_question_does_not_ask shows summaries were already being promoted. Gating on policy_.conversational is the root fix.
- Independent of C05. C05 bounds genuine asks; this one stops the loop from inventing asks. Even with C05, an invented ask still returns a non-ok result.
- No hard constraint is touched, and agent.cpp is not one of the measured engines.

RESIDUAL RISK
- A genuine prose question in working mode now gets the act-or-finish nudge instead of a halt.
- If the model keeps answering in text on a run with no check and no checklist, it can end `ended` with completed=true. That is the same existing behavior as any other text ending, and orchestration packets normally carry --check.
- The explicit ask_user tool is unchanged.

FREQUENCY
- Not measured, and evals hide it. Each occurrence costs up to timeout_s (default 600 s) of stall, a false FAIL/timeout card, and stderr heartbeat spam (the full summary every 5 s) that the cloud may end up reading.
- To size it on the Mac: grep worker events.jsonl for "kind":"ask_user" with "promoted_from_text" on agent/debug runs.

Back to the [index](README.md).
