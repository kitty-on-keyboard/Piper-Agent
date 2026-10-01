# C15: Give worker runs a worker prompt profile: state the acceptance check, drop IDE-only ask/sidebar guidance, cross-session recall and remember

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| UNVERIFIED | piper | success-rate | medium | M |

> **Not yet verified.** Open every cited line below and confirm the problem still holds before you implement. Drop or reshape the task if it does not.

Merged from these finder reports:

- Tell worker runs about their acceptance check, and drop the IDE-only guidance from the worker prompt
- Tell the model the operator check runs automatically after each edit
- Stop advertising cross-session recall (and `remember`) in worker slices

## Problem

A worker slice gets the VSIX prompt and tool policy word for word. build_start_message sends the brief as the mission and the check only as settings.verify_contract, with no system_prompt, so kPersona applies. This causes three problems.
(1) The model is never told the check exists, that the harness runs it after every turn that writes a file, or that `completed` requires its latest reading to pass. The contract only goes to the event log. The live-state block shows the check only after the first reading. Meanwhile the brief orders 'BUILD OR TEST AFTER YOU EDIT, every time', and the cloud's 'Done when' section names the same command. The model therefore spends a turn re-running a check whose output it already has, and it can call finish while the check is red.
(2) The persona and four nudge strings point the model to ask_user with '2 to 4 options' (the VSIX click-card) and describe `plan` as the operator's sidebar view. In orchestration an ask is the most expensive way to stop (see C05).
(3) From slice 2 on, the system prompt says 'Earlier sessions here left N stored items across M sessions … reach for it before re-reading a file'. context_recall searches every session by default, which steers the worker toward earlier slices' read_file bodies of files that have since been edited. N and M grow with every slice, so the system message changes each time, which defeats C18's prefix reuse. `remember` writes .lmp-memory.md at the workspace root. That file is deliberately not git-excluded, so it shows up unrequested in the slice diff and changes the next slice's prompt.

## Why it matters

Priorities 2 and 4: on every slice with a check, saves roughly one redundant verify turn and one check run per edit cycle. It makes 'done means the check is green' explicit, so the model finishes on red less often, and it stops steering the worker to stale file bodies from earlier slices. Priority 5: fewer asks, each of which costs a cloud wake or a C05 needs_input. Priority 1: no unexplained .lmp-memory.md in the slice diff. It also keeps the system message stable across slices, which C18 needs.

## Proposed fix

Have build_start_message send the setting `client: 'worker'`, mapped to AgentConfig::worker. Then:
(1) Tell the model about the check. When operator_verify_contract is non-empty, the harness appends a trailer to the MISSION message. The mission is stable for the run and comes after the system message, so the system prefix stays identical across slices. Trailer text: 'Acceptance check (the harness runs it, not you): `<cmd>`. It runs automatically after every turn in which you change a file, and its output is shown to you as the operator check. The slice is complete only when its latest reading passes; then call finish. Do not run it yourself; run narrower commands only when you need detail it does not give.' Render '# Operator check — NOT YET RUN <cmd>' in live state before the first reading. Make the 'Verifying' bullets in kWorkingDiscipline conditional on a contract existing. Pair this with C12's adoption of the model's verbatim run, so a self-run never runs twice.
(2) Use a worker variant of the persona, mode brief and nudges. Drop the click-card ask guidance and the sidebar framing. Instead: 'the brief is the full spec; if it is ambiguous or you are blocked, call finish and say what blocked you; ask_user pauses the slice and wakes the orchestrator'.
(3) In worker runs, skip the recall-scope paragraph but keep the in-run rehydrate hint. Declare context_recall with this-session-only scope, or not at all. Withhold `remember`, and keep loading .lmp-memory.md read-only.
Interactive runs stay unchanged. This is a tool and prompt policy change, not a PCC rewrite; changing recall's default scope globally would need a PCC re-score first.

## Evidence (file:line)

- src/surface/worker.cpp:476-478: the check is sent only as settings.verify_contract. 562-568: lmp/start carries {mission: prompt, settings}, with no system_prompt and no worker marker. src/surface/sidecar.cpp:1296: an empty system_prompt falls back to kPersona
- src/loop/agent.cpp:771: the contract is not part of the mode brief. 794-805: it is only emitted as an operator_contract event. src/context/context.cpp:328-338: '# Operator check' is rendered only once a reading exists
- src/loop/turn.cpp:143-158: 'OPEN A MULTI-STEP TASK BY CALLING plan … operator's only view' and 'BUILD OR TEST AFTER YOU EDIT, every time'
- src/loop/agent.cpp:3176-3182: the harness already runs the check after every turn that writes. 3445-3453: completed = list_clear && the last check passed
- src/context/context.cpp:26-29: the persona says to ask with ask_user and offer 2 to 4 options. More ask_user nudges: agent.cpp:2133, 3150, 3167; registry.cpp:2466-2468; session.cpp:368
- docs/ORCHESTRATOR_WORKER_VISION.md:68-70: the prompt.md 'Done when' section names the command. The task.json example uses the same command as its check
- src/surface/context_journal.cpp:117-137: the journal is keyed by run_id. sidecar.cpp:1320-1323: set_recall_scope(items, sessions) runs every run. context.cpp:189-209: the counts are rendered into the system message ahead of conventions, skills and memory
- src/tools/context_tools.cpp:121-127: 'Use it before re-reading a file … searches every past session by default'. context_journal.cpp:43-45: turn rows hold full observations. pcc/recall.cpp:28-30: rows are marked SUPERSEDED only if they are keyed
- src/tools/registry.hpp:199: kMemoryFileName is '.lmp-memory.md'. context_journal.hpp:88-97: it is deliberately not git-excluded. worker.cpp:986-1022: collect_git adds it to git.diff and diff_stat

## How the finder suggested verifying it

Unit tests on the Mac: with a contract, render() puts the command in the mission message, not the system message. Under the worker profile, render() with set_recall_scope(500, 9) contains no recall paragraph. A/B on the Mac with `python3 scripts/agent_eval.py run --seed 7,13,42` over tasks that have a contract (failing_test_median, build_error_cpp, email_regex_edges): count shell calls equal to the contract, turns to solve, ask_user events, and runs ending completed=false after a red final check. Dispatch two slices and confirm the system message is identical across them (same prompt hash) and that git.diff never contains .lmp-memory.md.

Back to the [index](README.md).
