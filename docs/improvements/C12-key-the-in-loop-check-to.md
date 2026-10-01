# C12: Key the in-loop check to real workspace changes and take a current reading at finish, with one bounded retry on red

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| UNVERIFIED | piper | success-rate | high | M |

> **Not yet verified.** Open every cited line below and confirm the problem still holds before you implement. Drop or reshape the task if it does not.

Merged from these finder reports:

- Check the operator check at finish: run a fresh reading and allow one bounded retry on red
- Stop re-running the operator check after every successful shell and every remote call

## Problem

The loop schedules the operator check off one counter that mixes up two questions: 'did the workspace change?' and 'might observations be stale?'. It also never checks that the reading used to judge the run is current. This causes two problems.
(1) Too many re-runs. invalidate_workspace_freshness() bumps workspace_writes_ after every successful shell and every executed remote/MCP call, including readOnlyHint tools. The post-write trigger keys on that same counter. So the full check re-runs after `ls`, `grep` or `cat`, after every read-only Godoer call in a trust_mcp slice, and right after the model ran the same build or test itself (which the brief tells it to do). Each re-run adds a full-output observation, which brings compaction sooner.
(2) Ending on a stale or red reading. `finish` ends the run unconditionally. If the model finishes on a red reading, the slice ends and the post-run check just confirms red. In a batched [replace_in_file, finish] turn, the loop breaks on halted_ before the post-write reading. The final reading runs only if no reading ever existed, so `completed` is judged on a reading taken before the last edit. Calls batched after a halting call still execute: [finish, write_file] still writes. The text-only nudge invites finish and never mentions a red check.

## Why it matters

Priority 2: a small model's most common failure is declaring done on a red or unseen check. One more in-slice turn with the failing output costs seconds locally, compared with a failed slice the cloud must re-brief. Priority 1: `completed` gets judged on a current reading. Priority 4: removes a full check run after every read-only shell or MCP call and every passing self-run. Checks are builds or test suites, and trust_mcp slices run up to 60 mostly-MCP turns, so this also cuts the context growth that triggers compaction re-prefills.

## Proposed fix

(a) Split the counter. freshness_epoch_ keeps today's meaning for the RepeatDetector and repeat cache (bumped by writes, shell, remote and external changes). A new mutations_ counter is bumped only by record_deliverable and by executed remote calls whose decl->mutates_workspace is true. The post-write trigger keys on mutations_.
(b) Store the freshness epoch with each reading; a reading is current iff that epoch is unchanged. When the model's shell command equals the contract verbatim, store its result as the reading (set_last_check plus on_verification) instead of running it again.
(c) In finish, when a contract exists:
- If the reading is not current, call run_operator_check('finish').
- If the current reading FAILED and finish has not already been challenged at this epoch, do not halt. Return ok, saying the check is FAIL as of the last change (its output is the observation just added) and that the model should fix it or call finish again to hand back unfinished with a reason. Record the challenged epoch.
- A second finish with no write in between halts as usual.
- COULD NOT RUN never blocks.
This allows at most one extra turn per red reading, so it cannot deadlock.
(d) In step(), once a call sets halted_, record the remaining batched calls as refused ('not run: the run ended at call i') instead of executing them.
(e) In run(), take the post-write reading before the halt break, and change the final-reading condition from 'no reading ever' to 'no current reading'.
(f) When the last reading is FAIL, the text-only nudge says so instead of inviting finish.
Shell-only edits are still caught at finish, because the finish and final readings key on freshness_epoch_.

## Evidence (file:line)

- src/loop/agent.cpp:2295-2302: every successful shell call triggers invalidate_workspace_freshness(); 2288-2293: so does every executed remote call, read-only or not
- src/context/context.hpp:133-154: record_deliverable and invalidate_workspace_freshness both bump workspace_writes_; the comment says this counter answers 'how much work has happened since that check last ran'
- src/loop/agent.cpp:3179-3182: the post-write check runs when workspace_writes() advanced; src/loop/agent.hpp:222-227 documents the contract as running 'after any turn that wrote a file'
- src/tools/mcp_host.cpp:262-275: readOnlyHint tools get mutates_workspace=false, yet still bump the counter at agent.cpp:2288
- src/loop/turn.cpp:156-158: the brief says 'BUILD OR TEST AFTER YOU EDIT, every time'. src/loop/agent.cpp:2756: each reading enters context with its full output. src/loop/agent.hpp:413-436: compaction re-prefill is the largest wall-clock cost
- src/loop/agent.cpp:2167-2188: finish sets halted_ without consulting ctx_.last_check()
- src/loop/agent.cpp:3130-3136: the halted_ break comes before the post-write reading at 3176-3182. 3448-3460: the final reading runs only if no reading exists, and completed uses that possibly stale reading
- src/loop/agent.cpp:1574-1605: the serial batch loop has no halted_ check (src/model/grammar.hpp:129 sets kMaxCallsPerTurn = 4)
- src/loop/agent.cpp:3371-3376: the text-only nudge invites finish without mentioning a failing check
- src/surface/worker.cpp:1076-1078: a red post-run check then demotes ok to error
- src/loop/agent.hpp:10-16 and turn.hpp:9-19: the deleted completion gate deadlocked, so any change here must be bounded

## How the finder suggested verifying it

ScriptedBackend tests in tests/loop/test_agent_step.cpp:
- Contract `exit 1`, turns [write, finish, finish]: the first finish does not halt, the second does, completed=false.
- Contract `grep -q good f.txt`, batch [write bad, finish] after an earlier PASS: completed=false, with a why=finish verification.
- Batch [finish, write_file]: the file is not written.
- Contract `echo ran >> .count`, turns [shell 'ls', shell 'true', finish]: .count gets at most one line, from the final reading.
- A readOnlyHint MCP fake: no verification event.
On the Mac, compare piper_stress pass rates and verification counts and check-seconds per slice, before and after.

Back to the [index](README.md).
