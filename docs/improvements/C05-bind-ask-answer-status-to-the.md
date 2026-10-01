# C05: Bind ask/answer/status to the active slice and the current run, and stop attached dispatch waiting on asks nobody can answer

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| VERIFIED (reproduced) | piper | reliability | high | M |

> **Verified.** The problem and fix below are the skeptic's corrected versions. They replace the finder's original claim where the two differ.

Merged from these finder reports:

- Bind ask/answer to the run and the active slice, and return to the parent on ask
- End the run with the question when nobody can answer, and never leave a stale ask behind
- Stop an attached `piper dispatch` from hanging until timeout_s when the model asks a question, and clean up stale ask files afterwards
- Resolve bare answer/status/await/review to the active slice, and delete the committed fake slice

## Problem

All four failures exist. A few details in the claim are wrong or overstated.

(1) An ask in an attached `piper dispatch` or `run` cannot be answered by the blocked parent. The ask can be an explicit ask_user, a prose turn promoted to ask_user (agent.cpp:2876-2893, Agent mode included), or an irreversible gate when the packet opted out of auto-approve (worker.cpp:1577-1640). The worker blocks in sidecar.cpp:2183-2205 until the whole run hits timeout_s. That clock starts at run start, so the wait is the remaining budget, up to about 600 s. The result is status=timeout and exit 2, even when the work was already finished and the check is green: is_incomplete_agent_stop (worker.cpp:454) only promotes errors that start with 'agent did not complete'.
- Correction: the message is not empty in real C++ output. compose_result_message (worker.cpp:891-938) falls back to the error 'timeout awaiting user answer (600s)' when there is no narration, and otherwise shows the start and end of the narration. The candidate's 'message: (empty)' came from a hand-written result.json.
- What the card really lacks: the question of an explicit ask_user, and the `error` field whenever narration exists.
- The question does reach the cloud through stderr. The non-daemon attached path prints it every 5 s, about 120 times in 600 s; the daemon path prints it once. So a misdiagnosed retry is plausible (SKILL.md:115 says bigger timeout or smaller slice), not certain. The heartbeat spam is also an unmentioned cloud-token cost.

(2) Bare answer, status, await and review resolve to cwd. This is a regression from 9f2060e (#223). That commit moved the default packet to .piper/slices/<id>/ and updated piper_ui (resolve_watch_dir) but not resolve_answer_dir, resolve_status_dir or resolve_result_path.
- Reproduced: bare `answer allow` writes <ws>/answer.json and exits 0; bare status says idle during an ask; bare await prints 'timed out still idle' with rc=2; bare review prints DIED.
- The bare `piper answer` form is taught in PIPER.md:97,169, both SKILL.md copies (lines 79, 92, 119) and the PARENT_CONTRACT_BANNER printed at every run start. Bare review is taught at PIPER.md:170, and bare status/await at VISION.md:36. PIPER.md step 5 does show --dir/--task for status, await and review.
- The committed root task.json/result.json from #177 make a bare `piper review` at the repo root print PASS.
- /api/answer writes to the workspace root, and app.js never calls it.

(3) awaiting_user.json is removed only when an answer is consumed. It survives timeout, cancel and crash (worker mode does not handle SIGTERM), is never cleared at run start, and outranks result.json in collect_run_status. This is worse than claimed. Reproduced: after a timed-out run, status says 'ask' with exit 0. In the NEXT run of the same slice, `piper await --dir` returns the old question immediately, because run start only removes result.json and events.

(4) answer.json is never cleared at run start and is consumed with no run_id/seq check (read_and_consume_answer). The failure chain follows naturally from (3): status shows the stale ask, the orchestrator follows 'next: piper answer', and the next run's first unrelated ask consumes that answer.
- The irreversible-gate bypass applies only when the packet was emitted with --no-auto-approve-irreversible; `piper packet` approves irreversible calls by default.

Not verifiable: whether any pinned agent_eval runs relied on the one unattended auto-reply. pins.json does not record unattended_replies.

## Why it matters

Priorities 4/5 and 1. Each ask in an attached slice wastes up to timeout_s (600 s) of wall-clock and leads to a misdiagnosed retry. Answers silently land in the wrong directory, finished runs ask to be answered, and a stale answer can be applied to a different question. Asks are common: the persona invites them (C15), and C06's promotion creates them from ordinary summaries.

## Fix to ship

PR A: ask-file lifecycle and resolution, mostly verifiable here.
(a) src/surface/sidecar.cpp execute_task_packet, right after archive_prior_events: remove any stale awaiting_user.json and answer.json in result_dir, logging an event when an answer is discarded. In the ask_user wait loop (sidecar.cpp:2183-2212) and in handle_irreversible_ask (worker.cpp:1607-1615), remove awaiting_user.json on every exit (timeout, cancel, answer) and again before write_result.
(b) write_awaiting_user also records the pid.
(c) Change read_and_consume_answer to take (answer_path, awaiting_path, run_id, seq). It consumes only an answer whose run_id and seq match, and removes and logs a mismatch.
(d) scripts/piper_worker.py:
- collect_run_status ranks result.json above awaiting_user.json. It treats an ask as live only while its pid is alive, so it does not depend on C08.
- cmd_answer refuses with exit 3 unless a live awaiting_user.json exists in the target dir. It writes {text, run_id, seq} copied from that file.
- One resolver, resolve_slice_dir(dir, task, result), shared by answer, status, await, review and piper_ui. Order: explicit flag first, then the nearest .piper/active.json walking up from cwd, then cwd. Move piper_ui.resolve_watch_dir into piper_worker so both use it.
(e) Delete the repo-root task.json/result.json. Delete /api/answer (nothing calls it) or route it through the resolver.
(f) Tests:
- self-test: after a default `piper packet`, the bare status, await, answer and review commands, run from the workspace root, reach .piper/slices/<id>.
- self-test: answer with no live ask exits 3.
- self-test: a timeout result plus a stale awaiting file reports timeout.
- C++ unit test: an answer with the wrong seq is not consumed.

PR B: attached ask policy, needs the Mac.
(a) Add an explicit `on_ask: wait|continue|end` as a packet field and as `--on-ask` on dispatch/run, forwarded in the daemon request the same way as the auto-approve flags.
(b) Choose the default by ATTACHMENT, not by whether a wake URL exists. main() picks up .piper/orch_webhook automatically, and the watch-only piper_ui leaves that file behind.
- Detached runs keep `wait`.
- Attached `--jsonl` keeps `wait`, since a program is reading the ask event.
- Plain attached dispatch/run defaults to `continue`.
(c) Under `continue`, the first ask_user gets the unattended reply agent_eval uses, from one shared constant (C++ header plus a Python mirror pinned by self-test). The question and reply are recorded in result.json as asks:[{question, options, answered_by:"unattended"}], and the card shows one line for them.
(d) A second ask, or an irreversible gate under a non-wait policy, ends the run at once with status `needs_input`:
- the question and options go into result.json and onto the card
- a new exit code (4), plus matching review, status and progress verdicts and a webhook kind
- no 5 s heartbeat loop
(e) Update PIPER.md, AGENT_WAKE.md, both SKILL.md copies, the cursor rule, and the init templates/banner embedded in piper_worker.py. The SKILL failure playbook should say: for needs_input, answer the question in the next brief, not with a bigger timeout.

## Evidence (file:line)

- src/loop/agent.cpp:2235-2244: ask_user halts with awaiting_user; 2876-2893: prose with two or more enumerated lines is promoted to ask_user
- src/surface/sidecar.cpp:2111-2212: writes awaiting_user.json and polls answer.json until packet.timeout_s, heartbeats every 5 s (2196-2202), and leaves awaiting_user.json behind on timeout; 2259-2262: timeout_awaiting_user becomes status timeout, exit 2
- src/surface/worker.cpp:1243-1297: read_and_consume_answer accepts any answer.json without checking run_id or seq; 1607-1616: Timeout and Cancelled leave awaiting_user.json
- src/surface/sidecar.cpp:1934-1938 and scripts/piper_worker.py:2553-2558: run start clears only result.json and events; emit_task_packet (685-729) leaves the previous run's files in the slice dir
- scripts/piper_worker.py:2088-2113: collect_run_status ranks awaiting_user.json above result.json
- scripts/piper_worker.py:555-562, 2066-2074, 1021-1028: resolve_answer_dir, resolve_status_dir and resolve_result_path fall back to cwd; 535-552: the answer payload is just {text}; 769-785: cmd_answer never checks that awaiting_user.json exists
- scripts/piper_worker.py:636-643, 657-672: packets default to <cwd>/.piper/slices/<id>/ and record .piper/active.json; scripts/piper_ui.py:176-189: resolve_watch_dir already follows active.json; 422-435: /api/answer writes to the workspace root and nothing calls it
- PIPER.md:63-64, 97, 144-145, 169-170, SKILL.md:79, 115, 119, and docs/AGENT_WAKE.md:97, 151-152 teach attached dispatch plus bare answer/review
- scripts/agent_eval.py:511-515, 708-717: the eval auto-replies once with '(unattended run) No operator is available…'
- Repo-root task.json/result.json from 4c8d204 (#177): cwd /Users/dev/…, diff_stat '+42 -3 lines', git_diff_path /tmp/slice-001.diff
- Repros (scratchpad):
- result.json {status: timeout} plus a leftover awaiting_user.json: `piper status` prints 'state: ask' with exit 0, and `piper await` exits 0
- bare `answer --text` from the workspace root wrote <ws>/answer.json
- bare review printed DIED; bare await printed 'timed out still idle' with rc=2
- at the repo root, bare review printed PASS
- the card for the timed-out ask read 'status: timeout / message: (empty) / test: not run / verdict: FAIL'

## Skeptic verdict

real=True, reproduced=True, keep=True, root-cause fix=True, already handled=False, violates constraint=False

I traced every cited line and the full C++ path: execute_task_packet, the ask loop, handle_irreversible_ask, read_and_consume_answer, compose_result_message, run_check, the run-start cleanup in sidecar.cpp:2662 and piper_worker.py:2553, and the daemon forward path. The Python half was reproduced in the scratchpad (repro_c05.sh). Every bare-command failure showed up. So did the stale ask and stale answer surviving a re-packet, and an immediate stale ask returned by `await --dir` at the start of a new run. `piper review` at the repo root printed PASS.

Line drift: the cited PIPER.md:63-64/144-145 hold the 'attached' text, and the bare-answer lines are actually 88/97/169-170. The content is as claimed.

Design flaw in the proposed fix: it keys the policy on 'without a wake URL'. But main() resolves .piper/orch_webhook automatically (piper_worker.py:2529-2533). piper_ui writes that file and never removes it (piper_ui.py:453-479), and the UI cannot answer anything: /wake only broadcasts, and /api/answer is dead and points at the wrong directory. I reproduced it: once `piper ui` has run in a workspace, every attached dispatch carries a wake URL. A URL-keyed policy would therefore choose 'wait' and hang again. The key has to be attachment.

The proposal also lets the first ask continue silently. That hides an assumption the model made, which works against priority 1, so the ask and the auto-reply must be recorded on the card.

'piper packet archives previous run files' adds nothing once run start clears ask/answer and already removes result.json. Liveness for a crash during an ask does not need C08: a pid in awaiting_user.json is enough.

Arguments against keeping it fail:
- It sits directly on the orchestration path.
- No code or doc handles it. The docs assume an attached parent can answer, which contradicts dispatch blocking.
- It violates no hard constraint: sidecar.cpp and worker.cpp are harness code, not measured engines.
- The per-occurrence cost is large: up to the full timeout of dead wall-clock, a false timeout on finished work, stderr spam into cloud context, an exit-0 silent failure on the taught answer path, broken await for later runs of the slice, and a gate bypass in the opt-in configuration.

I'd ship it as two PRs. The lifecycle and resolver half is low risk and mostly testable on Linux. The attached-policy half needs Mac validation.

Back to the [index](README.md).
