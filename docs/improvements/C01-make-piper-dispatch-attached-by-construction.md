# C01: Make `piper dispatch` attached by construction instead of inferring detach from stdin/stdout

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| VERIFIED (reproduced) | piper | false-success | high | S |

> **Verified.** The problem and fix below are the skeptic's corrected versions. They replace the finder's original claim where the two differ.

Merged from these finder reports:

- Make `piper dispatch` always attached instead of guessing detach from stdio
- Stop guessing 'detached' from stdin/stdout: under agent tool runners piper dispatch exits 0 with no card or refuses to start

## Problem

The claim is accurate in substance. Three details need correcting.

What happens: `cmd_dispatch` (scripts/piper_worker.py:1063-1105) calls `main(["run", ...])` in-process. `main()` decides `is_detached = --detach or LMP_DAEMONIZE=="1" or (LMP_DAEMONIZE!="0" and (stdin is /dev/null or stdout is a regular file))` at lines 2535-2538.

I confirmed the stdio claim inside this Claude Code session. A foreground Bash call and a run_in_background call both have fd0=/dev/null and fd1/fd2=a regular `.output` file, so the heuristic fires on every dispatch.

- **No URL:** line 2540 refuses with exit 3 and the 'detached launch needs a wake URL' message. Dispatch still prints a card. That card reads `verdict: DIED (no result.json)`, or it shows the previous run's result.json, because the removal at line 2553 comes after the gate.
- **With a URL** (LMP_ORCH_WEBHOOK, or `.piper/orch_webhook`, which piper_ui.run_server writes and never removes): `ae.detach_from_launch_session()` double-forks and the original process calls `os._exit(0)`. Dispatch exits 0 in about 0.1 s with no card while the run continues orphaned (ppid=1). The orphan later prints the card into the caller's stale stdout file and appends a review to orch.jsonl.

Corrections:
1. **Stale-result card is misleading, not a false PASS.** It shows the old `status: ok` and `test exit=0` but `verdict: FAIL`, because the exit is 3.
2. **The orphaning is entirely Python's double fork.** The C++ side never forks on the heuristic; it forks only on an explicit `--detach` (sidecar.cpp:2664). Its `is_detached_launch` heuristic (worker.cpp:1516-1529) only gates refusal (sidecar.cpp:2645-2653). That gate still matters: lmp_sidecar inherits the harness's /dev/null stdin, so once Python stops detaching it would refuse an attached run with no URL.
3. **The Claude Code completion notice points to the output file holding the card** rather than carrying the card inline.

An extra defect the claim missed: explicit intent is ignored in the other direction too. `detach_from_launch_session()` (agent_eval.py:117-121) re-applies the stdio heuristic as its own gate. So `piper run --detach` with piped stdio does not detach; I measured it running attached for 4.13 s. LMP_DAEMONIZE=0 plus `--detach` also passes the URL gate and then silently does not fork.

"A second model" is plausible but was not reproduced with a real model. The run path has no single-flight lock; only distill has /tmp/piper_distill.lock.

## Why it matters

Priorities 1 and 5, on the user's primary path. Every dispatch from Claude Code either gets refused with a misleading DIED or stale card, or returns exit 0 with no outcome while the model keeps running. The second case is a false success that invites `piper progress pass` or a second dispatch, which means two resident models. Once fixed, `run_in_background: true` with `piper dispatch` becomes a webhook-free wake for Claude Code, because the completion notice carries the card.

## Fix to ship

Root cause: detach policy is inferred from incidental stdio properties, and that policy is tangled with the detach mechanism. Fix: declare intent and never infer it.

1. **scripts/piper_worker.py**
   - Move the run body of `main()` (from `load_packet` onward) into `cmd_run(args, *, attached_only=False)`.
   - Set `detach = (not attached_only) and (args.detach or os.environ.get('LMP_DAEMONIZE') == '1')`, with no stdio probes. Keep the exit-3 refusal for `detach and not webhook_url`.
   - When `detach` is true, fork unconditionally: use a local mechanism-only `_daemonize()` (fork / setsid / fork, ignore HUP and PIPE, print the detached line), or `ae.detach_from_launch_session(force=True)`. Do not let the helper re-check stdio. That re-check is what makes explicit `--detach` a no-op today.
   - `cmd_dispatch` calls `cmd_run(..., attached_only=True)` directly instead of re-entering `main(argv)`. Dispatch is then attached by construction and ignores LMP_DAEMONIZE=1, since its help text says "Always attached".
   - Optionally, set LMP_DAEMONIZE=0 in the lmp_sidecar child env. This guards against a skewed, unrebuilt binary; the harness never forwards `--detach`, so the child is always attached to the harness.
   - Leave agent_eval.py's `run` bakeoff behaviour alone; it is not on the orchestration path.
2. **src/surface/worker.cpp `is_detached_launch`**
   - Make it return `cli_detach || getenv("LMP_DAEMONIZE") == "1"`.
   - Delete `stdin_is_devnull()` and `stdout_is_regular_file()`, along with their declarations at worker.hpp:246-247.
   - sidecar.cpp:2645 is otherwise unchanged.
3. **Tests**
   - Remove the global `os.environ['LMP_DAEMONIZE']='0'` pin in `self_test` (piper_worker.py:2739). It masks exactly this class of bug.
   - Add subprocess cases that run `piper_worker.py dispatch` with `stdin=DEVNULL`, stdout redirected to a file and LMP_DAEMONIZE scrubbed, using a fake that sleeps about 1 s. Cover three URL variants: none, LMP_ORCH_WEBHOOK pointing at the self-test's HTTP server, and a `.piper/orch_webhook` file. Cover both a `.py` fake and a non-`.py` executable for the C++ path.
   - Assert: rc 0, elapsed ≥ 1 s, 'verdict:  PASS' present in the file at process exit, no 'detached pid=' line, and exactly one `done` POST in the URL case.
   - Add `run --detach` with piped stdio plus a URL, and assert that the original process returns quickly with 'detached pid='.
   - In tests/surface/test_worker.cpp `detached_launch_detection_and_gating`, dup2 /dev/null onto stdin with LMP_DAEMONIZE unset, assert `!is_detached_launch(false)`, then restore stdin.
4. **Docs**
   - Update PIPER.md step 4 and the wake standard, plus their embedded copies (AGENT_WAKE_FALLBACK and CURSOR_RULE_CONTENT in piper_worker.py, and both piper-orchestration SKILL.md files).
   - State that Piper never guesses detach: background launches must pass `--detach` plus a wake URL.
   - State that in Claude Code you run `piper dispatch` with `run_in_background: true`, then read the card from the task output when the completion notification arrives. Foreground calls are capped at 10 minutes.
5. **Skip** pre-gate deletion of result.json and NOT STARTED rendering. Once dispatch cannot refuse, they do not matter. If a stale-result guard is still wanted, have dispatch render only a result written after its own launch, keyed by run_id or mtime.

## Evidence (file:line)

- scripts/piper_worker.py:1063-1088: cmd_dispatch builds run_argv and calls main(run_argv) ('Always attached: dispatch owns wait'), then loads whatever result.json exists and prints a card
- scripts/piper_worker.py:2535-2542: is_detached is true for --detach, LMP_DAEMONIZE=='1', or (flag != '0' and (stdin_is_devnull() or stdout_is_regular_file())); with no webhook it prints DETACHED_NO_WAKE_MSG and returns EXIT_INVALID
- scripts/piper_worker.py:2548-2558: detach_from_launch_session() runs before the stale result.json is removed
- scripts/agent_eval.py:79-98: the stdin/stdout probes; 104-126: double fork, then the original process calls os._exit(0)
- src/surface/worker.cpp:1516-1529: the C++ is_detached_launch uses the same heuristic and honours LMP_DAEMONIZE=0; src/surface/sidecar.cpp:2645-2653 refuses with exit 3
- scripts/piper_ui.py:465-468: run_server writes .piper/orch_webhook on start and never deletes it
- scripts/piper_worker.py:2739: self_test sets LMP_DAEMONIZE=0, so the heuristic is never tested
- PIPER.md:78 says 'piper dispatch does not detach'; PIPER.md:144-145 says 'piper dispatch stays attached and does not need a URL'
- Repros by two finders (Claude Code Bash stdio, fake sidecar): with no URL, 'detached launch needs a wake URL' plus a card showing 'exit: 3 verdict: DIED' or a stale 'status: ok'; with a URL, dispatch exited 0 after 0.09-0.1 s, stdout held only 'detached pid=… ppid=1', and the orphan appended the card about 4 s later

## Skeptic verdict

real=True, reproduced=True, keep=True, root-cause fix=True, already handled=False, violates constraint=False

Reproduced with Python here, launched from the Claude Code Bash tool. Scratch files are in <scratch, not kept>/c01 (driver.py, driver2.py, mkws.sh, fake_slow.py, fake_cpp_worker). The fake sidecars sleep 4 s. Results:

| Scenario | Exit | Time | What the caller saw |
|---|---|---|---|
| A: no URL | 3 | 0.11 s | Refusal message plus a DIED card; sidecar never started; orch.jsonl logged a DIED review |
| B: no URL, old result.json present | 3 | — | Card showed the old run's status ok, test exit=0 and old.py, with verdict FAIL |
| C: LMP_ORCH_WEBHOOK=http://127.0.0.1:9/h | 0 | 0.12 s | Only the banner and 'detached pid=18451 ppid=1'; no result.json at exit |
| D: .piper/orch_webhook file | 0 | 0.11 s | Same as C; PASS card appended to the stdout file after exit |
| E: non-.py C++-path stand-in plus env URL | 0 | 0.10 s | No card; the child saw LMP_DAEMONIZE unset |
| E2: C++ stand-in, no URL | 3 | — | DIED card |

In C, `ps` showed the orphaned dispatch (ppid 1) and its sidecar child. The orphan later wrote result.json and a PASS review to orch.jsonl, but the card went into the tool's already-consumed .output file and never reached the caller.

Controls behaved correctly: pipe stdio blocked 4.14 s and printed a PASS card, and LMP_DAEMONIZE=0 with /dev/null and a file blocked 4.15 s with a PASS card.

Why the tests miss it: self_test pins LMP_DAEMONIZE=0 globally (piper_worker.py:2739). The C++ test detached_launch_detection_and_gating (tests/surface/test_worker.cpp:1143-1157) only covers the env overrides. Nothing in the code or docs (PIPER.md, the skills, the cursor rule, AGENT_WAKE_FALLBACK) tells orchestrators to set LMP_DAEMONIZE=0, so nothing handles this elsewhere. Every doc says dispatch, and also `piper run` / `worker run`, are attached and need no URL. That makes this a contract violation on the primary path for the named orchestrator. Cursor's pty terminal does not trip the heuristic, which likely explains why it went unnoticed.

Design pushback considered:
- **Loss of the wake-standard guard.** Dropping inference means a bare `nohup piper run &` without `--detach` is no longer refused. That guard cannot be kept, because no stdio signal tells "nohup &" apart from "foreground under an agent tool runner". The standard's own 'Done when' item (an explicit detached launch with no URL exits before the sidecar starts) still holds for `--detach`.
- **Foreground timeout.** A foreground attached dispatch can exceed the Bash tool's 2–10 min cap. The right Claude Code pattern is `run_in_background: true`, and the current heuristic breaks that pattern too: the background task 'completes' at once with exit 0 and no card.
- **Hard constraints.** The fix violates none. It removes a path to two resident models.

Parts of the proposed fix I would trim:
- (3) Deleting result.json before any launch decision would destroy a prior run's evidence on a refused explicit `--detach`. Once dispatch is attached by construction, dispatch has no refusal path left, so the stale card goes away by itself. The NOT STARTED rendering is marginal.
- Keep-as-warning for the probes adds noise to every card the cloud reads. Delete the probes instead.

Back to the [index](README.md).
