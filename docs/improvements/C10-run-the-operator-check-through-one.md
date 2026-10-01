# C10: Run the operator check through one executor with one timeout, report timeouts as timeouts, and reuse an unchanged reading

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| UNVERIFIED | piper | bug | medium | M |

> **Not yet verified.** Open every cited line below and confirm the problem still holds before you implement. Drop or reshape the task if it does not.

Merged from these finder reports:

- Use one runner and one timeout for the operator check; reuse the in-loop reading when the workspace is unchanged
- Give the post-run check the same timeout as the in-loop contract and report timeouts as timeouts
- Run the in-loop check and the post-run acceptance check the same way
- Run the post-run check with the same time limit as the in-loop check, and expose that limit

## Problem

The packet `check` runs in two executors with different limits and environments. One steers the slice and the other judges it. Inside the loop, the check runs through the registry shell under the T1 Seatbelt profile with a 300 s wall clock, and that run decides completed=true. After the loop, run_check runs it again through a plain fork/exec of /bin/sh, unsandboxed, with check_timeout_s defaulting to 60 s. That value can only be changed by hand-editing task.json, which the protocol forbids.
Three failures follow.
- False FAIL on slow checks. A check that takes 60-300 s (a C++ or Swift build plus ctest, a Godot headless import, a cold cargo test) passes in the loop, then gets SIGKILLed after the loop. It is recorded as exit_code -1, and status drops from ok to error with 'check command failed (exit code -1)'. The verdict is FAIL and the card never says 'timeout'.
- Turns burned on sandbox-only failures. A sandbox denial (Go build cache, ~/.cargo, a network bind) shows the model a FAIL in the loop. Because the exit code is 1, it is not even reported as COULD NOT RUN. The acceptance run then passes, so the model wasted turns fixing nothing.
- A wasted re-run. On passing slices, the post-loop run repeats the last in-loop reading on an unchanged workspace, adding a full check execution to every slice.
The eval harness uses yet another limit, 120 s.

## Why it matters

Priority 1: false FAIL hits exactly the compiled-language and Godot checks this user runs, and each one costs a cloud review plus a re-dispatch or a hand edit. Priority 2: no turns burned on failures that only exist inside the sandbox. Priority 4: one fewer full check execution on every passing slice.

## Proposed fix

Make the operator check one component that both run_operator_check and run_check call, with the same executor, the same sandbox tier (the slice's), the same timeout, and the same output capture.
1. Default TaskPacket.check_timeout_s to the in-loop shell limit (300 s). Add `piper packet --check-timeout-s` to write it, and send it in the lmp/start settings so the loop uses the same bound. Document the field next to `check`.
2. Use run_cmd's return value. A timed-out check sets test.timed_out=true, records duration_s, sets the error 'check timed out after Ns', and shows TIMEOUT on the card (C04). finalize_run (C02) treats it as 'the check did not answer', not 'the check failed'.
3. Classify sandbox denials in the output (EPERM, 'Operation not permitted', network denied) as COULD NOT RUN (sandbox) instead of FAIL. Widening toolchain allowances, such as pointing GOCACHE or CARGO_HOME into the workspace, is a separate measured follow-up.
4. After the loop, if the latest in-loop reading is still current for the final workspace, copy it into result.test with `source: 'loop'`; otherwise re-run the check. 'Current' means C12's mutation epoch has not changed, or a fingerprint of C03's end tree matches.

## Evidence (file:line)

- src/surface/worker.hpp:55: `double check_timeout_s = 60.0;`. src/surface/worker.cpp:347-349: the only override is a task.json field
- scripts/piper_worker.py:685-716 and 2336-2354: `piper packet` has no check-timeout flag and never writes the field. PIPER.md:58 and ORCHESTRATOR_WORKER_VISION.md:33 forbid hand-writing task.json. No doc mentions check_timeout
- src/surface/session.cpp:191: shell_wall_clock_seconds = 300. src/loop/agent.cpp:2655-2671: run_operator_check runs through registry_.execute('shell', …, policy_.sandbox_tier); 3176-3182: after every turn that wrote; 3448-3459: completed requires the in-loop reading to pass
- src/tools/sandbox.cpp:124-128: denies network and file-write except the workspace and toolchain dirs (148-167). src/loop/agent.cpp:2668: ran=false only for Refused or exit 126/127. src/loop/agent.cpp:717-723: auto-detected contracts include 'go build ./…' and 'cargo build'
- src/surface/worker.cpp:45-77: run_cmd is fork plus execlp /bin/sh, no sandbox. 148-152: on timeout, exit_code becomes -1 and '[timeout after Ns]' is appended. 1054-1079: run_check ignores the return value and demotes ok to error
- src/surface/sidecar.cpp:2323: run_check always re-executes after the loop
- scripts/piper_worker.py:996-1000: the card renders `exit=-1`. scripts/agent_eval.py:803: the eval defaults to 120 s
- docs/AGENT_LOOP_WINS_BRANCH_PLAN.md:322: P6, a measured thrash when T1 blocked bind() for HTTP tests
- Repro (scratchpad repro_check_timeout.cpp, real worker.cpp, default TaskPacket): a passing `sleep 62; …; exit 0` check on an ok slice gives status=error, test.exit_code=-1, error 'check command failed (exit code -1)', tail '[timeout after 60.000000s]'

## How the finder suggested verifying it

Here: compile worker.cpp with the mach-o stub and run run_check with a default TaskPacket and a passing 62 s command. Today status flips from ok to error (shown this session). After the fix, status stays ok, and a command that runs past the limit reports test.timed_out. `python3 scripts/piper_worker.py packet --help` should list the new flag, and self-test should assert the emitted field. On the Mac: a `sleep 90; true` check runs once and gives status ok, and a Go module slice with `go build ./...` gives the same verdict in the loop and in result.test.

Back to the [index](README.md).
