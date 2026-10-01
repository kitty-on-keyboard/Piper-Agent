# G04: Red check cards carry 2000 raw tail bytes: run the check output through the measured log_triage engine and put the reason on the card

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| VERIFIED (reproduced) | piper | design | medium | M |

> **Verified.** The problem and fix below are the skeptic's corrected versions. They replace the finder's original claim where the two differ.

Merged from these finder reports:

- ideas-orchestration#1 Check-failure triage block

## Problem

All cited file:line references are accurate. Two claims are overstated.

What is true:
1. run_check (worker.cpp:1042-1080) throws away run_cmd's return value. A timeout becomes exit_code -1 with error "check command failed (exit code -1)". A command that never ran (exit 126/127) is also labeled "failed". The in-loop path calls that case COULD NOT RUN (agent.cpp:2667, ToolResult::never_executed).
2. run_cmd keeps only the first 1 MB, so the 2000-byte "tail" of a larger log comes from the middle. Reproduced: on a 1.43 MB output, the final error line is gone from every artifact.
3. The post-run check defaults to 60 s; the in-loop verify_contract gets the 300 s shell clock (session.cpp:191). With a contract set, completion requires the last in-loop check to pass (agent.cpp:3450-3460). So a green check that takes 60-300 s gives status ok, then the post-run timeout demotes it to error. The model's success message is left unchanged. `piper packet` has no flag for check_timeout_s, so every emitted packet gets 60 s.
4. The review card never prints result.error for any non-PASS verdict: check failure, stalled reason, or irreversible-tool denial. It also prints no check output. Yet the skill tells the cloud to "Read `error` or the card" and to aim the next slice "at test.output_tail" (SKILL.md:105,108).

Overstated:
- "The cloud gets almost nothing" is true of the card only. result.json, `dispatch --json` and `review --json` all carry the 2000-byte raw tail. The timeout marker is in output_tail; an existing test asserts it.
- The 6/32 vs 23/32 score reproduces exactly, but the corpus was built to defeat tail truncation. Compact finds 23 of 23 scoreable primary locators; the tail finds 6. On pytest, ctest, Python and unittest logs the two tie on locators except python_assert_midway. The gain is concentrated in compiler output (C++/Swift/Rust) and logs over 1 MB. Compact already reaches its ceiling at 2000-2048 bytes.

## Why it matters

A red check is the most common non-PASS outcome, and the one where the cloud spends the most tokens re-reading result.json, events.jsonl or git.diff, or re-running the check. A deterministic 8-12 line block that names the failing path:line in about 3 of 4 cases is distill-style value on the core orchestration path, with zero cost on green and no model call. It also stops slow-but-green checks from failing slices.

## Fix to ship

Ship a trimmed, Piper-only version. Godoer needs no change.

1. src/surface/worker.cpp, run_check:
   - Give run_cmd a capture policy, either a parameter or a run_check-specific capture. Keep a 1 MB head plus a 256 KB tail ring with a "[... N bytes not captured ...]" marker. The git callers keep today's head-only capture.
   - Use run_cmd's return value.
   - Extend TestBlock and write_result with:
     - timed_out
     - could_not_run (exit 126/127, the same rule as ToolResult::never_executed)
     - seconds
     - output_path
     - triage {runner, passed, failed, failing_tests<=6, primary[{path,line,message}]<=4, paths<=8}
   - Set output_tail = log_triage::compact(captured, 2000). Logs of 2000 B or less come back byte-identical, so the four existing run_check tests still hold.
   - When the digest is partial, write the captured text to <dirname(result_path)>/check.log and set output_path. This follows the shell tool's S14 rule of spooling only when content was dropped.
   - The error text names the condition: "check timed out after Ns", "check could not run (exit 127)", or "check failed (exit N)". Timeouts and could-not-run never promote.

2. One clock:
   - Hoist 300 into a shared constant, e.g. tools::kShellWallClockSeconds in registry.hpp.
   - Use it at session.cpp:191 and as the default for TaskPacket::check_timeout_s, so a check that passed in the loop cannot be killed by the post-run check.
   - Optionally add `piper packet --check-timeout`.

3. scripts/piper_worker.py, format_review_card:
   - On any non-PASS verdict, print `why: <result.error>`. This also surfaces stalled reasons and irreversible-denial escalations that are hidden today.
   - On a red check, print at most 3 extra lines: failing tests, the first located primary diagnostic, and `log: <output_path>`.
   - The PASS card stays byte-identical. `review --json` and `dispatch --json` already carry the result, so they need no change.

4. Update SKILL.md line 108 in both .agents and .cursor to say that output_tail is a triaged digest and output_path is the full log.

5. Tests:
   - tests/surface/test_worker.cpp:
     - `cat tests/testdata/log_triage/logs/build_missing_semi_early.log; exit 2` puts the locator in output_tail and triage.primary.
     - Output over 1 MB with the error printed last keeps the error.
     - `sleep 2` with a 1 s clock gives timed_out plus the error text.
     - Exit 127 gives could_not_run.
   - piper_worker.py self-test: a red result prints the why and triage lines, and the PASS card is unchanged.
   - Before shipping, score the card block itself on corpus plus holdout. It is a new consumer of analyze(), which is a measured engine.

Drop four parts:
- the repeat fingerprint across slices
- the optional G01 local-model job
- the 32 MB stream-and-reread
- the Godot locator follow-up

## Evidence (file:line)

- piper:src/surface/worker.cpp:1042-1080 run_check ignores run_cmd's return value (:1056), keeps `out.substr(out.size() - 2000)` (:1060), and demotes ok to 'check command failed (exit code N)' (:1078) while result.message stays as the model wrote it
- piper:src/surface/worker.cpp:121,132 run_cmd appends only while output is under 1 MB, so later bytes are dropped; :148-151 a timeout becomes exit_code -1 plus '[timeout after Ns]' and returns false
- piper:src/surface/worker.hpp:55 check_timeout_s = 60.0. The same command is the in-loop verify_contract (src/surface/worker.cpp:477), which src/loop/agent.cpp:2655-2676 runs through the shell tool under shell_wall_clock_seconds = 300 (src/surface/session.cpp:191)
- piper:src/tools/registry.cpp:2140-2144 in-loop shell results already get log_triage::compact plus analyze; src/surface/session.cpp:188 max_result_bytes = 8192; src/tools/log_triage.hpp:376 compact(), :870 analyze()
- piper:scripts/piper_worker.py:996-1002 the card's test line holds only exit and cmd, with no error and no output
- piper:src/surface/sidecar.cpp:2323 run_check is the single fill point for CLI and daemon runs
- Re-measured here: scratchpad lt_bench (compiled from src/tools/log_triage.hpp) plus score.py over tests/testdata/log_triage corpus.jsonl and holdout.jsonl at 2048 B. Tail: primary locator 6/32, local locators 36, messages 41. Compact: 23/32, 71, 76

## Skeptic verdict

real=True, reproduced=True, keep=True, root-cause fix=True, already handled=False, violates constraint=False

VERIFIED BY RUNNING CODE (scratch files under scratchpad/g04 only; neither repo was modified)

- Built the shipped src/surface/worker.cpp on Linux (g++ -std=c++20 -Werror, with a mach-o/dyld.h stub) and drove the real run_check:
  - Timeout: status=error, exit -1, error "check command failed (exit code -1)", success message untouched.
  - 1.43 MB output with the error printed last: the real error is absent from output_tail.
  - Corpus build_missing_semi_early: locator absent from output_tail.
  - Exit 127: labeled "check command failed".
- Rendered format_review_card in Python for a check-demoted run. It shows the model's message "...tests pass", then `test: exit=-1`, then FAIL, with no reason anywhere.
- Recompiled lt_bench from log_triage.hpp and reran score.py. Got 6/32 vs 23/32 at 2048 B, the same at 2000 B. Compact stays at 23 at 4096 and 8192 while the tail climbs to 10.
- A scratch prototype of the fix found the locator in all four cases where the shipped run_check found it in one:
  - head+tail capture
  - compact(…, 2000)
  - an error taxonomy: timed out / could not run / failed
- Confirmed run_check sits inside execute_task_packet and runs for both daemon (sidecar.cpp:2580) and CLI (sidecar.cpp:2731) runs. Python never runs the check itself. Godoer never reads Piper's result.json, so this does nothing for crash analysis.

WEAKNESSES FOUND IN THE PROPOSAL

1. The card's triage block uses analyze() in a role nobody has measured. Scored on the corpus:
   - The corpus primary path:line shows up in the format_annotation output for 21 of 30 failing cases, and as the first primary line in 17.
   - On Python tracebacks the 4-line cap shows "Traceback (most recent call last):" plus the outermost frames and drops the exception line (python_traceback).
   - On ho_unittest_failures all 4 lines are status noise. That matters for Godoer's pytest checks.
2. "repeat: same primary failure as <slice>" is brittle. primary_fingerprint includes line numbers, which shift after an edit. It also needs a lookup across slices, and the orchestrator already remembers previous slices.
3. The optional G01 local-model job adds a model call on every red check where the triage comes back empty. On this path the cloud orchestrator, which wrote the brief, is the stronger diagnostician.
4. Streaming 32 MB to disk and reading it back is not needed for correctness; a head+tail capture fixes the middle-tail bug with bounded memory.
5. Setting the default to "300" as a second literal lets the two clocks drift apart again; they should share one constant.
6. The Godot res:// follow-up is not needed now. On a synthetic Godot log, compact already keeps the SCRIPT ERROR line and its `at: res://…gd:42` line through severity and proximity, where the tail kept neither.

WHY IT STILL EARNS ITS PLACE

Trimmed, it fixes three verified defects on the user's daily orchestration path:
- the false FAIL on slow green checks
- the middle-of-log "tail" for outputs over 1 MB
- timeouts and never-ran checks labeled as failures

It also puts the missing "why" on the card. It uses an already-measured engine only as a caller, costs nothing on green, needs no model, and touches no hard constraint.

ASIDES
- The repo's piper_worker.py self-test currently has one failure unrelated to this candidate ("expected 'piper: failed to parse event log:' in stderr"). The review-card scenario passes.
- The in-loop sandbox also keeps only the head (4 MB cap, sandbox.cpp:571). That is a separate, larger-threshold version of the same problem.

Back to the [index](README.md).
