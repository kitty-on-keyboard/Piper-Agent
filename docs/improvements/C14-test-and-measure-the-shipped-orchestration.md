# C14: Test and measure the shipped orchestration path end to end: retire the Python mission fork, gate the C++ worker, add a real-model orchestration eval with per-slice timing

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| UNVERIFIED | piper | measurement | medium | L |

> **Not yet verified.** Open every cited line below and confirm the problem still holds before you implement. Drop or reshape the task if it does not.

Merged from these finder reports:

- Add a minimal end-to-end orchestration eval: piper packet, then piper dispatch, scored against an immutable checker
- Retire the Python mission driver and give the shipped C++ worker an end-to-end gate test
- Put a per-slice timing breakdown in result.json (warm/cold, load, prefill/reuse, decode, check)

## Problem

Nothing tests or measures the path the user actually runs: packet → dispatch → C++ worker → result.json → card.
- The agent eval drives the sidecar's JSON-RPC loop directly. It never produces or scores a result.json or a card.
- The gate's only worker integration test (`piper_worker.py self-test`) uses .py fake sidecars. main() skips the C++ worker for those, so every scenario runs run_mission instead. run_mission is a roughly 660-line Python copy of the worker that has drifted from execute_task_packet: it drops `check` and max_iterations and has no ask/answer flow. A packet with `check: false` dispatched through it prints 'test: not run' and 'verdict: PASS' with exit 0. In production the same copy is reachable through LMP_USE_CPP_WORKER=0 or a .py LMP_SIDECAR, which makes it a hidden false-success switch.
- execute_task_packet and the C++ LiveJournal sit outside lmp_surface and have no test. That is how C02 shipped green.
- result.json reports only wall_seconds, and it is taken before git collection and the check. Nothing records warm vs cold, load time, prefill vs reuse, decode, or check time, so the wins from C10, C13 and C18 cannot be confirmed in real runs.
- Documented commands drift from the parser. `piper distill --input` and the stdin form are documented, but both exit with code 3.

## Why it matters

Makes priorities 1-4 measurable on the real path: false_PASS (which must be 0), solve rate, turns, cold vs warm wall-clock, and the bytes the cloud reads per slice. It is the regression guard that would have caught C02 and the accounting gaps. It also removes roughly 660 lines of Python that must mirror the C++ behavior, a hidden false-success switch, and about 40 s from every gate run.

## Proposed fix

(a) Retire the Python mission fork.
- Delete run_mission, LiveJournal, apply_feature_flags, the Python collect_*, compose_message and archive helpers, the FAKE_* NDJSON templates, LMP_USE_CPP_WORKER and scripts/test_live_journal.py.
- Fail loudly if the C++ worker binary is missing.
- Move SIDECAR and the stdin/stdout/detach helpers out of agent_eval, so the CLI stops importing the eval harness.
- Rewrite the self-test fake as a fake worker binary that parses `--worker --task P [--auto-approve-*] [--orch-webhook U]` and either writes a C++-schema result.json with a chosen status and test block, or exits without one. Assert what the parent owns: argv forwarding, verdict mapping, orch.jsonl, and exactly one died.

(b) Make the shipped path testable.
- Change Session::backend to std::unique_ptr<model::InferenceBackend>.
- Move execute_task_packet and LiveJournal into lmp_surface.
- Add a gate test that runs packet → Agent(ScriptedBackend, lmp_mini_vocab) → result.json for: completed+green, max_turns+green, completed+red, backend_error, irreversible deny, ask timeout.

(c) Add scripts/orch_eval.py. For each task in evals/agent/tasks:
- Copy the workspace into a temp git repo and commit it as ground truth.
- Write prompt.md from the mission, with protect paths listed as DO NOT TOUCH.
- Run `piper packet … --check '<captured contract.check>'`, adding --protect once C09 lands.
- Run a timed `piper dispatch --json`, once cold and once with a warm server.
- Score against a fresh run of the captured checker, verify_task_integrity, and the true git delta.
- Emit verdict, solved, intact, false_PASS, false_FAIL, files_touched precision and recall, turns, result wall time vs dispatch wall time, and the bytes of card, result.json and git.diff.
- Pin false_PASS == 0 and a solve-rate floor beside pins.json.

(d) Add per-slice timing to result.json, collected through the on_turn hook: load_ms (0 when warm), a warm flag, prompt and prefill_reused tokens, prefill ms, decode ms, check_ms and git_ms. Measure wall_seconds at the very end, and add one line for it on the card.

(e) Add a doc-lint self-test that parses every `piper …` line in fenced code in PIPER.md, docs/AGENT_WAKE.md and the skill with build_parser(), so documented commands cannot drift from --help again.

## Evidence (file:line)

- scripts/agent_eval.py:425-457: build_start_request and drive_sidecar send lmp/start straight to the sidecar. evals/agent/README.md:66-73: scores only solved, completed, verified and intact. prove_deterministic_harness.py:1-14: checks only shapes, against synthetic results
- scripts/piper_worker.py:2560-2562: the C++ worker is skipped when LMP_USE_CPP_WORKER=0 or the sidecar path ends in .py. The self-test fakes are .py files (2755-2763; scenario 19 at 3331-3352)
- scripts/piper_worker.py:1589-1592: run_mission builds TaskContract(check=''). 1669-1677: empty test block. load_packet (80-172) drops check and max_iterations
- tests/gate/CMakeLists.txt:27-31: the comment says 'Fake sidecar only'. CI job 110125460270: test_piper_worker took 42.16 s of a 72.81 s gate
- src/surface/sidecar.cpp:1814-1905 (LiveJournal) and 1907-2367 (execute_task_packet) are outside lmp_surface (src/surface/CMakeLists.txt:1-2). src/surface/session.hpp:66: the MlxBackend member leaves no test seam, even though tests/loop already run the real Agent with ScriptedBackend and lmp_mini_vocab
- tests/surface/test_worker.cpp:1645-1663: promotion is tested on a hand-built status shape, which is why C02 shipped green
- scripts/agent_eval.py:708-717: the eval auto-replies to awaiting_user, while the worker blocks on it (sidecar.cpp:2111-2212)
- src/surface/sidecar.cpp:2226-2233: wall_seconds is taken before collect_git (2311) and run_check (2323). worker.hpp:106-128: RunResult has no timing breakdown. sidecar.cpp:840-873: ModelLoad{elapsed_ms} exists. agent.cpp:1384-1412: per-turn reuse and TTFT are already journaled
- PIPER.md:118,121 and SKILL.md:82 document `piper distill --input` and stdin, but the parser rejects both (scripts/piper_worker.py:2454; rc 3 when run)
- Repro (scratchpad repro_python_path.py): a packet with check `false` dispatched through a .py fake gives status ok, test.ran False, card 'verdict: PASS'

## How the finder suggested verifying it

Here:
- repro_python_path.py shows the fork's PASS-without-check today.
- After (a), `python3 scripts/piper_worker.py self-test` passes with the fake worker binary, and grep finds no run_mission or LMP_USE_CPP_WORKER.
- The orch_eval scorer can be unit-tested with synthetic result.json files and git workspaces; a PASS card with an edited protected file must count as false_PASS.
- The doc-lint self-test runs here and flags `piper distill --input`.

On the Mac:
- `ctest -L gate` passes with the new ScriptedBackend worker test (bump the manifest count).
- Real orch_eval runs complete both cold and warm.

Back to the [index](README.md).
