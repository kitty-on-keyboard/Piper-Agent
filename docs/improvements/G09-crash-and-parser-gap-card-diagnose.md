# G09: Crash and parser-gap card: diagnose the failed runs where Godoer today says 'read the raw output'

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| VERIFIED (reproduced) | both | feature | medium | S |

> **Verified.** The problem and fix below are the skeptic's corrected versions. They replace the finder's original claim where the two differ.

Merged from these finder reports:

- ideas-godoer#2 Crash & parser-gap card

## Problem

The substance is true, but several details are wrong.

(1) Native crash. DEBUG_ENABLED builds include the editor/debug binary godoer runs. In them the fork's crash handler prints to stderr (godot-godoer/platform/macos/crash_handler_macos.mm:102-201; linuxbsd and windows print the same strings):
- a '====' rule, then 'handle_crash: Program crashed with signal N' and the engine version;
- C++ frames symbolized by atos, shaped '[i] Fn(..) (in godot..) (/src/x.cpp:L)';
- '-- END OF C++ BACKTRACE --';
- a 'GDScript backtrace (most recent call first):' block with frames like '[0] func (res://file.gd:L)', closed by '-- END OF GDSCRIPT BACKTRACE --'.
Then it calls abort().

godoer/errors.py does not know this shape. parse_diagnostics finds nothing and root_causes is empty. report() then says 'killed by SIGABRT ... read the raw output', which is the wrong signal: abort() hides the real SIGSEGV. Downstream, the extractor makes no incident, auto_distill returns None, last_play.json records 0 fatals, and watch prints 'FAILED: 0 fatal(s)'.

RunResult.printed puts stderr after all of stdout. After 200 game prints the crash block starts at line 201, so MCP with_output (head 60) and CLI _printed_block (head 40) both cut it off. That includes a res:// culprit the engine already printed.

The finder's synthetic input was unrealistic: exit -11 and '-- END OF BACKTRACE --'. A handled crash on macOS exits -6, and the delimiter is '-- END OF C++ BACKTRACE --'.

(2) Parser gap. It has the same no-incident outcome. But it almost never happens: engine/README.md says the count and the parse agreed on all 507 suite runs after the rid_owner.h fix.

(3) The predicate is not exactly the two runner.py branches. When a gap and parsed errors occur together, report() returns early (runner.py:544-554) and hides the parsed root cause. In that case root_causes is non-empty, so the proposed collector would not fire. Reproduced.

## Why it matters

This is the most direct extension of the feature the user already likes. It targets the failures that cost a cloud agent the most: no file:line, an engine backtrace, an unknown shape. A parser_gap card also turns 'treat the gap as a bug in the parser' into a ready fixture span.

## Fix to ship

Keep the problem and replace the LLM card with a deterministic parser fix. The change is Godoer-only; Piper needs nothing.

1. godoer/errors.py
- Add Category.ENGINE_CRASH.
- In parse_diagnostics, recognize the fork's crash block as ONE Diagnostic. It runs from `^handle_crash: Program crashed with signal (\d+)$` through '-- END OF C++ BACKTRACE --', plus any following '<Lang> backtrace (most recent call first):' frames up to '-- END OF <LANG> BACKTRACE --' and their '====' rules.
- Its message reads 'the engine crashed with SIG<name> in <first C++ frame that is not _sigtramp/libsystem>', taken verbatim.
- Set engine_file and engine_line from atos's '(path:line)' tail when it is present.
- Build backtrace from the script frames with the existing _FRAME regex. _finalize then sets source_file and source_line from frame 0 with no new logic. Set raw to the whole block.
- game_output must drop the same block, including the leading rule, so the parser and game_output keep agreeing.
- recognized_errors must exclude ENGINE_CRASH. The engine's counter never sees crash-handler prints, so this keeps the count cross-check exact.

2. godoer/runner.py report()
- In the unexplained branch, list the parsed root causes first, then the gap sentence.
- Then quote the leftover engine-stream lines, numbered: the tail (at most 20 lines) of game_output(self.stderr). Engine errors go to stderr, so this is the unknown shape plus any printerr. It is the exact fixture span for tests/test_errors.py.

3. godoer/mcp_server.py with_output and godoer/cli.py _printed_block
- When not result.ok, show a head-plus-tail excerpt instead of the head. play_digest._excerpt already keeps the tail. This way the last lines before an abort, kill or quit(N) survive.

4. godoer/telemetry/extractor.py
- Use kind 'ENGINE_CRASH' for that category, and put the real signal and the top 3 C++ frames in metadata.
- Send an incident to distill only when it has a res:// frame or source_context. The existing piper distill already consumes {kind, message, file, line, source_context}.
- An engine-internal crash with no script on the stack gets a deterministic card and no model call. The card says it is an engine or environment fault and to attach the backtrace for engine/patches.

5. Tests (tests/test_errors.py, tests/test_runner.py, tests/test_telemetry_extractor.py)
- Fixtures in the exact crash_handler_macos.mm format: with a GDScript backtrace, without one, after 200 prints, and an OS.crash('x') run, which adds a parsed 'ERROR: FATAL' line before the block.
- Assert: one engine_crash root cause at res://...:346; game_output excludes the block; recognized_errors is unchanged; report names SIGSEGV, not SIGABRT; extract_incidents yields 1 incident; a green run is unchanged; a partial gap shows the parsed cause plus the leftover lines.
- On the Mac, capture 3-5 real blocks into tests/fixtures: OS.crash() called from _process, and `kill -SEGV <pid>` mid-run.

Drop the verdict enum, the model-produced parser_fixture, the new Piper job kind, and the dependency on G01, G05 and G13.

## Evidence (file:line)

- godoer:godoer/runner.py:544-554 the unexplained branch: 'diagnostic(s) are in a shape godoer/errors.py does not parse -- read the raw output, and treat the gap as a bug in the parser'
- godoer:godoer/runner.py:556-571 the no-diagnostic branch: 'FAILED ... {exit_reason}, and godoer parsed no diagnostic that explains it -- read the raw output'
- godoer:godoer/telemetry/extractor.py:95-107 incidents come only from run_result.root_causes; godoer/telemetry/distiller.py:184-185 returns None when nothing was extracted
- godoer:godoer/mcp_server.py:169-179 with_output keeps lines[:60], the head, so a crash after 60 game prints is cut off
- godoer:godoer/errors.py:277 every Diagnostic keeps raw; :952 game_output strips the parsed blocks, so the residual unparsed lines can be computed deterministically
- Finder repro (scratchpad repro1.py, repro3.py): exit -11 and GODOT_ERROR_COUNT|2 with 0 parsed both give auto_distill None; the handle_crash block lands in RunResult.printed

## Skeptic verdict

real=True, reproduced=True, keep=True, root-cause fix=False, already handled=False, violates constraint=False

I opened every citation and each matches the code: runner.py:544-571, extractor.py:95-107, distiller.py:184-185, mcp_server.py:169-179, errors.py:277 and :952. auto_distill is called from mcp_server.py:186, cli.py:1832 and watch.py:636.

**Reproductions.** pytest is not installed system-wide. I used the existing venv in the scratchpad. Scripts are in scratchpad/g09:
- **repro_crash.py.** Uses the fork's real crash format after 200 prints. Result: 0 diagnostics, report says SIGABRT, the crash block is at printed line 201, it is missing from MCP head-60 and CLI head-40, 0 incidents, auto_distill None.
- **repro_gap.py.** Engine count 2, recognized 0, so auto_distill is None and the unknown lines are missing from head-60. But game_output(stderr) isolates exactly those lines plus printerr output, with no model needed.
- **repro_partial_gap.py.** report() hides a parsed invalid_property error at res://scripts/player.gd:164.
- **proto_det.py.** About 40 lines of crash-block parsing produce one Diagnostic. With it, the existing report() prints 'res://scripts/vehicle/vehicle_rig.gd:346 in _spawn_wheels(): the engine crashed with SIGSEGV in Node::get_child'. The existing extractor emits a grounded incident with a source window. No Piper change is needed.

**Why the proposed design should not ship:**
- **(a) It is a band-aid.** The root cause is a parser that does not know the crash block. This is the same class of bug as the missing SHADER ERROR label that CLAUDE.md records. A side-channel LLM card would leave report(), the --json errors list, last_play.json fatals and watch's fail-closed line all reporting 0 errors.
- **(b) The model jobs are redundant or unreliable.**
  - The verdict is deterministic: unexplained_errors > 0 means a parser gap; a crash block with script frames means a crash from a script call; a crash block without them means an engine-internal crash; a positive exit other than 70 means quit(N).
  - Separating unknown-shape lines from game prints cannot be verified. The block-count check is weak, and the stderr residual already narrows the candidates.
  - Asking for a culprit on an engine-internal crash invites a made-up script line.
  - On local_model_can_do_it: a 27B could classify the closed set, but the classification is deterministic anyway, and the separation and culprit jobs are not reliable.
- **(c) It depends on G01, G05 and G13, which have not landed.**
- **(d) Impact is overstated.** Parser gaps are about 0 in 507 runs. Native crashes in the repo's history are few (the patch 0005 logger crash and the null-function-pointer import crash), though each was expensive. I rate impact medium, not high.

No hard constraint is violated, and neither repo already handles this.

**Adjacent bug (not part of G09).** distiller.py:264 evaluates `diag_data.get("model", Path(model_dir).name)` eagerly. When find_local_model() returns None, Path(None) raises, so every successful warm-daemon distill is reported as status=error. tests/test_telemetry_distiller.py::test_auto_distill_daemon_online_invokes_distill fails here: 1 failed, 7 passed. Baseline for test_errors and test_runner: 189 passed, 33 skipped.

Back to the [index](README.md).
