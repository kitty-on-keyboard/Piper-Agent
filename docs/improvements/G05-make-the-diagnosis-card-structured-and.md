# G05: Make the diagnosis card structured and grounded: one grammar-enforced report tool with enum incident ids, a deterministic grounding check, and a versioned report with per-incident status

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| VERIFIED (reproduced) | both | feature | high | M |

> **Verified.** The problem and fix below are the skeptic's corrected versions. They replace the finder's original claim where the two differ.

Merged from these finder reports:

- distill-piper#4 Diagnosis card is unstructured, ungrounded free text (schema, grounding, per-incident status)
- service-design#2 Structured output from the tool-call grammar (report tool, enum ids, RequireCallMask)
- distill-godoer#4 Distill should be a tool-less, schema-constrained diagnosis (grounding validator, bounded card)
- ideas-godoer#1 One shared analyze path (grounding of cited spans, shared card envelope)

## Problem

Distill's output has no typed contract and no record of where each piece of text came from.

1. Warm path: piper_worker.py:1300-1308 takes `message or error` from result.json as the diagnosis without reading result.status. Stall bookends, 'status=error; reason=...' and 'Daemon finished but no result.json was produced' are all filed as diagnoses. The report status is hard-coded to 'diagnosed' (:1353). Only cold-path subprocess failures get a per-item status.
2. Nothing checks the text against the incident or its source window.
3. Godoer ignores the report status and prints the text verbatim (distiller.py:127-128) into the CLI/watch output and the MCP reply (mcp_server.py:184-193).

Correction to 'unbounded': each item is capped upstream. The warm path caps at 2000 chars via strip_think_leak. The cold path caps at --max-tokens 1500; there, a think block cut by the cap has no </think>, so split() keeps the whole block. The card as a whole has no bound.

The decode-side claims hold. TurnGrammar's Text phase accepts <|im_end|> with no call required (grammar.cpp:107-109). ToolCallGuard enforces required params, Booleans and Text/Number enums.

Two parts of the proposed fix are wrong:
- Array item types are NOT enforced (items_type only affects tools_json), so 'cited_lines: Array of Number' accepts ["a",{}] and [400].
- The 'before is a verbatim substring' check refers to a field the proposed schema does not define.

The decode-time half also has no host today. The daemon only handles ping, stop and run (sidecar.cpp:2543-2582), and run is a full agent loop with auto_approve_all. It depends on G01's analyze job.

## Why it matters

This is the deliverable the user says they like: a compact, grounded card that a cloud agent or a human can act on. Enum-restricted ids and required fields mean the model cannot cite a nonexistent incident or omit the fix. 'Insufficient evidence' becomes a legal typed answer instead of pressure to invent. The deterministic check catches what decoding cannot.

## Fix to ship

Ship this in the same change as G01, as the output contract of its analyze job. Do not bolt it onto the agent run path.

**Piper C++ (verify on Mac):**
1. Add RequireCallMask to src/model/grammar.{hpp,cpp}, taking (MaskSource& inner, TurnGrammar& g, tok, prose_cap):
   - In the Text phase before any call, deny <|im_end|>.
   - Once text_ids().size() >= prose_cap, allow only <tool_call>.
   - After the call closes, allow only <|im_end|>.
   - If the forced token is illegal in inner, return inner.mask() (the same floor ThinkCapMask uses).
   - Not block-stable at the cap or after the call.
   - Delegate checkpoint, rollback, probe_advance, is_block_boundary and budget_exhausted.
   - Compose RequireCallMask(ToolCapMask(ThinkCapMask(TurnGrammar))) in the analyze job only; the agent path stays unchanged.
   - Add tests in tests/model/test_grammar.cpp next to the ThinkCapMask/ToolCapMask cases.
2. In the analyze handler, build one ToolSpec, report_diagnosis, per incident:
   - line_start and line_end: Text, required, with enum_values set to the window's line numbers parsed from the source_context gutter. Use Text, not Number: Registry::tools_json (registry.cpp:563-573) quotes every enum value, so a Number enum would be advertised as strings but enforced as numbers.
   - root_cause: Text, required.
   - fix_kind: Text, enum code|config|engine|unknown.
   - after: Text, required.
   - insufficient_evidence: Boolean, required.
   - Advertise it through the existing ToolDecl -> tools_json path.
   - Return {status: ok|no_report|length_capped|cancelled|error, fields}, mapped from GenStatus plus grammar.has_tool_call(). Never return free text.
3. Do not invoke the model for incidents without a source window (status no_source).
4. Delete the mlx_lm cold path, so --allow-cold runs the same in-process analyze.

**Piper Python (testable here), scripts/piper_worker.py cmd_distill:**
1. parse_window(source_context) using the regex `^(?: > |   )\s*(\d+) \| (.*)$`.
2. judge(incident, job) is the authority:
   - Re-check that both lines are in the window and line_start <= line_end. The enum pins them, but LMP_ENUM_MASK=0 disables enum masking.
   - Require a non-empty `after` when fix_kind=code.
   - Derive `before` from window lines a..b.
3. Write the report as `{schema:'piper.distill/1', status: clean|diagnosed|partial|undiagnosed|skipped_daemon_offline, items:[...]}`.
   - Each item: incident_id, file, line, kind, status (ok|abstained|no_source|rejected|length_capped|timeout|error), reason, lines, before, after, root_cause, fix_kind.
   - Write it atomically after each item, so Godoer's 180 s timeout still finds the finished items.
   - Never put raw text or error strings in typed fields.
4. Add contract tests against a fake daemon socket.

**Godoer, godoer/telemetry/distiller.py:**
1. Require the schema version, and map the report status to DistillationResult.status.
2. Render only typed fields:
   - About 4 lines per ok item: `[INC-001] res://p.gd L163-164 code (lines verified)` / `cause: <=200 chars` / `- before` / `+ after`, each block capped.
   - One line per item that is not ok, without repeating the error the run report already shows.
   - At most 5 items, then '+N more in .godoer/distilled_diagnosis.json'.
3. Fix the eager Path(model_dir).name at :264.
4. Update tests/test_telemetry_distiller.py.

**Drop:** the later-card-kinds envelope, deferred_daemon_busy, confidence, the cited_lines array, and the model-quoted 'before'.

## Evidence (file:line)

- piper:scripts/piper_worker.py:1330-1342 clean_diagnosis = raw_output.split('</think>')[-1]; entries hold only the free-text diagnosis and raw_output
- piper:scripts/piper_worker.py:1256-1262 grounding is requested only in prose ('Do NOT invent, assume, or hallucinate'); :1352-1357 report status is always 'diagnosed'
- piper:src/model/grammar.hpp:29-32 ToolCallGuard is schema-aware byte by byte, with 1000/1000 valid constrained generations and a 17.7 ns mask cost; third_party/parsephony/include/parsephony/toolcall.hpp:27,61-66 enum_values are enforced in the value-phase mask
- piper:src/model/grammar.cpp:105-109 advance_text accepts <|im_end|> unconditionally in the Text phase, so a call is never required
- piper:src/model/grammar.hpp:222 ThinkCapMask and :271 ToolCapMask wrap TurnGrammar without changing it; that is the pattern a RequireCallMask follows
- piper:src/model/backend.hpp:112-114 GenStatus::LengthCapped is distinct from a complete generation
- godoer:godoer/telemetry/extractor.py:59-75 the source window carries an exact line gutter (' > 164 | ...'), so cited lines can be checked deterministically
- godoer:godoer/telemetry/distiller.py:104-133 format_distillation_card prints d['diagnosis'] verbatim and unbounded; godoer/mcp_server.py:184-193 appends it to the MCP reply

## Skeptic verdict

real=True, reproduced=True, keep=True, root-cause fix=True, already handled=False, violates constraint=False

Every cited file:line checks out. Reproductions are in scratchpad/g05.

**(a) Fake daemon.** I ran a fake unix-socket daemon against the real scripts/piper_worker.py distill.
- Four modes: stalled, error, no result.json, and ok carrying a hallucinated diagnosis ('line 400 of enemy.gd' for a player.gd:164 incident).
- All four give report.status='diagnosed' with the fragment filed as the diagnosis.
- Godoer's format_distillation_card renders all four the same way. Nothing on the card tells a cloud agent which is real.

**(b) Compiled parsephony ToolCallGuard (g++) against the proposed schema.**
- Rejected: INC-999 outside the enum, fix_kind=refactor, insufficient_evidence=maybe, and a missing required 'fix'.
- Accepted: cited_lines=["a",{"x":1}] and [400].
- line_start/line_end with enum_values set to the window's line numbers rejects 400, '16' and '164.0'. This works for both Text and Number types, so line grounding can be enforced while decoding, not only checked afterwards.

**(c) Godoer telemetry tests** (/root/.local/bin/pytest with PYTHONPATH set): 7 pass, 1 fails off-Mac.
- Cause: distiller.py:264 `diag_data.get('model', Path(model_dir).name)` is evaluated eagerly. It raises when find_local_model() returns None, which turns a successful warm distill into status=error.

**Arguments against, and why it survives:**
1. **Dependency.** Without G01 there is nowhere to register a report tool. Adding one to the agent run path would be a band-aid. This should be G01's output contract, shipped in the same change, not a rework later.
2. **Speculative scope.** Drop the shared envelope (verdict, next_command, 'later card kinds'); it gives orchestration nothing today. Drop deferred_daemon_busy because nothing can detect a busy daemon: the serve loop handles one client at a time (sidecar.cpp:2527-2582), so a ping to a busy daemon times out and reads as offline.
3. **Schema.**
   - Drop confidence: an uncalibrated self-report invites misplaced trust.
   - Drop incident_id: there is one incident per job, and the harness already binds it.
   - Drop the cited_lines array: its item types are not enforced.
   - Drop the model-quoted 'before': derive it from the window instead.
4. **Card duplication.** The MCP reply already prints the root causes with file:line above the card (runner.py:578-580). An item that fails should get one status line, not the same facts repeated.
5. **Wording.** 'Grounded' here only means the cited lines exist in the window, not that the fix is correct. Label it 'lines verified'.
6. **Cold path.** The mlx_lm cold path cannot honor a grammar contract and loads a second model, so it must go.

**Why RequireCallMask is justified:** grammar.hpp:212-215 records measured turns that produced no tool call (TextOnly). The ThinkCapMask/ToolCapMask wrapper pattern and their tests in tests/model/test_grammar.cpp give a template to copy.

**Constraints:** none violated (one model, no subagents, no git, C++20). A 27B/A3B model reliably fills a 5-field XML tool call of the kind it was trained on when the mask enforces the shape, and abstaining is a legal answer.

**Scratchpad collision:** my first repro overwrote a scratchpad/fake_daemon.py that another agent wrote earlier. That agent's process, started at 16:03, is still running from it. My files are now in scratchpad/g05.

Back to the [index](README.md).
