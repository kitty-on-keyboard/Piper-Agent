# G13: Move Godoer's telemetry hook out of the renderers: one post-grade hook emitting typed, content-addressed failure packets, with cards delivered asynchronously on every surface including --json

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| UNVERIFIED | godoer | design | medium | L |

> **Not yet verified.** Open every cited line below and confirm the problem still holds before you implement. Drop or reshape the task if it does not.

Merged from these finder reports:

- distill-godoer#7 Distill bolted onto three presentation layers
- ideas-godoer#1 One shared analyze path (Godoer typed packets, collectors, async card queue, cards_banner)
- distill-godoer#3 No budget contract (non-blocking MCP delivery via with_notes)

## Problem

The hook lives in three renderers: the CLI text branch, MCP from_run, and watch. Each is a copy of try / auto_distill / print / except: pass.
- `godoer run --json`, the machine path SKILL.md tells agents to trust, never writes incidents or a diagnosis and has no distillation field.
- Several CLI paths call _emit without project=, so CLI route, explore and replay never distill, although their MCP equivalents do.
- The bare except-pass hides hook failures.
- The MCP reply blocks on the whole distill.

The only incident shape is SCRIPT_FATAL with a file and a line. Route stalls and probe failures arrive as synthetic USER_ERROR diagnostics with raw='' and no file; a finder reproduced a stall reaching distill as {kind SCRIPT_FATAL, file null, line null, source_context null}. Every new distill-style card would therefore need its own bespoke script.

## Why it matters

It gets the diagnosis to the surfaces cloud agents actually read (--json and every CLI and MCP path) and keeps red tool calls fast. It is the shared base the new card kinds (G09, G14-G16) plug into. It also stops per-surface drift, which is how the missing cleanup and the swallowed errors arose.

## Proposed fix

1. One hook, telemetry.on_graded_run(project, result), called where a run result for a project is finalized, beside write_digest and grading.
   - It returns None on green, after G07's cleanup.
   - Pass project through the CLI paths that drop it.
   - Replace the three except-pass blocks with the hook's own status.

2. Typed packets. godoer/telemetry/packet.py defines:
   Packet{schema:1, kind (script_fatal, engine_crash, parser_gap, route_stall, assertion_fail, human_note), packet_id = sha256(kind + canonical facts + evidence), facts, evidence spans capped at ~120 lines / 8 KB, input_fingerprint}.
   - packet_id plays the role of G07's fp.
   - A COLLECTORS map holds pure functions of (project, result). Each starts from a predicate the result already exposes, so a green run returns before any file I/O.

3. Non-blocking submit.
   - Write .godoer/telemetry/cards/pending/<packet_id>.json atomically. An existing id is a no-op, so a recurring failure is analyzed once.
   - Submit to Piper with wait=false and the ready path as out. Piper's daemon owns the queueing (G02), so Godoer runs no drainer process of its own.
   - Return one line: 'card queued: <kind> <id>'.
   - Packets refused because Piper is offline stay pending until the next hook call.

4. Delivery.
   - cards_banner sits beside notes_banner in with_notes and in CLI main. It prints ready cards (12 lines or fewer each), moves them to delivered/, and drops any whose input_fingerprint no longer matches.
   - --json gets 'distillation': {status, cards[...]} with the same typed fields.
   - play --report looks cards up by digest-bound packet_id.

## Evidence (file:line)

- godoer:godoer/cli.py:1770-1817 the --json payload is built and printed with no auto_distill call and no distillation key
- godoer:godoer/cli.py:1830-1836 only the text branch calls auto_distill, inside `except Exception: pass`
- godoer:godoer/cli.py:2088,2417,2492,2519,2544 call _emit(...) without project=, so the `project is not None` guard is false
- godoer:godoer/mcp_server.py:181-193 is the second copy (blocking, except-pass); godoer/watch.py:632-640 is the third
- godoer:godoer/telemetry/extractor.py:16 the kind comment lists SCRIPT_FATAL, FRAME_SPIKE, PHYSICS_ANOMALY; :132-141 every root cause becomes kind 'SCRIPT_FATAL'
- godoer:godoer/route.py:555-570 and godoer/probe.py:467-478 turn stalls and failed assertions into USER_ERROR diagnostics with raw='' and no source file
- godoer:skills/godoer/SKILL.md:41,109 tell agents to use and trust --json
- godoer:godoer/watch.py:297-310 notes_banner delivers notes and clears them; godoer/cli.py:2009 and godoer/mcp_result.py:184 with_notes are an existing next-command delivery channel

## How the finder suggested verifying it

Godoer pytest (scratchpad venv) with stub RunResult, ProbeResult and RouteResult:
- A green result writes nothing under .godoer/telemetry and never opens the socket (socket spy).
- `godoer run --json` output contains a distillation key, and the route and explore CLI paths invoke the hook.
- Packets are deterministic, byte-stable, and deduplicated by packet_id.
- A busy fake daemon leaves the packet pending, and no cold flag is passed.
- cards_banner delivers once, and drops a card after its cited file changes.

Back to the [index](README.md).
