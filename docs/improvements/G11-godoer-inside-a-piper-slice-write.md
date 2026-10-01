# G11: Godoer inside a Piper slice: write incidents into the slice and show them on the review card instead of calling the busy model

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| UNVERIFIED | both | feature | medium | M |

> **Not yet verified.** Open every cited line below and confirm the problem still holds before you implement. Drop or reshape the task if it does not.

Merged from these finder reports:

- ideas-orchestration#4 Godoer inside a Piper slice: defer distill, incidents to slice card
- distill-piper#1 Busy daemon is misread as offline (piper review as consumer of deferred diagnoses)

## Problem

When a failing godot_run happens inside a Piper slice (trust_mcp godoer), the only model is busy running that slice. Today Godoer's 1 s probe times out and appends 'Piper keep-warm daemon offline ... Run piper worker serve' to the MCP reply that the exec-auto-approved local agent reads. A finder reproduced this, and it matches the busy-socket repro in G02.

If GODOER_AUTO_DISTILL=1 is set in the server's .mcp.json env, which reaches the child unfiltered, Godoer also adds --allow-cold.

Even once G02 turns this into 'queued', the deterministic incidents never reach result.json or the review card. These are Godot res:// file:line fatals, which log_triage does not parse. The cloud reviewer has to dig through .godoer/ to learn that the game crashed during the slice.

## Why it matters

The user runs Godoer slices heavily, and this joins their two workflows. Runtime fatals land on the card the cloud already reads. The in-loop agent gets an actionable deterministic summary instead of false advice. The model runs only after the slice, and only on failure.

## Proposed fix

1. Piper: build_start_message adds explicit env PIPER_SLICE_DIR=<result_dir> and PIPER_SLICE_ID=<id> for trusted MCP servers, using the same mechanism as LMP_WARM_LSP.

2. Godoer, when PIPER_SLICE_DIR is set:
   - The telemetry hook still writes its own incidents, and also merges them into $PIPER_SLICE_DIR/godoer_incidents.json, deduplicated by site or fp (G07).
   - It opens no socket and launches no model call.
   - It returns status deferred_to_piper_slice with a deterministic summary of 5 lines or fewer (id, kind, res://file:line, message) that the in-loop agent can act on.

3. Piper post-run (sidecar.cpp near the collection at 2298): if godoer_incidents.json exists, add incidents:{count, items<=5} to result.json. The card prints one line per incident, e.g. 'incidents: SCRIPT_FATAL res://player.gd:4 Invalid call ... Nil'.

4. After the slice, with the daemon free, dispatch or review queues one G01 analyze job for those incidents, with its out path in the slice dir, and the card links the diagnosis. `piper review` thereby becomes the consumer of deferred diagnoses.

Green runs write no file and add no card lines.

## Evidence (file:line)

- godoer:godoer/mcp_server.py:181-193 from_run runs auto_distill on every failed run and appends its summary to the MCP reply
- godoer:godoer/telemetry/distiller.py:203-214 the offline summary says 'Piper keep-warm daemon offline (~/.piper/worker.sock). Run piper worker serve'; :204,229-230 GODOER_AUTO_DISTILL=1 adds --allow-cold
- piper:src/surface/worker.cpp:509-541 trusted MCP servers get explicit env (from .mcp.json env, plus the LMP_WARM_LSP and GODOT_BIN pattern); src/mcp/spawn_env.cpp:64-66 the denied prefixes include LMP_ but not GODOER_; :146-151 explicit extra env is merged unfiltered
- piper:src/surface/sidecar.cpp:2298-2307 the post-run collection point; scripts/piper_worker.py:1006-1018 the card has no incidents line
- Reproduced here: the busy-socket probe returns False after 1.00 s (see G02). A finder reproduced auto_distill returning skipped_daemon_offline with the 'Run piper worker serve' summary

## How the finder suggested verifying it

Here:
- A Godoer test with PIPER_SLICE_DIR set asserts no socket probe, no subprocess, status deferred_to_piper_slice, and the slice file written (pytest from the scratchpad venv).
- A piper_worker.py self-test puts a fixture godoer_incidents.json in the result dir and checks that it appears in result.json and on the card, and that a green fixture leaves the card byte-identical.

Mac: a trusted-MCP run confirms the env reaches the Godoer child.

Back to the [index](README.md).
