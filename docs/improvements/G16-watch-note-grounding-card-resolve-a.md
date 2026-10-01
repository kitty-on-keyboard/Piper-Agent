# G16: Watch-note grounding card: resolve a person's 'that tree is too big' to node path, .tscn line and property before the agent's next command

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| UNVERIFIED | both | feature | medium | M |

> **Not yet verified.** Open every cited line below and confirm the problem still holds before you implement. Drop or reshape the task if it does not.

Merged from these finder reports:

- ideas-godoer#5 Watch-note grounding card

## Problem

Watch mode exists so a person can say 'not like that' and an agent can act on it. The rig's own comment says a note it cannot resolve to a node is a note it cannot act on.

Yet the note payload carries only name, type and position for each visible node: no node path, size, screen rect or owning scene. The banner shows only the first 8 names.

Resolving 'that tree' among dozens of meshes, and mapping 'too big' to a property, is left to the cloud agent. That typically costs several inspect, search and capture turns.

## Why it matters

It turns the only human-in-the-loop channel from free text into an actionable target. Entity resolution over a closed candidate list plus an intent enum suits a local model, and the validator stops it inventing a node. Nothing runs when no note is filed.

## Proposed fix

1. Deterministic prerequisite in the rig: _objects() also emits
   - path (scene-relative);
   - scene_file (owner.scene_file_path);
   - AABB size;
   - screen_rect (bbox corners projected through the active Camera3D; for a Control, get_global_rect()).

2. Hook: WatchSession checks session_dir/notes on its existing heartbeat and submits a human_note packet (G13) for each new note. The card is then ready by the time notes_banner delivers the note. With no notes, nothing runs.

3. Collector. facts = {note text, scene, frame path}. Evidence = the candidate table (path, type, size, screen_rect, scene_file), capped at the 60 nodes nearest the screen centre or matching nouns in the note, plus each candidate's .tscn node-header line and attached script path.

4. Report schema (a Piper job kind):
   - intent: enum {scale, position, rotation_facing, material_color, lighting, visibility_missing, extra_object, behavior, input_feel, ui_text, other}
   - targets: [{node_path, scene_file, tscn_line, why}], at most 3, validated as a subset of the candidate table
   - property_hint: enum
   - next_command

Behavior and feel intents return targets plus a script ref but no property change, so the model never authors gameplay.

## Evidence (file:line)

- godoer:godoer/runtime/watch_rig.gd:514-521 'A note it cannot resolve to a node is a note it cannot act on.'
- godoer:godoer/runtime/watch_rig.gd:549-565 _objects() emits only {name, type, position}: no node path, AABB, screen rect or scene_file
- godoer:godoer/watch.py:249-256 Note.report shows objects[:8] names, then '(+N more)'; :297-310 notes_banner delivers and clears notes on the next engine-bound command
- godoer:godoer/mcp_result.py:184 with_notes and godoer/cli.py:2009 are the delivery points the card would ride on

## How the finder suggested verifying it

Here: write synthetic note JSONs, with the extended object fields, into .godoer/watch/notes. Assert the packet, the subset validator, delivery through notes_banner, and that no notes means no packet.

Mac (the rig change needs Godot): file about 20 notes against games/ scenes ('the lantern floats', 'that house is huge', 'too dark here') and hand-label the target node. Require at least 85% top-1 target accuracy with zero out-of-table targets.

Back to the [index](README.md).
