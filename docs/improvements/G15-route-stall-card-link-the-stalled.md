# G15: Route-stall card: link the stalled stage's action to control, connect, handler and change_scene call site, with cited lines

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| UNVERIFIED | both | feature | medium | M |

> **Not yet verified.** Open every cited line below and confirm the problem still holds before you implement. Drop or reshape the task if it does not.

Merged from these finder reports:

- ideas-godoer#3 Route-stall card

## Problem

Per the README, a broken scene path is the defect humans found first in seven friction logs, and route walks exist to catch it. When a route stalls, Godoer reports which scene was showing, but the two readings need opposite fixes:
- 'still on the previous scene' means the trigger never fired;
- 'a third scene' means it fired to the wrong place.

Today the stall reaches distill with no file, line or code; a finder reproduced {kind SCRIPT_FATAL, file null, line null, source_context null}. The model is told 'do not invent' with nothing to ground on.

Real games wire buttons in code (onready vars, connect calls), so a regex linker would be brittle. A model given 20-60 cited lines is not.

## Why it matters

Route stalls are rarer than script errors, but each costs many turns: the stage spec, the showing scene, its scripts and the autoloads. The card does that in one grounded step, and some cases resolve with no model call at all.

## Proposed fix

1. Hook: a G13 collector with the predicate `isinstance(r, RouteResult) and r.stalled is not None`. Completed routes are untouched.

2. Collector. facts = {stage index, wanted, showing, visited, waited, within, failed stage checks, the previous stage's do/drive}. Evidence:
   (a) node paths parsed from the previous stage's `do` expressions (click_control/click_ref/act literals);
   (b) the showing scene's .tscn via verify/tscn.py _Scene: the clicked node's line plus its [connection] rows;
   (c) scripts attached to the showing scene's nodes and to autoloads, keeping line-numbered hits for `.connect(`, `func _on_`, `change_scene_to_file`, `change_scene_to_packed` and `get_tree().quit`, with ±3 lines each;
   (d) for each call site, target_literal_matches_wanted.

3. Deterministic short-circuit: exactly one change_scene call site whose literal differs from wanted gives status=deterministic, verdict=wrong_target, with no model call.

4. Otherwise, the report schema (a Piper job kind):
   - verdict: enum {trigger_not_wired, handler_guarded, wrong_target, went_elsewhere, action_missed_control, deadline_too_short, deferred_load_pending, unknown}
   - chain: [{ref, line, role control|connect|handler|change_scene}], each element validated as inside the evidence
   - fix: {ref, line, change}
   - next_command: rerun the same route spec

## Evidence (file:line)

- godoer:godoer/route.py:555-570 the stall becomes a USER_ERROR with raw='' and no source file
- godoer:README.md:387 a broken scene path is 'the only defect in seven of this repo's friction logs that a human found first'; :415 the two stall readings 'want opposite fixes and are identical from inside the game'
- godoer:games/barrow/scripts/main_menu.gd:15-30 wiring is in code: `_begin.pressed.connect(_on_begin)`, then _on_begin calls get_tree().change_scene_to_file(res://scenes/level.tscn)
- godoer:godoer/verify/tscn.py:209 _Scene and :226 by_path() provide node lookup in the showing scene's .tscn

## How the finder suggested verifying it

Here: build RouteResults from synthetic GODOER_ROUTE_STALLED stdout against copies of games/barrow. Mutate main_menu.gd three ways: remove the connect line, change the target literal, and add a third-scene target. Assert that the collector cites main_menu.gd:15-30, that the short-circuit fires on the literal mismatch, and that a completed route makes no packet.

Mac: about 10 seeded stall variants across games/, scoring verdict accuracy and chain grounding.

Back to the [index](README.md).
