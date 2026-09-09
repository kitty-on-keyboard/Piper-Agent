# Cloud orchestrator ↔ Piper local worker

**End goal.** Save cloud **output** tokens by having a fast local Qwen (A3B)
do the writing. The cloud agent stays the long-horizon brain: plans in detail,
points at files, reviews outcomes at a high level, troubleshoots/restarts Piper,
records issues, and keeps slicing until the big task is done.

```
┌─────────────────────┐         task packet          ┌──────────────────────┐
│ Cloud agent         │ ───────────────────────────► │ Piper CLI worker     │
│ (Cursor Grok / Bot) │                              │ (A3B on Apple Silicon)│
│                     │ ◄─────────────────────────── │                      │
│ plan · review · QA  │       result.json + diff     │ write · tools · loop │
│ restart · log issues│                              │                      │
└─────────────────────┘                              └──────────────────────┘
         │                                                      │
         └──────── repeat until horizon acceptance passes ──────┘
```

## Roles

| Who | Owns | Does not own |
|---|---|---|
| **Cloud** | Goal decomposition, file-level direction, acceptance criteria, high-level review, restart/troubleshoot, issue log, “are we done?” | Bulk code tokens, local tool thrash |
| **Piper (A3B)** | Edits, tool use, local iteration inside `cwd` | Long-horizon judgment, multi-repo strategy |

A3B is not the smartest — **packets must be fat and scoped**, and **every turn gets a cloud review**. Trust outcomes (diff + tests + `message`), not vibes.

## Loop (until horizon complete)

1. **Frame the horizon** — one sentence goal + acceptance checklist (tests, UX, “file X exists”).
2. **Pick the next slice** — smallest useful step; name the files; say what *not* to touch.
3. **Write a detailed packet** — see template below. Cloud burns *input* tokens here on purpose so Piper burns the *output* tokens locally.
4. **Dispatch** — `piper worker run --task …` (later: submit to keep-warm `serve`).
5. **Wait on process exit** — `0` ok, `1` error, `2` timeout, `3` bad packet.
6. **Review high-level**
   - Read `result.message` (summary only).
   - Skim `files_touched` / `git.diff` — expected paths? Surprises?
   - Run / check the slice’s acceptance (test command, open the HTML, etc.).
   - Do **not** re-read every line unless something smells wrong.
7. **Decide**
   - **Pass** → next slice (or mark horizon done).
   - **Fail (agent)** → thinner/clearer packet, same worker.
   - **Fail (infra)** → restart sidecar / clear lock / bump timeout; log the issue.
   - **Fail (model ceiling)** → cloud does that one hard edit itself, or shrink scope.
8. **Record** — append to an issue/progress log so the next cloud turn has memory.
9. Repeat from 2 until horizon acceptance is green.

## Packet template (`task.json`)

Cloud should write packets like a senior briefing a mid-level implementer:

```json
{
  "id": "horizon-slug/slice-003",
  "cwd": "/absolute/path/to/workspace",
  "mode": "agent",
  "model_dir": "/Users/dev/Desktop/Models/Qwen3.6-35B-A3B-MLX-4bit",
  "prompt": "…see prompt.md shape below…",
  "auto_approve_exec": true,
  "timeout_s": 600,
  "result_path": "/absolute/path/to/slice-003/result.json",
  "commit_think": true,
  "shadow_compact": true
}
```

### `prompt` / `prompt.md` shape (recommended sections)

```markdown
## Horizon context (short)
What the overall project is trying to achieve. 3–6 lines max.

## This slice only
Exactly what to do in this turn. One primary outcome.

## Files
- EDIT: path/a.py — what to change
- CREATE: path/b.html — what it should contain
- DO NOT TOUCH: path/c.py, secrets, lockfiles

## Constraints
- No new dependencies unless listed
- No drive-by refactors
- Prefer single-file / no CDN when possible (example)

## Done when
1. Concrete check (e.g. `index.html` exists and has a canvas game loop)
2. Optional command: `python -m pytest …` or “open in browser, paddle moves”
3. Short summary in your final answer: what changed + how to verify

## If stuck
Stop after N failed attempts. Explain blockers in the final message; do not invent scope.
```

**Why this saves cloud tokens:** the cloud’s expensive generation is *direction*, not thousands of lines of HTML/JS/Python. Piper emits those locally.

## Review rubric (cloud, keep it cheap)

For each `result.json`:

| Check | Pass signal | Fail → |
|---|---|---|
| Status | `status == "ok"` | restart or repacket |
| Paths | `files_touched` ⊆ expected (+ maybe new allowed) | ask undo / narrow |
| Diff size | not a 2k-line drive-by | reject slice, restate DO NOT TOUCH |
| Acceptance | done-when checks hold | new slice to fix, not a novel |
| Message | readable summary | ignore think-leak; fix worker later |

Log one line per slice: `slice-id | pass/fail | note`.

## Troubleshooting / restart (cloud owns this)

- **Timeout** → smaller slice or higher `timeout_s`; check for tool loops.
- **Worker crash / no result** → kill `lmp_sidecar`, clear stale locks, re-run same packet once.
- **Model thrash** → switch slice to “read-only plan” (`mode: plan`) then a tiny edit slice.
- **Repeated same miss** → record in `ISSUES.md`; cloud patches that spot or changes approach.
- **GPU contention** → never run IDE Piper + CLI worker together (one MLX process).

## Keep-warm (v1.1)

`piper worker serve` keeps A3B loaded between slices so the orchestrator loop
isn’t paying cold load every packet. Semantics stay the same: packet in →
result out → cloud reviews → next packet. Still one task at a time on 48GB.

## What “complete” means

Horizon is done when **the cloud’s acceptance checklist** is green — not when
Piper says it finished a slice. Piper can be wrong; the loop exists to catch that.

## Near-term build outs (product)

1. Packet + result schema stable (mostly done).
2. Cleaner `message` (strip think leakage) + reliable `git_diff` / `diff_stat`.
3. Optional `check` command in task.json → fill `result.test`.
4. `worker serve` keep-warm.
5. Thin orchestrator helper script: `dispatch → wait → print review card`
   (still driven by cloud decisions, not autonomy theater).

## Demo path that matches the vision

1. Horizon: “ship a tiny browser game with score + restart.”
2. Slice 1: create `index.html` Breakout (done in `piper_cli_tests/html_game`).
3. Slice 2: packet says “add a Restart button and high-score in localStorage; only edit index.html.”
4. Cloud reviews result + opens the file mentally / in browser.
5. Slice 3: polish copy / mobile note — or stop when acceptance passes.

That is the whole product thesis in miniature: **cloud directs, local writes, cloud verifies, repeat.**
