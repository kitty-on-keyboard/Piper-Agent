# Piper Local Worker — Orchestrator Guide & Parent Runbook

## Overview
**Core Principle:** Cloud directs, local writes, cloud verifies, repeat.

Piper is a fast headless coding worker running locally on Apple Silicon (via MLX). It executes edits, shell commands, and file operations inside `cwd` with zero cloud output token cost.

The cloud orchestrator (Cursor, Claude, Gemini, Antigravity, or custom script) acts as the high-level brain:
- Maintains the long-horizon plan and acceptance criteria.
- Decomposes complex tasks into bounded slices (1–3 files per slice).
- Directs Piper with a slice brief (`prompt.md`). The harness emits `task.json`.
- Verifies outcomes with the review card, the packet `check`, and `result.json`.
- Loops until the entire mission is verified complete.

## The Long-Horizon Execution Loop
```
┌─────────────────────────┐    prompt.md → task.json    ┌──────────────────────────┐
│   Cloud Orchestrator    │ ───────────────────────────► │       Piper Worker       │
│  (Cursor, Claude, etc.) │                              │  (MLX on Apple Silicon)  │
│                         │ ◄─────────────────────────── │                          │
│ plan · review · verify  │   review card + result.json  │  edits · tools · loop    │
└─────────────────────────┘                              └──────────────────────────┘
             │                                                         │
             └────────── repeat until horizon acceptance passes ───────┘
```

### Roles

| Who | Owns | Does not own |
|---|---|---|
| **Cloud Orchestrator** | Goal decomposition, file-level direction, acceptance criteria, high-level review, troubleshooting, "are we done?" | Packet JSON shape, bulk code generation tokens, local tool thrash |
| **Piper Worker** | Edits, tool execution, test commands, local iteration inside `cwd` | Long-horizon judgment, multi-repo strategy |

Local models work best on scoped slices: **the brief must be specific**, and **every slice gets an orchestrator review**. Trust the review card, the `check`, and `result.json`.

### Step-by-Step Procedure

1. **Frame the Horizon**
   Define the overarching objective and an acceptance checklist (e.g. unit tests pass, new command works, UI renders).

2. **Write `prompt.md` for this slice**
   Pick the smallest incremental step. Scope it to 1–3 files. The cloud writes this file only:
   - Horizon context (short).
   - This slice only — one outcome.
   - Files: EDIT, CREATE, and DO NOT TOUCH.
   - The acceptance command.
   - When to stop if stuck.

3. **Emit `task.json`**
   Do not hand-write the packet. The harness owns the shape.
   ```bash
   piper mcp-list --cwd /abs/ws                 # only when the slice needs MCP
   piper packet --id slice-001 --cwd /abs/ws \
     --prompt-file prompt.md --check "ctest -R test_validator" \
     --out task.json
   piper packet ... --trust-mcp godoer          # names must exist in cwd .mcp.json
   ```
   The emitter writes `id`, `cwd`, `prompt`, auto-approve flags, `timeout_s` (default 600), and `result_path` (sibling `result.json` unless `--result-path` is set). Optional `--model-dir` and `--check`.
   - **`check`**: operator acceptance command. Also becomes `verify_contract` during the run. A green post-run check yields `status=ok` / wake `done` even if the loop hit `max_turns` without `completed=true` (not a crash). Timeouts and irreversible denials stay failures.
   - **`max_iterations`**: turn budget sent to the agent loop. Default **30**; default **60** when `trust_mcp` is set (Godoer-heavy). Raise it for a long slice — no rebuild.
   - **`trust_mcp`**: explicit server names from `piper mcp-list`. No guessed JSON array.

4. **Dispatch**
   Default parent launch. Attached: wait, then print the review card.
   ```bash
   piper dispatch --task task.json
   # Unattended irreversible tools (godot_project, delete_file, whole-file overwrite):
   piper dispatch --task task.json --auto-approve-irreversible
   # Or approve exec + writes + irreversible:
   piper dispatch --task task.json --auto-approve-all
   ```
   Exit codes:
   - `0`: Completed normally.
   - `1`: Worker error.
   - `2`: Execution timed out (`timeout_s`).
   - `3`: Invalid task packet or missing wake URL for a detached run.

   Lower-level attached run, when you are not using the review card helper: `piper run --task task.json` (same as `piper worker run --task task.json`). Keep weights warm across slices with `piper worker serve`, then `piper worker run`. `piper dispatch` does not detach.

   Detached or background (`--detach`, nohup, screen) requires a wake URL before start. Resolve it with `piper wake-url`. Do not copy a URL from the panel.
   - `--orch-webhook URL`, or
   - task field `orch_webhook`, or
   - env `LMP_ORCH_WEBHOOK`, or
   - file `.piper/orch_webhook` (written by `piper_ui`)

   Without a URL the CLI exits before the sidecar starts.

   Irreversible tools pause unless auto-approve was set. Answer with `piper answer`. Do not write `answer.json`.

5. **Review**
   Read the card `piper dispatch` printed. If you already waited some other way:
   ```bash
   piper review --task task.json              # or: piper review --result result.json
   piper status --dir /path/to/slice          # idle/ask/done/stalled/error; no cat/jq
   piper await --dir /path/to/slice           # wait for ask/done; no sleep-loop
   ```
   On `ask`: `piper answer allow`, `piper answer deny`, or `piper answer --text "..."`. Do not restart the process.

   The card is the rubric. A pass is `status == "ok"`, `files_touched` inside the brief, a proportional diff, and a green `check` (`result.test`). `"stalled"` is not a pass. Green `test.exit_code=0` after an incomplete loop is still `ok` when `check` was set.

6. **Record and continue**
   ```bash
   piper progress --id slice-001 pass --note "validator + test"
   ```
   Verdicts: `pass`, `fail`, `stalled`, `timeout`, `died`, `skip`. The line is appended to `.piper/progress.log`.
   - **Pass**: emit the next slice.
   - **Fail**: a narrower brief, or do that edit yourself.
   - **Done**: when the horizon checklist is green, stop.

---

## Piper worker wake standard

Any agent that starts Piper is the parent. Piper does not come find you.
A human must not copy a URL from a panel. A timer that checks the folder is
not the wake. It is a fallback, and it is how a finished or stalled run sits
until someone asks.

## Who owns the wake

The agent that launches `lmp_sidecar --worker` or `piper worker run` already
has a way to be woken, or it stays attached to the process. Piper does not
know who that agent is. There is no default host. Grok, Gemini, a script,
and a human CI job each pass their own URL.

Turn-based agents cannot stay attached. They pass a wake URL or they do not
detach. `piper dispatch` stays attached and does not need a URL.

## How to launch

Stay attached only if your process waits and reads the exit. That is `piper dispatch` and `piper run`.

Otherwise, before start, resolve the URL with `piper wake-url` (do not copy it from a panel):

- `--orch-webhook URL`, or
- task field `orch_webhook`, or
- env `LMP_ORCH_WEBHOOK`, or
- file `.piper/orch_webhook` (written automatically by `piper_ui` — **do not copy a URL from the panel**)

Detached with no URL: the CLI exits before the sidecar starts. `--help`
says this in one paragraph. Read that before the first launch. The help
text is the contract, not this chat.

## Events

One short JSON POST. No prompt bodies. Failed POST does not fail the
mission. No URL means no POST.

| kind | when | parent does |
| --- | --- | --- |
| `ask` | `awaiting_user.json` written, or an irreversible call is paused | `piper answer allow`, `piper answer deny`, or `piper answer --text "..."` (writes `answer.json`). Do not restart. Do not freehand the JSON. |
| `done` | `result.json` written and the slice completed | `piper review` (or the dispatch card). Send the next slice or stop. |
| `stalled` | `result.json` written and the harness stopped the run (`stalled`, `max_turns`, not completed) | read what landed. Do not treat it as success. Next slice or stop. |
| `died` | process exited and no `result.json` was written | launch parent sends this. Tell the user. Do not relaunch blindly. |

`stalled` is its own kind. Do not hide it inside `done` with `status: error`.
A parent that only handles `done` will miss a stall, which is the bug this
standard exists to kill. `result.json` uses the same `status: "stalled"` for
`max_turns` / no-progress stalls so parents need not parse error strings.

Body:

```json
{
  "kind": "stalled",
  "task_id": "mc-live-2",
  "run_id": "r-...",
  "cwd": "/absolute/path",
  "result_path": "/absolute/path/result.json",
  "seq": 0,
  "status": "stalled",
  "question": ""
}
```

`ask` fills `question` and `seq`. `done` and `stalled` fill `status`
(`ok` / `error` / `timeout` / `stalled`).
`died` fills `cwd` and `task_id`.

Do not POST every turn. Do not POST tool output.

## What the parent must not do

- Do not hand-write `task.json`, `answer.json`, or a progress-log line.
- Do not poll Piper, and do not poll the cloud from Piper.
- Do not ask the human if the sidecar is still there. The event is the notice.
- Do not bake another agent's webhook into the binary.
- Do not start a silent screen or nohup without the URL.

A folder watch is allowed only as a backup when the platform has no wake
URL yet. It is not the standard, and it must not be the path a second
agent copies.

## Done when

- `--help` names this standard in one paragraph.
- A detached launch with no URL exits before the sidecar starts.
- A run that writes `result.json` POSTs `done` if `status=ok` (model completed,
  `plan_ready`, or a green packet `check` after an incomplete loop stop such as
  `max_turns`), `stalled` if it did not.
- The launch parent POSTs `died` if the sidecar exits with no result.
- An irreversible call and `ask_user` both POST `ask` and wait.
