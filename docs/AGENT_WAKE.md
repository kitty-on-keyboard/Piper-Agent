# Piper Local Worker — Orchestrator Guide & Parent Runbook

## Overview
**Core Principle:** Cloud directs, local writes, cloud verifies, repeat.

Piper is a fast headless coding worker running locally on Apple Silicon (via MLX). It executes edits, shell commands, and file operations inside `cwd` with zero cloud output token cost.

The cloud orchestrator (Cursor, Claude, Gemini, Antigravity, or custom script) acts as the high-level brain:
- Maintains the long-horizon plan and acceptance criteria.
- Decomposes complex tasks into bounded packets (1–3 files per packet).
- Directs Piper by writing task packets (`task.json`).
- Verifies outcomes (diffs, test execution, acceptance checks).
- Loops until the entire mission is verified complete.

## The Long-Horizon Execution Loop
```
┌─────────────────────────┐         task.json            ┌──────────────────────────┐
│   Cloud Orchestrator    │ ───────────────────────────► │       Piper Worker       │
│  (Cursor, Claude, etc.) │                              │  (MLX on Apple Silicon)  │
│                         │ ◄─────────────────────────── │                          │
│ plan · review · verify  │     result.json + diff       │  edits · tools · loop    │
└─────────────────────────┘                              └──────────────────────────┘
             │                                                         │
             └────────── repeat until horizon acceptance passes ───────┘
```

### Roles

| Who | Owns | Does not own |
|---|---|---|
| **Cloud Orchestrator** | Goal decomposition, file-level direction, acceptance criteria, high-level review, troubleshooting, "are we done?" | Bulk code generation tokens, local tool thrash |
| **Piper Worker** | Edits, tool execution, test commands, local iteration inside `cwd` | Long-horizon judgment, multi-repo strategy |

Local models work best on scoped packets: **packets must be specific**, and **every turn gets an orchestrator review**. Trust outcomes (diff + tests + `result.json`), not vibes.

### Step-by-Step Procedure

1. **Frame the Horizon**
   Define the overarching objective and an acceptance checklist (e.g. unit tests pass, new command works, UI renders).

2. **Slice into Discrete Packets**
   Pick the smallest incremental step towards the goal.
   - Scope each slice to 1–3 files.
   - Explicitly list which files to EDIT, CREATE, and DO NOT TOUCH.

3. **Write `task.json`**
   Write a task packet in the workspace or a task directory:
   ```json
   {
     "id": "slice-001",
     "cwd": "/absolute/path/to/workspace",
     "mode": "agent",
     "model_dir": "/Users/dev/Desktop/Models/Qwen3.6-35B-A3B-MLX-4bit",
     "prompt": "## Horizon Context\nBuilding feature X.\n\n## This Slice Only\nAdd validator in src/validator.cpp and test in tests/test_validator.cpp.\n\n## Files\n- EDIT: src/validator.cpp\n- CREATE: tests/test_validator.cpp\n- DO NOT TOUCH: src/core.cpp\n\n## Done When\n- Unit test passes with `ctest -R test_validator`",
     "auto_approve_exec": true,
     "auto_approve_writes": true,
     "auto_approve_irreversible": true,
     "timeout_s": 600,
     "result_path": "/absolute/path/to/result.json"
   }
   ```

4. **Dispatch Piper**
   Run the CLI command:
   ```bash
   piper run --task /path/to/task.json
   # or equivalently:
   piper worker run --task /path/to/task.json
   # For autonomous unattended execution without prompts or pauses:
   piper run --task /path/to/task.json --auto-approve-irreversible
   # Or approve all (exec + writes + irreversible):
   piper run --task /path/to/task.json --auto-approve-all
   ```
   - **Attached mode (standard)**: Process waits and exits when the slice completes.
     - `0`: Completed normally.
     - `1`: Worker error.
     - `2`: Execution timed out (`timeout_s`).
     - `3`: Invalid task packet or missing wake URL for detached run.
   - **Detached mode**: If launching in the background (nohup, screen), you MUST pass `--orch-webhook <URL>`. Silent background launches without a webhook are refused.
   - **Irreversible tools**: Destructive tools or project managers (like `godot_project`, `delete_file`, or whole-file overwrites) escalate to `gate: irreversible`. Set `"auto_approve_irreversible": true` or pass `--auto-approve-irreversible` / `--auto-approve-all` for unattended runs; otherwise Piper pauses and writes `awaiting_user.json` for `answer.json`.

5. **Review `result.json` & Inspect Changes**
   Piper writes a structured result upon completion:
   ```json
   {
     "task_id": "slice-001",
     "status": "ok",
     "message": "Implemented validation logic and verified with unit test.",
     "files_touched": ["src/validator.cpp", "tests/test_validator.cpp"],
     "diff_stat": "+52 -2",
     "git_diff_path": "/path/to/slice.diff"
   }
   ```

   **Review Rubric (Keep it cheap):**
   - **Status**: Is `status == "ok"`? If `"error"` or `"stalled"`, inspect the message.
   - **Files Touched**: Are changes confined to expected paths? Reject drive-by edits.
   - **Diff**: Skim git diff for regressions or unnecessary churn.
   - **Acceptance**: Run verification commands or tests to validate the slice.

6. **Iterate or Complete**
   - **Pass**: If acceptance criteria for the slice pass, dispatch the next slice.
   - **Fail**: Send a narrowed/clarified packet, or perform that specific edit yourself.
   - **Done**: When all acceptance checklist items are verified, complete the mission.

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
detach.

## How to launch

Stay attached only if your process waits and reads the exit.

Otherwise, before start:

- `--orch-webhook URL`, or
- task field `orch_webhook`, or
- env `LMP_ORCH_WEBHOOK`

Detached with no URL: the CLI exits before the sidecar starts. `--help`
says this in one paragraph. Read that before the first launch. The help
text is the contract, not this chat.

## Events

One short JSON POST. No prompt bodies. Failed POST does not fail the
mission. No URL means no POST.

| kind | when | parent does |
| --- | --- | --- |
| `ask` | `awaiting_user.json` written, or an irreversible call is paused | write `answer.json` as `{"text":"..."}`. Do not restart. `allow` or `deny` for an irreversible call. Guidance for a real question. |
| `done` | `result.json` written and the slice completed | read the files. Send the next slice or stop. |
| `stalled` | `result.json` written and the harness stopped the run (`stalled`, `max_turns`, not completed) | read what landed. Do not treat it as success. Next slice or stop. |
| `died` | process exited and no `result.json` was written | launch parent sends this. Tell the user. Do not relaunch blindly. |

`stalled` is its own kind. Do not hide it inside `done` with `status: error`.
A parent that only handles `done` will miss a stall, which is the bug this
standard exists to kill.

Body:

```json
{
  "kind": "stalled",
  "task_id": "mc-live-2",
  "run_id": "r-...",
  "cwd": "/absolute/path",
  "result_path": "/absolute/path/result.json",
  "seq": 0,
  "status": "error",
  "question": ""
}
```

`ask` fills `question` and `seq`. `done` and `stalled` fill `status`.
`died` fills `cwd` and `task_id`.

Do not POST every turn. Do not POST tool output.

## What the parent must not do

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
- A run that writes `result.json` POSTs `done` if it completed, `stalled`
  if it did not.
- The launch parent POSTs `died` if the sidecar exits with no result.
- An irreversible call and `ask_user` both POST `ask` and wait.
