# Wake the cloud when Piper asks, finishes, or dies

Gemini. Piper worker only. Do not stash or reset the dirty tree. Do not edit
Godoer. Do not add a cloud poll loop. The local side already waits. The cloud
side must be woken.

## Goal

A smarter cloud agent owns the long-horizon task. Piper is the local coder.
The cloud gives a fat slice, reviews the outcome, answers when Piper is
unsure, and sends the next slice. Piper does not own the whole task and does
not poll the cloud.

## What already works

Worker writes `awaiting_user.json` and blocks on `answer.json` (same directory
as `result.json`). `answer.json` is `{ "text": "..." }` or `{ "answer": "..." }`.
Trust voucher is `trust_mcp` on the task. See
`docs/WORKER_MCP_TRUST_HANDOFF.md`.

What failed on 2026-09-08: the cloud never watched. A finished run looked like
a dead sidecar. An `ask_user` would have sat until someone opened the folder.

## Fix

The worker (or the process that launches it) POSTs one small JSON event to a
URL the operator supplies. No URL, no POST. Do not invent a default host.

Task field, also accepted as env `LMP_ORCH_WEBHOOK`:

```json
"orch_webhook": "https://..."
```

POST when:

1. `awaiting_user.json` is written. Kind `ask`.
2. `result.json` is written. Kind `done`. Include status
   (`ok`, `error`, `timeout_awaiting_user`).
3. The sidecar process exits without either of those. Kind `died`.

The sidecar cannot send `died` after it is killed. The launch parent (the
screen/caffeinate wrapper, or a few lines around the worker exec) must wait
on the pid and POST `died` if no `done` webhook was sent. Do not start a
second poller that watches the cloud.

### Body

JSON, short, no prompt bodies except the question text already in
`awaiting_user.json`:

```json
{
  "kind": "ask",
  "task_id": "room-crate2-a3b",
  "run_id": "r-...",
  "cwd": "/Users/dev/Desktop/Piper tests/room_crate2",
  "result_path": "/Users/dev/Desktop/Piper tests/room_crate2/result.json",
  "seq": 0,
  "status": "",
  "question": ""
}
```

`ask` fills `question` and `seq`. `done` fills `status`. `died` fills neither
beyond cwd and task_id. Timeout a few seconds. A failed POST must not fail
the mission. Log it and keep waiting on `answer.json` if this was an ask.

Do not POST on every turn. Do not POST tool output.

## CLI is the agent path

`piper worker run` is the orchestrated path. Whoever starts it is the cloud
half, or is launching for one. Piper does not know who that is. Do not
hardcode a host, a Grok Bot routine, or any other listener.

Two legal ways to own the horizon:

1. Stay attached. The parent waits. It reads `awaiting_user.json` /
   `result.json` and the process exit. No webhook.
2. Detach (screen, nohup, background). Then a wake URL is required. Pass
   `--orch-webhook`, task field `orch_webhook`, or env `LMP_ORCH_WEBHOOK`.
   The URL belongs to the caller. Gemini creates its own. This chat's
   webhook is only one listener, not the default.

If the launch is detached and no webhook is set, the CLI exits non-zero
before the sidecar starts and prints one line: create YOUR wake URL and
pass it, or stay attached. Do not daemonize a silent worker.

On start, print the contract once to stderr:

`agent path: parent owns the horizon; events ask/done/died; pass --orch-webhook or stay attached`

`piper worker run --help` must say the same thing in one short paragraph.
The help text is how a different agent (Gemini, a script, a human) learns
the rule without this chat.

## Cloud side

Whoever owns the wake URL reads the payload, then:

- `ask`: read `awaiting_user.json`, write guidance to `answer.json` as
  `{ "text": "..." }`. Do not restart the process.
- `done`: read `result.json` and the files. If the slice is actually done,
  either stop or write the next task and start the next worker. If the model
  called `finish` but the files are wrong, that is a new slice, not an answer
  to a question that was never asked.
- `died`: tell the user. Do not relaunch blindly.

The cloud holds the goal. Piper never polls this chat.

## Done when

- A worker task with `orch_webhook` POSTs `ask` when it writes
  `awaiting_user.json`, and stays blocked until `answer.json` or timeout.
- The same task POSTs `done` when `result.json` is written.
- A parent POSTs `died` if the sidecar exits with no `done`.
- No webhook field and an attached parent: no POST, today's file wait.
- Detached launch with no webhook: CLI exits non-zero, sidecar never starts.
- Help text and the start line name this as the agent path. No default URL.
- No Godoer edits. No cloud poll inside the sidecar.

