# Piper worker wake standard

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
