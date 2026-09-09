# Worker must vouch for MCP the way the IDE does

Gemini. Piper only. Do not stash or reset the dirty tree. Do not edit Godoer.
Do not set `trusted` inside a project's `.mcp.json`. That file arrives with a
checkout and must stay unable to vouch for itself.

## What failed

A3B worker run on `/Users/dev/Desktop/Piper tests/room_crate`
(`events.ndjson`, about 96s, 11 turns).

`godot_guide` (readOnlyHint) now runs. Good.

`godot_build_scene` does not. Every call:

- `approval` gate `write`, `destroys_data=1`, why
  `destroys data, which overrides auto_approve_writes`
- `approval` gate `irreversible`, `answer=denied`
- `tool_denied` why `irreversible, not approved`

Then the model called `ask_user` and the process exited. `result.json` status
is `error`. No live `awaiting_user.json` was left to answer.

## Why

The IDE trusts a server only when the operator names it in settings
(`params.settings.mcp_servers[]` with `"trusted": true`). See
`parse_mcp_servers` in `src/surface/mcp_settings.cpp`. That boolean is the
voucher: tools run without a per-call card.

`build_start_message` in `src/surface/worker.cpp` never sends `mcp_servers`.
The worker therefore only sees `.mcp.json`, which forces `trusted=false`.
Untrusted MCP tools are marked `irreversible` in `apply_mcp_decl_flags`
(`src/tools/mcp_host.cpp`) unless `readOnlyHint` saves them. A build is a
write, so it is irreversible. The worker approver denies every irreversible
call on the spot.

A checkout must not be able to set `trusted: true` and have it stick. The
voucher has to come from the operator side of the worker, same as the IDE.

## Fix 1: pass the voucher on start

When the task packet names servers to trust, copy those servers from the
workspace `.mcp.json` (command, args, env) into
`params.settings.mcp_servers` on `lmp/start`, with `"trusted": true`.

Task field, operator-owned, not read from the file's own `trusted` key:

```json
"trust_mcp": ["godoer"]
```

Match by the `.mcp.json` server name. If the name is absent from the file,
fail the task with a clear error. Do not invent a command. Convert `.mcp.json`
env objects to the `KEY=VALUE` array `parse_mcp_servers` expects.

Default for a worker packet that omits the field: trust nothing. The room_crate
task will set `trust_mcp` to `["godoer"]`. Do not auto-trust every server in
the file.

After this, a trusted `godot_build_scene` must not take the irreversible deny
path. `readOnlyHint` behavior stays as it is. An untrusted write stays
contained.

Done when a worker run with `trust_mcp: ["godoer"]` logs `mcp_server`
`trusted=1` for godoer, and `godot_build_scene` executes instead of
`tool_denied`. The event `mcp_trust` should say the operator vouched.

## Fix 2: an irreversible deny must not kill the question wait

Even with trust, a real deny has to reach the cloud. Today it does not.

`hooks.approver` in `src/surface/sidecar.cpp` sets `denied_irreversible` and
returns false. The wait loop is:

```text
while (termination_reason == "awaiting_user" && !denied_irreversible)
```

So the first irreversible deny skips the wait. The model can still call
`ask_user` and emit the question, then the process exits. Measured on
room_crate: `run_end` `termination_reason=awaiting_user`, then the process is
gone, no `awaiting_user.json`.

Change that. An irreversible call in the worker should not be a silent final
deny that also disables the wait. Prefer: refuse the call (the model sees the
refusal) but do not latch a flag that skips `awaiting_user`. When the run
halts on `ask_user`, write `awaiting_user.json` and block for `answer.json`
until the mission timeout, whether or not an earlier tool was denied.

Do not auto-answer `ask_user`. Do not let the model set `trusted`.

## Done when

- `trust_mcp: ["godoer"]` on a worker task connects godoer as trusted and
  `godot_build_scene` is not denied as destroy.
- A task that omits `trust_mcp` still denies an untrusted write.
- A run that reaches `ask_user` after any deny writes `awaiting_user.json`
  and stays alive until `answer.json` or timeout.
- No Godoer edits. `.mcp.json` `trusted` stays ignored.

