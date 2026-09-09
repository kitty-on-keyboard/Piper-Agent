# Cloud answers for ask_user, and the godot_guide deny

Cursor Grok. Two defects. Do not stash or reset the dirty tree. Do not treat
this as a Godoer change. The live proof is
`/Users/dev/Desktop/Piper tests/guide_floor/events.ndjson`, run
`r-18d36c5d01ce9508-9bdd6c1e`.

## Bug: godot_guide is denied as destroying data

Hypothesis, verify and discard if wrong.

`godot_guide` is read-only. Godoer annotates it `_RO` / `readOnlyHint`. The
worker still denied it:

- `tool_call` `godot_guide`
- `approval` gate `write`, `destroys_data=1`, `own_output=0`,
  why `destroys data, which overrides auto_approve_writes`
- `approval` gate `irreversible`, `answer=denied`
- `tool_denied` why `irreversible, not approved`

The MCP entry came from the project `.mcp.json`. That file is never trusted
(`mcp_server` `trusted=0`). In `src/tools/mcp_host.cpp`,
`apply_mcp_decl_flags` returns early when `!trusted` and sets
`mutates_workspace`, `needs_execution`, and `irreversible` without reading
`readOnlyHint`. The comment above that early return says untrusted is
containment. The worker approver then denies every irreversible call.

So a read-only tool on an unvouched server is classified as destroy. That is
the bug. Trust is the wrong signal for "this call deletes something."

### Invariant

`readOnlyHint` true means the call is not a write and not destroy, trusted or
not. Untrusted still must not silently run a mutating MCP tool. A read-only
hinted tool must be allowed to execute without an approval card, including in
the worker, whose approver denies irreversible and has nobody to ask.

Do not fix this by marking every `.mcp.json` server trusted. That turns
containment off for writes. Do not special-case the name `godot_guide`.

Done when a worker run with `auto_approve_exec` can call `godot_guide` (no
topic, then `topic=scene`) and the event log shows the tool result, not
`tool_denied`. A mutating untrusted MCP tool is still denied or asked, not
auto-run.

## Feature: ask_user must reach the cloud orchestrator

This headless worker is the local half. The cloud half is the agent watching
the run. When Piper is unsure it already calls `ask_user`, emits the question,
and halts (`halt_reason=awaiting_user`). Measured on the same run, seq 35,
after the guide deny. Nobody received it. The worker approver only returns
yes or no. Its inbox is empty.

The product behavior:

1. Piper calls `ask_user` with a question and options.
2. The worker does not invent an answer and does not treat the halt as a
   finished failure.
3. The question is written where the orchestrator can see it: the existing
   `ask_user` event is enough, plus a small `awaiting_user.json` next to
   `result.json` (question, options, run id, event seq).
4. The worker blocks on a same-directory `answer.json` (the human/cloud text),
   with the mission timeout still applying.
5. That text is injected as the user reply to that `ask_user`, the same way the
   IDE card resumes a halted run. Then the loop continues.
6. A repeated identical question still gets the previous answer, not a second
   wake-up.

The orchestrator (this chat, or the worker CLI's parent) watches
`events.ndjson` / `awaiting_user.json`, writes the guidance, and deletes or
renames the question file so a stale question cannot be answered twice.

Do not auto-answer `ask_user` inside the sidecar. Do not send the question only
to stdout. A screen session has no reader.

### Done when

- A fixture worker mission whose model calls `ask_user` writes
  `awaiting_user.json` and does not exit `completed` or `error` until an
  answer arrives or the timeout fires.
- Dropping `answer.json` resumes the run and the next turn can see that text.
- Timeout with no answer is a distinct status, not a silent death.
- The godot_guide deny is fixed first, so the cloud is not asked to bless a
  read that should never have been a card.

