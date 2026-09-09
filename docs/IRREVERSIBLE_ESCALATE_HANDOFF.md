# Worker irreversible calls must escalate, not hard-deny

Gemini. Piper worker only. Do not stash or reset the dirty tree. Do not edit
Godoer. Do not loosen blast_radius so `> /tmp` looks safe. Do not change the
IDE approval card.

## What failed

On 2026-09-08, `mc_tiny` slice 1 ran `cat > /tmp/floor_spec.json << 'EOF'`.
The classifier did the right thing: parsed, risk 0.6, `destroys_data` plus
`writes_outside_workspace`. The own-output exception does not cover `/tmp`
(`approval.cpp` `redirects_only_own_output`: a target outside the workspace
returns false). `mkdir -p game/scenes` had already auto-approved. `write_file`
to `specs/floor.json` was allowed on a later turn. The shell redirect was the
wrong tool, not an unreadable command.

The worker then failed its own contract. The approver in
`src/surface/sidecar.cpp` treats every `loop::is_irreversible(hint)` as a hard
deny, sets `denied_irreversible`, and returns false. It never writes
`awaiting_user.json` and never POSTs `ask`. The error string says
"orchestrator must escalate" and then nothing is sent.

The model kept going and hit `max_turns` at 30. Result status was still
`error` with that deny text, because the sticky flag overrides the real
termination. The cloud saw a fatal deny. The process had already left.

## Root cause

Worker mode has no path from "irreversible" to the existing ask wait.
`is_irreversible` is correct. The bug is the worker approver, plus the result
stamp that pretends a denied tool ended the mission.

Do not fix this by telling the model not to use shell. Do not auto-approve
writes outside the workspace. `/tmp` stays irreversible.

## Contract

In worker mode, an irreversible tool call is an ask. The mission pauses on
that call. The model does not get a denial and another turn until the
orchestrator answers.

1. Write `awaiting_user.json` next to `result.json` before returning from the
   approver. Question text is the tool name plus the full command (or preview
   if there is no command). Say that the call is irreversible and the
   orchestrator must allow or deny it.
2. If `orch_webhook` is set, POST kind `ask` with that question. No URL means
   no POST. A failed POST must not fail the mission.
3. Block on `answer.json` (`{"text":"..."}` or `{"answer":"..."}`), same
   helpers the post-mission wait already uses. Honor the remaining
   `timeout_s`. Timeout writes the usual timeout status and stops. Do not
   continue the model loop on timeout.
4. Allow: return true so the tool runs. Deny: return false and give the model
   a normal denied tool result. Do not set a sticky flag that rewrites the
   final status.
5. A later `max_turns` or `finish` keeps its own termination. Mention a prior
   deny only if you add a separate field. Do not replace `max_turns` with the
   escalate error string.
6. Repeated identical irreversible command: if the same command was already
   denied in this run, deny without a second wait. An allow applies only to
   the call that was asked, not to every later shell.

IDE path is unchanged. This is the headless worker approver only.

## Where

- `src/surface/sidecar.cpp` worker approver, around the `is_irreversible`
  branch. Reuse `write_awaiting_user`, `read_and_consume_answer`, and the
  webhook POST already used when `termination_reason == "awaiting_user"`.
- Result assembly below that: the `else if (denied_irreversible)` branch must
  not override a completed run or a non-ask termination.
- Do not change `loop::is_irreversible` or `redirects_only_own_output`.
- Do not change blast_radius.

## Done when

- A worker task whose model (or a test double) issues `cat > /tmp/x << 'EOF'`
  writes `awaiting_user.json` and does not take another model turn until
  `answer.json` exists.
- An allow answer lets that one command proceed. A deny answer returns a tool
  denial and the loop may continue. The result is not stamped
  "orchestrator must escalate" unless the mission actually stopped on that ask
  (timeout, or the ask never got an answer before exit).
- No webhook still writes the file and waits. Parent can watch the folder.
- `is_irreversible` still true for `> /tmp`. A workspace `mkdir` still
  auto-approves when `auto_approve_exec` is on.
- No Godoer edits. No cloud poll inside the sidecar.
