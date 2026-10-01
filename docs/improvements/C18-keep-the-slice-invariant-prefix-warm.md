# C18: Keep the slice-invariant prefix warm: anchor KV checkpoints and a byte-stable system message

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| UNVERIFIED | piper | latency | medium | L |

> **Not yet verified.** Open every cited line below and confirm the problem still holds before you implement. Drop or reshape the task if it does not.

Merged from these finder reports:

- Reuse the static system+tools KV prefix across slices in serve mode
- Keep an anchor KV checkpoint at the end of the slice-invariant prefix so serve mode and post-compaction turns stop re-prefilling it
- Make mid-run load_skill observation-only, add `piper packet --skill`, and keep the parent's orchestration skill out of the worker

## Problem

Even with resident weights, three cases fall back to a full Reset re-prefill:
- the first generate of every serve-mode slice
- every turn after a compaction
- the turn after a mid-run skill load
The root causes are the following.
(1) The backend keeps only one checkpoint: the end of the current run's stable prefix (system + mission + history). A new mission, or a compacted history, diverges inside that span, so plan_turn_reuse returns Reset. The gated-delta layers can only restore to positions that were snapshotted in advance, and nothing is snapshotted at the system-message boundary. That boundary is byte-identical across slices and across a compaction, by design. Measured: 48 of 48 post-compaction turns reused zero tokens, with a median TTFT of 43.5 s.
(2) The system message changes anyway.
- The recall advert embeds the PCC item and session counts, which grow every slice. They sit ahead of the conventions (up to 64 KiB), skills, memory and the tools JSON.
- A mid-run load_skill both returns the whole body (up to 64 KiB) as an observation and rewrites '# Loaded skills' inside the system message. The skill is then carried twice, and the next turn resets.
(3) The cache-safe preload path is unreachable from orchestration: `piper packet` has no --skill flag, and hand-writing task.json is forbidden.
(4) `piper init` installs the parent-only piper-orchestration skill into discovery roots, so every initialised workspace advertises the orchestrator's recipe in the worker's system-prompt catalog.

## Why it matters

Priority 4.
- After a compaction, turns stop re-prefilling system+mission. Today that costs a measured median TTFT of 43.5 s per compaction.
- In serve mode, each slice's first turn prefills only the brief instead of the full system prompt. Estimated savings: 3-4 s per slice on A3B, about 15-20 s on 27B, and tens of seconds for Godoer's ~30k-token prefix.
- A mid-run skill load no longer costs a full re-prefill plus up to ~16k duplicated tokens.
Model behaviour does not change.

## Proposed fix

(1) Replace the single turn checkpoint with a small ordered set of verified checkpoints:
- a 'system' anchor at the end of the system message, including tools (offsets[1])
- optionally one at the end of the mission, for compaction
- today's turn checkpoint
InferenceTask gains `system_prefix_at`. prefill_tokens makes each boundary a chunk edge and snapshots there; a snapshot is the SSM conv/delta state plus seq_len. plan_turn_reuse picks the longest valid checkpoint whose length is within the verified reusable prefix, still id+tag verified per S5.10. It restores that checkpoint, truncates the ledger, invalidates longer checkpoints and calls mtp_reset. A Reset or a model reload clears all checkpoints. shadow_compact warms from the anchor instead of resetting.
(2) Make the system message byte-stable.
- Across slices: drop the per-slice recall counts from the advert. C15 already removes the whole paragraph for worker runs; if the interactive wording changes, A/B context_recall usage.
- Within a run: a mid-run load_skill returns its body as an observation only and stops rewriting '# Loaded skills'. The skill is promoted into the system message at the next compaction, which re-prefills anyway, and is exempt from floor stubbing until then.
(3) Add a repeatable `piper packet --skill ID`. Validate the id against the same discovery roots and write it into task.json `skills`, which the worker already preloads before turn 0.
(4) Add an `audience: orchestrator` frontmatter field to the shipped piper-orchestration SKILL.md, and have discover_skills and load_skill skip it.

## Evidence (file:line)

- src/model/mlx_backend.cpp:43-51: the backend holds a single TurnCheckpoint {cp, len, valid}. src/loop/agent.cpp:1169-1172: checkpoint_at is the end of system+mission+history, and it is the only boundary passed
- src/model/kv_cache.cpp:143-160: when the prompt diverges and checkpoint_len > reusable, the plan is Reset. mlx_backend.cpp:1065-1121: Reset prefills from 0. 1293-1297: the shadow warm always Resets (docs/KV_SHADOW_SWAP.md:46-47: 'Warm always Resets first')
- src/model/mlx/qwen35_moe_model.hpp:178-241: linear layers can only be reached through snapshots. A checkpoint is seq_len plus an SSM snapshot; KV is restored with truncate_to
- src/loop/agent.hpp:418-424: measured 48/48 post-compaction turns reusing zero tokens, median TTFT 43.5 s. context.cpp:190-195: compaction keeps system+mission byte-identical, which a single checkpoint cannot use
- src/surface/sidecar.cpp:2523: serve keeps one Session across runs. 2605: --worker --task forwards to a live daemon
- src/context/context.cpp:189-209: the recall advert, with item and session counts, sits inside the system string before conventions, skills and memory. sidecar.cpp:1320-1323 and pcc/store.cpp:345-352: the counts change every slice. chat_template.cpp:87-99: the tools JSON is appended after the system content. session.cpp:221-229: conventions can be up to 64 KiB
- src/tools/registry.cpp:941-946: load_skill calls skill_loaded_sink_ and also returns the body. sidecar.cpp:1259-1268: the sink calls add_loaded_skill. context.cpp:116-121 and 216-221: loaded skills render inside the system message, which is documented as never changing within a run. kv_cache.hpp:162-165: rewriting the stable prefix falls through to Reset. skills.hpp:21: kSkillMaxBytes is 64 KiB
- scripts/piper_worker.py:788-819: `piper packet` takes no skills argument, while worker.cpp:426-445 parses skills and preload_skills. 1971-2020: `piper init` installs piper-orchestration into .cursor/skills and .agents/skills, both discovery roots (skills.cpp:47-58), so the worker catalog advertises the parent's recipe (reproduced in the scratchpad)
- docs/hardware_squeeze/HS1_27B_RESULTS.md:73-80: 27B cold prefill runs about 300 tok/s; restore TTFT is 407-459 ms. mlx_backend.cpp:52-58: A3B prefill runs 1317-1684 tok/s. docs/AGENT_LOOP_WINS_BRANCH_PLAN.md:331: 'First-turn ~50s TTFT / ~30k tokens with Godoer'

## How the finder suggested verifying it

Model-free gate tests in tests/model/test_kv_reuse.cpp: ledger [S|M1|T|G], checkpoints at |S| and |S|+|M1|+|T|, prompt [S|M2|L]. Expect Restore at |S|; today the result is Reset. A ContextStore test: two stores that differ only in recall counts render identical system messages. A test_loop case: a mid-run load_skill leaves render()[0] byte-identical, and the body appears once. Here: check `packet --skill` emission and the audience filter against a replica of discovery. On the Mac, run `piper worker serve` with two slices in one workspace. On slice 2, turn 1, `generate: reuse=` should change from reset with prefill_from=0 to a restore at the system length. In long_context, TTFT after each compaction event should drop from about 43 s.

Back to the [index](README.md).
