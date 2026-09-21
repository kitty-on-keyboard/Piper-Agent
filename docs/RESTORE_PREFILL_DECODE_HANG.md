# Restore + large suffix prefill: decode never starts (MTP)

## Symptom (bakeoff A0, 2026-09-21)

Mac bakeoff arm A0 (`max_think_tokens=8192`, tip ~`dd23640`), Qwen3.8-27B-MLX-4bit + MTP:

- Turns 1–3: generate + tools succeed; `[spec] blocks=…` lines print.
- Turn 4 after batched `read_file`: `reuse=restore`, `prefill_from≈6411`, `prompt≈10613`.
- Sidecar stderr reaches `prefill_done` / `mem at=prefill_done`.
- **No** subsequent `[spec]` line, no decode journals, no `generate` completion event.
- Harness SIGKILL at ~900s wall.

Sibling A1 (`max_think_tokens=4096`) on the same day kept journaling (~734s) and hit think_budget — same model/MTP/seed. ThinkCap defaults are **not** the bug; A0 is the hang case because the large tool-observation restore is what exercises the broken handoff.

Attached artifacts for that run: `uploads/sidecar_8f3c.stderr`, `uploads/events_c5cf.jsonl`, `uploads/run_14e3.log`.

## Root cause

Two cooperating defects on the MLX MTP path after a **large Restore + multi-chunk suffix prefill**:

1. **Lazy `last_hidden` across `clear_cache`.** Prefill chunks call `clear_cache` after each chunk (jetsam fix). `logits_to_host` only forces the final-position lm_head slice. With MTP loaded, `last_hidden_` still held the full-chunk lazy graph. MTP’s first `propose()` evals that array; after reclaim the Metal eval did not return. Matches “prefill_done then silence.”

2. **MTP KV not reset on Restore.** Target caches rewind; the MTP head’s own KV did not. The first draft after a large suffix prefill could attend over positions the restored target no longer owns.

## Fix

- `Qwen35MoeModel::pin_last_hidden_for_decode()`: keep only the last row, `mx::eval` it, **before** the final prefill chunk’s `mx::clear_cache()` when MTP is loaded.
- `mtp_reset()` on Restore (and from `reset_cache()`), plus again at `decode_speculative` entry so Extend turns start clean too.
- stderr breadcrumbs: `decode_begin speculative=1 …` and `decode_first_token ttft_ms=…` so a future hang can be placed on either side of first token without a 27B CI job.

## Gate test (no 27B)

`tests/model/test_mtp_proposer.cpp` → `after_a_restore_mtp_reset_the_next_propose_starts_cold` locks the SpecForward contract: after `mtp_reset` + proposer `reset`, the next propose is cold (no carried seed).

## Mac repro (full 27B)

```bash
# Same product flags as the hang arm; do not change ThinkCap/MTP defaults for the repro.
piper run --task /path/to/bowling_seed7_task.json \
  --model /Users/dev/Desktop/Models/Qwen3.8-27B-MLX-4bit \
  # MTP draft head as in product; commit_think + shadow_compact on
```

Mission: Aider polyglot bowling, seed 7. Confirm hang: after a Restore prefill that jumps prompt by ~4k tokens (batched file reads), stderr shows `prefill_done` and then neither `decode_begin` nor `[spec]` until timeout. Fixed build should print `decode_begin` then `decode_first_token` and `[spec]` stats.
