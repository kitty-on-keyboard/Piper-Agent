# Dense 27B `generate_begin` death — resolved

Companion: `docs/GENERATE_BEGIN_CRASH.md`. Bakeoff remains paused for the operator.

## What killed it

Not an uncaught MLX exception. Cold prefill is `Extend` from an empty KV ledger, so the Reset `clear_cache` never ran. Prefill-chunk activations stacked in the MLX allocator cache to the ~38 GB jetsam band while `pre_generate` still reported ~16.3 GB active.

## What changed

- `prefill_tokens`: `mx::clear_cache()` after each chunk (after `eval_caches` / logits copy).
- `set_memory_ceiling`: memory limit 4 GiB below `max_recommended_working_set_size`; cache limit unchanged.
- `drive_sidecar`: log `waitpid` returncode / signal.
- Accidental `dbg_log` ofstreams from the think/shadow PR were stripped on this branch.

## Do not

- Resume the 34-exercise bakeoff from this handoff.
- Revert to an always-on small `set_cache_limit` (5–6 s TTFT).
- Mix Qwen4_exp / Flash-Next into this branch.
