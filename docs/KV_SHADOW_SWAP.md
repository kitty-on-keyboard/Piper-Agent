# KV shadow-swap (v1)

Flag: `LMP_SHADOW_COMPACT` unset/`0` = off. `=1` warms the live cache after
compaction rewrites the prompt. Compact policy stays **75 / 35 summarize**.

v1 is **not** a second GPU cache. Dual KV on 48 GB is unsafe. After
`compact_to_budget` applies collapses **or** drops turns, the live ledger no
longer matches the next prompt, so today's `plan_turn_reuse` Resets and
re-prefills everything (historically ~0 reuse, high TTFT).

## What v1 does

Warm the **stable prefix** of the *next* `render_with_offsets` prompt into the
**live** ledger/checkpoint. Single-threaded. No overlap with `generate`.

Reuse must be planned with `render_with_offsets` / `plan_turn_reuse`. Message
`[0:k]` tokens are not a prefix of the full generation prompt.

## When to warm

Any `compact_to_budget` pass that applied pending collapses **or** dropped
turns. Collapse-only (`compaction_avoided_by_collapse`) still invalidates the
live prefix today.

Not immediately after `compact_to_budget()` in `run()`. Operator check,
plan-spin notes, inert nudges, and `take_steering()` all mutate context after
that. Warm at the **start of the next `step()`**, after `take_steering()` in
`run()`, once the next prompt is known: `render` + `render_with_offsets` +
`stable_message_count` → `checkpoint_at`. Prefill tokens `[0, checkpoint_at)`,
snapshot the checkpoint. The following `generate()` should `Extend`/`Restore`
and only prefill the live-state tail.

Warm **always Resets** first. The cache it inherited is the pre-compact prompt;
trying to Extend/Restore that against the rewritten prefix is the S5.10 failure.

## Invalidate / fallback

Emit `shadow_compact_fallback` (why) and leave `generate` to today's
Reset-at-generate:

- cancel during warm
- `checkpoint_at == 0` (no stable prefix)
- backend error / MLX throw
- `generate` still reports `prefill_reused_tokens == 0` after a successful warm
  (id mismatch — S5.10)
- images that fail `prepare_images`
- any doubt

A steer or `tools_guidance_` / mode change that lands *after compact and before
the next `step()`* is already folded into that step's prompt, so v1 warms the
new prefix (Reset + prefill). Steering *during* generate is the next turn.

## Concurrency

No second `generate`/prefill on the same MLX model. Tool-idle dual-cache is v2.

## GO

Cannot reuse against a wrong prefix. Flag off is zero behavior change.
