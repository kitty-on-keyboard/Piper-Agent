# Enabling agent-loop wins measurement (PR1)

## Tier A (always on)

Every generate / tools refresh / tool_result now emits attribution fields:

- `kv_reuse` — `mode`, `reused_tokens`, `prompt_tokens`, `reason` (required on Reset),
  `stable_prefix_tokens`, `shadow_armed`
- `tools_refresh` — `trigger`, short guidance hashes, `changed`, spec counts, `noop`
  (`noop=1` when Phase A1 skips an identical mid-run rewrite)
- `generation` — aggregate `draft_len_hist` / `accept_at_depth` / `reject_at_depth`,
  `grammar_empty_mask`, `grammar_phase_end`, …
- `tool_result` — existing `status` plus `error_class` / optional `error_code`
  (`error_class` values include `parse_args`, `schema_enum`, `edit_miss`, `exec`,
  `sandbox`, `unknown`; Phase C maps `ErrorClass::SchemaEnum` → `schema_enum`)

### Phase C grammar kill switch

Enum value masking in `ToolCallGuard` is on by default when `ParamSpec::enum_values`
is non-empty. To disable (empty-mask kill):

```bash
export LMP_ENUM_MASK=0   # exact "0" only
```

Or construct `parsephony::Options{ .enforce_enum_values = false }` for tests.

## Tier B firehose

```bash
export LMP_AGENT_LOOP_TRACE=1   # exact "1" only
```

Adds per-block `spec_block` events and full sha256 on `tools_refresh` hashes.

## Summarize

```bash
python3 scripts/agent_loop_wins/summarize_events.py path/to/events.jsonl
python3 scripts/agent_loop_wins/prefix_saboteur_stub.py   # synthetic matrix smoke
```

Keep multi-MB journals under local `piper-bench/agent_loop_wins/` (gitignored).
