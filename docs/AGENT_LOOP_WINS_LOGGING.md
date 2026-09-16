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
- `run_end` — plus loop hygiene counts: `degenerate_text_count`, `text_only_turns`,
  `nudged_count`, `tool_error_count`
- `nudged` / `stalled` — `why` (`degenerate_text` | `length_capped_no_tool` |
  `text_only_turn` | `loop_cut` | `no_progress`), `consecutive`, `cap`

### Degenerate-text / no-tool-call recovery

Default **on**. Extends the existing inert nudge/stall path:

- Length-capped think/text turns with no tool call join the inert counter (they used
  to leave it alone and burn turn budget — bowling seed7 class).
- Degenerate prose streaks end as `stalled` (not handback `ended`) after the nudge cap.
- Mid-tool length caps stay exempt.

```bash
export LMP_DEGENERATE_RECOVERY=0   # exact "0" disables recovery behavior
export LMP_DEGENERATE_NUDGE_CAP=2  # optional; default = kRunNudgesBeforeEnding (3)
```

Worker / bakeoff: durable journal is `events.jsonl` beside `result.json` (also copied to
`events.ndjson`). Before overwrite, any prior journal is archived to
`events-<UTC>.jsonl`.

### Phase A1 keep/revert kill switch

Identical `tools_refresh` no-op is **on** by default. Disable for the Mac A/B:

```bash
export LMP_A1_NOOP_TOOLS_REFRESH=0   # exact "0" only
```

Tests set `AgentConfig::noop_identical_tools_refresh`; they do not race on setenv.

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
python3 scripts/agent_loop_wins/summarize_events.py baseline.jsonl treatment.jsonl
python3 scripts/agent_loop_wins/prefix_saboteur_stub.py   # synthetic matrix smoke
```

Keep multi-MB journals under local `piper-bench/agent_loop_wins/` (gitignored).

`scripts/agent_eval.py` copies each task's `events.jsonl` when `LMP_KEEP_EVENTS` is a directory:

```bash
LMP_KEEP_EVENTS=piper-bench/agent_loop_wins/a1_on \
  python3 scripts/agent_eval.py run --split corpus --seed 7
```
