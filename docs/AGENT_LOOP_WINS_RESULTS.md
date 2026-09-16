# Agent-loop wins — A/B results

Rows for the stacked PRs in `docs/AGENT_LOOP_WINS_BRANCH_PLAN.md`. Fill after each
measurement land; kill criteria live in the plan (§1).

**How to enable tier-B firehose:** `LMP_AGENT_LOOP_TRACE=1` (exact `1` only).
See also `docs/AGENT_LOOP_WINS_LOGGING.md`.
Tier A attribution (`kv_reuse`, `tools_refresh`, `error_class`, aggregate
`accept_at_depth` / `draft_len_hist`) is always on.

**Summarize / A/B delta:**

```bash
python3 scripts/agent_loop_wins/summarize_events.py path/to/events.jsonl
python3 scripts/agent_loop_wins/summarize_events.py baseline.jsonl treatment.jsonl
```

Local multi-MB dumps belong under `piper-bench/agent_loop_wins/` (gitignored), not
in the public tree.

---

## Baseline (post-PR1 logging)

| Change | Metric | Baseline | Treatment | n | Kill/keep | Notes |
|--------|--------|----------|-----------|---|-----------|-------|
| *(logging only)* | — | — | — | — | keep instrumentation | Gate covers field presence. A1 Mac pair is in Phase A below. |

---

## Phase A — prefix hygiene

| Change | Metric | Baseline | Treatment | n | worth | stability | Kill/keep | Notes |
|--------|--------|----------|-----------|---|-------|-----------|-----------|-------|
| **A1** no-op `tools_refresh` when allowlist text unchanged | TTFT; `kv_reuse.reason=tools_guidance_changed`; reuse rate; task solved | `LMP_A1_NOOP_TOOLS_REFRESH=0`: mean TTFT 4167 ms; 2 `tools_guidance_changed` Resets; 8 tools_refresh (6 init + 1 plan_lock + 1 plan_unlock); solved 6/6 | default on: mean TTFT 3563 ms (−14.5%); 0 `tools_guidance_changed` Resets; 6 init-only tools_refresh; solved 6/6 | Mac corpus n=6 seed=7, Qwen3.6-35B-A3B-MLX-4bit, same binary; CPU freeze vs kill-switch tests | **yes** TTFT −14.5% (bar ≥10%); Reset reason drop 2→0 | **yes** solved 6/6 both; empty-mask 0; ToolError 3→2; decode tok/s +5.7% | **keep** | Do not revert A1. Absolute `prefill_reused_tokens` mean fell (−21%) because prompt lengths differed; reuse *rate* rose +5.1%. Plan-lock refresh barely fired (1 vs 0), so this is not a clean plan-lock pair. **A2 blocked** until a confirmation set with matched `plan_lock` refreshes. Journals under `piper-bench/agent_loop_wins/a1_{on,off}/` (gitignored). |

Mac pairing (done): same sidecar, `LMP_A1_NOOP_TOOLS_REFRESH=0` vs unset, Qwen3.6-35B-A3B-MLX-4bit, `scripts/agent_eval.py run --split corpus --seed 7`. Compare with `summarize_events.py a1_off/all.jsonl a1_on/all.jsonl`. CPU freeze/kill-switch tests remain the lock that identical allowlist text does not rewrite. Confirmation set (disjoint n≥3 with matched `plan_lock`) not run — required before A2, not to keep A1.

---

## Phase B — SuffixProposer / draft_cost_ratio

| Change | Metric | Baseline | Treatment | n | Kill/keep | Notes |
|--------|--------|----------|-----------|---|-----------|-------|
| `draft_cost_ratio` | net decode tok/s / accept-at-depth | default `0.60` | sweep not run | — | **keep 0.60** | No Mac agent-generate sweep. Do not change the default without a ≥5% net win + confirmation set. |

---

## Phase C — grammar / ToolError class

| Change | Metric | Baseline | Treatment | n | Kill/keep | Notes |
|--------|--------|----------|-----------|---|-----------|-------|
| **C1** enforce `enum_values` in `ToolCallGuard` value phase | museum reject/accept matrix; empty-mask on golden paths | Pre-C: Text interiors unconstrained; `enum_values` in ToolSpec but not masked — bad enum strings ACCEPT at guard and only fail later as ToolError | Bad enum / missing required / unknown tool / truncated / bare text REJECT; valid enum + golden echo ACCEPT; `allowed_bytes()` non-empty on every golden prefix | Gate: `test_malformed_call_museum` + `test_toolcall_enum_mask` | **keep (CPU lock)** | Kill if enum enforcement empty-masks or sticks a golden good call. Kill switch: `Options::enforce_enum_values=false` or `LMP_ENUM_MASK=0`. Journal: `ErrorClass::SchemaEnum` → `error_class=schema_enum` (for post-guard inject paths). Live A3B ToolError A/B optional later — museum is the regression lock. |

### Museum matrix (before → after)

| Fixture | Pre-C (guard) | Post-C (guard) |
|---------|---------------|----------------|
| well-formed `echo` | ACCEPT | ACCEPT |
| unknown tool | REJECT | REJECT |
| truncated call | REJECT | REJECT |
| missing required param | REJECT | REJECT |
| wrong param name | REJECT | REJECT |
| **bad enum** (`color=yellow`) | **ACCEPT** (escape) | **REJECT** |
| valid enum (`color=green`) | ACCEPT | ACCEPT |
| number enum valid / invalid | valid ACCEPT; invalid **ACCEPT** (escape) | valid ACCEPT; invalid **REJECT** |
| bare text / non-tool garbage | REJECT | REJECT |
| golden enum path empty-mask? | n/a | **never** (kill criterion) |

**Mac A3B live A/B optional (parent):** bowling replay is flaky for forcing storms; re-run only if a journal shows residual `schema_*` / ToolError loops after this lands.

---

## Explicit skips (no A/B)

Not taken:

- Second PLD codebase beside `SuffixProposer`
- MTP / neural draft on MoE
- Tree-sitter / fuzzy AST apply in `apply_patch.hpp`
- Changing `LMP_PREFILL_CHUNK` default to 512 (2048 is the measured knee)
- Prefill token streaming during prefill
- Prompt-window n-grams, unique-token mini-prefill, and composite suffix+MTP (unproven; discarded)

---

## Degenerate-text / no-tool-call recovery

| Change | Metric | Baseline | Treatment | n | Kill/keep | Notes |
|--------|--------|----------|-----------|---|-----------|-------|
| Bounded recovery on degenerate / length-capped text-instead-of-tool | stall rate; wall; success; `ToolError`; `degenerate_text_count` / `nudged_count` | Tip ~79d4667 bowling seed7: A3B stalls after ~5× `degenerate_text`/(text) while tools Ok | Length-capped think/text joins inert nudge path; degenerate streak → `stalled` after cap; metrics on `run_end` | Gate: detector + nudge-cap + length-capped recovery tests | **pending Mac A/B** | Kill bar below |

**Kill bar (Mac A3B bowling seed7 A/B — do not claim win without it):**

- **Keep** if stall≤0 on same seed **OR** wall↓≥20% with success≥5/6 and `ToolError` not up.
- **Else** revert recovery behavior; keep metrics / journal fields.

**Enable / disable:**

```bash
export LMP_DEGENERATE_RECOVERY=0   # disable (default on)
export LMP_DEGENERATE_NUDGE_CAP=2  # optional hard cap (default = 3 agent / 2 plan)
```

**Benchbot rebuild for A/B:** tip of this branch vs `main` @ 79d4667 (or pre-merge main), same Qwen3.6-35B-A3B-MLX-4bit, NAX on, bowling seed7. Preserve `events.jsonl` beside `result.json` (auto-archived to `events-<UTC>.jsonl` on retry).
