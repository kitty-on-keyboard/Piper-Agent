# Agent-loop wins — A/B results

Rows for the stacked PRs in `docs/AGENT_LOOP_WINS_BRANCH_PLAN.md`. Fill after each
measurement land; kill criteria live in the plan (§1).

**How to enable tier-B firehose:** `LMP_AGENT_LOOP_TRACE=1` (exact `1` only).
See also `docs/AGENT_LOOP_WINS_LOGGING.md`.
Tier A attribution (`kv_reuse`, `tools_refresh`, `error_class`, aggregate
`accept_at_depth` / `draft_len_hist`) is always on.

**Summarize a journal:**

```bash
python3 scripts/agent_loop_wins/summarize_events.py path/to/events.jsonl
```

Local multi-MB dumps belong under `piper-bench/agent_loop_wins/` (gitignored), not
in the public tree.

---

## Baseline (post-PR1 logging)

| Change | Metric | Baseline | Treatment | n | Kill/keep | Notes |
|--------|--------|----------|-----------|---|-----------|-------|
| *(logging only)* | — | — | — | — | keep instrumentation | Gate covers field presence; **Mac A3B live run still needed** for real Reset-reason histogram + plan-lock `tools_refresh` |

---

## Phase A — prefix hygiene

| Change | Metric | Baseline | Treatment | n | Kill/keep | Notes |
|--------|--------|----------|-----------|---|-----------|-------|
| **A1** no-op `tools_refresh` when allowlist text unchanged | `tools_refresh.noop` / `changed`; `kv_reuse.reason!=tools_guidance_changed` on identical refresh; TTFT / `prefill_reused_tokens` on plan-lock workloads | Pre-A1: every refresh rewrote `tools_guidance_`/`mode_specs_` even when hash-identical (`noop` always `0`) | Mid-run identical refresh skips rewrite (`noop=1`, `changed=0`); real plan-lock still `noop=0`/`changed=1` and drops `plan` from guidance | Gate: `phase_a1_noop_tools_refresh_freezes_identical_guidance` | **keep pending Mac A/B** | Kill if Mac plan-lock workloads show &lt;10% TTFT/reuse win vs baseline *and* no drop in spurious `tools_guidance_changed` Resets. A2 (tools-delta outside stable prefix) only if A1 leaves real allowlist-change Resets as the dominant cost. |

**Mac A3B live A/B still needed (parent):**

1. Same binary flags except A1 on/off (or tip vs pre-A1), Qwen3.6-35B-A3B-MLX-4bit, shadow+commit_think on.
2. 3–5 short agent loops that hit plan-lock (or recorded `events.jsonl` replays).
3. Compare mean TTFT, mean `prefill_reused_tokens`, Reset reason histogram (`tools_guidance_changed` count), `tools_refresh` `noop`/`changed` rates via `summarize_events.py`.
4. Kill/keep per row above; do not land A2 without that proof.

---

## Phase B — SuffixProposer / draft_cost_ratio

| Change | Metric | Baseline | Treatment | n | Kill/keep | Notes |
|--------|--------|----------|-----------|---|-----------|-------|
| *(not started)* | accept-at-depth / net decode tok/s | default `draft_cost_ratio=0.60` | sweep `{0.50,0.55,0.60,0.70,0.80}` on **agent-shaped** drafts | — | — | Needs Mac speculative-on generates + tier-B `spec_block` / `accept_at_depth`. Kill if no ≥5% net win vs 0.60. |

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
