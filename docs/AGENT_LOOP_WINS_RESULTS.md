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
| *(logging only)* | — | — | — | — | keep instrumentation | Fill after first short agent run with shadow on |

---

## Phase A — prefix hygiene

| Change | Metric | Baseline | Treatment | n | Kill/keep | Notes |
|--------|--------|----------|-----------|---|-----------|-------|
| | | | | | | |

---

## Phase B — SuffixProposer / draft_cost_ratio

| Change | Metric | Baseline | Treatment | n | Kill/keep | Notes |
|--------|--------|----------|-----------|---|-----------|-------|
| | | | | | | |

---

## Phase C — grammar / ToolError class

| Change | Metric | Baseline | Treatment | n | Kill/keep | Notes |
|--------|--------|----------|-----------|---|-----------|-------|
| | | | | | | |
