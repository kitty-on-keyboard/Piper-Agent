# Branch plan: agent-loop wins (prefix hygiene, SuffixProposer, grammar)

**Audience:** Cursor Grok (or whoever implements).  
**Repo:** `kitty-on-keyboard/Piper-Agent` (local working tree: `LM_Pipe_2` on Sean’s Mac).  
**Date:** 2026-09-15.  
**Context:** Gemini notebook suggested research techniques; code audit shows Piper already has shadow/reuse, adaptive `SuffixProposer`, and L1 `ToolCallGuard` masking. This branch chases the *remaining* A3B agent-loop wins, not greenfield MTP/oQ/SSD KV.

---

## 0. Coordination (do this first)

1. **Do not land on a tree Jules/Grok are still stress-testing.** Wait until the current stress pass is green (or explicitly branched *from* their tip after merge).
2. Create branch from current `main` (or post-Jules tip):
   ```bash
   git fetch origin
   git checkout main && git pull
   git checkout -b exp/agent-loop-wins
   ```
3. **One heavy model at a time** on the 48GB M5 Pro. No bakeoff + this bench in parallel.
4. Default product model for all A/B: **Qwen3.6-35B-A3B-MLX-4bit**, NAX on (same metallib story as bakeoff). Dense MTP is out of scope for this branch.
5. Stress re-runs use `/Users/dev/Desktop/seans_projects_local/piper_stress/` only — never cwd inside `LM_Pipe_2`.

### Non-goals (explicit skip)

- MTP / neural draft on MoE (hard-refused; leave alone).
- oQ / OptiQ / sensitivity mixed-bit cook.
- Disk / paged SSD KV.
- New “PLD” stack alongside `SuffixProposer`.
- GBNF-for-JSON tool calls (trained format is Qwen XML + nested JSON PDA).
- Expert-aware draft steering (already measured dead in `docs/MOE_ROUTING_FINDINGS.md`).

---

## 1. Prove-it protocol (applies to every phase)

Every change ships with:

| Gate | Requirement |
|------|-------------|
| **Unit / grammar tests** | `ctest` gate (or the project’s usual gate target) green on the branch. |
| **A/B before claim** | Same binary flags except the one change under test. Same model dir, seed, prompt corpus. |
| **Metrics logged** | Write a short `docs/AGENT_LOOP_WINS_RESULTS.md` row: change, metric, baseline, treatment, n, kill/keep. |
| **Kill criteria** | If treatment is within noise or regresses the primary metric, **revert the change** (keep the harness/tests). No “land and hope.” |

**Noise floor (use these unless you measure tighter):**

- Wall / TTFT: treat &lt;5% as noise unless n≥10 paired runs say otherwise.
- Speculative accept rate / net tok/s: need ≥5% net decode win *or* clear accept-rate win with no TTFT regression.
- ToolError rate: need absolute drop that survives a fixed replay corpus (see Phase C).

**Harnesses already in tree (prefer these over inventing new ones):**

- Shadow / reuse: `tests/model/test_kv_compact_realmodel.cpp`, `lmp_diag reuse` (diag).
- Speculative / SuffixProposer: `tests/model/test_speculative.cpp`, `lmp_diag` spec cmds, `docs/MOE_ROUTING_FINDINGS.md` method.
- Grammar: `TurnGrammar` + parsephony tests; `lmp_diag specgrammar`.
- Agent events: journal fields like `prefill_reused_tokens`, `shadow_compact_*`, tool failures / `ToolError`.

---

## 2. Measurement logging (required before behavior PRs)

Today’s journal is good for **symptoms** (TTFT, reused tokens, aggregate spec counters, tool ok/fail). It is **not** enough to attribute prefix busts, tune `draft_cost_ratio`, or prove grammar wins. **PR1 must land this logging with no behavior change**; later phases depend on it.

**Overkill is fine during this branch.** Prefer too many structured fields over missing the unknown unknown. Verbose / high-cardinality emits (per-block `spec_block`, full arg dumps, hash traces) should be easy to turn up for experiments — and **must be cleaned up or gated when the branch closes** (see §2.8 and §8).

### 2.1 Already present (keep; scrape in baselines)

| Signal | Where | Notes |
|--------|--------|------|
| `ttft_ms`, `tokens_generated`, `prefill_reused_tokens` | generate / turn end emit | Symptom of reuse health |
| `spec_blocks`, `spec_drafted`, `spec_accepted` (and abandoned if emitted) | same | Totals only — not enough for cost-ratio sweep |
| `shadow_compact` (`tokens_warmed`, …) | agent emit | Warm succeeded |
| `shadow_compact_fallback` (`why`) | agent emit | Warm skipped / generate still Reset |
| `tool_result` (`tool`, `ok`, args, summary) | agent emit | Failures not typed |
| `mode_tools` | agent emit | Allowlist withheld count; not a tools-text hash |
| `phase` `generate_begin` | agent emit | Ordering anchor |

### 2.2 Must add (PR1) — prefix / reuse attribution

Emit on **every** generate (or immediately before prefill) a structured reuse decision:

**Event suggestion:** `kv_reuse` (or fields on the existing generate emit)

| Field | Purpose |
|-------|---------|
| `mode` | `Extend` \| `Restore` \| `Reset` (from `ReuseMode`) |
| `reused_tokens` | same as today’s `prefill_reused_tokens` |
| `prompt_tokens` | full prompt length |
| `reason` | **required on Reset** (and useful on Restore): stable string enum, e.g. `first_turn`, `tools_guidance_changed`, `compact_no_shadow`, `shadow_id_mismatch`, `persona_or_system_changed`, `image_hash_mismatch`, `ledger_mismatch`, `explicit_reset`, `unknown` |
| `stable_prefix_tokens` | checkpoint / stable render length when known |
| `shadow_armed` | `0`\|`1` — `kv_invalidated_by_compact_` / warm pending |

**Event suggestion:** `tools_refresh`

| Field | Purpose |
|-------|---------|
| `trigger` | `init` \| `plan_lock` \| `plan_unlock` \| `other` |
| `guidance_hash_before` / `guidance_hash_after` | FNV/sha256 of `tools_guidance_` bytes |
| `changed` | `0`\|`1` — byte-identical or not |
| `spec_count_before` / `spec_count_after` | `mode_specs_.size()` |
| `noop` | `1` if caller skipped applying a rewrite (post-A1); always `0` until A1 |

Without `reason` + `tools_refresh`, Phase A cannot prove A1 did anything.

### 2.3 Must add (PR1) — SuffixProposer / speculative depth curve

Aggregate `spec_*` totals stay. Add **per-block** (or per-generate histogram) detail, either:

- journal event `spec_block` sampled every block, or
- a ring/histogram rolled into generate end: `draft_len_hist`, `accept_at_depth` (counts for depth 1..K), `reject_at_depth`, `verify_ms_sum`

Minimum viable for the cost-ratio sweep:

| Field | Purpose |
|-------|---------|
| `draft_len` | tokens proposed this block |
| `accepted` | how many draft tokens survived |
| `bonus` | bonus/residual token committed or not |
| `phase` | Think/Text/ToolCall (speculation should be gated) |
| `abandoned` | block abandoned (grammar/special) |

Optional but high value: `proposer` = `suffix`\|`mtp`\|`none`.

Expose the same counters via `lmp_diag` so sweeps don’t require full IDE runs.

### 2.4 Must add (PR1) — ToolError / grammar class

Today `tool_result.ok=0` lumps parse escapes, exec failures, edit-misses, policy. Add:

**On `tool_result` (extend fields):**

| Field | Purpose |
|-------|---------|
| `status` | mirror `ToolResult::Status` (`Ok`, `ToolError`, `PolicyRefusal`, …) |
| `error_class` | stable enum, e.g. `exec`, `parse_args`, `schema_enum`, `schema_type`, `schema_required`, `edit_miss`, `sandbox`, `mcp`, `unknown` |
| `error_code` | optional short machine code if one exists |

**Grammar health (generate end or dedicated emit when mask goes empty / turn stuck):**

| Field | Purpose |
|-------|---------|
| `grammar_empty_mask` | count of sample steps with empty allow-set inside ToolCall |
| `grammar_phase_end` | phase at turn end |
| `grammar_reject_bonus` | if speculative bonus rejected by grammar (if tracked) |

Production need not log every mask probe; counters per generate are enough. Keep deep probes in `lmp_diag specgrammar`.

### 2.5 Must add (PR1) — small analysis helpers (scripts, not protocol)

Under local `piper-bench/agent_loop_wins/` (gitignored or docs-only summaries):

1. **`summarize_events.py`** — from `events.jsonl`: mean TTFT, reuse rate, Reset reason histogram, tools_refresh changed rate, ToolError class histogram, spec accept-at-depth.
2. **Fixture exporters** — pull `error_class=schema_*` / malformed raw tool XML into `tests/.../corpus/` for Phase C.

### 2.6 Optional tease-out harnesses (same PR1 or PR1.5)

Cheap, deterministic suites that *use* the new logs (no new inference stacks):

| Harness | What it forces | Pass signal |
|---------|----------------|-------------|
| **Prefix saboteur** | no-op plan-lock vs real allowlist change vs compact±shadow vs one-byte system poke | expected `kv_reuse.reason` + TTFT |
| **Malformed-call museum** | frozen bad `<tool_call>` strings through `ToolCallGuard` only (CPU) | reject/accept matrix; feeds Phase C |
| **Spec histogram** | repetitive import/XML tails with speculative on | accept-at-depth table for ratio sweep |

### 2.7 Logging acceptance for PR1 merge

- [ ] One short agent run’s `events.jsonl` shows `kv_reuse.reason` on every Reset
- [ ] A plan-lock transition emits `tools_refresh` with hashes; unchanged allowlist is visible as `changed=0` or still `changed=1` (pre-A1 truth)
- [ ] At least one speculative-on generate produces accept-at-depth (or per-block) detail
- [ ] Failed tools carry `error_class` (even if many are `unknown` initially — map the common paths)
- [ ] `summarize_events.py` prints histograms used in `AGENT_LOOP_WINS_RESULTS.md`
- [ ] Doc fix: `KV_SHADOW_SWAP.md` defaults match `session.hpp`
- [ ] **No** behavior change to reuse/grammar/draft defaults in PR1
- [ ] Verbose paths are behind an explicit flag/env (see §2.8), not unconditionally flooding default IDE runs

### 2.8 Verbosity policy + cleanup when done

Two tiers so we can fish for unknown issues without permanently bloating every run:

| Tier | Default | Examples | After branch |
|------|---------|----------|--------------|
| **A — Keep forever (cheap attribution)** | ON in product | `kv_reuse.mode` + `reason`, `tools_refresh` changed/hash (short), `error_class` on failures, aggregate `spec_*`, existing TTFT/reuse | **Stay.** These are the find/fix signals. |
| **B — Experiment firehose** | OFF unless `LMP_AGENT_LOOP_TRACE=1` (or similar) | per-block `spec_block` every step, accept-at-depth histograms every generate, full tools_guidance hashes every refresh, grammar empty-mask step counters, oversized arg/result snippets | **Gate or delete** before merge to main closes the effort. Keep the *codepath* if useful, default off. |

**Rules while chasing wins**

1. New high-volume emits go in tier B by default; promote to tier A only if a results row proved they catch real bugs cheaply.
2. Local artifacts (`piper-bench/agent_loop_wins/**`, raw `events.jsonl` dumps, sweep CSVs) are **local/gitignored** — never dump multi‑MB journals into the public tree.
3. Do not leave debug `printf` / stderr spam; everything goes through the event log with stable field names.

**Cleanup checklist (required before calling the branch done — see §8)**

- [ ] Tier B defaults **off** on main (env/flag required to re-enable).
- [ ] Remove any temporary always-on traces added “just for the sweep.”
- [ ] Delete or archive local `piper-bench/agent_loop_wins/` raw logs; keep only summarized rows in `docs/AGENT_LOOP_WINS_RESULTS.md`.
- [ ] Confirm a normal agent run’s event volume is back to ~pre-branch levels (spot-check event count / file size on a short mission).
- [ ] Docs mention how to re-enable trace mode for a future regression hunt.

---

## 3. Phase A — Prefix hygiene (shadow hit rate)

### Problem

Architecture is already “stable prefix + live-state suffix” (`context` render + `plan_turn_reuse`). Mid-run **`refresh_mode_tools()`** rewrites `tools_guidance_` (system `<tools>` block) on plan-lock transitions, which **invalidates the KV prefix from token 0**. Compaction without shadow warm also forces Reset.

### Goals

1. **Measure** how often and how costly tool-list rewrites are on real agent runs.
2. **Reduce unnecessary prefix busts** without breaking plan/mode tool allowlists.
3. Keep `shadow_compact` / `commit_think` **default ON**; fix doc drift (`docs/KV_SHADOW_SWAP.md` still implies unset=off; `session.hpp` precedence is C++ default true + env overlay).

### Implementation sketch (investigate, then change)

1. **Depend on §2 logging** (`kv_reuse.reason`, `tools_refresh`). No extra one-off prints.
2. **Baseline capture (after PR1):** 3–5 short Godoer or coding-agent loops (or recorded `events.jsonl` replays) with shadow on; report mean TTFT, mean `prefill_reused_tokens`, count of tools-rewrite busts.
3. **Candidate fixes (try in order; each must beat baseline):**
   - **A1. Freeze tools text for the run when the allowlist is unchanged.** If `refresh_mode_tools` produces identical `tools_guidance_` / `mode_specs_`, skip rewrite and skip any grammar rebuild that would change the rendered system block.
   - **A2. Park allowlist deltas outside the stable system prefix** (e.g. append a small “tools delta” live-state note instead of rewriting `<tools>` in the system prefix) — only if A1 is insufficient and reuse still dies on plan-lock. This is the bigger design change; prove A1 first.
   - **A3. Doc + setting hygiene:** update `KV_SHADOW_SWAP.md` to match `session.hpp`; assert bakeoff/worker paths don’t silently set `LMP_SHADOW_COMPACT=0` unless intentional.

### Tests / proof

| Test | Pass bar |
|------|----------|
| Unit: `refresh_mode_tools` no-op when allowlist unchanged | hash stable; no Reset reason `tools` |
| Realmodel: existing `shadow_compact_reuses_the_warmed_prefix_and_matches_flag_off` still green | unchanged |
| A/B agent loop (same mission) | mean TTFT ↓ or reused tokens ↑ by ≥10% on runs that hit plan-lock; no task-success regression |
| Kill | no measurable reuse/TTFT win on plan-lock workloads → keep instrumentation only, drop A2 |

### Success

Plan-lock / mode transitions no longer systematically zero `prefill_reused_tokens` when the tool *set* did not meaningfully change; docs match defaults.

---

## 4. Phase B — SuffixProposer tune (MoE “PLD”)

### Problem

Adaptive `SuffixProposer` already *is* the PLD-shaped path (`src/model/draft_proposer/`). Default `draft_cost_ratio = 0.60` was tuned on **synthetic** data. `MOE_ROUTING_FINDINGS.md` says re-measure on **real** SuffixProposer drafts; fixed-k loses.

### Goals

1. Capture accept/reject traces from real A3B agent generates (tool-heavy).
2. Sweep `draft_cost_ratio` (and only if needed: `min_match_len` / `min_support`) against that corpus.
3. Land a new default **only if** net decode tok/s improves without hurting TTFT or accept-quality.

### Implementation sketch

1. **Harness:** use §2.3 `spec_block` / accept-at-depth logs (plus `lmp_diag` if needed). Sweep must not rely on aggregate `spec_accepted` alone.
2. **Corpus:** ≥3 agent sessions or a fixed replay of tool-call generations (imports, paths, XML tags — where trie drafts should hit).
3. **Sweep:** `draft_cost_ratio ∈ {0.50, 0.55, 0.60, 0.70, 0.80}` (band from MOE doc 0.56–0.81). Fixed-k control must remain worse (sanity).
4. **Optional micro-win:** if history trie misses coding PLD cases, allow a **prompt-window** match source (copy n-grams from the *current prompt* not only decode history) behind a flag — **only after** cost-ratio sweep; treat as separate A/B.

### Tests / proof

| Test | Pass bar |
|------|----------|
| Existing `tests/model/test_speculative.cpp` | green |
| Sweep table in `AGENT_LOOP_WINS_RESULTS.md` | best ratio vs 0.60 |
| Keep new default | ≥5% net generate tok/s on agent corpus **or** same speed + clearly higher accept rate with no TTFT loss |
| Kill | no winner vs 0.60 → leave default; keep harness |

### Success

Documented, re-measured default (or explicit “0.60 confirmed”) on **real** A3B drafts; no second PLD codebase.

---

## 5. Phase C — Grammar holes (ToolError escapes)

### Problem

L1 already masks via `TurnGrammar` → `parsephony::ToolCallGuard` (Qwen XML tool calls). Gaps:

- `ParamType::Text` interiors are nearly unconstrained.
- `enum_values` are parsed into specs / tools JSON but **not enforced in the mask** (see `toolcall.hpp` / value phases).
- Past ToolError storms were often **post-decode** malformed args / edit loops, not missing masking entirely.

### Goals

1. Build a **regression corpus** of real malformed tool calls (from bakeoff miss notes / journals).
2. Enforce **enums** (and other cheap schema edges) in the guard when `enum_values` non-empty.
3. Do **not** replace XML tool-call format with JSON-GBNF.

### Implementation sketch

1. **Corpus first:** mine journals with `error_class` / malformed-call museum (§2.6); fixture files under `tests/` with raw tool-call strings that emit but fail at execution (bad enum, wrong type, …).
2. **Enum masking:** when entering a param whose `enum_values` is non-empty, restrict value phase to those literals (XML text or JSON string form as appropriate).
3. **Required / type:** only tighten where fixtures prove escape; avoid over-constraining Text paths/paths that need freeform.
4. **MCP path:** ensure `mcp_host` → `ToolSpec` already fills `enum_values` (it does); guard must use them.

### Tests / proof

| Test | Pass bar |
|------|----------|
| New parsephony / grammar unit tests for enum params | invalid enum token never in mask; valid path still completes |
| Replay corpus | ToolError count ↓ vs baseline on same prompts; zero false-block on golden good calls |
| Agent smoke | 1 short run; no new “empty mask” / stuck-in-ToolCall failures (`lmp_diag specgrammar` style) |
| Kill | enum enforcement causes stuck generations or empty masks → feature-flag off or narrow to JSON-typed enums only |

### Success

Fewer repair turns from schema-invalid tool calls; grammar tests lock the wins; XML tool-call format unchanged.

---

## 5b. Stress-test product fixes (2026-09-15 orchestration log)

Roll these into the same branch effort (can be parallel PRs to the wins stack). Source: `piper_stress/ISSUES.md`. Sandbox: `/Users/dev/Desktop/seans_projects_local/piper_stress/` — **never** cwd inside `LM_Pipe_2` for stress runs. Worker: `LM_Pipe_2/build/src/surface/lmp_sidecar`. Model: `Qwen3.6-35B-A3B-MLX-4bit`.

### Must fix (Piper product) — priority order

| ID | Issue | Fix direction | Proof |
|----|-------|---------------|-------|
| **P1** | `files_touched` only counts `kind=write`; MCP/Godoer writes show `[]` while `git.diff` has files | Collect paths from MCP `tool_result` / workspace freshness **or** union with `git.diff` paths when trust_mcp | Re-run lt-001 / lt-002c: `files_touched` non-empty and ⊆ expected; orchestrator review usable |
| **P2** | `result.status` never `"stalled"` (only ok/error/timeout) though wake contract has stalled | Set `status=stalled` for max_turns / no_progress stalled | result.json + webhook kind agree; parent need not parse error strings |
| **P3** | Incomplete runs dump think/progress diary into `result.message` | Prefer last finish summary; else short first/last of final assistant text | Incomplete slice messages are short and useful |
| **P4** | Hard `max_iterations=30` in `build_start_message`; not in task.json | Packet field `max_iterations` (and/or higher default when `trust_mcp` set) | Godoer-heavy slice can raise limit without rebuild; document default |
| **P5** | Passing check + max_turns still `status=error` | Treat verify_contract success as complete **or** document check as advisory and surface distinctly | lt-004-class: green check does not look like a crash |
| **P6** | Sandbox tier 1 blocks `bind()`; HTTP integration tests thrash | Allow loopback bind in auto-approved exec **or** hard clear “network bind denied” | ha-002-class: model stops or loopback server test can run |
| **P7** | Noisy `mcp_config_file skipped` then successful trust_mcp | Downgrade/clarify log when settings trust path succeeds | Grep no longer looks like spawn failure |
| **P8** | `conventions_name_absent_tools: probe_args` (Godoer AGENTS.md) | Fix Godoer docs and/or worker warn list | Warning gone or accurate |

### Live with / orchestrator lessons (do not block Piper PRs)

- **A1** Structural asserts not substring grep (packet/check quality).
- **A2** Stall detector may be aggressive with huge MCP context — measure before retuning.
- **A3–A4** Godot playability / asset axes — Godoer/content, not worker infra.
- **A5** First-turn ~50s TTFT / ~30k tokens with Godoer — expected; budget for it.
- Setup: metallib copy needed for sidecar build on that machine; Cursor MCP `godot_check_scene` None vs CLI OK — Godoer wrapper, not Piper.

### Suggested PR interleave with wins stack

| PR | Contents |
|----|----------|
| **PR-S1** | P1 files_touched + P2 stalled status + P3 message trim (orchestration contract) |
| **PR-S2** | P4 max_iterations packet + P5 check-vs-complete policy |
| **PR-S3** | P6 loopback bind or hard deny + P7 MCP log + P8 probe_args |

Re-prove with targeted `piper_stress` slices (lt-001, lt-002c, lt-004, ha-002) after each PR — keep-warm daemon OK; one model at a time.

---

## 6. Suggested PR slice order


Land as **stacked PRs** (easier kill/revert), not one megadiff:

| PR | Contents | Merge iff |
|----|----------|-----------|
| **PR1** | **§2 measurement logging** (tier A on, tier B flagged) + `summarize_events.py` + doc default fix + results skeleton + optional prefix-saboteur / museum stubs | always (merge when §2.7 checklist passes) |
| **PR2** | Phase A1 no-op refresh / freeze identical tools text | A/B using `kv_reuse.reason` + TTFT/reuse histograms |
| **PR3** | Phase B ratio default (if any) | sweep using accept-at-depth, not totals only |
| **PR4** | Phase C enum (or other) grammar tighten | corpus `error_class=schema_*` ↓, no stuck mask |
| **PR5 (optional)** | A2 tools-delta layout or B prompt-window PLD | only if PR2/PR3 left clear headroom |
| **PR-final** | §2.8 cleanup: tier B default off, remove temp always-on traces, artifact purge notes | required to close the branch |

Each PR: gate CI green + `AGENT_LOOP_WINS_RESULTS.md` updated.

---

## 7. Baseline checklist (run once on branch tip after PR1 logging)

```text
[ ] ctest gate (CPU) green
[ ] Note model path + MLX/NAX build flags
[ ] PR1 logging merged (§2.7 checklist)
[ ] lmp_diag reuse (or realmodel shadow test) — record reused tokens + TTFT
[ ] One short agent loop with LMP_SHADOW_COMPACT=1, LMP_COMMIT_THINK=1 — save events.jsonl
[ ] summarize_events.py: Reset reason histogram, tools_refresh changed rate
[ ] ToolError class histogram (even if many `unknown`)
[ ] Speculative-on generate: accept-at-depth table present
[ ] Prefix saboteur smoke (if landed): expected reasons match
```

Store artifacts under something like `piper-bench/agent_loop_wins/baseline/` (local only; do not dump huge logs into the public repo root).

---

## 8. Definition of done for the branch

- Stacked PRs merged or explicitly killed with numbers in `docs/AGENT_LOOP_WINS_RESULTS.md`.
- PR1 **tier A** logging remains even if behavior PRs are killed (attribution is permanent value).
- **§2.8 cleanup checklist complete** — tier B firehose off by default; local raw logs gone; event volume sanity-checked.
- No new inference stacks (MTP/oQ/SSD/PLD-dupe).
- Product defaults still: shadow_compact on, commit_think on, A3B without MTP.
- Stress/Jules line remains green; this branch does not regress gate tests.

---

## 9. Handoff one-liner for implementer

> Branch `exp/agent-loop-wins` from post-Jules main (or tip after stress rebuild). Parallel tracks: (1) wins plan PR1 logging → A/B/C; (2) stress P1–P8 orchestration fixes (§5b). PR1 first on the wins track: overkill-OK structured logging with tier B behind `LMP_AGENT_LOOP_TRACE`. Every behavior change needs proof or revert. When done: firehose off, purge local dumps. Skip MTP/oQ/SSD/new PLD. Owner: Agentic Engineer bot (not Cursor Grok unless Sean escalates).
