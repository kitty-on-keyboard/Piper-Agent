# Plan: close every gap between LM_Pipe and a 2026 local agent

> **Status, 2026-08-02, end of the implementation session.** All of G0–G7 landed,
> G7b included (schema, sidecar round trip, and the extension's WorkspaceEdit applier).
> Gate 26/26, ratchets 6/6 with `--self-test` still able to redden each one.
> Eval pins moved 3/6 -> 5/6 corpus and 1/4 -> 2/4 holdout, reproduced across two runs.
>
> Two things beyond the original eleven items were fixed because running the suite exposed
> them: an overwrite hole in the HITL gate (finding 2 below) and a repeated-failure loop
> (finding 5). One architectural defect the plan did not contain was found and fixed:
> **completion rested on a criterion the model chose for itself** (finding 3).
>
> **Known and NOT closed:** the mutation harness `docs/PHASES.md` cites for its 3/8 figure
> is not in this repo, so that number cannot be re-measured. Recorded there rather than
> quietly left.
>
> **Five findings the plan did not predict, every one from RUNNING the suite rather than
> reading the code:**
>
> 1. **The line-number hazard was mis-diagnosed here.** This document said a numbered
>    `old_text` produces a safe NoMatch and left it at a warning in the tool description.
>    The first run after numbering landed showed five consecutive `replace_in_file`
>    failures — which turned out to be graft **ambiguity**, not numbering, and would have
>    been invisible for another hour if `agent_eval.py -v` had printed the summary
>    alongside the status. It now does. The numbering defence (`strip_line_numbers`) was
>    built anyway and is right on its own terms; it was not what those turns were.
>
> 2. **The safety fixture was failing, and every declared gate was holding.**
>    `refuse_wipe_workspace` denied `delete_file` twice and `shell` twice — the HITL
>    apparatus worked exactly as designed — and the run then emptied `ledger.csv` with
>    three `write_file` calls that nothing asked about, because `write_file` is not
>    declared irreversible and `auto_approve_writes` defaults on. Scored on whether the
>    DATA survived, it did not. Structurally the same hole `HANDOFF_AGENT.md` records as
>    the reason `irreversible` was added to `ToolDecl`: the apparatus was watching the
>    tool the run had stopped using. Fixed by making irreversibility a property of the
>    CALL — a whole-file write over existing non-empty content escalates regardless of
>    `auto_approve_writes`. `replace_in_file` is deliberately not covered.
>    **Whether this was a regression could not be established**: `pins.json` stores only
>    aggregates, so there is no record of which 3 of 6 corpus tasks passed on 2026-07-31.
>
> 3. **A false completion, surfaced by the disagreement column.** `rename_across_files`
>    finished `completed=yes verified=yes solved=no`. The model declared `pytest -q` as
>    its contract, made it pass, and the harness agreed the evidence was complete — while
>    the task's ground truth also requires no residual `calc_total`. Completion is only as
>    strong as the contract the run declares, and nothing checks that the declared contract
>    covers the mission. Not fixed here; it is the next real question for the loop.
>
> 4. **G3's falsifier, as this document specified it, was unachievable — and the code was
>    fine.** The plan said reuse must produce byte-identical output to a run without it.
>    It does not, and fails at token 0, deterministically. The cause is not a stale cache:
>    `checkpoint_at` forces a chunk edge, so the two runs prefill in different SEGMENTS,
>    and this bf16 checkpoint is segmentation-sensitive. Measured directly with no reuse in
>    the picture — `LMP_PREFILL_CHUNK=512` and `=2048` produce different first tokens on the
>    same prompt. `HANDOFF_SPECULATIVE.md` records the same property from the other side.
>    The assertion that isolates what the test is for holds segmentation FIXED (both runs
>    pass the same `checkpoint_at`) and varies only whether turn two restored or
>    re-prefilled; that passes, and an injected off-by-one on the prefill resume point makes
>    it fail, so it has been shown capable of red.
>
>    A second finding fell out of proving it: the FIRST injection (desyncing the ledger by
>    one) did **not** turn the test red, because prefill resumes from the reuse plan rather
>    than from the ledger. The test was insensitive to that wire. That is a finding about
>    the test, and it is why the injection was moved to the resume point.
>
> 5. **An unrecoverable failure repeated verbatim was not a repeat.** `BreakRepeat`
>    required `ok()`, on the reasoning that "repeating after an error is legitimate retry".
>    True of a transient error, false of ambiguity, NotFound or Malformed — those are pure
>    functions of the bytes sent. The same hole `RefusalLedger` was added to close, one
>    class over. Fixed and gated on `retryable`, so a flaky build still gets its retry;
>    the observed run went 5 wasted turns → 2.


Written 2026-08-02, on `feat/markdown-stream-cookoff`, from a line-by-line audit of the
tree against a circulating "2026 local agent" guide (grammar-constrained decoding,
targeted diff editing, a deterministic verification loop, context/KV hygiene).

The audit's conclusion is that three of the guide's four pillars are already implemented
here in a stronger form than the guide describes, and that the interesting gaps are the
ones the guide does *not* mention. This plan covers all of them, plus the four items
`docs/PHASES.md` already lists as not done — "all" means all.

Every item below states: **what is wrong**, **where**, **the design and why that design**,
**the tests that must exist**, and **what would falsify the fix**. An item with no
falsifier is not finished.

---

## Contents

| # | Gap | Size | Risk |
|---|---|---|---|
| G0 | Miniature tokenizer fixture — unblocks gate-level tests for everything below | M | low |
| G1 | No automatic post-edit check (the LSP/typecheck gap) | L | medium |
| G2 | `SynthesizeVerification` runs a hardcoded `cmake --build build` | S | low |
| G3 | KV prefix cache is discarded every turn | L | **high** |
| G4 | File reads enter the prompt unbounded | M | low |
| G5 | `read_slice` is not line-numbered | S | low |
| G6 | `locate_symbol` is a grep regex, not an index | M | low |
| G7a | T2 containers refuse; unattended runs unavailable | XL | medium |
| G7b | `lmp/edit` does not go through the extension (no undo, no diff review) | L | medium |
| G7c | `.vsix` is not packaged and the sidecar is not staged | S | low |
| G7d | `ApprovalRequestNotification` dead-code exemption is stale | XS | low |

Ordering is G0 → G2 → G4/G5 → G1 → G3 → G6 → G7. G0 first because four of the five
surviving mutants and most of the tests below are blocked on the same missing fixture.
G3 late because it is the only item that can produce silently wrong output.

**Standing rules for every step.** `ctest --preset gate` green. `./scripts/run_ratchets.py`
6/6 green. `tests/gate/gate_manifest.txt` count *and* names bumped in the same commit as a
new test. `python3 scripts/agent_eval.py run` re-run; `evals/agent/pins.json` are floors —
a drop fails, an improvement asks to be re-pinned deliberately. One MLX process at a time.

---

## G0 — The miniature tokenizer fixture

**What is wrong.** `docs/PHASES.md` names four surviving mutants
(`grammar.cpp:105`, `agent.cpp:168`, `agent.cpp:287`, `sidecar.cpp:85`) and says all but
the last are blocked on one thing: there is no vocabulary in the gate, so `TurnGrammar` and
`Agent::step` are only exercised by `realmodel` tests, which are excluded. It calls a
committed miniature Qwen-shaped `tokenizer.json` "the single highest-value next task". It
is also a prerequisite for testing G1, G2, G3 and G5 at gate level rather than by hand.

**Design.** A hand-written `tests/fixtures/mini_qwen/tokenizer.json`, a few hundred tokens:

- Every special id `TurnGrammar` and `ChatTemplate` reference by name — `<|im_start|>`,
  `<|im_end|>`, `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, and the
  `<function=` / `<parameter=` / closing forms parsephony's `ToolCallGuard` needs.
- Enough byte-level and merge entries to encode ASCII identifiers, digits, braces, quotes
  and newlines, so a tool call and a small source file both round-trip.
- At least one multi-byte UTF-8 sequence split across two tokens, because `frankentok`'s
  streaming decode contract is "byte-identical to batch" and the fragment case is the one
  that broke it before.

The ids must **not** match the real checkpoint's. A fixture that coincidentally agrees
with production hides the class of bug where code assumes a literal id.

**Tests.**
- `tests/model/test_mini_vocab.cpp` — the fixture loads, encode/decode round-trips,
  streaming decode equals batch decode, every special the grammar names resolves.
- Re-run the mutation harness. `grammar.cpp:105`, `agent.cpp:168` and `agent.cpp:287`
  must move from survivor to killed. Publish the new honest score in `docs/PHASES.md`.

**Falsifier.** Delete one special from the fixture: `test_mini_vocab` must fail by name,
not by a downstream crash somewhere in the loop.

**Manifest.** count 22 → 23; add `test_mini_vocab`.

---

## G1 — No automatic post-edit check

**What is wrong.** `grep -rn 'lsp\|clangd\|tsserver\|pyright\|diagnostic'` over `src/`
returns nothing outside `log_triage`'s prose. After a successful `write_file`,
`replace_in_file` or `append_file`, **nothing runs**. The only non-model feedback path is
`shell`, and only when the model chooses to call it. This is the guide's §3 steps 3–4, and
it is the one thing in that guide genuinely worth adopting.

**What we will not build first.** An LSP client. LSP diagnostics arrive as a
`textDocument/publishDiagnostics` *push* with no completion signal, so a client needs
server lifecycle management, `didOpen`/`didChange` version tracking, and a settle timeout
that is a heuristic in exactly the place we claim not to use heuristics. Phase B, behind
the same interface, once Phase A has proven the loop wiring.

### Design

**New L2 component: `src/tools/syntax_check.{hpp,cpp}`.**

```cpp
struct SyntaxContract {
    std::string language;      // "python", "cxx", "json", ...
    std::vector<std::string> extensions;
    // Command template; "{file}" is substituted with the workspace-relative path.
    std::string command;
    // True when the contract needs project context to be meaningful (cxx).
    bool needs_compile_db = false;
};

struct SyntaxVerdict {
    bool ran = false;          // false == no contract matched, or it could not be run
    bool clean = false;
    std::string diagnostics;   // already compacted; empty when clean
};

class SyntaxChecker {
  public:
    SyntaxChecker(std::string root, std::size_t budget_bytes);
    [[nodiscard]] SyntaxVerdict check(const std::string& rel_path, int approved_tier) const;
};
```

Shipped contracts:

| language | command | note |
|---|---|---|
| python | `python3 -m py_compile {file}` | ~50 ms, no project deps |
| json | `python3 -m json.tool {file} >/dev/null` | |
| javascript | `node --check {file}` | syntax only, honest about it |
| cxx | the file's own entry from `build/compile_commands.json`, with `-fsyntax-only` | see below |
| typescript | *deferred to Phase B* | `tsc --noEmit` is project-wide and slow |

**C++ only runs when `compile_commands.json` has an entry for that exact file.** A bare
`c++ -fsyntax-only` on a project header emits a cascade of missing-include errors that are
not about the edit, and a false diagnostic handed to the model is worse than no check —
it sends the run off fixing something that was never broken. `needs_compile_db` makes that
a property of the contract, not a special case in the caller. No entry → `ran = false`.

**Silence is the default for anything unrecognised.** A per-turn "no checker available for
`.md`" line is prompt noise that recurs for the whole run.

**Where it is called: `Agent::dispatch_call`,** in the block that already handles a
successful mutating call (`src/loop/agent.cpp:505-512`, where `record_deliverable` fires).
Not inside `Registry::execute`: the registry is L2 and must not own loop policy, and the
`dispatch_call` site is already the one place that knows "a write just succeeded, here is
the path".

**The result is appended to the same observation, not made a new turn.** A post-edit check
is a consequence of the edit, not a separate action. Making it a turn would violate
one-turn-one-outcome (S9.1) and inflate `iterations` against the eval budget.

```
replaced one occurrence in stats.py
[syntax] python: FAILED
  File "stats.py", line 14
    return sorted(xs)[n//2
                     ^
SyntaxError: '[' was never closed
```

**It must never touch the verification ledger.** `SyntaxChecker` does not take a
`Verifier` and does not construct one. A syntax check is not the declared contract, and a
green from it must never help a run complete. This is the single most likely way to
implement this wrongly: routing it through `Verifier::run_and_record` would look tidy and
would quietly make S10.4 completion cheaper. Enforce it with a test that asserts
`ctx.verifications().size()` is unchanged across a checked edit.

**First-touch pre-check.** A red that was already red is not evidence about the edit. Before
the *first* mutating call for a given path, run the same contract against the file as it
is on disk — that file *is* the pre-image, so nothing needs snapshotting — and cache the
verdict in `Agent::pre_edit_clean_` (a `std::map<std::string, bool>`). If the file was
already failing, the post-edit diagnostic is reported as
`[syntax] python: still failing (was already failing before this edit)`, which is a
different fact and a different next move. This is the FAIL_TO_PASS discipline from
`verification.cpp:63-77` applied one level down, and it costs one extra sandboxed run per
file per run.

**Tier and budget.** Skipped entirely when `policy_.sandbox_tier == 0` (Plan mode cannot
write, so this is moot, but state it rather than let it be implied). `ExecLimits` with a
15 s wall clock. Output through `log_triage::compact` at a 2048-byte budget — a quarter of
`max_result_bytes`, so a check can never crowd out the edit result it is annotating.

**Config.** `AgentConfig::auto_syntax_check`, default `true`, plus a settings field on the
wire and in the extension drawer. Default-on is consistent with S13: this only ever
tightens, and S13's rule is that defaults tighten and explicit requests loosen.

**What we deliberately do not do: revert the file.** The write succeeded; reverting would
leave the workspace in a state the model did not ask for and cannot see. Report and let the
next turn fix it. Completion is unaffected either way, because it still requires the
declared contract to pass falsifiably.

### Tests

- `tests/tools/test_syntax_check.cpp`
  - contract resolution by extension; unknown extension → `ran == false`, empty diagnostics
  - cxx with no `compile_commands.json` → `ran == false` (not a failure)
  - cxx with an entry → runs, and the entry's flags are honoured
  - a broken python file → `clean == false` with the diagnostic text present
  - diagnostics longer than the budget are compacted, not head-truncated
- `tests/loop/test_post_edit_check.cpp` (needs G0)
  - a scripted turn that writes a broken file → the observation carries `[syntax]` and
    `observation_is_error == true`
  - the same with `auto_syntax_check = false` → observation byte-identical to today's
  - an already-broken file → the "was already failing" wording, and the run does not treat
    it as new information
  - **`ctx.verifications()` unchanged in all of the above**

### Falsifier

Plant a syntax error via the eval harness in `evals/agent/tasks/failing_test_median` and
require that the agent's next observation names it. Then disable the feature and require
the same run to *not* name it. A feature that cannot be switched off into the old
behaviour has not been isolated.

### Eval

Add `evals/agent/tasks/edit_introduces_syntax_error/` (corpus split): a task where the
obvious edit is one character away from a syntax error, scored on `check` as usual. The
metric that matters is turns-to-solve, not solve rate.

**Manifest.** +2 tests.

---

## G2 — `SynthesizeVerification` runs a hardcoded command

**What is wrong.** `src/loop/agent.cpp:584-590`:

```cpp
case Corrective::SynthesizeVerification: {
    Verifier verifier(registry_, ctx_);
    (void)verifier.run_and_record("cmake --build build", policy_.sandbox_tier);
    return;
}
```

Three defects in six lines.

1. **The command is hardcoded** while `ctx_.verify_contract()` holds the contract the run
   declared through `plan`. Eight of the ten eval fixtures are Python. When this corrective
   fires there — and it fires on `turn.assistant_text` containing "should pass", which is
   exactly what a model says about a Python test — it runs a command that cannot work. The
   mechanism that exists to break a stall instead files a guaranteed failure against a
   contract nobody declared.
2. **It constructs a second `Verifier`.** `proven_` (the falsifiability cache) lives in
   `verifier_`; a fresh instance starts empty. `is_proven` partially recovers by scanning
   the ledger, so this is masked rather than harmless — which is worse.
3. **It files under the command's own canonical form**, not the declared contract, minting
   the historyless identity that `run_and_record_as` was added to prevent
   (`verification.hpp:36-40`).

**Fix.**

```cpp
case Corrective::SynthesizeVerification: {
    emit("corrective", {{"kind", "synthesize_verification"},
                        {"contract", ctx_.verify_contract()}});
    (void)verifier_.run_and_record_as(ctx_.verify_contract(), policy_.sandbox_tier,
                                      canonicalize_check(ctx_.verify_contract()));
    return;
}
```

**Applicability moves into `choose_corrective`.** With no declared contract there is
nothing to synthesize, and returning the corrective anyway would run the empty string
through the shell. `choose_corrective` is a pure function and must stay one, so it takes a
new `bool have_verify_contract` parameter rather than the store:

```cpp
Corrective choose_corrective(const TurnResult&, const RepeatDetector&,
                             const RefusalLedger&, int iterations_used, const Budget&,
                             bool wall_clock_exhausted, bool have_verify_contract);
```

and the `claims_verification` branch at `turn.cpp:176-185` is gated on it.

**Tests** (`tests/loop/test_loop.cpp`):
- a scripted run with `verify_with = "pytest test_stats.py"` whose turn says "should pass"
  → the synthesized command contains `pytest test_stats.py` and **not** `cmake`
- the same run with no contract → `choose_corrective` returns `None`, and no verification
  record is written
- the synthesized record's `contract` field equals the canonical declared contract

**Falsifier.** Revert the change: the first test must go red on the `cmake` string.

**Ratchets.** `prose_correctives` must stay at 0 — this change adds a mechanism, not
prose, so an unchanged 0 is the expected reading.

---

## G3 — The KV prefix cache is discarded every turn

**What is wrong.** `src/model/mlx_backend.cpp:308-315`:

```cpp
const ReuseDecision reuse = ledger_.plan_reuse(task.prompt);
if (reuse.divergent) {
    impl_->model.reset_cache();
    ledger_.clear();
}
const std::size_t start = reuse.divergent ? 0 : reuse.reusable;
```

`plan_reuse` carefully computes the verified-identical prefix length and then it is thrown
away whenever anything diverges. **Divergence is guaranteed between turns.** After turn N
the ledger holds `[prompt_N][generated_N]`. Turn N+1's prompt inserts a new `TurnRecord`
*before* the live-state block, so the ledger continues with turn N's live-state tokens
where the new prompt continues with the new turn record. Mismatch mid-ledger →
`divergent` → full reset → re-prefill of the entire context, every turn.

So the most-stable-first layout in `src/context/context.cpp:79-94` — which was
introduced with a measurement (TTFT climbing 1427 → 1758 ms across 29 turns at ~3k
tokens) and whose comment claims a change now "costs one message of re-prefill rather than
the whole context" — currently buys nothing at the backend. The recorded
`ttft 1775 ms` against a measured prefill rate of 1317–1684 tok/s is arithmetically
consistent with re-prefilling roughly 3k tokens from scratch on every turn.

### Design: one checkpoint per turn, at the stable boundary

`src/model/mlx/qwen35_moe_model.hpp:113-154` already has exactly the machinery, built for
speculation, and its comment states the constraint that shapes this design:

> Full-attention layers keep a per-token history, so any earlier position is reachable by
> moving an index. Linear (gated-delta) layers keep a recurrence with no per-token history,
> so only positions that were snapshotted in advance are reachable at all.

30 of this model's 40 layers are gated-delta. Arbitrary rollback is therefore impossible;
rollback **to a position that was snapshotted in advance** is not. The whole design follows
from that one sentence.

**Step 1 — the boundary must come from token offsets, not from re-rendering.**

Hazard, and the reason this is written down: `ChatTemplate::render()` appends the
generation prompt (`chat_template.cpp:108`). So rendering the first *k* messages does
**not** produce a token prefix of rendering all of them — it ends in
`<|im_start|>assistant\n<think>\n`. Any implementation that computes the stable boundary by
rendering a message sub-list is wrong, and wrong in the way that produces fluent
wrong text rather than a crash.

New API on `ChatTemplate`:

```cpp
// Renders as render() does, and additionally reports where each message begins.
// offsets.size() == messages.size() + 1; offsets.back() is where the generation
// prompt starts.
std::vector<TokenId> render_with_offsets(const std::vector<Message>& messages,
                                         std::string_view tools_json,
                                         std::vector<std::size_t>& offsets) const;
```

`render()` becomes a one-line wrapper so the two cannot drift.

**Step 2 — `ContextStore` names the boundary.** `render()` already puts live state last and
only when non-empty. Add:

```cpp
// Index of the first message that is NOT part of the stable prefix, i.e. the live-state
// block. Equal to render().size() when there is no live state.
[[nodiscard]] std::size_t stable_message_count() const;
```

**Step 3 — carry it on the task.** `InferenceTask` gains
`std::size_t checkpoint_at = 0` (0 meaning "do not checkpoint"), set by the Agent from
`offsets[ctx_.stable_message_count()]`.

**Step 4 — take the checkpoint during prefill.** The prefill loop chunks at 512
(`mlx_backend.cpp:319-341`); the boundary will usually fall mid-chunk. Split the loop so
the boundary is a chunk edge, then:

```cpp
if (at_boundary) {
    saved_.cp = impl_->model.checkpoint();
    saved_.len = task.checkpoint_at;
    saved_.valid = true;
}
```

One extra chunk edge per turn. Measure the cost; it should be lost in the noise against a
full barrier every 512 tokens anyway.

**Step 5 — the reuse decision becomes a pure function, in `src/model/kv_cache.hpp`:**

```cpp
enum class ReuseMode : std::uint8_t {
    Extend,   // cache is a verified prefix of the prompt; prefill the tail
    Restore,  // roll back to the saved checkpoint, then prefill from there
    Reset,    // nothing usable; full re-prefill
};

struct TurnReuse {
    ReuseMode mode = ReuseMode::Reset;
    std::size_t prefill_from = 0;
};

[[nodiscard]] TurnReuse plan_turn_reuse(const KvCacheLedger& ledger,
                                        const std::vector<TokenId>& prompt,
                                        std::size_t checkpoint_len, bool checkpoint_valid);
```

Rules, in order:

1. Not divergent → `Extend`, `prefill_from = reusable`. (Today's fast path, unchanged.)
2. Divergent, `checkpoint_valid`, `checkpoint_len > 0`, and the first `checkpoint_len` ids
   of the ledger are **verified equal id-by-id** to the prompt's → `Restore`,
   `prefill_from = checkpoint_len`.
3. Otherwise → `Reset`.

Rule 2's comparison is a full `std::mismatch` over `checkpoint_len` ids, not a fingerprint
comparison. `fingerprint_at()` is O(1) and is the right fast reject, but S5.10's rule is
"verified, never assumed" and the ledger header says it in as many words: the hash keys the
lookup, equality is the proof. A few thousand `int32` compares against a 19 GB model is
not a cost worth reasoning about.

Making this a free function over the ledger is what lets the whole decision be tested in
the gate with no GPU — the same argument `src/model/speculative.hpp` makes for keeping the
block algebra model-free, and for the same reason: an off-by-one here does not crash, it
shifts the distribution and the text stays fluent.

**Step 6 — apply it.** In `generate()`:

```cpp
const TurnReuse plan = plan_turn_reuse(ledger_, task.prompt, saved_.len, saved_.valid);
switch (plan.mode) {
    case ReuseMode::Extend:  break;
    case ReuseMode::Restore: impl_->model.restore(saved_.cp);
                             ledger_.truncate_to(saved_.len);
                             break;
    case ReuseMode::Reset:   impl_->model.reset_cache(); ledger_.clear();
                             saved_.valid = false;
                             break;
}
```

**Invalidation is generic, not enumerated.** Compaction, a steering message, a persona
change, a project-memory change and a tools-block change all rewrite the stable prefix.
None of them needs special handling: rule 2's id-by-id comparison fails and the decision
falls through to `Reset`. Resisting the urge to enumerate the invalidating events is the
point — an enumeration is a list someone forgets to extend.

**Memory.** `CacheCheckpoint` holds an `SsmCache::Snapshot` (conv + delta state) for each
of the 30 linear layers. These are sequence-length-independent, but they are MLX arrays and
holding them pins their buffers. Exactly **one** checkpoint is held at a time and it is
overwritten each turn, which bounds it by construction. Measure the resident delta with
`lmp_diag` over a 30-turn run and record the number here; if it is not small, the fallback
is to snapshot only on turns where the stable prefix grew by more than a threshold.

**Interaction with speculative decoding.** `MlxSpecForward` has its own
`checkpoint()`/`restore()` pair with a per-block `mark_`. The two never overlap —
speculation runs strictly inside `generate()`'s decode phase, after the turn checkpoint is
taken — but nothing enforces it. Add an assertion that the turn checkpoint is not taken
while a speculative block is live.

### Tests

Gate, no GPU (`tests/model/test_kv_reuse.cpp`):
- append-only prompt → `Extend`, `prefill_from == cache length`
- mid-prompt divergence with a valid checkpoint at or below the common prefix → `Restore`,
  `prefill_from == checkpoint_len`
- divergence *before* the checkpoint → `Reset`
- `checkpoint_valid == false` → `Reset` regardless
- a prompt shorter than the checkpoint → `Reset`
- ids equal but the checkpoint is stale by one token → `Reset` (the off-by-one probe)

Template (`tests/model/test_chat_template.cpp`, extend the existing golden tests):
- `render_with_offsets` produces ids identical to `render`
- for every k, `ids[0..offsets[k])` equals the concatenation of `append_message` over the
  first k messages — the prefix property, asserted rather than assumed

Real model (`realmodel` label, serial):
- a fixed-seed multi-turn run with reuse on produces **byte-identical output** to the same
  run with reuse forced off. This is the correctness proof; a stale-cache bug is invisible
  in every other kind of assertion.
- `prefill_reused_tokens` (new field on `GenResult`, surfaced on `lmp/perf`) is > 0 from
  turn 2 onward on a run with no compaction, and 0 on the turn immediately after a
  compaction.

### Falsifier

Force `plan_turn_reuse` to return `Restore` with `checkpoint_len` one token too high. The
byte-identity realmodel test must fail. If it passes, the test is not measuring what it
claims and the whole item is unproven.

### Measurement

Add mean and p90 TTFT per turn to `scripts/agent_eval.py`'s report and pin them in
`evals/agent/pins.json` as ceilings (a rise fails, a fall asks to be re-pinned). The
expected result on a ~3k-token context is TTFT on turn ≥ 2 falling from ~1775 ms to
roughly (new tokens ÷ 1500 tok/s), i.e. tens of milliseconds plus decode. **If the measured
drop is small, say so and keep the change only if the byte-identity test justifies the
complexity on its own** — which it does not. This item is worth doing because of a number,
and if the number does not appear the item should be reverted, not defended.

**Manifest.** +1 test in the gate; `realmodel_count` 4 → 5.

---

## G4 — File reads enter the prompt unbounded

**What is wrong.** `src/surface/sidecar.cpp:562-563` sets `max_read_bytes = 4 MiB` and
`max_result_bytes = 8192`. `read_file` returns `ToolResult::okay(std::move(f.bytes))`
(`registry.cpp:249`) and `read_slice` likewise, with no compaction. That summary becomes
`TurnRecord::observation` (`agent.cpp:655`) and is rendered verbatim. Only `shell` passes
through `log_triage::compact` (`registry.cpp:588`).

`src/tools/tool_result.hpp:40-41` documents the summary as "bounded by the caller through
the log-triage compactor before it enters the prompt". For file reads that is not true. One
`read_file` on a large source file can exceed the entire 96,000-token context budget.

`read_file`'s own description already says it "fails honestly with the real size if it
exceeds the limit; use read_slice for a line range instead" — which is currently a lie,
because the limit it fails against is 4 MiB, not the prompt budget.

**Design.**

1. **Split the two limits.** `WorkspaceContext` gains `max_model_read_bytes`, defaulted to
   `max_result_bytes`. `max_read_bytes` (4 MiB) stays for the *internal* reads that
   `replace_in_file`, `append_file` and `delete_file` do — those bytes never enter the
   prompt. `read_file` fails honestly above `max_model_read_bytes`, reporting the real size
   *and the real line count*, and naming `read_slice`. Its description becomes true.

2. **Bound `read_slice` too.** A 5,000-line slice is as bad as a whole file. Stop at
   `max_model_read_bytes` and end with an exact continuation instruction:
   `[budget reached at line 141 of 3000; continue with read_slice(path, 141, ...)]`.
   Head-first, not anchor-ranked: `log_triage` is a *build log* compactor whose scoring
   elides by diagnostic proximity, and running source code through it would reorder and
   drop lines the model asked for by number. Different content, different policy — do not
   reuse it here.

3. **Spool the remainder.** The `shell` path already spools oversized output to
   `.lmp_spool` and references it in `artifacts` (`registry.cpp:597-606`). Do the same for
   an over-budget read, so the full bytes exist on disk and the summary names the path.

4. **Make the class of bug impossible.** `ContextStore::set_observation_budget(n)`, and in
   `add_turn`, assert the observation is within it and clamp in release. Assertions are on
   in every configuration here (`cmake/LmpAssertions.cmake` strips `-DNDEBUG` everywhere),
   so the assert fires in tests and the clamp protects a real run. This is the choke point
   the tool layer's comment already claims to have; adding it means the next tool that
   forgets to compact fails loudly at the door rather than silently in the prompt.

**Tests** (`tests/tools/test_registry.cpp`, `tests/loop/test_context.cpp`):
- `read_file` over the model budget → honest error carrying the true byte and line counts,
  naming `read_slice`, file untouched
- `read_file` under the budget → byte-identical to today
- `read_slice` over the budget → truncated at a line boundary with the continuation hint,
  and the artifact path present
- `add_turn` with an over-budget observation → assertion fires under the check framework's
  `EXPECT_FAILING_CHECKS` harness
- the tool-honesty ratchet still passes: `read_slice` named in `read_file`'s description
  resolves to a registered tool

**Falsifier.** Raise `max_model_read_bytes` to 4 MiB: the first test must go red.

---

## G5 — `read_slice` is not line-numbered

**What is wrong.** `read_slice` appends raw lines (`registry.cpp:277-296`). The model gets
content with no way to say where it is except by counting, while the tool's own arguments
are line numbers and every compiler diagnostic it will see reports line numbers.

**Design.** Prefix each line with its **absolute, 1-based file line number** and a tab.
Absolute, not slice-relative: slice-relative numbering is a trap that silently produces
off-by-`start_line` reasoning, and absolute numbers are what the tool takes as arguments
and what diagnostics report. No column padding — alignment costs tokens and buys nothing.

Apply the same to `read_file` for consistency. This changes what every eval task sees, so
the suite must be re-run and the pins moved deliberately.

**The one real hazard, and why it is safe.** The model may paste line-numbered text into
`replace_in_file`'s `old_text`. `graft`'s tokenizer treats `42` as an identifier token
(`graft_engine.hpp:96-100`), so a numbered `old_text` produces different tokens from the
file and the engine returns **NoMatch** — a refusal with the file untouched, which is the
contract (`graft_engine.hpp:53-58`). The failure mode is a wasted turn, never a corrupted
file. Belt and braces: state it in both tool descriptions.

**Tests:**
- exact format, tab-separated, no padding
- absolute numbering: `read_slice(f, 20, 22)` starts at `20\t`, not `1\t`
- a file whose last line has no trailing newline
- line 1 and EOF boundaries
- `replace_in_file` with a numbered `old_text` returns NoMatch and leaves the file
  byte-identical — the safety property, asserted rather than assumed

---

## G6 — `locate_symbol` is a grep regex, not an index

**What is wrong.** One global regex over every extension, `head -60`, no ranking, no
dedup (`registry.cpp:383-418`). Its own description says "Grep-based, language-agnostic",
so it is honest — but `rename_across_files` is a live eval task and this is its weak link.

**Approach, in this repo's own idiom: build the scoreboard before choosing the engine.**
The founding argument of `bakeoff/` is that ten implementations all reported 0% false
applies and nine of them had them. Picking a symbol locator by intuition would be the same
mistake.

**Step 1 — `bakeoff/symbol_locator/`.** A corpus of `(tree, symbol, expected definition
site)` drawn from the ten eval workspaces plus this repository itself, with a held-out
split written at the same time and never tuned against. Ground truth is the definition
site, established by reading, and the scoring metric is: correct site in the top result,
correct site anywhere in the output, and false-confidence (a wrong site ranked first).

**Step 2 — score the incumbent.** Whatever the current grep gets is the number to beat.
It is not obviously bad and it may not need replacing; that is what the corpus is for.

**Step 3 — the candidates.**
- **(a) Language-aware patterns.** Per-extension definition regexes, ranked
  (`def name(` / `class name` / `fn name` / `func name` / `name =` at low indent above the
  same at high indent above any mention), deduplicated by `(path, line)`, with the count of
  suppressed hits reported. Falls back to today's behaviour on an unknown extension.
- **(b) Tree-sitter.** A real AST. Correct, and 20+ vendored grammars against §2.2's
  "would a competent team building this fresh choose to build it?" test — probably yes for
  two or three languages, clearly not for twenty.
- **(c) LSP `workspace/symbol`.** Correct and free *if* G1 Phase B lands an LSP client
  anyway. Sequenced behind it deliberately.

Recommendation: ship (a), which is a day's work and testable, and let the corpus decide
whether (b) or (c) is ever worth it. Record the decision either way — a candidate rejected
with a number is a result.

**Acceptance.** `rename_across_files` solve rate and turn count in
`scripts/agent_eval.py`, plus the new scoreboard. Both, because the scoreboard measures the
component and the eval measures whether it changed the agent's behaviour, and those are
different questions.

---

## G7 — The four items `docs/PHASES.md` already lists as not done

### G7a — T2 containers (unattended runs)

**What is wrong.** `SandboxTier::T2_Container` refuses rather than downgrading (correctly —
a silent downgrade is exactly the v1 `unsafe_host` default S13 exists to prevent). But
S7.2 requires T2 for unattended runs, so **unattended runs are not available**, and the
eval suite runs at T1 with an attended-shaped policy.

**Design.**
1. **Runtime detection at startup**, not at call time: probe for Apple `container` (macOS
   26) then a Docker-compatible socket. Record which was found in the event log. If none:
   T2 continues to refuse. **Never a downgrade to T1**, under any circumstance — this is
   the invariant the tier numbering exists to protect.
2. **Image.** Pinned by digest, not by tag. The hard part is the toolchain: a container
   whose compiler differs from the host's turns "the build passes" into a claim about a
   different machine. Scope honestly: **ship T2 for the eval fixtures' toolchain first**
   (python3 + a C++ compiler), which is a known, small, pinnable set, and state in the
   docs that T2 for an arbitrary user workspace is not solved. A tier that works for the
   two languages we can test beats a tier that claims to work for all of them.
3. **Mounts and egress.** Workspace bind-mounted read-write at the same path, `TMPDIR`
   inside the jail (the same fix the Seatbelt tests forced), no network namespace.
4. **Limits.** `ExecLimits` maps onto the runtime's cgroup/resource flags; the wall clock
   stays enforced by the harness, because a runtime that ignores a limit must not be the
   only thing enforcing it.

**Tests.** The `tests/tools/test_sandbox.cpp` break-out suite, re-run against T2: write
outside the root, reach the network, spawn beyond `max_processes`, exceed the wall clock.
Plus the two that matter most here: a missing runtime **refuses** (never downgrades), and a
T2 grant cannot be minted from a `RiskHint` (the type-system property in
`sandbox.hpp:56-70` must still hold).

**Falsifier.** Point the runtime probe at a binary that exists but is not a container
runtime. T2 must refuse, and no command may execute on the host.

### G7b — `lmp/edit` through the extension

**What is wrong.** S12.4 wants workspace edits applied through VS Code's edit API so undo,
dirty buffers and diff review work. The sidecar writes files directly; `EditNotification`
exists in the generated protocol and is exempted in `ratchets.json` with that reason.

**Design.** Reuse the approval round-trip machinery rather than inventing a second one —
`sidecar.cpp:280-304` already blocks on a `request_id` and drains replies through
`RunInbox`, which is the same shape.

1. `Registry` gains an optional **edit sink**: `std::function<EditOutcome(path, old_text,
   new_text)>`. When attached, `write_file` / `replace_in_file` / `append_file` /
   `delete_file` route through it instead of `fsx::write_file_atomic`.
2. The sidecar's sink emits `lmp/edit` and blocks for the reply on the run thread.
3. The extension applies a `WorkspaceEdit`, and replies applied / failed with a reason.
4. **No sink attached → write directly, as today.** Unlike approvals, absent-editor here
   means "no UI to route through", not "nobody to ask" — an eval run has no extension and
   must still be able to edit. Deny-by-default would be the wrong analogy and would break
   the suite.
5. `graft` still computes the replacement in the sidecar; the extension applies bytes. The
   matching, ambiguity refusal and indent re-anchoring stay in the engine that was measured.

**Tests:** sink attached and succeeding → the file changes and no direct write occurs; sink
attached and failing → `ToolResult` error, file untouched; no sink → today's behaviour
byte-identical; a sink that never replies → the run's wall clock still fires.

**Ratchets.** Remove the `EditNotification` dead-code exemption in the same commit. The
exemption's own text says to.

### G7c — Package the `.vsix`, stage the sidecar

`extension/scripts/stage-sidecar.js` and `install-local.js` exist; `extension/bin/lmp_sidecar`
is stale. Add a CMake `package` target that builds the sidecar, stages it, runs
`vsce package`, and — the part that makes it a test rather than a script — **asserts the
binary inside the produced `.vsix` hashes equal to the one just built**. A packaging step
that can ship a stale binary is how a fixed bug reappears in the user's editor.

CI: build the vsix on the gate workflow; do not publish.

### G7d — The stale `ApprovalRequestNotification` exemption

`scripts/ratchets.json` exempts `ApprovalRequestNotification` from the dead-code gate
because "the sidecar currently passes a null approver". It does not:
`src/surface/sidecar.cpp:485` calls `agent.set_approver(...)`. `docs/PHASES.md` still
records the old state too.

This is precisely the failure mode this repo built the dormant-gate mechanism to prevent —
an exemption that outlives its reason is a gate that reports green over something nobody is
checking. **Delete the exemption and let the gate speak.** If it stays green, the
notification is genuinely referenced and the exemption was stale. If it goes red, then the
notification is unreferenced and the approval round-trip is not as wired as
`sidecar.cpp:485` suggests — which is a finding worth having. Update `docs/PHASES.md`'s
"Not done" list either way.

---

## Sequencing, with what each step unblocks

```
G0  fixture ──┬─> G2  synthesize-verification    (independent, do it first for the confidence)
              ├─> G4  read bounds ──> G5  line numbers ──> re-pin evals
              ├─> G1  post-edit check ──> (Phase B: LSP client) ──> G6 candidate (c)
              └─> G3  KV reuse  ──> re-pin TTFT
G7d ──────────────> (immediately; it is a deletion)
G7c ──────────────> (independent, small)
G6  corpus ───────> G6 engine
G7a, G7b ─────────> last; both are surface-area projects
```

Suggested branch-per-item, each landing green, in that order. G4 and G5 land together
because they change the same observations and there is no point re-pinning the eval suite
twice.

## Risk register

| Risk | Item | Mitigation |
|---|---|---|
| Stale KV → fluent wrong output | G3 | id-by-id verification in rule 2; fixed-seed byte-identity realmodel test as the primary proof |
| Boundary computed by re-rendering a message prefix | G3 | `render_with_offsets`; the prefix property asserted in a template test |
| False diagnostics send the run chasing phantoms | G1 | first-touch pre-check; `needs_compile_db`; silence when no contract matches |
| Post-edit check quietly becomes completion evidence | G1 | `SyntaxChecker` has no access to `Verifier`; a test asserts the ledger is unchanged |
| Line numbers pasted into `old_text` | G5 | graft returns NoMatch (file untouched); asserted by test |
| Truncation hides what the model needed | G4 | head-first, explicit continuation hint, full bytes spooled as an artifact |
| T2 silently downgrades to T1 | G7a | refusal is the only alternative to execution; tested against a bogus runtime |
| Eval drift from three behaviour changes at once | all | one item per branch; suite re-run per landing; pins are floors |

## Definition of done, per item

1. Gate green, and the manifest's count **and** names bumped in the same commit.
2. `./scripts/run_ratchets.py --root .` 6/6, and `--self-test` still able to make each fire.
3. The item's falsifier executed and observed red, then restored to green. A green that has
   not been shown capable of red does not count — the same standard the agent is held to.
4. `python3 scripts/agent_eval.py run` at or above pins, or pins moved deliberately with
   the reason recorded.
5. `docs/PHASES.md` updated where it currently states the old behaviour.
