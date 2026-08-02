# Handoff: speculative decoding, the model-layer half

> **Update, 2026-08-01 (later session). ALL FOUR STEPS ARE DONE.** Speculative decoding is
> wired, correct, gate-tested without a GPU, and measured end-to-end on the real agent loop:
> **+4.5% to +11% decode on turns where it fires, ~+3.8% overall, 84% acceptance.** It is
> OFF by default (`LMP_SPECULATIVE=1`, or `MlxBackendConfig::speculative`). The binding
> constraint turned out to be the TRIGGER rate, not the acceptance rate -- see
> [Findings](#findings-from-building-steps-1-and-2). Two
> measurements taken while doing it change the design and are recorded in
> [Findings](#findings-from-building-steps-1-and-2) at the end of this document — read them
> before writing step 4. The short version:
>
> - **Pass `draft_probs[i] = 1.0` to the verifier.** With a deterministic proposer the
>   brief's residual reduction is exactly distribution-preserving; with anything else it is
>   not. Proven by exact enumeration, not sampling.
> - **The batched verification pass is ~4% TV from sequential decode** on this bf16
>   checkpoint. Speculation is distribution-preserving with respect to the rows it is
>   handed, and those rows are not bit-identical to the ones one-at-a-time decoding would
>   produce. Not a defect; a property to state rather than discover.

Written 2026-08-01, end of the session that landed `7cb3139` on
`perf/streaming-and-parallel-dispatch`. Read
[PLAN_PARALLELISM.md](PLAN_PARALLELISM.md) and
[MOE_ROUTING_FINDINGS.md](MOE_ROUTING_FINDINGS.md) first; this says what to do next and what
is arriving from elsewhere.

---

## Where things stand

Landed and green (21/21 gate, 6/6 ratchets):

- **Token streaming.** `src/loop/token_stream.cpp`. 51 notifications for one turn against
  the old 2, verified on the real sidecar.
- **Parallel read-only tool dispatch.** `src/loop/parallel_calls.cpp`, gated on
  `blast_radius` capability flags, results recollected in call order.
- **SPSC cached index.** `src/platform/spsc_channel.hpp`, TSan clean.
- **Two cook-off amalgamations** under `bakeoff/`: `SuffixProposer` (the drafter) and
  `moetrace` (the analyser). Both beat every entrant on a shared scoreboard.
- **MoE routing instrumentation and its first real measurement.**
  `src/model/mlx/moe_trace.hpp`, 16,914 decode steps.

Not done: speculative decoding itself. That is this handoff.

---

## The work, in order

### 0. Read this before estimating: the rollback blocker is SMALL

`PLAN_PARALLELISM.md` lists "the KV cache cannot roll back" as a structural blocker and
implies major surgery. **That estimate was wrong, and the correction is the most useful thing
in this document.** Having read the cache layout properly:

- **Attention layers** (`KVCache`, `src/model/mlx/kv_cache.hpp:18`) hold an OVER-ALLOCATED
  buffer written in place by `slice_update`, with `int offset` as the true token count.
  `update_and_fetch` returns `slice(keys, ..., offset, ...)`. Nothing ever reads past
  `offset`. **Rollback is therefore `offset = n`** — an integer assignment, roughly three
  lines including a bounds clamp. Stale data past the new offset is overwritten by the next
  append and never read before then.

- **Linear / gated-delta layers** (`SsmCache`, same file, line 79) are the real subtlety and
  are still tractable. `conv_state` and `delta_state` are **fixed-size tensors that do not
  grow with sequence length** — they are a running recurrence, so they cannot be rolled back
  by moving an index. But because they are fixed size, **snapshot and restore is cheap and
  bounded**: copy both tensors per linear layer before a speculative block, restore on
  rejection. Cost is independent of draft length and of context length.

So the shape of the job is: one integer per attention layer, two small tensor copies per
linear layer. That is a session, not a project. Qwen 3.6 is a hybrid — do not assume all 40
layers are the same kind; `cfg_.is_linear_layer(layer)` is the discriminator.

### 1. All-position logits (`src/model/mlx/qwen35_moe_model.hpp`)

`forward_logits` slices the hidden state to the final position and says so in a comment:
*"only the final position's logits are ever consumed downstream."* Verification needs a row
per drafted position.

Add an opt-in path rather than changing the default. The decode path is tuned and the
existing contract is load-bearing — `HANDOFF_PERF.md` records that folding the float32 cast
into this graph measured *worse* (84.8 -> 83.9 tok/s). Suggested shape: a second entry point
(`forward_logits_all`) that skips the slice, leaving `forward_logits` byte-identical.
Re-run `lmp_diag bench` and confirm decode tok/s is unmoved before and after.

### 2. Cache rollback

Per section 0. Add `truncate_to(int n)` to both cache types and a matching
`Qwen35MoeModel::rollback_to(int n)` that walks all layers. The test that matters is
equivalence by construction: process 100 tokens then roll back to 70, versus process only
the first 70 — the caches must produce identical logits for the next token. Assert on the
logits, not on the cache internals.

### 3. Ledger truncation

`KvLedger` (`src/model/kv_cache.hpp`) offers `append` and `clear` and nothing between.
Needs `truncate_to(n)` with the content hash staying consistent. **Jules Brief D is exactly
this problem** — see below; prefer wiring their result to writing it again.

### 4. Wire it together

`MlxBackend::generate` gains a speculative path: propose from `SuffixProposer`, forward all
draft positions, verify, commit the accepted prefix, roll back the rest. The grammar must
advance over the committed tokens only, and the mask for the next step comes from the
grammar's post-commit state.

**Two constraints from the measurements, both non-negotiable:**

- **Draft length must be adaptive.** Fixed-length drafting is a NET LOSS on this model:
  0.678x at k=4, 0.454x at k=8. A fixed `k` ships a regression.
- **Do not build expert-aware drafting.** Tested three ways, does not pay; the best-looking
  variant turned out identical to fixed-length-1. Section 3 of the findings.

---

## Incoming from Jules — do not build these

Three cook-offs launched 2026-08-01, 5 entrants each, on `cat-collector-king`. All are pure
C++20 with no GPU and no platform syscalls, so unlike earlier rounds they should be real,
buildable code. Brief text is in the session scratchpad as `JULES_BRIEFS_ROUND2.md`.

| brief | what it is | where it lands |
|---|---|---|
| **C — `SpecVerifier`** | The acceptance rule: which drafted tokens to keep so the output distribution stays EXACTLY what ordinary sampling would give. Rejection-sampling maths, easy to get subtly wrong. | Step 4 above. This is the piece that makes speculation correct rather than merely fast. |
| **D — `PrefixLedger`** | Token-sequence ledger with prefix matching, `truncate_last`, and a fingerprint. | Step 3 above, directly. |
| **E — `MarkdownStream`** | Incremental markdown/code-fence state machine so streamed text renders live without half-open fences. | The webview, independent of speculative decoding. |

**How to review them.** Do not read five implementations and form an opinion — build a
neutral scoreboard, compile every entrant against it, and score. That is what separated the
winner from the pack in round 1, and it caught a `moetrace` entrant whose Gini disagreed
with the other five. `bakeoff/draft_proposer/scoreboard.cpp` and
`bakeoff/moetrace/scoreboard.cpp` are the pattern.

For Brief C specifically, the decisive test is statistical: histogram the committed tokens
over many runs with a deliberately bad drafter and check convergence to the target
distribution. A naive implementation passes everything else and fails only that.

---

## Gotchas earned today

- **Suspect your harness before the subjects.** Twice a measurement said every entrant was
  broken and the harness was wrong both times: once the training and held-out corpora were
  built from independently drawn phrase sets (nothing to match, all five correctly proposed
  nothing), once a shell-quoting error made a build "fail". When N independent
  implementations agree, they are probably right.
- **The obvious UTF-8 test is the wrong test.** Byte-level BPE fragments concatenate back
  correctly, so "does the total match" passes for an implementation that does no buffering.
  Assert per-message validity.
- **A traced run's throughput is meaningless.** `LMP_MOE_TRACE` forces a mid-graph sync;
  52 tok/s traced against 85 untraced. Routing is unaffected. Never quote a traced tok/s.
- **The gates will catch you.** Adding a test requires editing `gate_manifest.txt` twice
  (count AND name) by design. (The 800-line size gate this used to also warn about was
  removed on 2026-08-02 — put new loop code wherever it belongs, not wherever it fits.)
- **Never two MLX processes.** One model is 19 GB on a 48 GB host. `ctest --preset
  realmodel` pins jobs=1; `scripts/drive.py` must run alone.

---

## How to verify anything here

```
cmake --build --preset dev -j8 && ctest --preset gate && python3 scripts/run_ratchets.py --root .
```

Then the real thing, which is what actually counts:

```
python3 scripts/drive.py --workspace /tmp/ws --mission "..." --auto
```

Re-measure speculative speedup against the real `SuffixProposer` rather than the trigram
stand-in used for the findings doc — the existing trace at
`scratchpad/qwen_routing_big.jsonl` can be replayed without another model run, and the
numbers in section 4 of the findings should improve.

---

## Findings from building steps 1 and 2

Written 2026-08-01, the session after the one above. What landed:

- **`Qwen35MoeModel::forward_logits_all`** (`src/model/mlx/qwen35_moe_model.hpp`). Second
  entry point, `forward_logits` untouched. Both now share a private `forward_hidden` so the
  layer stack cannot drift between them; the only divergence is the final-position slice.
  Decode measured before and after by stashing the change: **89.0 -> 89.1 tok/s** median
  (`lmp_diag bench 3 512 256`), **91.4 -> 91.7 tok/s** (`lmp_diag step 60`). Unmoved.
- **Cache rollback** (`src/model/mlx/kv_cache.hpp`). `KVCache::truncate_to(n)` is the
  integer assignment the correction predicted. `SsmCache::snapshot()/restore()` is cheaper
  than predicted: `mx::array` is an immutable refcounted handle and `forward_gated_delta`
  *replaces* the state rather than writing through it, so a snapshot is two refcount bumps,
  not two tensor copies.
- **`Qwen35MoeModel::checkpoint()/restore()`**, and NOT the `rollback_to(int n)` the plan
  sketched. A bare `rollback_to(n)` is not implementable for the 30 linear layers — their
  recurrence has no per-token history, so only positions snapshotted in advance are
  reachable. An API promising otherwise would be a lie whose only symptom is output drift.
- **`tests/model/test_spec_cache.cpp`**, 4 cases, labelled `realmodel`
  (`realmodel_count` 3 -> 4). All green.

### 1. Rollback is exact, including across a buffer growth boundary

Process 200 tokens, checkpoint at 140, process 60 more, roll back, decode: **bit-identical**
to a cache that only ever saw 140. Repeated across `KVCache::kStep` (256) — where the
rolled-back run holds a 512-token buffer and the fresh run holds 256 — also **0.00e+00**.
That second one was written expecting to need a tolerance and did not.

### 2. The batched pass is ~4% TV from sequential decode, and that is inherent

Forwarding k tokens in one pass and reading row i does NOT reproduce the logits of decoding
those k tokens one at a time. Measured at k=8, both cold and at cache offset 64:

| | max abs logit diff | max TV | top-1 agreement |
|---|---|---|---|
| offset 0 | 0.75 | 0.059 | 16/16 |
| offset 64 | 0.63 | 0.041 | 16/16 |

**This is numerics, not a mask bug**, and the evidence is specific: at offset 0 the FIRST
row — the one case where the batched and sequential paths are the same computation on the
same shapes — agrees at exactly `0.00e+00`, and the error across the remaining rows is flat
rather than growing with position. A misaligned causal mask fails the first row and
compounds; this does neither.

The consequence is worth stating plainly, because Brief C's whole premise is exactness: the
acceptance rule preserves the distribution of **the rows it is given**, and those rows sit
~4% TV from the ones ordinary decoding would have produced. Speculation here is
distribution-preserving up to batched-forward numerics, not absolutely. Every implementation
of this technique on a quantized model has this property; state it, do not rediscover it.

### 3. Brief C's prescribed residual is only exact for a DETERMINISTIC drafter

Found by exact enumeration over 20,000 random `(p, q)` pairs — no sampling. The brief tells
the implementer to subtract the scalar `q` from the rejected token's mass, clamp, and
renormalise. That is **not** distribution-preserving in general (worst TV **0.23**, against
5.6e-17 for the textbook full-row residual). It becomes exact precisely when
`q(proposed) = 1`.

**So pass `draft_probs[i] = 1.0f`.** `SuffixProposer` proposes a concrete continuation from
matched history; it is not a sampling model, and it has no calibrated probability to offer.
Acceptance then reduces to `u < p(t)` — keep the drafted token with the target's own
probability for it — and the procedure is exact. Feeding a confidence score into that slot
instead buys measurable bias in exchange for nothing. Full table in
[bakeoff/spec_verifier/README.md](../bakeoff/spec_verifier/README.md).

### 4. Cook-off status

| brief | entrants | verdict |
|---|---|---|
| **C — SpecVerifier** | **5 / 5** | All five correct and at the sampling-noise floor. Scored, falsified, ready to adopt; `e3` is the most compact. `bakeoff/spec_verifier/` |
| **D — PrefixLedger** | **3 / 5** | All three correct — and all three share a fingerprint that **collides at 1024 tokens** by Thue-Morse construction, which their tests and the first version of mine both missed. `bakeoff/prefix_ledger/amalgam/` fixes it and makes bulk truncation O(1) (0.02 us against 49-56 us). Adopt the amalgam; re-run when 4 and 5 land. |
| E — MarkdownStream | 0 / 5 | Nothing landed yet. |

Both scoreboards ship with `falsifiers/` and build them on every run. A scoreboard on which
every entrant passes is not evidence until it has been shown red — and on Brief C that
exercise produced a correction to the brief itself:

**The bad-drafter histogram does not catch the bug the brief says it exists to catch.** The
`naive_argmax` falsifier scores `tv_det_bad = 0.0022` (clean) and `tv_det_good = 0.8557`
(catastrophic), while passing floor, determinism, perfect-drafter and degenerate tests. A
drafter proposing the least likely token is never the argmax, so the naive rule never
accepts and falls through to sampling the target row — which converges perfectly. Any
reissue of Brief C should put the GOOD-drafter histogram first.

### 6. Step 4, as built and measured

`src/model/speculative.hpp` / `.cpp`, plus `spec_verifier.{hpp,cpp}` (Brief C entrant e3,
adopted) and the ledger above. `MlxBackend::generate` branches to `decode_speculative` --
two loops, not one with an `if` per step, so the tuned plain path runs the code it was
tuned in.

**Model-free by construction.** Everything MLX-shaped is behind `SpecForward`, so the block
algebra is a GATE test (`tests/model/test_speculative.cpp`, 5 cases, no GPU). That matters
because a wrong speculative decoder does not crash: it commits a token from the wrong
position and the text stays fluent. The headline case drives the loop with a deterministic
scripted model, where the correct output is known exactly, and asserts token-for-token
equality over 200 tokens.

**Measured, `scripts/drive.py`, same mission with and without:**

| turn | blocks | acceptance | decode ON | decode OFF |
|---|---|---|---|---|
| 1-2 | 0 | -- | 83.4, 81.8 | 81.5, 82.1 |
| 3 | 1 | 100% | 83.7 | 81.0 |
| 4 | 13 | 91% | **90.8** | 81.8 |
| 5 | 6 | 100% | **87.5** | 82.4 |
| 6 | 8 | 67% | **86.8** | 83.1 |

Two honest caveats. These are separate runs, so the agent takes slightly different paths and
this is indicative rather than a controlled A/B -- but the correlation between block count
and speedup is monotone and in the right direction. And `lmp_diag bench`, which generates
novel prose from a synthetic prompt, shows **no change at all** (89.0 against 88.8): it
speculated 6 times in 238 steps.

**THE BINDING CONSTRAINT IS THE TRIGGER RATE, NOT ACCEPTANCE.** Across the agent run: 28
blocks against 1269 fallbacks, but 84% of drafted tokens accepted -- far above the ~65%
break-even the routing table implies. When it fires it is very good; it just rarely fires.
The lever is `SuffixProposer`'s `min_support = 2` and `min_match_len = 3`, which are
deliberately conservative and were tuned on synthetic data. Sweeping those on real agent
traces is the highest-value next move by a distance, and needs no new machinery.

### 7. What is left

- Sweep `min_support` / `min_match_len` / `draft_cost_ratio` on real agent traces (above).
- Fold the partial-acceptance re-forward into the next block as a pending prefix. Currently
  a partial block pays one extra pass of m+1 positions; 14 of 50 blocks in the gate test hit
  that path.
- Re-score `bakeoff/prefix_ledger/` when Brief D's PRs 4 and 5 land, in case they bring an
  idea worth folding into the amalgamation.

### 5. Earlier note: what step 4 needed

Steps 1 and 2 are the whole model-layer half and they are done. **Step 4 is no longer
blocked**: `bakeoff/prefix_ledger/amalgam/` is the ledger, and any Brief C entrant is the
verifier. Two integration constraints and one design note:

- **Verify against the POST-SAMPLER distribution, not raw softmax.** Every Brief C entrant
  samples from the rows it is handed, and the brief told them those rows are plain
  probabilities. `MlxBackend::generate` never samples from a plain row — `Sampler::sample`
  applies repetition penalty, the grammar mask, temperature, top-k, top-p and min-p first. If
  verification uses raw softmax the committed tokens follow neither law, and worse,
  speculation can commit a token the grammar forbids and emit malformed tool-call JSON. Build
  each target row through the same `Sampler` transform before verifying.
- **Filter drafts through the grammar mask.** A masked-out proposal has p = 0, is rejected
  with certainty, and wastes a draft slot.

One design note earned while sketching the loop:

- The gated-delta layers cannot roll back to an arbitrary position, so on PARTIAL acceptance
  the accepted tokens have to be re-forwarded. Do not pay a separate pass for that: carry
  them as a pending suffix and prepend them to the next block's forward, which fuses the
  re-forward into a pass that was happening anyway. Cap the pending length so a run of
  partial acceptances cannot grow it without bound. On FULL acceptance the end-of-batch
  state is already correct and no rollback happens at all.
- Because the pending suffix is re-forwarded, the ledger only ever needs to record
  *committed* tokens. That makes truncation a safety net rather than a load-bearing
  requirement — worth knowing if Brief D stalls, but not a reason to hand-roll it.
