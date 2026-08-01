# Handoff: speculative decoding, the model-layer half

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
- **The ratchets will catch you.** Adding a test requires editing `gate_manifest.txt` twice
  (count AND name) by design. `agent.cpp` is near the 800-line size ratchet; new loop code
  should go in its own file.
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
