# Handoff: LM_Pipe v2 vs LM Studio, second pass

Paste this whole file as the opening prompt of a fresh session in
`/Users/dev/Desktop/seans_projects_local/LM_Pipe_2` (branch `main`).

Everything below was re-measured in this repo on 2026-07-30. No number here is quoted
from the previous handoff without being reproduced first (S19.6).

---

## Baselines, re-derived

`scripts/lmstudio_baseline.py` rebuilds LM Studio's numbers from `~/.lmstudio/server-logs`
(34 files, real prior usage of this exact checkpoint). Timestamps are 1-second
resolution, so it reports two views: all windows, and windows >= 5 s where the clock
error is small.

```
all windows:      prefill n=1963 median 1145.0   decode n=134 median 78.5
windows >= 5s:    prefill n=740  median 1338.9   decode n=17  median 87.7
```

The previous handoff's 78.5 / 1347 reproduce. **Exit criterion stays decode > 78.5,
prefill > 1347.**

## Where we are

Matched before/after, same prompt and seed, `test_realmodel`:

| | before | after | LM Studio |
|---|---|---|---|
| decode | 22.7 tok/s | **28.4 tok/s** | 78.5 |
| prefill | 45.8 tok/s | **119.1 tok/s** | 1347 |

`lmp_diag bench 4 512 200` (547-token prompt, 200 new tokens) — a new instrument, so
there is no pre-change measurement at this shape:

```
prefill  n=4 median 591.5 tok/s  min 520.7 max 592.6    [0.44x LM Studio]
decode   n=4 median  27.9 tok/s  min  27.8 max  28.0    [0.35x LM Studio]
```

Not there. Two of the three named causes are closed; the third turned out not to be a
cause at all, and the real one is now located precisely.

---

## Closed

### 1. The grammar mask — 22.8 ms/token to 0.00

`Sampler::sample` took `std::function<bool(TokenId)>` and called it once per vocabulary
id. `TurnGrammar::permitted` is now the *definition* of the mask, not the hot path; the
sampler consults `TurnGrammar::mask()`, which returns a `TokenMask` bitset
(`src/model/token_mask.hpp`).

Outside a tool call the legal set is "everything except a handful of structural ids", so
it is a bitset cached per (phase, saw_tool_call) — no vocabulary walk at all. Inside one
it is parsephony's `TokenMaskT<ToolCallGuard>`, finally wired up.

`lmp_diag mask`:

```
permitted() over full vocab :    28.29 ms/token
bulk mask, warm             :     0.000 ms/token
sampler, no mask            :     0.18 ms/token   (was 1.7 -- top_p no longer
sampler + mask (as shipped) :     0.17 ms/token    std::sort's 248k indices)
```

Two correctness results fell out of it, both pinned by tests:

- **parsephony's free-text mask was more permissive than its own automaton.**
  `classify()` only simulated tokens containing the parameter terminator, but
  `ToolCallGuard` also rejects control bytes in a raw text value — 93 vocabulary entries
  the fast mask allowed and the grammar denies. Fixed in
  `third_party/parsephony/include/parsephony/mask.hpp`. Found by the new test
  `the_bulk_mask_and_the_predicate_agree_over_the_whole_vocabulary`, which compares
  `mask()` against `permitted()` for every id at every state of a real tool call. **Keep
  that test.** It is the only thing standing between a 22 ms saving and a mask that lies
  to the sampler.
- **Ids past the vocabulary are no longer emittable.** The logits row is 248,320 wide and
  the tokenizer has 248,077 entries; the old predicate permitted the difference.

### 2. Prefill's op-per-timestep scan — 20-25x

`gated_delta_update` ran one MLX op-set per token position: ~8,600 sequential launches
for a 287-token prompt.

The fix was not a chunked associative scan. **mlx-lm already ships a fused Metal kernel
for this** (`mlx_lm/models/gated_delta.py`), and LM Studio runs it — our C++ had ported
mlx-lm's *reference* loop, the one it labels `gated_delta_ops`.
`gated_delta_update_kernel` is that kernel: the whole T-step recurrence in one launch,
state held in registers.

Same arithmetic in the same order, so the deviation is fp32 association only.
`lmp_diag scan`:

```
     T         ops ms      kernel ms   speedup   rel|dy|   rel|dstate|
     1            1.4           0.73      1.9x   1.9e-07     0.0e+00
   287           27.2           1.15     23.6x   1.1e-07     9.4e-08
  1024           79.1           3.12     25.3x   1.5e-07     9.4e-08
```

The reference loop is kept as `gated_delta_update_ops` — it is the definition the kernel
is tested against. If you touch the kernel, `lmp_diag scan` is the check.

One note on that diagnostic: feed it raw normals for `k` and *both* implementations
diverge, because `(I - beta k k^T)` is only contractive once `k` is rms-normed and scaled
by `1/sqrt(Dk)` the way `forward_gated_delta` does. The first version of the check got
that wrong and reported 1e16 deviations that were entirely its own inputs.

### 3. The logits copy — measured, then left alone

0.07 ms/token (`copy=14ms` over 200 tokens). It was the smallest of the three and it is
not worth touching. Closed by measurement rather than by work.

---

## The remaining gap is one block

`lmp_diag layers 1` runs the model's own block functions, chained the way a step chains
them:

```
  linear layer (delta+moe)   1.139 ms/layer  x30 =  34.17 ms
  attn layer (attn+moe)      0.791 ms/layer  x10 =   7.91 ms
    of which moe             0.674 ms/layer  x40 =  26.98 ms
```

**The MoE block is 27 ms/token of a 35 ms decode step.** And `lmp_diag blocks 1` says its
actual GPU work is 0.085 ms/layer:

```
  moe gate                   latency  0.343  cpu  0.000  marginal  0.020 ms
  moe switch_glu (8/256)     latency  0.232  cpu  0.003  marginal  0.048 ms
  moe shared expert          latency  0.172  cpu  0.001  marginal  0.019 ms
```

So ~0.59 ms per MoE block is neither compute nor CPU graph construction. It is the cost
of ~27 MLX ops that cannot overlap. `lmp_diag chain` prices that directly:

```
  dependent chain   :  5.22 us/op        independent :  4.08 us/op
```

Three things were tried against it and **all three did nothing**, which is worth knowing
before trying them again:

- **`mx::compile` on the elementwise clusters** (silu, swiglu, precise_rms_norm_gated,
  compute_g). In isolation it is real — compute_g goes 18.4 us -> 3.3 us per call. End to
  end: 28.1 -> 27.9 tok/s, inside noise. Reverted; the comment on `compute_g` records why.
- **`MLX_MAX_OPS_PER_BUFFER`** (50 / 200 / 500). Moves the `chain` micro-benchmark 1.9x,
  moves real decode by 0.1 tok/s. MLX already batches inside our single `mx::eval`.
- **CPU-side graph construction** as a suspect: measured at under 1 ms per step. It is not
  the problem, so `mx::async_eval` has nothing to hide and was not pursued.

### What to try next

The cost is in the MoE's *heavy* ops, not its elementwise ones — that is what the failed
fusion experiment established. Two leads, cheapest first:

1. **MLX version.** LM Studio ships **MLX 0.31.2**; we link **0.29.3** (the pip package
   `src/model/CMakeLists.txt` probes). Its engine source is on this machine, and its
   `SparseMoeBlock` and `SwitchGLU` are structurally identical to ours:
   `~/.lmstudio/extensions/backends/vendor/_amphibian/`
   `app-mlx-generate-mac14-arm64@29/lib/python3.11/site-packages/mlx_lm/models/`.
   Same op graph, 2.8x the throughput, newer MLX. Upgrading and re-running
   `lmp_diag layers 1` is a one-line experiment and should be the first one.

2. **Time forward_moe's pieces chained, not batched.** `lmp_diag blocks` batches them,
   which is exactly what hides this; that measurement is the one this session did not get
   to. Build it incrementally — the shared-expert path is naturally chainable through
   [1,1,2048] — and diff. Prime suspects: the three `gather_qmm` calls, then
   `argpartition` + `take_along_axis` over 256 experts. If one op carries most of the
   0.59 ms, that is a different fix from "too many ops".

Prefill is the same story: the scan is fixed, so prefill is now MoE-bound too.

---

## Instruments

`tests/model/diag_main.cpp` (`cmake --build --preset dev --target lmp_diag`):

| | |
|---|---|
| `lmp_diag scan [T...]` | fused kernel vs reference loop: deviation and wall time |
| `lmp_diag mask` | one decode step outside the forward pass |
| `lmp_diag blocks [T]` | per-block cost, batched — latency / cpu / marginal |
| `lmp_diag layers [T]` | per-layer cost, chained, via the model's own blocks |
| `lmp_diag chain [n]` | what a dependent MLX op costs on this machine |
| `lmp_diag bench [runs] [prompt] [max_new]` | N-run ledger against the LM Studio numbers |

`scripts/lmstudio_baseline.py` re-derives the other side.

`Qwen35MoeModel::forward_{linear_layer,full_attn_layer,moe}` are public so `layers` times
the code the model actually runs rather than a copy of it.

`GenResult` now carries `forward_ms`, `logits_copy_ms`, `sample_ms`, so "decode is slow"
is answerable with *where* without reaching for a profiler.

## Guardrails — all green as of this handoff

`ctest -L gate` 19/19 · `./scripts/run_ratchets.py --root .` 6/6 clean ·
`ctest -L realmodel` 2/2 · `./scripts/eval.py --root . score` unmoved
(corpus wmiss=0 179/179 pinned, holdout wmiss=15 34/42 pinned).

## Do not

- Do not delete `gated_delta_update_ops`, or the equivalence tests in
  `tests/model/test_grammar.cpp`. They are what make the fast paths falsifiable.
- Do not re-apply `mx::compile` on the strength of a micro-benchmark. It was measured end
  to end and it was zero.
- Do not weaken the grammar to make the mask cheap. Speed came from *how* the mask is
  computed; the tool-call automaton constrains exactly as much as it did before.
- Do not reach for speculative decoding. The MoE block is 27 ms of a 35 ms step; stacking
  a throughput trick on top would hide it, not fix it.
- Do not re-litigate settled decisions: Apple Silicon only, MLX in-process, Qwen3 only,
  XML tool-call syntax (see `docs/PHASES.md`).
