# Handoff: LM_Pipe v2 vs LM Studio, fourth pass

Paste this whole file as the opening prompt of a fresh session in
`/Users/dev/Desktop/seans_projects_local/LM_Pipe_2` (branch `perf/mask-and-scan`).

Every number here was measured in this repo on 2026-07-31. Nothing is quoted from a
previous handoff without being reproduced first (S19.6) — and this pass had to retract
several things the third pass asserted, so read the retractions before trusting anything
you remember.

**Decode now passes.** 28.6 -> 84.8 tok/s against an exit criterion of 78.5. Prefill went
586.7 -> 1110.7 against a criterion of 1347, and is the only thing still open.

---

## Build note, read this first

The third pass claimed "MLX is now 0.31.2, matching LM Studio exactly." **That was
false, and the instrument that reported it was the reason.**

`find_package(MLX CONFIG PATHS ... NO_DEFAULT_PATH)` resolves through the cached
`MLX_DIR` and ignores `PATHS` once that cache entry exists. Passing
`-DLMP_MLX_PYTHON=...` to an existing build directory therefore kept linking the
*previous* interpreter's MLX. The `message(STATUS ...)` line asked the **interpreter**
for `mx.__version__` rather than the package `find_package` had actually found, so it
printed 0.31.2 while the link line, the rpath and `MLX_LIBRARY` all still pointed at
0.29.3 in `~/Library/Python/3.9`.

Fixed three ways in `src/model/CMakeLists.txt`: `MLX_DIR` is re-pointed when the probe
disagrees with it, the stale `MLX_LIBRARY` is unset alongside it, and the status line
now prints `MLX_VERSION` from the found package. Configure with:

```bash
cmake --preset dev -DLMP_MLX_PYTHON=$HOME/.venvs/lmp-mlx/bin/python
```

Building against real 0.31.2 then needed one source change: `scaled_dot_product_attention`
changed its mask parameter from `const std::vector<array>&` to `std::optional<array>`.
The single call site now says "no mask" via `mask_mode` instead of naming that argument,
so it compiles on both.

**Verify with the binary, not the configure log:**

```bash
otool -l build/tests/model/lmp_diag | grep -A2 LC_RPATH | grep path
```

---

## Baselines

`scripts/lmstudio_baseline.py` rebuilds LM Studio's numbers from `~/.lmstudio/server-logs`:

```
all windows:      prefill n=1963 median 1145.0   decode n=134 median 78.5
windows >= 5s:    prefill n=740  median 1338.9   decode n=17  median 87.7
```

**Exit criterion: decode > 78.5, prefill > 1347.**

New this pass, and much more useful than the log-derived numbers: **LM Studio's own
stack runs directly on this machine.**

```bash
~/.lmstudio/extensions/backends/vendor/_amphibian/app-mlx-generate-mac14-arm64@29/bin/python
```

That interpreter has mlx 0.31.2 + mlx_lm 0.31.3 — exactly what LM Studio ships. On the
same checkpoint, same machine, 522-token prompt, 200 new tokens:

```
mlx_lm reference:   prefill 728.5 tok/s     decode 103.1 tok/s
```

Use it. An apples-to-apples reference you can instrument and ablate in Python, with no
rebuild, is worth more than any log-derived median.

## Where we are

`lmp_diag bench 4 512 200`, 547-token prompt:

| | third pass (really 0.29.3) | real 0.31.2 | + wired limit | + dtype fix | + kernel dtypes |
|---|---|---|---|---|---|
| decode  | 27.6 | 27.5 | 28.6 | 84.8 | **85.4** |
| prefill | 586.7 | 856.4 | 871.0 | 1110.7 | **1126.5** |

Decode **PASS** at 1.09x. Prefill **FAIL** at 0.84x — but note we are already 1.55x the
Python reference's prefill (728.5), so 1347 may not be an apples-to-apples target; see
"Still open".

**The honest apples-to-apples decode comparison is 85.4 vs 80.9, not 85.4 vs 103.1.**
The reference reaches 103 by running one step ahead with `mx.async_eval`, which a
host-side sampler cannot do (see "Why async_eval is not available"). Constrain the
reference the way our loop is constrained — synchronous eval, full logits to host, CPU
sampling — and it does **80.9 tok/s** on this machine. Our forward pass is already
faster than mlx-lm's; what is left is a loop-structure difference, not a kernel one.

---

## The cause: one strongly-typed scalar

`forward_gated_delta` scaled the rms-normed q and k like this:

```cpp
const mx::array inv2 = mx::array(inv_scale * inv_scale);   // float32 array
q = mx::multiply(inv2, mx::fast::rms_norm(q, std::nullopt, 1e-6f));
```

mlx-lm writes `(inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)`. A **Python float is
weakly typed** and leaves q in bf16. `mx::array(float)` is a **strongly typed float32
array** and promotes it.

That promotion does not stay local. q,k go float32 into the delta kernel; y comes back
float32; `precise_rms_norm_gated` returns `hidden.dtype()`, so float32; `out_proj` emits
float32; and `h + attn_out` makes **the residual stream float32 from layer 1 to the end
of the model**. Every quantized matmul downstream — all 40 MoE blocks included — then ran
its float32 activation path against bf16 weights.

Confirmed directly before the fix (`LMP_DEBUG_DTYPE`, since removed):

```
[layer 0] residual in = bf16
[delta]   inputs=bf16 qkv=bf16 conv_out=bf16 q=f32 k=f32 v=bf16 z=bf16
[layer 1] residual in = f32
[layer 2] residual in = f32
```

The size of the effect was measured independently on mlx-lm's own `SparseMoeBlock`, 40
chained layers: **12.17 ms in bf16, 28.03 ms in fp16** (and fp32 is worse). That is the
whole gap.

The fix is `mx::astype(mx::array(...), qkv.dtype())`. Two related alignments with mlx-lm
went in beside it and are worth keeping even though neither moved the clock on its own:
`gated_delta_update` now returns `y` in `q.dtype()` with only the state in float32
(mlx-lm's `output_dtypes=[input_type, state_type]`), and the router softmax passes
`precise=true`.

The same dtype confusion had a second instance in the fused kernel itself, worth +0.6
tok/s decode and +16 prefill on its own. `gated_delta_update_kernel` cast q,k,v,g,beta to
float32 before the launch, reasoning that the recurrence runs in float32 anyway. It does
— the accumulators are `float` and every read widens on load — so the casts bought
nothing arithmetically and cost five materialised float32 copies of the inputs per call,
thirty times a token. They existed only because the kernel templated `InT` on one type
and used it for both `y` and the output state, so keeping the state in float32 forced the
activations to float32 too. mlx-lm splits these (`InT` = input dtype, `StT` = state
dtype); ours now does the same and passes the activations through untouched.

**Any substitute value spliced into the delta block must carry the block's dtype.** The
`LMP_ABLATE=deltakernel` stand-in was initially `mx::zeros(..., mx::float32)` and
re-created the original bug exactly — 11.8 -> 32.8 ms/token — turning the ablation into a
measurement of the promotion rather than of the kernel.

**Watch for this class of bug anywhere a Python reference is transcribed to C++.** MLX's
Python bindings give scalars weak dtype semantics; the C++ API has no such notion, so the
literal transcription silently changes the dtype of everything downstream. It is invisible
in output quality (eval is unmoved, realmodel generates the same tokens) and shows up only
as uniform slowness.

---

## Retractions — things the third pass asserted that are not true

Re-derive before reusing any of these.

| third-pass claim | what measurement shows |
|---|---|
| "MLX is now 0.31.2" | It was 0.29.3. The version line reported the interpreter, not the linked package. |
| "MLX 0.31.2 -> decode 27.9 -> 27.6. Nothing." | Never tested. Real 0.31.2 moves **prefill 586.7 -> 856.4** (+46%); decode is genuinely unaffected. |
| "gather_qmm fed an unevaluated index array costs ~2.5x" | An artifact of the isolation benchmark. mlx-lm's own block shows 0.058 (evaluated) vs 0.102 (unevaluated) — real but small — and our full chained routed path measures 3.86 ms against the reference's 3.54. Not a cause. |
| "forward_moe is 27 ms of a 35 ms step" | `layers` fed **float16** into a bf16 model, so every number it ever printed was an fp16 number. That is why it matched mlx-lm's fp16 figure (0.701 ms/layer) so exactly. Fixed to bf16. |
| "the dependency is innocent: chained 0.693 vs independent 0.647" | Same fp16 instrument. |

**Ablations are not additive, and reading them as if they were is what sent the third
pass after the MoE.** Removing the routed experts saved 25.8 ms and removing the
gated-delta block saved 27.8 ms — inside the same 34.9 ms step. Neither block cost what
its ablation appeared to charge it; both were slow for one shared reason, and deleting
either one took the float32 residual with it.

## Also measured, also not the cause

- **Memory traffic.** Peak 19.00 GB vs the reference's 20.18 GB. `moestream` sustains
  218 GB/s on the routed experts under a real access pattern.
- **Scheduling knobs.** `MLX_BFS_MAX_WIDTH` (1/2/8/40), `MLX_MAX_MB_PER_BUFFER`,
  `MLX_METAL_FAST_SYNCH`: all within noise. `MLX_MAX_OPS_PER_BUFFER` was already ruled out.
- **CPU graph construction.** 0.81 ms of a 34.9 ms step; `lmp_diag step` prints the
  build/eval split, so this is now a one-command check rather than an assumption.
- **Host logits copy and CPU sampling.** Imposed on the reference's loop, they cost it
  nothing (81.4 -> 81.1 tok/s).

---

## Still open

1. **Prefill, 1110.7 vs 1347.** First establish whether 1347 is a real target: it comes
   from LM Studio's server logs at 1-second resolution, and we already run prefill 1.52x
   faster than the mlx_lm reference on this machine. If the reference is the honest bar,
   prefill is done. Settle that before optimising.
2. **The wired limit is set but barely earns its keep** (+4%, 27.5 -> 28.6). Kept because
   it is what mlx-lm does and it costs nothing; do not expect more from it.
3. **Weak-scalar audit.** The bug above was one instance of a general hazard. Worth
   grepping the forward path for other `mx::array(<float literal>)` constructions and
   checking each one's dtype against the reference.
4. **Small quantized matmuls.** With the ablations now additive, our gated-delta block
   costs 4.42 ms against the reference's 3.41, and our MoE gate/topk/shared-expert 1.32
   against 0.77. Both "everything except the big gather" categories are ~2x. Same op,
   same shapes, same `quantized_matmul` arguments, so this is unexplained and is the
   largest remaining item that is actually ours to fix.

## Why async_eval is not available

The third pass listed `mx::async_eval` double-buffering as the clearest remaining win.
It is worth ~20% in the reference (12.33 -> 9.86 ms/token, re-measured). **We cannot
have it, and the reason is structural rather than an implementation gap.**

mlx-lm samples on the GPU, so `y` is an `mx.array`. It builds step N+1's forward from
that *unevaluated* array and submits it before ever reading step N's token, which is what
keeps the pipeline full. Our sampler is a pure CPU function over a host logits row —
deliberately, per `sampler.hpp`: mask-first constrained decode, repetition penalty over
`recent`, top-k/min-p/top-p, and a seeded splitmix64 draw whose determinism the tests and
the replay path depend on. The next forward's embedding lookup needs the sampled id as a
host value, so `logits_N -> token_N -> forward_{N+1}` is a hard data dependency and there
is nothing to overlap it with.

Closing it would mean reimplementing constrained sampling on the GPU. That cannot be done
bit-exactly — the heap top-k's tie-break, the accumulation order in softmax/min-p/top-p,
and the cumulative-sum traversal of the draw would all have to match — so it would change
generated tokens and trade a documented determinism invariant for ~20%. That is a product
decision, not a perf fix, and it should not be made silently.

## Measured and rejected — do not redo these

- **Folding the float32 logits cast into the forward graph** ("one round-trip instead of
  two"). Reads better, measures worse: 84.8 -> 83.9 tok/s, reproduced across three runs.
  As a separate small dispatch the cast is free; on the end of the step's graph it
  extends the critical path. `logits_to_host` carries a comment saying so.
- **Caching resolved weight handles.** The model rebuilds every weight key as a
  `std::string` and hash-probes it on each access, ~800 times a step. Measured before
  writing the refactor: that is 0.27 ms of a 0.94 ms CPU build, and the other ~0.7 ms is
  MLX op construction, which no cache removes. Worth ~2%; not worth the blast radius.

---

## Instruments

`tests/model/diag_main.cpp` and `tests/model/diag_moe.cpp`
(`cmake --build --preset dev --target lmp_diag`):

| | |
|---|---|
| `lmp_diag step [n]` | **the real decode forward**, split CPU-build vs GPU-eval |
| `lmp_diag moestream [n]` | routed experts under a real step's access pattern, in GB/s |
| `lmp_diag bench [runs] [prompt] [max_new]` | N-run ledger against the LM Studio numbers, plus peak/active/cache memory |
| `lmp_diag scan [T...]` | fused kernel vs reference loop: deviation and wall time |
| `lmp_diag mask` | one decode step outside the forward pass |
| `lmp_diag layers [T]` \| `moe [T]` \| `blocks [T]` | isolation benchmarks — **hypothesis generators only**, see below |

`LMP_ABLATE=routed|mlp|delta|deltakernel` deletes one block from a real forward pass.
Output is garbage; only the rate means anything.

`scripts/lmstudio_baseline.py` re-derives the log side; the mlx_lm interpreter above
re-derives the live side.

**The standing warning, now with three convictions against it.** `blocks`, `moe` and
`layers` call one block repeatedly on a fixed input, which is not what a decode step
does. `blocks` hid a 6x effect by hoisting an evaluated `inds`; `moe` invented a
gather_qmm penalty that the reference does not reproduce; `layers` ran the whole model in
the wrong dtype. Every one of them was believed at the time. Prefer `step` and
`moestream`, prefer ablation on a real run to any per-block timing, and when a number
surprises you, suspect the instrument before the code.

## Guardrails — all green

`ctest --preset gate` 19/19 · `./scripts/run_ratchets.py --root .` 6/6 clean ·
`ctest --preset realmodel` 2/2 · `./scripts/eval.py --root . score` unmoved
(corpus wmiss=0 179/179, holdout wmiss=15 34/42).

## Do not

- Do not delete `gated_delta_update_ops`, or the equivalence tests in
  `tests/model/test_grammar.cpp`. They are what make the fast paths falsifiable.
- Do not re-apply `mx::compile`, or re-test `MLX_MAX_OPS_PER_BUFFER`, without a new
  reason. Both were measured end to end and both were zero. **MLX versions are no longer
  on this list** — 0.31.2 was never actually tested until this pass.
- Do not trust a per-block timing that disagrees with `step` or `bench`.
- Do not weaken the grammar to make the mask cheap.
- Do not reach for speculative decoding.
- Do not re-litigate settled decisions: Apple Silicon only, MLX in-process, Qwen3 only,
  XML tool-call syntax (see `docs/PHASES.md`).
