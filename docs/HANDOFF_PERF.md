# Handoff: LM_Pipe v2 vs LM Studio, fifth pass

Paste this whole file as the opening prompt of a fresh session in
`/Users/dev/Desktop/seans_projects_local/LM_Pipe_2` (branch `perf/mask-and-scan`).

Every number here was measured in this repo on 2026-07-31. Nothing is quoted from a
previous handoff without being reproduced first (S19.6) — and each of the last two passes
had to retract things its predecessor asserted, so read the retractions before trusting
anything you remember.

**Both criteria now pass, and one of them was the wrong criterion.**

Decode 84.7 tok/s against 78.5. Prefill 1324 tok/s at a 547-token prompt and 1818 at
8240 — against a target of 1347 that was never comparable to what it was being measured
against. 1347 is the median over LM Studio log windows >= 5 s, and **those windows have a
median prompt of 9172 uncached tokens**, while the bench ran 547. LM Studio's own prefill
runs 824 tok/s at 512-1024 tokens and 1529 at 8192-16384; there is no single number to
beat. Against length-matched bars we now pass everywhere.

The fix that got us there was one constant: prefill chunked at 512 where mlx-lm's
`prefill_step_size` default is 2048.

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

**Decode criterion: > 78.5. There is no single prefill criterion** — see below.

### Prefill has no scalar baseline

LM Studio's logged prefill is a strong function of prompt length. Bucketed by
`uncached_tokens` over the same 2097 windows:

| uncached tokens | n | LM Studio median | median window | trustworthy? |
|---|---|---|---|---|
| 512-1024 | 449 | 824 | 1 s | no |
| 1024-2048 | 341 | 1127 | 1 s | no |
| 2048-4096 | 376 | 1155 | 2 s | no |
| 4096-8192 | 234 | 1350 | 4 s | yes |
| 8192-16384 | 447 | 1529 | 7 s | yes |
| 16384+ | 195 | 1334 | 15 s | yes |

The short buckets are not measurements. Timestamps are 1-second resolution, so a
sub-second prefill either straddles a tick and reads as a full second — understating
throughput badly — or does not straddle one and is dropped by the `dt > 0` filter. They
are a floor. Only the 4096+ buckets have windows long enough to mean anything.

`lmp_diag bench` now picks the bar for the length it actually measured and prints no
verdict at all for the untrustworthy buckets. Do not reintroduce a scalar target.

### The live reference

Much more useful than the log-derived numbers: **LM Studio's own stack runs directly on
this machine.**

```bash
~/.lmstudio/extensions/backends/vendor/_amphibian/app-mlx-generate-mac14-arm64@29/bin/python
```

That interpreter has mlx 0.31.2 + mlx_lm 0.31.3 — exactly what LM Studio ships.
`scripts/mlxlm_reference.py` sweeps it; run it with that interpreter. At
`prefill_step_size=2048`:

| prompt | reference prefill | reference decode |
|---|---|---|
| 522 | 699.9 | 98.0 |
| 2062 | 801.9 | 95.2 |
| 8213 | 773.0 | 88.4 |

Use it. An apples-to-apples reference you can instrument and ablate in Python, with no
rebuild, is worth more than any log-derived median.

**Unexplained, and worth knowing before you trust the logs again:** LM Studio's logged
prefill (1529 at 8-16k) is about 2x what stock mlx_lm does on the same machine and
checkpoint (773 at 8213). Either their backend is not stock mlx_lm, or the progress lines
the parser brackets do not span the whole prefill. It did not need settling this pass —
we now beat both — but do not treat the two as interchangeable.

## Where we are

`lmp_diag bench 4 512 200`, 547-token prompt:

| | third pass (really 0.29.3) | real 0.31.2 | + wired limit | + dtype fix | + kernel dtypes | + chunk 2048 |
|---|---|---|---|---|---|---|
| decode  | 27.6 | 27.5 | 28.6 | 84.8 | 85.4 | **84.7** |
| prefill | 586.7 | 856.4 | 871.0 | 1110.7 | 1126.5 | **1324** |

Decode **PASS** at 1.08x. Prefill passes at every length once the bar is length-matched:

| prompt | LM_Pipe | LM Studio (same bucket) | mlx_lm live |
|---|---|---|---|
| 547 | 1324 | 824 (floor only) | 700 |
| 2090 | 1751 | 1155 (floor only) | 802 |
| 4130 | 1779 | 1350 | — |
| 8240 | 1818 | 1529 | 773 |
| 16430 | 1684 | 1334 | — |

**The honest apples-to-apples decode comparison is 84.7 vs 80.9, not 84.7 vs 98.**
The reference reaches 98 by running one step ahead with `mx.async_eval`, which a
host-side sampler cannot do (see "Why async_eval is not available"). Constrain the
reference the way our loop is constrained — synchronous eval, full logits to host, CPU
sampling — and it does **80.9 tok/s** on this machine. Our forward pass is already
faster than mlx-lm's; what is left is a loop-structure difference, not a kernel one.

Decode falls with context at the same rate as the reference — ours 82.4 at 547 tokens to
75.1 at 8240 (0.91x), the reference 98.0 to 88.4 (0.90x) — so the 8k decode reading
below 78.5 is the aggregate-bar problem again, not a regression. The 78.5 bar is a median
over logged decodes at every context length.

---

## The cause of the prefill gap: one chunk size

Prefill was roughly **flat in prompt length** — 1118 tok/s at 547 tokens, 1170 at 8240 —
where LM Studio's climbs from 824 to 1529. A fixed chunk size predicts exactly that
shape, and ours was 512 against mlx-lm's `prefill_step_size` default of 2048.

Every chunk ends in a full synchronous barrier (`eval_caches`), so the chunk size sets
how often prefill drains the GPU and rebuilds a 48-layer graph from the host. At 512 we
paid that barrier four times as often as the reference. At 547 tokens it was worse than
that: a 512-token chunk plus a **35-token second chunk** that was almost entirely
overhead, which is why the single biggest relative jump is at the shortest prompt.

| prompt | chunk 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| 547 | 1117.7 | 1315.6 | **1317.1** | — |
| 2090 | 1209.0 | 1513.0 | **1751.5** | — |
| 8240 | 1169.6 | 1512.4 | **1684.5** | 1527.7 |

2048 is the knee; 4096 turns over again. Peak memory is unchanged on short prompts and
19.08 -> 20.41 GB at 8240, against the 20.18 GB mlx-lm peaks at on this checkpoint.

`LMP_PREFILL_CHUNK` overrides it so the sweep can be re-run without a rebuild.

---

## The cause of the decode gap: one strongly-typed scalar

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

## Retractions

Re-derive before reusing any of these.

**Fourth pass:**

| fourth-pass claim | what measurement shows |
|---|---|
| "Exit criterion: prefill > 1347." | Not comparable to anything the bench measured. 1347 is the median over >= 5 s windows, whose median prompt is 9172 tokens; the bench ran 547. LM Studio's own prefill is 824 at 512-1024 tokens and 1529 at 8192-16384. |
| "prefill n=1963 median 1145.0" as a baseline | The buckets under 4096 tokens have 1-2 s median windows against a 1-second clock. They are a floor, not a measurement — sub-second prefills either read as a full second or are dropped by `dt > 0`. |
| "mlx_lm reference: prefill 728.5, decode 103.1" | Reproduces as 699.9 / 98.0 at 522 tokens with `prefill_step_size=2048`. Close enough to be the same measurement, but the reference's prefill is also length- and chunk-dependent (802 at 2062, 773 at 8213) — one number is not a baseline for it either. |

**Third pass:**

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

Both exit criteria pass. Nothing below is blocking; this is the list of things that are
known-unfinished rather than known-broken.

1. **Small quantized matmuls — the largest item that is actually ours.** With the
   ablations additive, our gated-delta block costs 4.42 ms against the reference's 3.41,
   and our MoE gate/topk/shared-expert 1.32 against 0.77. Both "everything except the big
   gather" categories are ~2x. Same op, same shapes, same `quantized_matmul` arguments,
   so this is unexplained.
2. **LM Studio's logged prefill is ~2x stock mlx_lm** on the same machine and checkpoint
   (1529 vs 773 at 8k). Either their backend is not stock mlx_lm, or the progress lines
   `lmstudio_baseline.py` brackets do not span the whole prefill. Unsettled — it stopped
   mattering once we beat both, but it means the two references are not interchangeable.
3. **The wired limit is set but barely earns its keep** (+4%, 27.5 -> 28.6). Kept because
   it is what mlx-lm does and it costs nothing; do not expect more from it.

**Closed this pass:** prefill (chunk 2048, and the target was wrong); the weak-scalar
audit (three sites, one latent bug in the weight loader's norm shift, two correct — see
the comments in `gated_delta.hpp` and `qwen35_moe_model.hpp`, both of which explain why
they look like the bug and are not).

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
| `lmp_diag bench [runs] [prompt] [max_new]` | N-run ledger against the **length-matched** LM Studio bar, plus peak/active/cache memory. Sweep `[prompt]` — one length is not a prefill result. |
| `lmp_diag scan [T...]` | fused kernel vs reference loop: deviation and wall time |
| `lmp_diag mask` | one decode step outside the forward pass |
| `lmp_diag layers [T]` \| `moe [T]` \| `blocks [T]` | isolation benchmarks — **hypothesis generators only**, see below |

`LMP_ABLATE=routed|mlp|delta|deltakernel` deletes one block from a real forward pass.
Output is garbage; only the rate means anything.

`LMP_PREFILL_CHUNK=N` overrides the prefill chunk (default 2048) without a rebuild.

`scripts/lmstudio_baseline.py` re-derives the log side. `scripts/mlxlm_reference.py`
re-derives the live side — run it with the LM Studio interpreter, and use `--prompts` /
`--chunks` rather than trusting any single number from it.

**The standing warning, now with three convictions against it.** `blocks`, `moe` and
`layers` call one block repeatedly on a fixed input, which is not what a decode step
does. `blocks` hid a 6x effect by hoisting an evaluated `inds`; `moe` invented a
gather_qmm penalty that the reference does not reproduce; `layers` ran the whole model in
the wrong dtype. Every one of them was believed at the time. Prefer `step` and
`moestream`, prefer ablation on a real run to any per-block timing, and when a number
surprises you, suspect the instrument before the code.

**A fourth conviction, and this one was the baseline rather than a benchmark.** The
prefill target was a median over a population whose prompts were 17x longer than the one
being measured against it. Nothing in the instrument was wrong — `bench` reported its own
number honestly — but the *bar* it printed came from somewhere else, and three passes
read the resulting FAIL as a fact about the code. When a comparison is against an
aggregate someone else produced, check what population it aggregates before believing the
verdict.

## Guardrails — all green

`ctest --preset gate` 20/20 · `./scripts/run_ratchets.py --root .` 6/6 clean ·
`ctest --preset realmodel` 2/2 · `./scripts/eval.py --root . score` unmoved
(corpus wmiss=0 179/179, holdout wmiss=15 34/42).

## Do not

- Do not delete `gated_delta_update_ops`, or the equivalence tests in
  `tests/model/test_grammar.cpp`. They are what make the fast paths falsifiable.
- Do not re-apply `mx::compile`, or re-test `MLX_MAX_OPS_PER_BUFFER`, without a new
  reason. Both were measured end to end and both were zero. **MLX versions are no longer
  on this list** — 0.31.2 was never actually tested until this pass.
- Do not trust a per-block timing that disagrees with `step` or `bench`.
- Do not reintroduce a scalar prefill target, and do not quote a prefill number without
  the prompt length beside it. Both stacks' prefill varies ~2x across the length range.
- Do not lower `LMP_PREFILL_CHUNK` back to 512 to save memory without re-measuring: it
  costs 15-35% of prefill and buys 1.3 GB only on prompts over ~4k.
- Do not weaken the grammar to make the mask cheap.
- Do not reach for speculative decoding.
- Do not re-litigate settled decisions: Apple Silicon only, MLX in-process, Qwen3 only,
  XML tool-call syntax (see `docs/PHASES.md`).
