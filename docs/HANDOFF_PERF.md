# Handoff: LM_Pipe v2 vs LM Studio, sixth pass

Paste this whole file as the opening prompt of a fresh session in
`/Users/dev/Desktop/seans_projects_local/LM_Pipe_2` (branch `perf/mask-and-scan`).

Every number here was measured in this repo on 2026-07-31. Nothing is quoted from a
previous handoff without being reproduced first (S19.6) — the fifth pass's headline
numbers were re-run at the top of this one and held (decode 83.7 median over 10 runs
against its claimed 84.7). Each of the last three passes had to retract something its
predecessor asserted, so read the retractions before trusting anything you remember.

**Read "Never run two MLX processes at once" in the instruments section before running
anything.** Ignoring it crashed the machine this pass.

**Both criteria pass, and the last open item was misnamed.**

**Read this before quoting any absolute number below.** This machine's throughput drifts
~9% with background load. The same binary measured decode 88.2 tok/s early in the sixth
pass and 80.3 two hours later, both as tight 5-to-8-run medians. **Absolute numbers taken
at different times are not comparable, and a difference between two of them is not
evidence about a code change.** The sixth pass committed a claim built exactly that way
and had to retract it. Use a same-session A/B — ideally one env var on one binary — for
any before/after.

Clean sequential sweep, 5 runs each, `bench 5 <p> 200`, nothing else on the GPU:

| prompt | prefill | decode | peak mem |
|---|---|---|---|
| 547 | 1368 | 82.7 | 19.00 GB |
| 2090 | 1887 | 80.6 | 19.98 GB |
| 4130 | 1855 | 79.1 | 20.02 GB |
| 8240 | 1679 | 74.9 | 20.35 GB |
| 16426 | 1495 | 70.0 | 21.01 GB |

Prefill beats the length-matched LM Studio bar everywhere it can be compared (1.37x at
4130, 1.10x at 8240, 1.12x at 16426; the sub-4k buckets are floors, not bars). Decode
passes the 78.5 aggregate bar to 4130 and falls under it beyond — which is the
aggregate-bar problem, not a regression: 78.5 is a median over logged decodes at every
context length, and our decay with context (0.85x from 547 to 16426) tracks the
reference's.

Two things got us here. The fifth pass found one constant — prefill chunked at 512 where
mlx-lm's `prefill_step_size` default is 2048. This pass found the other by building the
first instrument in this repo that is not a stopwatch: **a primitive histogram of one
decode step's graph, diffed against the same dump from mlx-lm.** It showed our quantized
matmuls were identical in count to the reference's and that the entire difference was 919
un-fused elementwise dispatches. See "What the '~2x on small quantized matmuls' actually
was".

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

The progression below is **each pass's own reading**, and the passes are hours or days
apart, so the deltas between adjacent columns carry the machine drift described at the
top. Read it as history, not as attribution.

| | third pass (really 0.29.3) | real 0.31.2 | + wired limit | + dtype fix | + kernel dtypes | + chunk 2048 | + fused chains |
|---|---|---|---|---|---|---|---|
| decode  | 27.6 | 27.5 | 28.6 | 84.8 | 85.4 | 84.7 | 82.7 |
| prefill | 586.7 | 856.4 | 871.0 | 1110.7 | 1126.5 | 1324 | 1368 |

**The only attribution here that survives an A/B** is the fusion itself, measured on one
binary in one session with `MLX_DISABLE_COMPILE=1` as the control:

| | decode | prefill |
|---|---|---|
| compile ON | 80.3 | 1223 |
| compile OFF | 77.6 | 1239 |

**Fusion is worth +3.5% of decode and nothing measurable on prefill.** The sixth pass
first claimed +4.8% decode and +5.5% prefill by comparing runs an hour apart; the decode
figure was inflated and the prefill figure was drift, retracted in full. What the fusion
provably does is structural and drift-proof: the decode graph goes from 3529 primitives
to 2629 against the reference's 2610.

**The honest apples-to-apples decode comparison is ours vs 80.9, not ours vs 98.**
The reference reaches 98 by running one step ahead with `mx.async_eval`, which a
host-side sampler cannot do (see "Why async_eval is not available"). Constrain the
reference the way our loop is constrained — synchronous eval, full logits to host, CPU
sampling — and it does **80.9 tok/s** on this machine (fifth-pass measurement, not
re-run this pass, and subject to the same drift caveat). What is left against the
unconstrained 98 is a loop-structure difference, not a kernel one.

Decode still falls with context at the reference's rate — ours 82.7 at 547 tokens to
74.9 at 8240 (0.91x), the reference 98.0 to 88.4 (0.90x) — so an 8k decode reading below
78.5 is the aggregate-bar problem again, not a regression.

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

**Fifth pass:**

| fifth-pass claim | what measurement shows |
|---|---|
| "Small quantized matmuls — the largest item that is actually ours... same op, same shapes, same `quantized_matmul` arguments, so this is unexplained." | Not the matmuls, and not unexplained. `lmp_diag graph` counts 391 QuantizedMatmul and 120 GatherQMM on both sides. The gap was 919 un-fused elementwise dispatches; fusing them is worth 4.8% of decode. |
| "Left uncompiled deliberately... `mx::compile` moved decode by nothing: 28.1 -> 27.9 tok/s" (in `gated_delta.hpp`) | True when written, void now. It was measured while the float32-residual bug made a step 34.9 ms, so ~0.5 ms was 1.3%. At an 11.3 ms step the same four sites are worth 4.8%. |
| "decode 84.7 at 547 tokens" | Reproduces at 83.7 median over 10 runs (min 82.1, max 84.6) on the fifth pass's own binary. 84.7 was the top of the spread, not the middle. The conclusion is unchanged — it passed then and passes now — but quote medians. |

**Sixth pass, against itself:**

| sixth-pass claim | what measurement shows |
|---|---|
| "decode 83.7 -> 87.7, prefill 1329 -> 1402" (commit 83a18c6) | Built by comparing runs an hour apart, and that is drift, not attribution. A same-session A/B on one binary (`MLX_DISABLE_COMPILE=1`) gives **80.3 vs 77.6 decode, +3.5%**, and **1223 vs 1239 prefill — no effect**. The fusion is worth keeping on the decode number and on the 900-node graph reduction; the prefill claim is withdrawn. |
| "decode 87.7 -> 88.2, prefill 1402 -> 1409" (commit 779157e) | Below this machine's drift floor. The split/expand_dims/scalar changes are justified by the graph converging to 2629 nodes, not by a measured speedup. Do not quote a number for them. |

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

1. **LM Studio's logged prefill is ~2x stock mlx_lm** on the same machine and checkpoint
   (1529 vs 773 at 8k). Either their backend is not stock mlx_lm, or the progress lines
   `lmstudio_baseline.py` brackets do not span the whole prefill. Unsettled — it stopped
   mattering once we beat both, but it means the two references are not interchangeable.
2. **The wired limit is set but barely earns its keep** (+4%, 27.5 -> 28.6). Kept because
   it is what mlx-lm does and it costs nothing; do not expect more from it.
3. **The last 19 graph nodes** are a Reshape difference (ours 340, reference 320).
   Named for completeness; at this size it is not worth a commit on its own.
4. **4130 and 16430 were not re-measured** after the fusion commits. Every length that
   was re-measured improved, so the table's numbers for those two are a floor.

**Closed this pass:** the "small quantized matmuls are ~2x" item, which was misnamed —
see below.

## What the "~2x on small quantized matmuls" actually was

The fifth pass left this as the largest item that was ours, and described it as
unexplained: same op, same shapes, same `quantized_matmul` arguments, yet our
gated-delta block measured 4.42 ms against the reference's 3.41 and our MoE
gate/topk/shared-expert 1.32 against 0.77.

It was never the matmuls. `lmp_diag graph` dumps one decode step's graph as dot,
unevaluated, and `scripts/graph_histogram.py` histograms it and diffs it against the
same dump taken from mlx-lm on the LM Studio interpreter. The diff settled it in one
run:

| | ours (before) | reference |
|---|---|---|
| QuantizedMatmul | 391 | 391 |
| GatherQMM | 120 | 120 |
| RMSNorm | 191 | 191 |
| CustomKernel | 30 | 30 |
| **TOTAL** | **3529** | **2610** |

Every heavy op issued in identical counts. The whole 919-node difference was elementwise
and shape work: mlx-lm fuses four chains into 170 `Compiled*` kernels — `swiglu` (x80),
`silu` (x30), `compute_g` (x30), the gated-norm tail (x30) — and we were issuing ~1090
separate dispatches for the same arithmetic. Compiling the same four sites took the graph
to 2799 and decode 83.7 -> 87.7; three shape/scalar alignments (`split` instead of
slices, scalars built in the target dtype instead of cast into it, one multi-axis
`expand_dims`) took it to 2629 and decode to 88.2.

**The instrument is the point.** Every other subcommand in `lmp_diag` answers "how long
did this take", and on this project that question has now been answered wrongly four
times. A primitive histogram answers "what work is in the graph" — a fact about the
program rather than about the machine it ran on — so it cannot be wrong about the thing
it measures, only incomplete. Reach for it first when two stacks that should agree do
not.

**And a null result has an expiry date.** `compute_g` carried a comment stating that
`mx::compile` had been measured end to end at "28.1 -> 27.9 tok/s, nothing" and reverted
on principle. That measurement was honest. It was also taken while the float32-residual
bug made a decode step 34.9 ms, where the ~0.5 ms it saves is 1.3% and invisible. The
step is now ~11.3 ms. **A null measured under a bug that has since been fixed by 3x has
to be re-run, not inherited** — check the denominator every negative result was divided
by before trusting it.

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
| `lmp_diag graph [prompt]` | one decode step's graph as dot, unevaluated — **not a timing**, see below |
| `lmp_diag layers [T]` \| `moe [T]` \| `blocks [T]` | isolation benchmarks — **hypothesis generators only**, see below |

`LMP_ABLATE=routed|mlp|delta|deltakernel` deletes one block from a real forward pass.
Output is garbage; only the rate means anything.

`LMP_PREFILL_CHUNK=N` overrides the prefill chunk (default 2048) without a rebuild.

`scripts/lmstudio_baseline.py` re-derives the log side. `scripts/mlxlm_reference.py`
re-derives the live side — run it with the LM Studio interpreter, and use `--prompts` /
`--chunks` rather than trusting any single number from it.

`scripts/graph_histogram.py` is the other half of `lmp_diag graph`, and the one
instrument here that does not measure time:

```bash
./build/tests/model/lmp_diag graph 547 > ours.dot
python3 scripts/graph_histogram.py --dot ours.dot --json ours.json
```

```bash
~/.lmstudio/extensions/backends/vendor/_amphibian/app-mlx-generate-mac14-arm64@29/bin/python scripts/graph_histogram.py --reference --json ref.json
```

```bash
python3 scripts/graph_histogram.py --compare ours.json ref.json
```

Run those three **one at a time** — see the memory-safety note below.

## Never run two MLX processes at once

One loaded checkpoint is 19 GB resident and peaks over 20 GB on a 16k prompt; this
machine has 48 GB and normally has several IDEs open. Two concurrent MLX processes
exhaust it. This is not hypothetical: backgrounding a `bench` length sweep and running
`graph_histogram.py --reference` on top of it **crashed the machine** on 2026-07-31.

Everything that loads a model counts — `lmp_diag` (any subcommand), `mlxlm_reference.py`,
`graph_histogram.py --reference`, `ctest --preset realmodel`. Run them sequentially, in
the foreground, and wait for each to exit.

It also invalidates the numbers, which is how the contamination was noticed: the same
8240-token prefill read 1686 tok/s under contention and 1932 alone.

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

**A sixth, and it is the simplest one yet: this machine drifts ~9%.** The same binary
read decode 88.2 and 80.3 two hours apart, each a tight multi-run median. Every
before/after in this document that spans more than one sitting is therefore suspect,
including two the sixth pass committed before catching it. **Use a same-session A/B.**
`MLX_DISABLE_COMPILE=1` is the model: one binary, one process lifetime, one variable.
When no env-var control exists, build both variants and alternate them.

**A fifth, and the reason `graph` exists.** The "small quantized matmuls are ~2x" item
survived a whole pass as unexplained because every tool available to question it was a
timing. The primitive histogram answered it immediately and unambiguously: the matmul
counts were identical and the difference was 919 un-fused elementwise dispatches. When
two stacks that should agree do not, compare *what they do* before comparing how long
they take.

## Guardrails — all green

`ctest --preset gate` 20/20 · `./scripts/run_ratchets.py --root .` 6/6 clean ·
`ctest --preset realmodel` 2/2 · `./scripts/eval.py --root . score` unmoved
(corpus wmiss=0 179/179, holdout wmiss=15 34/42).

## Do not

- Do not delete `gated_delta_update_ops`, or the equivalence tests in
  `tests/model/test_grammar.cpp`. They are what make the fast paths falsifiable.
- Do not re-test `MLX_MAX_OPS_PER_BUFFER` without a new reason; it was measured end to
  end and was zero. **`mx::compile` has come off this list** — it is now applied at the
  four sites mlx-lm compiles and is worth 4.8% of decode. It was on the list because of a
  null measured under a since-fixed 3x bug, which is the trap described above. **MLX
  versions came off it in the fifth pass** for the same kind of reason.
- Do not run two MLX processes concurrently. It crashes the machine and it corrupts the
  numbers; see the instruments section.
- Do not trust a per-block timing that disagrees with `step` or `bench`.
- Do not answer "why is our X slower than theirs" with a timing when `lmp_diag graph`
  can answer it structurally. That question has been answered wrongly four times here by
  timings and correctly once, immediately, by an op histogram.
- Do not reintroduce a scalar prefill target, and do not quote a prefill number without
  the prompt length beside it. Both stacks' prefill varies ~2x across the length range.
- Do not lower `LMP_PREFILL_CHUNK` back to 512 to save memory without re-measuring: it
  costs 15-35% of prefill and buys 1.3 GB only on prompts over ~4k.
- Do not weaken the grammar to make the mask cheap.
- Do not reach for speculative decoding.
- Do not re-litigate settled decisions: Apple Silicon only, MLX in-process, Qwen3 only,
  XML tool-call syntax (see `docs/PHASES.md`).
