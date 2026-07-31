# Handoff: LM_Pipe v2 vs LM Studio, third pass

Paste this whole file as the opening prompt of a fresh session in
`/Users/dev/Desktop/seans_projects_local/LM_Pipe_2` (branch `perf/mask-and-scan`).

Every number here was measured in this repo on 2026-07-31. Nothing is quoted from a
previous handoff without being reproduced first (S19.6).

**The target is known to be reachable on this hardware, because LM Studio reaches it
with the same checkpoint.** Two of three original causes are closed. The remaining gap is
one block, and this session narrowed it to one measurement.

---

## Build note, read this first

MLX is now **0.31.2**, matching LM Studio exactly. It was 0.29.3, and the reason was
invisible: `src/model/CMakeLists.txt` probed `python3 -m mlx --cmake-dir`, this machine's
`python3` is /usr/bin/python3 (3.9), and **MLX stopped shipping cp39 wheels after
0.29.3**. The whole project was pinned two minor versions back by an interpreter nobody
chose.

The interpreter is now a cache variable. Configure with:

```bash
cmake --preset dev -DLMP_MLX_PYTHON=$HOME/.venvs/lmp-mlx/bin/python
```

That venv exists (`python3.12 -m venv ~/.venvs/lmp-mlx`, `pip install mlx==0.31.2`).
**Reconfigure without the flag and you silently drop back to 0.29.3.** The configure line
now prints the resolved version — read it before believing any performance number.

---

## Baselines

`scripts/lmstudio_baseline.py` rebuilds LM Studio's numbers from `~/.lmstudio/server-logs`
(34 files, real prior usage of this exact checkpoint). 1-second timestamps, so it reports
all windows and windows >= 5 s where clock error is small.

```
all windows:      prefill n=1963 median 1145.0   decode n=134 median 78.5
windows >= 5s:    prefill n=740  median 1338.9   decode n=17  median 87.7
```

**Exit criterion: decode > 78.5, prefill > 1347.**

## Where we are

`lmp_diag bench 4 512 200` (547-token prompt, 200 new tokens), MLX 0.31.2:

```
prefill  n=4 median 586.7 tok/s  min 571.0 max 587.1    [0.44x LM Studio]
decode   n=4 median  27.6 tok/s  min  27.6 max  27.7    [0.35x LM Studio]
```

Matched before/after from the first pass, same prompt and seed (`test_realmodel`):
decode 22.7 -> 28.4 tok/s, prefill 45.8 -> 119.1 tok/s.

---

## Closed (first pass)

### The grammar mask — 28.3 ms/token to 0.00

`Sampler::sample` took `std::function<bool(TokenId)>` and called it once per vocabulary
id. `TurnGrammar::permitted` is now the *definition*; the sampler reads a `TokenMask`
bitset (`src/model/token_mask.hpp`) — cached per phase outside a tool call, parsephony's
`TokenMaskT<ToolCallGuard>` inside one. `lmp_diag mask`:

```
permitted() over full vocab :    28.29 ms/token
bulk mask, warm             :     0.000 ms/token
sampler, no mask            :     0.18 ms/token   (was 1.7 -- top_p no longer
sampler + mask (as shipped) :     0.17 ms/token    std::sort's 248k indices)
```

Two correctness results, both pinned by
`the_bulk_mask_and_the_predicate_agree_over_the_whole_vocabulary`:

- **parsephony's free-text mask was more permissive than its own automaton** — it only
  simulated tokens carrying the parameter terminator, but `ToolCallGuard` also rejects
  control bytes in a raw text value. 93 vocabulary entries the fast mask allowed and the
  grammar denies. Fixed in the vendored header.
- **Ids past the vocabulary** (logits row 248,320 wide, vocab 248,077) are no longer
  emittable.

**Keep that test.** It is the only thing between a 28 ms saving and a mask that lies.

### Prefill's op-per-timestep scan — 20-30x

`gated_delta_update` ran one MLX op-set per token position. The fix was not a chunked
scan: **mlx-lm already ships a fused Metal kernel** and LM Studio runs it; our C++ had
ported mlx-lm's *reference* loop. `lmp_diag scan` on 0.31.2:

```
     T         ops ms      kernel ms   speedup   rel|dy|   rel|dstate|
     1            1.6           0.25      6.7x   1.9e-07     0.0e+00
   287           33.0           1.08     30.6x   1.1e-07     9.4e-08
   512           40.1           1.78     22.5x   1.6e-07     1.1e-07
```

Deviation is fp32 association only. `gated_delta_update_ops` stays as the definition the
kernel is tested against. Feed that diagnostic raw normals for `k` and *both*
implementations diverge — `(I - beta k k^T)` is only contractive once `k` is rms-normed
and scaled by `1/sqrt(Dk)`, which is what `forward_gated_delta` does.

### The logits copy

0.07 ms/token. Closed by measurement, not by work.

---

## The gap: forward_moe, 27 ms of a 35 ms step

`lmp_diag layers 1`:

```
  linear layer (delta+moe) chained 1.067  independent 1.579 ms/layer  x30 = 32.02 ms
  attn layer (attn+moe)    chained 0.818  independent 0.764 ms/layer  x10 =  8.18 ms
    of which moe           chained 0.693  independent 0.647 ms/layer  x40 = 27.71 ms
```

### Five hypotheses, all falsified by measurement

Do not re-run these without a new reason:

| tried | result |
|---|---|
| **MLX 0.31.2** (this session's main lead) | decode 27.9 -> 27.6. Nothing. |
| `mx::compile` on the elementwise clusters | 5.5x in isolation on compute_g; 28.1 -> 27.9 end to end. Reverted. |
| `MLX_MAX_OPS_PER_BUFFER` 50/200/500 | moves `chain` 1.9x, moves decode by 0.1 tok/s |
| CPU graph construction | under 1 ms/step, so `mx::async_eval` has nothing to hide |
| **dependent-op stall** | **chained 0.693 vs independent 0.647 — the dependency is innocent** |

That last row killed the theory the previous handoff was built on. MLX serializes onto
one stream either way, so "make the ops independent" buys nothing.

### The one live lead

`lmp_diag blocks 1` times forward_moe's pieces and they sum to **0.109 ms**. The block is
**0.647 ms**. Composing them costs 6x, and `lmp_diag moe 1` bisects where:

```
  1 gate                               0.025 ms/call
  2 + moe_topk                         0.027 ms/call   (+0.002)
  3 + switch_glu                       0.256 ms/call   (+0.229)
  4 + combine                          0.120 ms/call
  control: switch_glu, evaluated inds  0.093 ms/call
```

**`gather_qmm` fed an unevaluated index array costs ~2.5x what it costs fed an evaluated
one** — same kernel, same weights, same data. That is the only thing `blocks` did
differently (it hoisted `mx::eval({inds, scores})` out of its loop), and it is why
`blocks` and `layers` disagreed by 6x.

Stage 4 being *cheaper* than stage 3 is not noise to ignore either: stage 3 returns
[1,1,8,2048] and stage 4 reduces it to [1,1,2048]. Understand that before trusting the
absolute numbers in stages 3-4; the control row is the clean comparison.

### Where to go next

1. **Find out what MLX does differently for gather_qmm with an unmaterialized index.**
   Read `mx::gather_qmm`'s Metal path in the 0.31.2 source
   (`~/.venvs/lmp-mlx/lib/python3.12/site-packages/mlx/include/mlx/`, and the ops source
   upstream). Candidates: an index-dependent kernel-selection or output-shape decision
   that forces the scheduler to break the batch; a fallback path when indices are not
   contiguous/evaluated.
2. **Test the obvious workaround directly**: `mx::eval(inds)` inside `forward_moe` before
   `switch_glu`. It adds a sync per layer, so it may well be a net loss — but it is a
   three-line experiment that either confirms the mechanism or kills it, and the answer
   is worth more than the change.
3. **Compare against LM Studio's actual call shape.** Its engine source is on this
   machine and its `SparseMoeBlock`/`SwitchGLU` are structurally identical to ours:
   `~/.lmstudio/extensions/backends/vendor/_amphibian/`
   `app-mlx-generate-mac14-arm64@29/lib/python3.11/site-packages/mlx_lm/models/`.
   Same ops, same MLX, same machine, 2.8x the throughput — so the difference is in *how*
   they are issued, and it is now down to a small surface. Note `generate_step` runs
   inside `with mx.stream(generation_stream)` on a **dedicated stream** and uses
   `mx.async_eval`; the stream has not been tried.

Prefill is the same story — the scan is fixed, so prefill is MoE-bound too.

---

## Instruments

`tests/model/diag_main.cpp` (`cmake --build --preset dev --target lmp_diag`):

| | |
|---|---|
| `lmp_diag scan [T...]` | fused kernel vs reference loop: deviation and wall time |
| `lmp_diag mask` | one decode step outside the forward pass |
| `lmp_diag blocks [T]` | per-block cost, batched — latency / cpu / marginal |
| `lmp_diag moe [T]` | bisects forward_moe stage by stage; has the gather_qmm control |
| `lmp_diag layers [T]` | per-layer cost, chained AND independent, via the model's own blocks |
| `lmp_diag chain [n]` | what a dependent MLX op costs on this machine (5.2 us) |
| `lmp_diag bench [runs] [prompt] [max_new]` | N-run ledger against the LM Studio numbers |

`scripts/lmstudio_baseline.py` re-derives the other side.

`Qwen35MoeModel::forward_{linear_layer,full_attn_layer,moe}` are public so `layers` times
the code the model actually runs rather than a copy of it. `GenResult` carries
`forward_ms`, `logits_copy_ms`, `sample_ms`.

A warning about this driver, learned twice: **it is easy to write a measurement that
answers a different question than the one asked.** `blocks` hoisted an evaluated `inds`
and hid a 6x effect; the first `scan` fed unnormalised `k` and reported 1e16 deviations
that were entirely its own inputs; the first `blocks` charged every block a 0.17 ms eval
round-trip and concluded they all cost the same. When a number surprises you, suspect the
instrument before the code.

## Guardrails — all green on MLX 0.31.2

`ctest -L gate` 19/19 · `./scripts/run_ratchets.py --root .` 6/6 clean ·
`ctest -L realmodel` 2/2 (same 536 tokens generated as on 0.29.3) ·
`./scripts/eval.py --root . score` unmoved (corpus wmiss=0 179/179, holdout wmiss=15 34/42).

## Do not

- Do not delete `gated_delta_update_ops`, or the equivalence tests in
  `tests/model/test_grammar.cpp`. They are what make the fast paths falsifiable.
- Do not re-apply `mx::compile`, or re-test MLX versions or `MLX_MAX_OPS_PER_BUFFER`,
  without a new reason. All three were measured end to end and all three were zero.
- Do not weaken the grammar to make the mask cheap. Speed came from *how* the mask is
  computed; the tool-call automaton constrains exactly as much as it did before.
- Do not reach for speculative decoding. The MoE block is 27 ms of a 35 ms step; stacking
  a throughput trick on top would hide it, not fix it.
- Do not re-litigate settled decisions: Apple Silicon only, MLX in-process, Qwen3 only,
  XML tool-call syntax (see `docs/PHASES.md`).
