# Handoff: make LM_Pipe v2 faster than LM Studio

Paste this whole file as the opening prompt of a fresh session in
`/Users/dev/Desktop/seans_projects_local/LM_Pipe_2` (branch `main`, all work merged).

---

## The job

We run Qwen3.6-35B-A3B in-process via MLX. LM Studio runs the **same checkpoint on the
same machine** through a generic stack. We are losing badly. Fix it.

We have advantages LM Studio does not: no HTTP hop, no process boundary, direct MLX
calls, our own SPSC transport, and a KV ledger we control. **Losing is not acceptable
and matching is not the target — beating these numbers is the minimum bar.**

| | LM Studio (measured) | Ours (measured) | Gap |
|---|---|---|---|
| decode | **78.5 tok/s** median (n=133) | 22.9 | **3.4× slower** |
| prefill | **1347 tok/s** median (n=62) | 19.2 | **70× slower** |

Exit criterion: **decode > 78.5 tok/s and prefill > 1347 tok/s**, reported as an N-run
ledger (≥3 runs, median + spread), on the same prompt shape. Do not report a single
number — v1 measured 462 s, then 275 s and 258 s on an identical binary (spec §11.5).

## Reproduce both sides before changing anything

Re-measure first (§19.6). Do not trust the numbers above because they are written down.

**Our numbers** — `tests/model/test_realmodel.cpp::model_generates_a_grammatical_turn`
prints a `[perf]` line:
```bash
cmake --preset dev && cmake --build --preset dev -j8
cd build && ctest -L realmodel -j1 --output-on-failure 2>&1 | grep perf
```

**LM Studio's numbers** — derived from `~/.lmstudio/server-logs/*/*.log`, which contain
real prior usage of this exact model. Timestamps are 1-second resolution, so short
requests carry real error; that is why the medians above use n=133 / n=62. The
extraction script is in this session's history; rewrite it, don't trust my summary.
Fields: `Prompt processing progress: 0.0%` / `100.0%` timestamps, `uncached_tokens=`,
`"completion_tokens":`, and the `Generated prediction` timestamp.

## Three confirmed causes, measured

### 1. The grammar mask costs 22.8 ms/token and rejects 8 tokens (decode killer)

Measured on this machine with `tests/model/diag_main.cpp`:

```
vocab = 248077
mask over full vocab       :  27.6 ms/token   (248069 of 248077 allowed)
sampler, no mask           :   1.7 ms/token
sampler + mask (as shipped):  22.8 ms/token
=> CPU-side ceiling from this alone: 43.8 tok/s
```

`Sampler::sample` (`src/model/sampler.cpp:131`) calls `mask(id)` for **every one of
248,077 ids, every token**. Each call lands in `TurnGrammar::permitted`
(`src/model/grammar.cpp:104`), and in the Think/Text phases that path does:

```cpp
TurnGrammar probe(tok_, tools_);   // ctor -> reset() -> make_unique<ToolCallGuard>(tools_)
```

**A heap allocation and a full ToolCallGuard construction, 248,077 times per token**, to
reject 8 structural ids.

The fix is not a micro-optimisation. In Think/Text the mask is "everything except a
handful of structural ids" — that is a tiny denylist, computable without touching the
vocabulary at all. Only inside `TurnPhase::ToolCall` does it need real work, and
**parsephony already ships the engine for that and we never wired it up**:
`third_party/parsephony/include/parsephony/mask.hpp`, `TokenMaskT`, which caches masks
by `state_signature()`, pre-classifies string-safe tokens, and buckets candidates by
first byte. Its own repo measures **17.7 ns per sampling step**. `ToolCallGuard` already
implements the full contract it needs (`state_signature`, `mask_class`, `allowed_bytes`,
`probe_byte`, `mute`) — verified.

Also in `src/model/sampler.cpp`: `apply_top_p` does a full `std::sort` over all 248,077
indices per token. Partial selection (`nth_element`) over the top-k survivors is enough,
and top-k already ran.

### 2. Prefill runs one MLX op-set per token position (prefill killer)

`src/model/mlx/gated_delta.hpp`, `gated_delta_update`:

```cpp
for (int t = 0; t < T; ++t) { ... gated_delta_step(...) ... }
```

30 of 40 layers take this path (`is_linear_layer` = `(idx+1) % 4 != 0`). A 287-token
prompt is therefore ~8,600 sequential kernel launches where LM Studio issues a handful
of batched ones. Decode is unaffected (T=1, one iteration) — which is exactly why decode
is 3.4× off and prefill is 70× off. That asymmetry is the evidence this is the cause.

The gated-delta recurrence is sequential in principle but has the standard associative-
scan structure: process in chunks, batch the intra-chunk math, carry state between
chunks. Reference implementations chunk it. **Numerics must not change** — this code was
debugged against this exact checkpoint. Prove equivalence against the current
implementation on a fixed seed before trusting any speedup.

### 3. Full logits row copied GPU→CPU every step

`src/model/mlx_backend.cpp`, `logits_to_host`: `mx::eval` + copy of all 248,077 floats
(~1 MB) per decode token, so the CPU sampler can run. Options: mask and sample
on-device; or narrow to top-k on-device and copy only that. This is the smallest of the
three — measure it before spending time on it.

## Do not

- **Do not change the numerics** in `src/model/mlx/` to gain speed. It is v1's debugged
  forward pass for this checkpoint. Equivalence first, speed second.
- **Do not weaken the grammar** to make the mask cheap. A malformed tool call must stay
  unrepresentable (§5.6). Speed comes from *how* the mask is computed, not from
  constraining less.
- **Do not quote a throughput number without a baseline next to it.** That is the
  mistake that produced this handoff: 22.9 tok/s was reported as a working result
  without ever comparing it to the LM Studio numbers already sitting on this machine
  (§16: no metric quoted without checking what it counts).
- Do not re-litigate settled decisions: Apple Silicon only, MLX in-process, Qwen3 only,
  XML tool-call syntax (the model's own `chat_template.jinja` — see
  `docs/PHASES.md` for why the spec's §5.6 JSON form was overruled).

## Guardrails

`ctest -L gate` (19 tests, ~6 s) must stay green, and all six ratchets
(`./scripts/run_ratchets.py --root .`) must stay clean. Both are required to merge.

Watch these specifically while optimising:
- `test_grammar_realmodel` pins that `permitted()` and `advance()` agree. A fast mask
  that disagrees with the walk is a mask that lies to the sampler — that test is the
  one that catches it. Keep it.
- `test_backend_seam::mask_is_applied_before_shaping` pins mask-before-top-k ordering.
  Masking after top-k can leave zero legal candidates.
- The bake-off pins (`./scripts/eval.py --root . score`) must not move.

## Also worth knowing

- **Prefix caching across turns.** LM Studio's logs show `cached_tokens` and
  `lifetime_efficiency`. We have `KvCacheLedger` doing verified id-by-id prefix reuse,
  but the agent loop rebuilds the prompt every iteration and the sidecar constructs a
  fresh backend per run. In a loop that re-sends a growing conversation, that is a large
  repeated prefill cost — and it compounds with cause #2.
- `tests/model/diag_main.cpp` is a scratch driver (built via
  `cmake --build --preset dev --target lmp_diag`, EXCLUDE_FROM_ALL). It currently holds
  the mask/sampler micro-benchmark above. Overwrite it freely; it exists so a failure
  can be attributed by observation rather than guessed at (§19.3).
- `docs/PHASES.md` lists what else is unfinished (T2 containers, speculative decoding,
  the approval round-trip, `.vsix` packaging). **None of that is this session's job.**
  Speculative decoding in particular is a tempting throughput lever — it is not the
  bug, and stacking it on top of a 22 ms/token mask would only hide the defect.
