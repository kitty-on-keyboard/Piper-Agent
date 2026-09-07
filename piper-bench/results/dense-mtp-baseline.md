# Dense 27B MTP — M0 baseline (M5 Pro 48 GB)

Quiet machine, one MLX process. Harness: `lmp_diag bench 5 512 256`.
Prompt tokenizes to **547**; TurnGrammar stops generation at **69–73** tokens (not 256).
Target: `/Users/dev/Desktop/Models/Qwen3.8-27B-MLX-4bit`.
MTP head: `/Users/dev/Desktop/Models/Qwen3.8-27B-MTP-4bit`.

Recorded 2026-09-05 ~07:44–07:48 MDT. Binary: `./build/tests/model/lmp_diag`.

## Commands

```bash
export LMP_QWEN_DIR=/Users/dev/Desktop/Models/Qwen3.8-27B-MLX-4bit
DIAG=./build/tests/model/lmp_diag
MTP=/Users/dev/Desktop/Models/Qwen3.8-27B-MTP-4bit

unset LMP_DRAFT_DIR LMP_SPECULATIVE
$DIAG bench 5 512 256                    # A

export LMP_DRAFT_DIR=$MTP LMP_SPECULATIVE=0
$DIAG bench 5 512 256                    # B

unset LMP_SPECULATIVE
export LMP_DRAFT_DIR=$MTP
$DIAG bench 5 512 256                    # C

unset LMP_DRAFT_DIR LMP_SPECULATIVE
$DIAG verify 4 547                       # verify-width microbench
```

## M0 table

| Arm | draft | spec | median decode | min–max | median TTFT | median prefill | peak mem | acceptance |
|---|---|---|---|---|---|---|---|---|
| A plain AR | no | off | **16.2 tok/s** | 15.6–16.9 | 5609 ms | 97.5 tok/s | 15.22 GB | n/a |
| B residency control | yes | `LMP_SPECULATIVE=0` | **16.1 tok/s** | 15.7–16.3 | 5607 ms | 97.6 tok/s | 15.45 GB | n/a |
| C MTP | yes | on (auto) | **20.5 tok/s** | 19.5–23.0 | 5899 ms | 97.1 tok/s | 15.45 GB | **65.6%** typical |

MTP stderr (median-like run 4): `blocks=31 drafted=61 accepted=40 (65.6%) committed=71 fallbacks=2`.

Per-block `SpecPhases` (run 4):

| verify | forward_last | mtp_logits | mtp_step | hidden | checkpoint | restore |
|---|---|---|---|---|---|---|
| **81.3 ms** | 15.3 ms (0.5 pos) | 9.2 ms (2.1 calls) | 4.6 ms | 0.4 ms | 5.2 ms | 0.0 ms |

Accounted ≈ 116 ms/block × 2.29 committed tokens/block ≈ 20 tok/s.

`lmp_diag verify 4 547` (eval only, no MTP, 10 iters):

| k | ms/pass | vs k=1 |
|---|---|---|
| 1 | 67.41 | 1.00× |
| 2 | 72.88 | 1.08× |
| 3 | 79.05 | 1.17× |
| 4 | 86.94 | 1.29× |

**GO.** AR is in the 15–20 band. MTP 20.5 is within ±20% of Sean’s ~25 (and matches an earlier in-tree note of 23.4 tok/s at 547 tokens). MTP > AR. Phases and acceptance printed.

## M1 — tax attribution

We are leaving **~81 ms/block on the batched target verify**, which is 70% of block wall and matches `lmp_diag verify` at **k=3** (79 ms). Every deferred-bonus block is `[bonus, d0, d1]`, so verify is a 3-position dense forward, not a host-copy problem (eval-only k=3 is 79 ms; e2e verify is 81 ms). AR is ~62 ms/token; k-scaling is already healthy (k=2 is 1.08× k=1). The remaining 35 ms/block is `forward_last` flushes/fallbacks (15 ms), MTP LM-head + layer (14 ms), and checkpoint (5 ms). Killing every non-verify bucket still leaves 2.29 / 0.081 ≈ **28 tok/s**. Perfect accept (3 tok/block) at k=3 with zero overhead is 3 / 0.079 ≈ **38 tok/s**. **≥40 needs near-full accept and essentially no overhead; ≥50 needs a faster 27B forward than MLX’s ~67 ms k=1, which is not a narrow verify/append op.**

## M2 — smallest sidecar change

Did not rewrite proposer algebra or the MoE graph. Two dense-path changes:

1. **Batched checkpoint eval** in [`src/model/mlx/qwen35_moe_model.hpp`](../../src/model/mlx/qwen35_moe_model.hpp): one `mx::eval` of all live SSM/PLE handles instead of ~60 per-layer `snapshot()` evals.
2. **Fused greedy MTP** in [`src/model/mlx_backend.cpp`](../../src/model/mlx_backend.cpp): `mtp_step_greedy` / `mtp_argmax` run layer + LM head + argmax in one eval and keep the vocab row off-host. Gate fakes still use the default `mtp_step` + `mtp_logits` path ([`speculative.cpp`](../../src/model/speculative.cpp), [`mtp_proposer.cpp`](../../src/model/mtp_proposer.cpp)). `test_mtp_proposer` and `test_speculative` PASS.

Same M0 C harness after the change:

| | median decode | verify | forward_last | mtp_logits | mtp_step | checkpoint |
|---|---|---|---|---|---|---|
| M0 C | 20.5 tok/s | 81.3 ms | 15.3 ms | 9.2 ms | 4.6 ms | 5.2 ms |
| M2 C | **22.8 tok/s** | 77.0 ms | 14.3 ms | 10.7 ms (fused) | 2.6 ms | **0.3 ms** |

Checkpoint tax is gone (5.2 → 0.3 ms). Decode **20.5 → 22.8 tok/s**. Acceptance unchanged (65.6%) — distribution-preserving path untouched. Still far from ≥40.

Verify remains ~77 ms/block (k=3 MLX forward). `mtp_logits` did not shrink in wall time because the 0.7 GB head still runs; we only stopped copying 248k floats to the host.

## M3 — custom Metal op: not taken

M2 plateaued at **22.8 tok/s** (low-20s). A narrow sidecar verify/append kernel cannot replace the 40-layer 27B forward that `lmp_diag verify` already prices at 67 ms (k=1) / 79 ms (k=3). MLX *can* express the batched verify; it is just that expensive on this quant. ≥50 tok/s is a faster generic matmul / packing problem (explicit non-goal). Stopped rather than thrash.

## M4 — agent smoke

Dense + MTP, `LMP_DRAFT_DIR` set, spec on:

- **Constrained** (`bench`, TurnGrammar): M2 table above. status=0, 2 fallbacks / 31 blocks, grammar completed the turn. Abandoned did not explode.
- **Plain completion** (`lmp_diag smoke 32`): think-open decode 29.2 tok/s, 83.3% accept, fallbacks=0, coherent “Paris”. think-closed 23.3 tok/s, 63.3% accept, answers Paris then unconstrained special-token chatter (no grammar).
- **KV reuse** (`lmp_diag reuse 1 256 64 16`): both turns status=0, turn2 `reused=11` (reuse_on path still fires with the draft head loaded).

MoE regression, **no** MTP dir, A3B `Qwen3.6-35B-A3B-MLX-4bit`:

- `lmp_diag smoke 32`: load 7.9 s, decode **83.4 / 83.0 tok/s**, “Paris”, peak 19.00 GB. MoE path untouched.

`agent_eval` / Aider 49-task bakeoff not run (handoff: not until tok/s + TTFT are in the target band).

## Report (handoff §8)

1. **M0:** AR 16.2 tok/s, spec-off+draft 16.1, MTP 20.5, acceptance 65.6%, phases printed. Lengths: prompt 547, completion 69–73 (grammar), 5 runs.
2. **M1:** ~81 ms/block on k=3 target verify (70% of the block). Overhead 35 ms. Ceiling with perfect accept ≈ 38 tok/s on this MLX forward.
3. **Diff:** `qwen35_moe_model.hpp` batched checkpoint eval; `mlx_backend.cpp` fused GPU greedy MTP; `speculative.hpp/.cpp` + `mtp_proposer.cpp` seam (`mtp_step_greedy` / `mtp_argmax`). Gate tests green.
4. **New tok/s:** **22.8 median** vs ≥40 / ≥50. Miss. Remaining gap is the 27B 4-bit forward (~67 ms/token), not proposer algebra.
5. **MoE:** smoke 83 tok/s, no draft head. Untouched.
6. **Deliberately not done:** HTTP dual-backend, MoE+MTP, MLX fork / custom GEMM, M3 Metal op, qwen4_exp, ToolError rewrite, full Aider bakeoff.
