# HS1_27B — fat-prefix suffix-only prefill results

**Status:** **KEEP** — full 4-arm PASS on Mac wall (Benchbot)  
**Living 1-pager:** Research `HS1_27B_SUFFIX_ONLY_PREFILL_2026-09-25` (supersedes A3B plan)  
**Branch:** `hs1-suffix-prefill` · draft PR #227 · **do not merge**  
**Tip SHA:** `4f132ec`  
**Suite TS:** `20260925-085953`  
**Model:** dense **Qwen3.8-27B-MLX-4bit** — **not A3B**, no dual-load  
**GJF PR #226:** completely separate queue; this branch must not touch grammar jump-forward.

## Verdict

**KEEP.** Bars met at P=2048 and P=8192. Tip Extend/Restore algebra already delivers
suffix-only prefill on the honest harness. **Hold `LMP_SUFFIX_PREFILL` — not needed**
(measure-only; do not implement).

All four trap arms: `END ec=0 DIED_EARLY=0`.

## Tip env name (confirmed)

Harness reads **`LMP_QWEN_DIR`** (`tests/model/diag_common.hpp` → `qwen_dir()`).

```bash
export LMP_QWEN_DIR=/Users/dev/Desktop/Models/Qwen3.8-27B-MLX-4bit
```

## Product change

**None.** Measure-only KEEP. No product patch and **no `LMP_SUFFIX_PREFILL`**.

## P_shared definition

`P_shared` = length of the **byte-identical leading token span** of `turn1.prompt` and `turn2.prompt`.

Harness construction (honest; avoids the 2026-09-03 `reused≈11` failure):

1. Encode filler with `QwenTokenizer::encode_content`.
2. Truncate to **exactly P** tokens.
3. That vector is the leading P tokens of **both** turns.
4. `checkpoint_at = P`.

`reuse_frac = prefill_reused_tokens / P_shared`.

## Exact binary path + Benchbot commands

```text
/Users/dev/Desktop/seans_projects_local/LM_Pipe_2/build/tests/model/lmp_diag
```

Prefer each arm **FOREGROUND under the trap as direct parent** (see Dig note below):

```bash
export LMP_QWEN_DIR=/Users/dev/Desktop/Models/Qwen3.8-27B-MLX-4bit
cd /Users/dev/Desktop/seans_projects_local/LM_Pipe_2/build/tests/model

../../scripts/lmp_diag_trap.sh ./lmp_diag reuse 3 2048 128 32
../../scripts/lmp_diag_trap.sh ./lmp_diag reuse 3 8192 128 32
../../scripts/lmp_diag_trap.sh ./lmp_diag reuse --cold 3 2048 128 32
../../scripts/lmp_diag_trap.sh ./lmp_diag reuse --cold 3 8192 128 32
```

Raw logs: `piper-bench/results/hardware_squeeze/hs1-27b-*.txt` (suite TS `20260925-085953`).

## Pass bar

At **P=2048 and P=8192**, S=128, max_new=32, runs≥3 on **Qwen3.8-27B-MLX-4bit**:

| Check | Bar |
|-------|-----|
| median `reuse_frac` | ≥ 0.95 of `P_shared` |
| turn2 TTFT (live) vs cold full prefill of P+S | ≥ 5% better |

## B0 medians (suite `20260925-085953`, tip `4f132ec`)

| P | arm | median TTFT ms | median reuse_frac | vs cold | PASS/FAIL |
|---|-----|----------------|-------------------|---------|-----------|
| 2048 | live turn2 | 407.4 | 1.0 | ~94.5% better than cold | **PASS** |
| 2048 | cold P+S | 7371.1 | — | — | (contrast) |
| 8192 | live turn2 | 458.8 | 1.0 | ~98.3% better than cold | **PASS** |
| 8192 | cold P+S | 27571.0 | — | — | (contrast) |

- **SHA:** `4f132ec` (`hs1-suffix-prefill`)
- **Model:** Qwen3.8-27B-MLX-4bit
- **Trap:** all arms `END ec=0 DIED_EARLY=0`
- **Verdict:** **KEEP** — bars met; hold `LMP_SUFFIX_PREFILL`

## Dig note (brief)

Earlier suite `DIED_EARLY` was **parent/nohup process-group pressure**, not a product
decode hang. Lone `reuse 2` and `reuse 3` both PASS with full plain-path breadcrumbs
(`decode_begin speculative=0` → `decode_first_token`). Suite fix: run each arm
**FOREGROUND under `scripts/lmp_diag_trap.sh` as the direct parent** (not nohup’d into a
shared process group).

Plain-path breadcrumbs (measure-only, tip `4f132ec`):

- `decode_begin speculative=0 mtp=%d prompt=%zu`
- `decode_first_token ttft_ms=%.1f`

## CPU unit tests (gate; no Mac)

`tests/model/test_kv_reuse.cpp` — fat-prefix Restore-at-P + chat-header false-friend.

```bash
ctest --preset gate -R test_kv_reuse --output-on-failure
```

## Out of scope / parked

`LMP_SUFFIX_PREFILL` (not needed), QuantizedKV (HS2), MTP draft_cost_ratio, parsephony FF,
**GJF PR #226** (separate Mac micro queue), warm LSP, Pulse/OpenJev/tiny-gate, **A3B**,
second heavy model, GEMM, MLX fork, dual KV, middle-drop.
