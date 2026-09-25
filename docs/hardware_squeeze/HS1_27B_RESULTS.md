# HS1_27B — fat-prefix suffix-only prefill results

**Status:** harness green for Benchbot · Mac wall numbers **pending** (AE does not burn Mac)  
**Living 1-pager:** Research `HS1_27B_SUFFIX_ONLY_PREFILL_2026-09-25` (supersedes A3B plan)  
**Branch:** `hs1-suffix-prefill` · draft PR only · **do not merge**  
**Model:** dense **Qwen3.8-27B-MLX-4bit** — **not A3B**, no dual-load  
**GJF PR #226:** completely separate queue; this branch must not touch grammar jump-forward.

## Tip env name (confirmed)

Harness reads **`LMP_QWEN_DIR`** (`tests/model/diag_common.hpp` → `qwen_dir()`).

```bash
export LMP_QWEN_DIR=/Users/dev/Desktop/Models/Qwen3.8-27B-MLX-4bit
```

(Confirm the folder exists on the Mac wall; abort if another MLX holder is live.)

## Product change

**Measure-only KEEP-seed (B0).** No product patch and **no `LMP_SUFFIX_PREFILL`** unless B0 fails the pass bar. If a fix is required later: minimal planner/ledger/cache-append change behind **`LMP_SUFFIX_PREFILL` default 0** (`=1` to enforce). No MLX fork, dual KV, middle-drop, GEMM, MoE, or GJF edits.

## P_shared definition

`P_shared` = length of the **byte-identical leading token span** of `turn1.prompt` and `turn2.prompt`.

Harness construction (honest; avoids the 2026-09-03 `reused≈11` failure):

1. Encode filler with `QwenTokenizer::encode_content`.
2. Truncate to **exactly P** tokens.
3. That vector is the leading P tokens of **both** turns.
4. `checkpoint_at = P`.

Do **not** wrap this in `ChatTemplate` and call it HS1 — that was the system-header false-friend.

`reuse_frac = prefill_reused_tokens / P_shared`.

## Exact binary path + Benchbot commands

After build on the Mac checkout (AE does **not** burn wall):

```text
/Users/dev/Desktop/seans_projects_local/LM_Pipe_2/build/tests/model/lmp_diag
```

(If the checkout path differs, use `<repo>/build/tests/model/lmp_diag` from `cmake --preset dev`.)

```bash
# Abort if another MLX holder is live. One model only.
export LMP_QWEN_DIR=/Users/dev/Desktop/Models/Qwen3.8-27B-MLX-4bit

cmake --preset dev
cmake --build --preset dev --target lmp_diag -j8
cd /Users/dev/Desktop/seans_projects_local/LM_Pipe_2/build/tests/model

# B0 live KV — turn2 should reuse ≈ P
./lmp_diag reuse 3 2048 128 32
./lmp_diag reuse 3 8192 128 32

# Alias
./lmp_diag suffix 3 2048 128 32

# Optional stretch if RAM allows on 48GB with 27B resident:
# ./lmp_diag reuse 3 32768 128 32

# Cold contrast — full KV reset each run; TTFT = full prefill of P+S
./lmp_diag reuse --cold 3 2048 128 32
./lmp_diag reuse --cold 3 8192 128 32
```

Per turn prints: TTFT ms, `prefill_reused_tokens`, `reuse_frac` vs `P_shared`, reuse mode/reason, decode tok/s.

### Raw logs

Prefer (when tree is live):

```text
piper-bench/results/hardware_squeeze/hs1-27b-*.txt
```

One-row verdict lands in **this file** (`HS1_27B_RESULTS.md`) on first burn.

## Pass bar (Mac wall = Benchbot on 27B dense)

At **P=2048 and P=8192**, S=128, max_new=32, runs≥3 on **Qwen3.8-27B-MLX-4bit**:

| Check | Bar |
|-------|-----|
| median `reuse_frac` | ≥ 0.95 of `P_shared` |
| turn2 TTFT (live) vs cold full prefill of P+S | ≥ 5% better |

Primary P∈{2048,8192}. Optional P=512 / P=32768 for characterization only.

## B0 medians (stub — Benchbot fills)

| P | arm | median TTFT ms | median reused | median reuse_frac | mode/reason | PASS/FAIL |
|---|-----|----------------|---------------|-------------------|-------------|-----------|
| 2048 | live turn2 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 2048 | cold P+S | _TBD_ | 0 | — | Reset | _TBD_ |
| 8192 | live turn2 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 8192 | cold P+S | _TBD_ | 0 | — | Reset | _TBD_ |

- **SHA:** _TBD after Benchbot run_
- **Model:** Qwen3.8-27B-MLX-4bit
- **Verdict:** _PENDING measure-only_

## CPU unit tests (gate; no Mac)

`tests/model/test_kv_reuse.cpp`:

- fat prefix P + suffix S → Restore at P, reuse_frac ≥ 0.95
- checkpoint at chat-header-only → pins ≈11-token false-friend

```bash
ctest --preset gate -R test_kv_reuse --output-on-failure
```

## Out of scope / parked

QuantizedKV (HS2), MTP draft_cost_ratio, parsephony FF, **GJF PR #226** (separate Mac micro queue), warm LSP, Pulse/OpenJev/tiny-gate, **A3B**, second heavy model, GEMM, MLX fork, dual KV, middle-drop.
