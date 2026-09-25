# HS1 — fat-prefix suffix-only prefill results

**Status:** harness landed; Mac wall numbers **pending Benchbot**  
**Branch:** `hs1-suffix-prefill`  
**Model (retarget, Sean):** `Qwen3.8-27B-MLX-4bit` at `~/Desktop/Models/Qwen3.8-27B-MLX-4bit`  
**Not A3B.** Cloud does not load this checkpoint.

## Product change

**Measure-only KEEP-seed.** Tip `plan_turn_reuse` Extend/Restore algebra is what the
honest harness exercises. **No `LMP_SUFFIX_PREFILL` flag and no product patch** unless
Benchbot B0 fails the pass bar below. If a fix is later required: gate it behind
`LMP_SUFFIX_PREFILL` default 0 at `plan_turn_reuse` / `ReuseMode` / generate prefill /
`kv_cache` — follow the tree. No MLX fork, dual KV, middle-drop, or GEMM.

## P_shared definition

`P_shared` is the length of the **byte-identical leading token span** shared by
`turn1.prompt` and `turn2.prompt`.

The harness constructs it explicitly (this is the whole point of HS1):

1. Encode filler text with `QwenTokenizer::encode_content`.
2. Truncate to **exactly P** tokens.
3. Use that vector as the leading P tokens of **both** turns.
4. Set `checkpoint_at = P`.

The 2026-09-03 failure (`reused = 11` every time) checkpointed at a chat-template
message boundary (system header ≈ 11 tokens), not at the end of a fat shared body.
Do **not** wrap this harness in `ChatTemplate` and call that HS1.

`reuse_frac = prefill_reused_tokens / P_shared`.

## Binary path (Benchbot)

```text
<repo>/build/tests/model/lmp_diag
```

Already wired in `tests/model/CMakeLists.txt` as `EXCLUDE_FROM_ALL` target `lmp_diag`.

```bash
cmake --preset dev
cmake --build --preset dev --target lmp_diag -j8
```

## Benchbot commands (27B dense)

```bash
export LMP_QWEN_DIR=$HOME/Desktop/Models/Qwen3.8-27B-MLX-4bit
# Abort if another MLX holder is live before loading 27B.
cd <repo>/build/tests/model   # real binary dir after cmake --preset dev

# Live KV — turn2 should reuse ≈ P (reuse_frac >= 0.95)
./lmp_diag reuse 3 2048 128 32
./lmp_diag reuse 3 8192 128 32

# Alias
./lmp_diag suffix 3 2048 128 32

# Cold contrast — full KV reset each run; TTFT is full prefill of P+S
./lmp_diag reuse --cold 3 2048 128 32
./lmp_diag reuse --cold 3 8192 128 32
```

Per turn the harness prints: TTFT ms, `prefill_reused_tokens`, `reuse_frac` vs
`P_shared`, reuse mode/reason, decode tok/s.

Raw logs (suggested): `piper-bench/results/hardware_squeeze/hs1-*.txt`.

## Pass bar (Mac wall = Benchbot on 27B dense)

At P=2048 and P=8192, S=128, max_new=32, runs≥3 on **Qwen3.8-27B-MLX-4bit**:

| Check | Bar |
|-------|-----|
| median `reuse_frac` | ≥ 0.95 of `P_shared` |
| turn2 TTFT (live) vs cold full prefill of P+S | ≥ 5% better |

## B0 medians (stub — fill on Mac wall)

| P | arm | median TTFT ms | median reused | median reuse_frac | mode/reason | PASS/FAIL |
|---|-----|----------------|---------------|-------------------|-------------|-----------|
| 2048 | live turn2 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 2048 | cold P+S | _TBD_ | 0 | — | Reset | _TBD_ |
| 8192 | live turn2 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 8192 | cold P+S | _TBD_ | 0 | — | Reset | _TBD_ |

- **SHA:** _TBD after Benchbot run_
- **Model:** Qwen3.8-27B-MLX-4bit
- **Verdict:** _PENDING measure-only_

## CPU unit tests

`tests/model/test_kv_reuse.cpp`:

- `hs1_fat_prefix_of_P_then_suffix_S_restores_exactly_P` — Restore at P, reuse_frac ≥ 0.95
- `hs1_checkpoint_at_chat_header_only_reuses_header_not_fat_body` — pins the ≈11-token false-friend

```bash
ctest --preset gate -R test_kv_reuse --output-on-failure
```

## Out of scope / parked

QuantizedKV, MTP draft_cost_ratio, parsephony FF / GJF, warm LSP, Pulse/OpenJev/tiny-gate,
**A3B**, second heavy model, GEMM, MLX fork, dual KV, middle-drop. Do not touch GJF PR #226.
