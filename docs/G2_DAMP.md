# G2 DAMP — delta-state asymmetric mixed precision (stub)

Research offline KEEP: Top-16 error-energy ≫ uniform on 30/30 GDN layers
(global mean Top-16 share 74.85% vs 12.5% uniform); `a_eff` not collapsed.
G1 (MoE wrap) was KILL and is closed. This is the next one-at-a-time queue item.

## What this is / is not

| | |
|---|---|
| **Is** | Quantize **`delta_state` only** with a static 2-tier mask: protect Top-K_hi key channels per head (default K_hi=16, Dk=128) ranked by offline error-energy. |
| **Seam** | `SsmCache` / `forward_gated_delta` / `gated_delta` — **not** `QuantizedKVCache`. |
| **Stub** | Store mixed precision; **compute in FP32 after dequant** (proves quality/GB). tok/s / fused Metal later. |
| **Not** | G1/MoE, SSD KV, SA QuantizedKV, `conv_state`, Hadamard fused kernel v1 (plain INT8 compressed tier is OK). |

## Kill switch (default OFF)

| Env | Meaning | Default |
|---|---|---|
| `LMP_DAMP` | Enable pack/dequant around GDN `delta_state` | unset → **off** (identical to main) |
| `LMP_DAMP_K_HI` | Protected channels per head | `16` |
| `LMP_DAMP_MASK` | Path to `g2_damp_mask_v1` JSON | required when `LMP_DAMP=1` on the live path |

Unset `LMP_DAMP` → no pack path runs. Product decode stays the FP32 recurrent-state path.

## Mask artifact format (`g2_damp_mask_v1`)

```json
{
  "format": "g2_damp_mask_v1",
  "source": "research_cal",
  "k_hi": 16,
  "dk": 128,
  "num_layers": 40,
  "num_heads": 32,
  "layers": [
    {
      "layer": 0,
      "heads": [
        [/* k_hi unique Dk indices for head 0 */],
        [/* head 1 */]
      ]
    }
  ]
}
```

- `num_heads` is **Hv** (value heads); `delta_state` is `[B, Hv, Dv, Dk]`.
- Indices are into the last axis `Dk`. Loader sorts and rejects duplicates / OOB.
- `source` must say where the rankings came from. **Do not invent fake production energy rankings.**

### Dropping in Research cal

Research’s measured dump lives on Research’s box (historically
`/workspace/research/g2_damp_cal/`), not in this repo by default.

1. Convert the cal dump to `g2_damp_mask_v1` JSON (same schema as above; `source` =
   `"research_cal"` or a dated cal id).
2. Point the live path at it: `LMP_DAMP=1 LMP_DAMP_MASK=/path/to/mask.json`.
3. Checked-in `tests/fixtures/damp/synthetic_mask.json` is **synthetic only** — sequential
   indices for unit tests, not measured energy.

## Stub packing

- **Protected tier:** float32 (exact).
- **Compressed tier:** plain INT8 affine, one scale per `(b, h, dv)` over the compressed
  channels only (`absmax/127`).
- Between steps the live path may keep the packed form on `SsmCache` and materialize FP32
  only for `gated_delta_update`. Host round-trip is acceptable for this stub; Metal/dequant
  speed is a kill bar for later A/B, not this PR.

## Hypothesis / kill

Hypothesis: 2-tier `delta_state` keeps greedy-identical (or within noise) vs FP32 state on
golden 2K/8K and frees measurable recurrent-state GB (or ≥5% decode if packed path exists).

Kill: greedy/logits disagree outside noise OR Metal/dequant slower than FP32 on UMA OR GB
win in noise at batch=1 → leave GDN untouched; flags can stay research-only or close PR.

**Mac A/B is Benchbot/Research after land — not this PR.**

## Code map

| File | Role |
|---|---|
| `src/model/mlx/damp.hpp` | Env, mask load, CPU quant/dequant |
| `src/model/mlx/kv_cache.hpp` | `SsmCache` optional `damp_delta` |
| `src/model/mlx/qwen35_moe_model.hpp` | `forward_gated_delta` pack/dequant when enabled |
| `tests/model/test_damp.cpp` | Gate: off==baseline, mask load, roundtrip |
| `tests/fixtures/damp/synthetic_mask.json` | Synthetic fixture |
