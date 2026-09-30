# HS2 — QuantizedKV B0/B1 Mac A/B (harness only)

Product QuantizedKV already ships on tip (`src/model/mlx/kv_cache.hpp`, adopted from
`forward_self_attn` in `src/model/mlx/qwen35_moe_model.hpp`). Default stays **OFF**.
This note is the Benchbot attribution recipe so B1 cannot be a silent no-op.

## Env recipe

| Arm | Env |
| --- | --- |
| **B0** (baseline) | Leave `LMP_KV_BITS` unset (or `0`). |
| **B1** (8-bit KV) | `export LMP_KV_BITS=8` — leave `LMP_KV_QUANT_AFTER` at default **8192**. |

Same tip/seed, long Godoer-class / high-context **27B** mission, **MTP ON**. Skip B2
unless split K/V bits are already exposed (they are not).

## Grep-able adopt breadcrumb

On the first successful adopt in a process (once per session, not per layer):

```text
kv_quant adopt bits=8 after=8192 offset=<n> gs=64
```

Hook: `Qwen35MoeModel::forward_self_attn` immediately after `qcache.adopt_from(...)`.

## Memory / context lines (existing MLX helpers)

Decode begin/end print real MLX allocator bytes via `log_mlx_mem` /
`mlx_memory_report()` (active / cache / peak — not invented GB):

```text
mem at=decode_begin tokens=<context> active=… cache=… peak=… sum=…
mem at=decode_end tokens=<context> active=… cache=… peak=… sum=…
```

`tokens=` is current ledger / context length. No process-RSS or Metal APIs were
invented for this harness; wire only what tip already exposes.

## Footguns (adopt never fires)

Adopt runs only when **all** of:

1. `LMP_KV_BITS > 0`
2. context `cache.offset >= LMP_KV_QUANT_AFTER` (default **8192**)
3. first **decode** step with `L == 1` (never during prefill)

If the mission never reaches 8192 tokens, or never enters single-token decode after
that, B1 is a no-op even with `LMP_KV_BITS=8` — the adopt breadcrumb will be absent.
