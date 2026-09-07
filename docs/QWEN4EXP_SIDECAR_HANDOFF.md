# Qwen4_exp / Flash-Next sidecar port — handoff for Grok

You are implementing this for Sean's C++ MLX sidecar. Be conservative: preserve the working Qwen3.5/3.8 path and prove every new numeric path against the canonical reference.

## 0. Non-negotiables

- Repo: `/Users/dev/Desktop/seans_projects_local/LM_Pipe_2`.
- Branch: `fix/greenfield-write-caps`.
- Working tree is dirty (26 files); HEAD is approximately `2a568b8` (Merge PR #54).
- Do not clean, stash, reset, checkout, or overwrite unrelated work.
- Hardware is an M5 Pro with 48 GB unified memory. Never load two 35B-class models.
- Existing models live under `/Users/dev/Desktop/Models/`.
- Do not reimplement the known bugs listed in §Known bugs.
- Prefer extending `Qwen35MoeModel` over a polymorphic rewrite (map risk #10).
- Milestone 0 is a GO/NO-GO gate. If Flash is near the dense 27B (~20 tok/s), stop and report before writing C++.
- Correctness is measured against `transformers` at tiny scale, never by “does the text look fluent.” Every documented bug still generates fluent text.

Why this lines up: Sean already ported Piper for Qwen3.8-27B (GDN hybrid). Flash-Next / `qwen4_exp` reuses that skeleton. This is an extension, not a rewrite.

Product goal:
1. Determine whether Flash beats the dense 27B (~20 tok/s) and is smarter than A3B (Piper 33/34 on the Aider Python seed-7 run).
2. More importantly, be ready when Qwen4 ships on this architecture.
A3B remains the bakeoff baseline until Flash wins on the harness.
The sidecar currently only speaks Qwen3.5-family `model_type`s; `qwen4_exp` is refused today.

## 1. Goal & success criteria

- Primary: load the `qwen4_exp` text backbone, with correct prefill and decode, while PLE is offloaded rather than resident.
- Secondary: MTP is optional; omit it in v1.
- “Done” requires tiny-scale, module-by-module parity with `transformers` (§5), then live tok/s plus an Aider smoke test on the REAP-320 Q2 GGUF through llama.cpp FIRST, and only then (if worthwhile) native sidecar on a real MLX quant.
- Do not silently download a 100GB+ checkpoint or claim full-expert MLX fits 48GB.

## 2. What already exists (do not rebuild)

The detailed source survey is `/workspace/lmpipe2-sidecar-map.md`; it is not available inside Grok's repo, so the relevant evidence is reproduced here.

| Existing capability | Location/evidence |
|---|---|
| Gated DeltaNet, including op reference and custom Metal scan kernel | `src/model/mlx/gated_delta.hpp:31-291`; dispatch at `:281-283` |
| GDN model block, causal depthwise conv, q/k/v split, A_log/dt_bias, gated output norm | `src/model/mlx/qwen35_moe_model.hpp:727-810` |
| GDN recurrent `SsmCache` snapshot/restore | `src/model/mlx/kv_cache.hpp:91-142` |
| MoE router, top-k, shared expert, `shared_expert_gate` | `qwen35_moe_model.hpp:255-286`; `switch_glu.hpp` |
| Affine quantization and expert `gather_qmm` | `weight_store.hpp:33-46,142-172`; `switch_glu.hpp:18-71` |
| Softmax attention, partial RoPE, q/k norms, GQA and output gate | `qwen35_moe_model.hpp:835-909`; quantized GQA in `quant_attention.hpp:60-77` |
| Hybrid layer loop | `qwen35_moe_model.hpp:652-673`, dispatch specifically `:660-666` |
| KV append-in-place, quantized KV, SSM rollback | `src/model/mlx/kv_cache.hpp:19-324` |
| Prefix reuse (`Extend`/`Restore`/`Reset`) and `reuse` diagnostic | `mlx_backend.cpp:833-887`; `tests/model/diag_main.cpp:1218-1416` |
| MTP load/forward infrastructure | `qwen35_moe_model.hpp:350-434`; gated to dense at `mlx_backend.cpp:234-241` |
| Weight loading, `embed_lookup`, tied logits | `weight_store.hpp:50-78,176-200` |
| `lmp_diag scan` no-checkpoint Metal-vs-reference pattern | `tests/model/diag_main.cpp:142-207` |

The current modulo schedule happens to match Flash: 48 layers = 12 repetitions of 3 GDN + 1 QSA. But the loader ignores authoritative `text_config.layer_types` (`qwen35_moe_config.hpp:54,66-68`; loop `qwen35_moe_model.hpp:660-666`).

## 3. What is missing (build these)

- QSA indexer and non-contiguous KV gather.
- Gated Residual with `hc_count=4`, `hc_lowrank=320`.
- PLE/n-gram table: 128 shards, applied at 0-indexed layer 1, out-of-core.
- `model_type` allowlist entries for `qwen4_exp` and `qwen4_exp_text`.
- Parse `layer_types` from config instead of relying on modulo (or verify modulo and retain fallback).
- Out-of-core weight machinery (mmap/lazy/prefetch): the hard blocker. `WeightStore::load_directory` is eager/all-resident and `wired_limit` pins the working set.

## 4. Milestones (ordered, with go/no-go)

### M0 — Prove the model is worth the port (NO C++)

- Download only after Sean approves:
  `hf download AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF --include 'Q2/*.gguf' --local-dir ~/models/Qwen3.8-Flash-Next-REAP-320-Q2-GGUF`
- Use current llama.cpp master with qwen4exp merge `6c84c7d5d8833c6e0df69628f75a0f599797934e`, fixes `36b10154383b60eb15baac2c7a40d2a5f784faa7`, and `--lazy-mode`.
- Recommended Q2 payload is 61,492,913,728 bytes (~61.493 GB), with ~32.693 GB non-PLE and ~28.8 GB PLE; it retains 320/512 experts and is lossy.
- Measure with `-c 8192 --lazy-mode on --parallel 1`, no MTP, on the M5 Pro. Start conceptually:
  `llama-server -m <first-Q2-shard> --lazy-mode on --load-mode mmap -ngl all -c 8192 --parallel 1 --jinja --reasoning-effort low`
- GO only if decode is clearly above dense 27B (~20 tok/s), preferably approaching A3B (~70 tok/s).
- NO-GO if <=~25 tok/s: report measurements and stop before C++.
- Optional cheap quality probe: one Aider Python task through OpenAI-compatible `llama-server`, compared to known A3B. Do not start a 34-task bakeoff.

Done when: Sean has a recorded M5 Pro tok/s result and an explicit GO decision; no C++ has been changed.

### M1 — Config + allowlist (tiny, GPU-free)

- Extend `ffn_kind_for` in `src/model/mlx/qwen35_moe_config.hpp:27-35` for `qwen4_exp` and `qwen4_exp_text`.
- Parse `hc_count`, `hc_lowrank`, `indexer_n_heads`, `indexer_kv_heads`, `indexer_head_dim`, `indexer_budget`, `indexer_compress_ratio`, `heads_per_ngram`, `ngram_size`, `ngram_vocab_size_base`, `split_ngram_parts`, `make_ngram_vocab_size_divisible_by`, `ple_embed_dim`, `ple_conv_kernel_size`, `ple_layer_ids`, and `layer_types`.
- Also parse/guard `attn_output_gate`/`output_gate_type`; current code assumes the gate is always present.
- Add cases to `tests/model/test_model_bounds.cpp` (GPU-free).
- Check tokenizer first: Flash vocab is 248320, inside `[140000,260000]`; change `src/model/qwen_tokenizer.cpp:25,41` only if an actual target falls outside it.

Done when: a tiny config classifies both spellings, every required field is asserted, layer types and PLE IDs round-trip, and GPU-free tests pass.

### M2 — Gated Residual

Implement reference semantics §C:

- Widen embedding once: `[B,S,2560] -> [B,S,10240]` by tiling four identical branches.
- For each attention and MLP sublayer, separate GR module:
  `Rbar_i = RMSNorm(R_i; gamma_i)` per branch;
  `G = unvec(sigmoid(W_u SiLU((1/nr) W_d vec(Rbar))))`;
  `x = (1/nr) sum_i G_i (*) Rbar_i`;
  `s = 2 sigmoid((1/nr) W_w vec(Rbar))` (one scalar/branch);
  `R'_i = R_i + s_i*y`.
- `hc=4`, `d=2560`, rank `r=320`; no H_res operator.
- GR read replaces the sublayer's pre-norm: do not add another norm before attention/GDN/MoE.
- The final `hyper_connection_mixer` is `use_combine=false`: it performs the same normalized/gated read and collapses 10240 to 2560.
- There is no `model.norm.weight`. Do not invent a final RMSNorm.

Done when: tiny random GR output matches a Python/transformers reference, including final collapse, and attention/MLP both use independent modules.

### M3 — QSA (the hard attention piece)

Implement QSA indexer with production values: MQA 4 query heads/1 key head, indexer dim 128, compression ratio 4, token budget 2048 = 512 blocks.

- Fused indexer projection is `self_attn.indexer.index_qk_proj.weight`, output `(4+1)*128=640`; split first 512 into Q and last 128 into shared K.
- Query: RMSNorm then partial RoPE on 64/128 dimensions at query position.
- Pool raw K into non-overlapping blocks by average first, then K RMSNorm, then partial RoPE at block START position.
- Score `I_ib = sum_h ReLU(dot(q_i^h,kbar_b))/sqrt(128)` only when `block_end <= query_position`; otherwise `-inf`.
- Select top `ceil(2048/4)=512` complete blocks; expand each to four token positions.
- Always union the query's own trailing partial block and query token.
- Required mask, per query: `keep = (keep | own_tail) & (key_pos <= qp)`.
- Never compute one global prefill selection. Every query position has its own selection.
- Never replace causal masking with sparse masking; combine masks (`&` for bool, additive negative floor for float).
- Below `kv_len <= 2048`, short-circuit to plain causal attention. Test just below/above this boundary.
- Extend `KVCache`: `update_and_fetch` currently returns one contiguous slice (`kv_cache.hpp:38-68`), which cannot serve selected blocks. Add selected-index gather while preserving append/rollback behavior and cache checkpointing.
- QSA main attention remains 24 Q / 2 KV / 256 dim, partial RoPE 64, fused q output gate; QSA replaces the current full-attention arm.
- Follow the `lmp_diag scan` pattern: op-level reference plus fused Metal kernel, random real shapes, max/relative deviation and wall time, no checkpoint.

Done when: below-budget QSA is bit-identical to dense at tiny scale, above-budget selection/mask matches reference except explicitly counted zero-score ties, and prefill/decode paths pass independently.

### M4 — PLE / out-of-core (hard memory piece)

This fights `WeightStore::load_directory` (`weight_store.hpp:50-78`) and `set_wired_limit` (`mlx_backend.cpp:79-95`).

- Keep the logical table flat but physically separate its storage from GPU-resident tensors. There are 128 shards `shard_0`…`shard_127`; nominal shard shape is `[2,500,012 x 160]`; logical total is ~320,001,536 rows x 160 (~51.2B params).
- Address: `shard = gid // rows_per_shard`, `row = gid % rows_per_shard`.
- N-gram heads: `(ngram_size-1)*heads_per_ngram = 2*8 = 16`; eight bigram heads then eight trigram heads, each row width `2560/16=160`.
- Per-head table sizes are distinct primes: for global head `g`, `size = nth_prime_after(ngram_vocab_size_base-1, g+1)`; offsets are cumulative. Prefer checkpoint `ngram_heads_vocab_sizes` and `ngram_heads_offsets` buffers.
- Hash multipliers must be checkpoint `layer_multipliers`, or the transformers default seed 1234. Never use seed 0.
- For each position: EOS-aware shifted history; bigram hash is `id[i-1]*m0 XOR id[i]*m1`; trigram is `id[i-2]*m0 XOR id[i-1]*m1 XOR id[i]*m2`; modulo that head's prime, add that head offset.
- N-gram output is concatenated 16x160 -> 2560. No unigrams.
- PLE applies at `ple_layer_ids:[2]`, which means 0-indexed layer 1 (`layers.1.ple.*`), not layer 2. Use the widened 10240 residual.
- `key_proj:2560->10240`, `value_proj:2560->2560`; normalize key/query/conv in branch groups of 2560.
- `gate=(key*query).sum(-1,keepdims=True)/sqrt(2560)`; transform `sign(gate)*sqrt(max(abs(gate),1e-6))`; gate the shared value with sigmoid and broadcast to all four branches.
- Add the widened gated value to its dilated depthwise causal short conv: kernel 4, dilation `ngram_size=3`, state span 9 prior steps.
- Design mmap/lazy row access and async prefetch overlapping layer-1 compute. Random row reads have poor sequential readahead; ordinary page cache has a hot subset, so measure before forcing prefetch.
- Validate with a tiny stub table first. Do not download the real 51B table until correctness and memory design are proven.
- Update `model_limits` / `max_affordable_context_tokens` so PLE is not counted as resident. Never wire the 51B table into unified memory.
- If converting/dequantizing, stream row bands; never materialize the whole table.

Done when: a tiny EOS-aware PLE table matches transformers, chunk boundaries preserve context, and a real-table design leaves PLE outside the wired/resident working set.

### M5 — Wire the layer loop

- Prefer `text_config.layer_types`; retain modulo only as a verified fallback.
- Dispatch three arms: GDN (existing), QSA (new), MoE FFN (existing).
- PLE injection occurs before that layer's attention GR read at 0-indexed layer 1.
- `mtp_forward` hardcodes `forward_self_attn`; leave MTP disabled in v1 (recommended). If later enabled, MTP attention is QSA, not dense attention.

Done when: a config-driven 48-layer loop selects 36 GDN and 12 QSA layers, injects PLE once at layer 1, and old Qwen3.5 tests remain green.

### M6 — Tiny-scale correctness (mandatory before a real checkpoint)

- Create a synthetic fixture under `tests/model/`; none currently exists for the MLX graph.
- Reproduce PipeNetwork's method: tiny random weights, perturbed norms, EOS mid-sequence, sequence longer than indexer budget, more experts than top-k, QSA last.
- Compare module-by-module against transformers, not only final logits.
- Test single-shot, token-by-token decode, and chunked prefill (e.g. 5+7) against the same reference.
- Deliberately re-break each known fix and prove the test fails.
- Use `lmp_diag layers/chain/blocks/verify` for integration and `LMP_ABLATE` for isolation; ablations are not additive.

Done when: all module boundaries pass with documented tolerances, tie-affected positions are counted/excluded only when genuinely tied, and every deliberate regression is detected.

### M7 — Real MLX quant load (only after M0 GO and M6 green)

- Prefer a full-expert MLX build only if Sean upgrades hardware. On 48GB, native MLX requires aggressive PLE offload and perhaps expert pruning; do not silently fetch 100GB+.
- Interim option: llama.cpp OpenAI-compatible server for Piper bakeoffs; Cline already has the adapter path.

Done when: Sean explicitly approves the artifact, observed memory stays safe, and native results beat the server route enough to justify ownership.

## 5. Known bugs — DO NOT REIMPLEMENT

1. **QSA prefill mask.** Buggy implementations use global tail (future leak), drop each query's own partial block, and replace causal instead of ANDing it. Fix exactly: `keep = (keep | own_tail) & (key_pos <= qp)`.
2. **GDN q/k norm.** Must be exact `x * rsqrt(sum(x*x)+1e-6)` in float32, cast back; not `rms_norm` (which introduces mean reduction, `sqrt(d)`, and epsilon*d). Apply q-only `* dk**-0.5`.
3. **N-gram hash seed.** Transformers default/checkpoint is 1234; PR default 0 produces unrelated rows. Prefer loaded `layer_multipliers = [23703573157769, 20109073645365, 8052911324071]`; otherwise hardcode default 1234.
4. **RMSNorm convention.** Every zero-centered norm is `x/rms * (1+w)`, not `x/rms*w`: all GR `hc_norm`s (including final mixer), q/k norms, indexer q/k layernorm, and PLE key/query/conv norms. Fold `+1` once at raw-HF sanitize, idempotently. Exception: GDN `linear_attn.norm` is gated, ones-initialized RMSNorm and must not receive the shift.
5. **No `model.norm.weight`.** Top-level `hyper_connection_mixer` (`use_combine=false`) is the final norm/collapse. Adding another norm is wrong.
6. **Prefill segmentation.** Chunking can change sampled tokens on the existing checkpoint; bit-identical decode is not universally available. Use module diffs, not “same text.”
7. **Float scalar promotion.** `mx::array(float)` can promote residuals to f32 and cost ~3x decode. Build scalars in the input dtype explicitly.
8. **GDN head repeat.** 16 key heads to 48 value heads uses interleaved repeat (`mx::repeat`), not tile; state is `[B,48,128,128]` float32 per layer.
9. **QSA ordering.** Pool K then normalize then RoPE; block position is block start; eligibility requires full block observed. Do not RoPE token K before pooling.
10. **PLE details.** No unigrams; EOS resets shifted history; 16 heads are bigram/trigram; PLE is widened-residual injection; dilation is 3, not 1.
11. **Raw-HF loading.** Map `model.language_model.` to the MLX `model.` convention; split fused `experts.gate_up_proj [E,2I,H]` gate rows first into gate/up; rename `experts.down_proj`; drop MTP for v1. Do not double-sanitize.
12. **Divisibility traps.** GDN Metal requires `Dk%32==0`; KV group size 64 and QSA/indexer dimensions require explicit divisibility checks, not silent integer truncation.

## 6. Touch set (smallest)

### Tier 1 — unavoidable

1. `src/model/mlx/qwen35_moe_config.hpp`:27-180, 349-371 — allowlist, fields/parser, layer types, KV accounting.
2. `src/model/mlx/qwen35_moe_model.hpp`:44-93, 101-105, 175-212, 326-335, 458-466, 486-492, 572-648, 652-673, 675-714, 727-810, 835-909 — graph, caches, sanitize, loop, QSA/GR/PLE seams.
3. `src/model/mlx/weight_store.hpp`:50-78, 89-107, 142-200 — lazy/mmap/offload and lookup primitives.
4. `src/model/mlx_backend.cpp`:30-45, 79-152, 234-241, 362-513, 833-930 — concrete model member, wiring, reuse, prefill, MTP.

### Tier 2 — likely

- `src/model/qwen_tokenizer.cpp:25,41` only if vocab/family actually requires it.
- `src/model/family_traits.hpp:86-97` only if chat/tool syntax differs.
- `src/model/model_limits.cpp:51-73` and `src/sidecar.cpp:1026-1042` for QSA KV and offloaded-weight accounting.
- `tests/model/test_model_bounds.cpp`; add `tests/model/` tiny fixture and diagnostics as needed.

### Do not touch

Do not edit `src/loop/`, `src/context/`, `src/tools/`, wire protocol/surface, extension, or unrelated dirty files. Do not introduce a second inference server as part of the native port.

## 7. Test plan & harnesses

- `tests/model/test_model_bounds.cpp`: config parser and allowlist, GPU-free.
- `tests/model/diag_main.cpp:142-207`: copy the GDN `scan` reference-vs-Metal pattern for QSA.
- `lmp_diag layers`, `chain`, `blocks`, `verify`: real-checkpoint integration after tiny tests.
- `LMP_ABLATE=routed|mlp|delta|deltakernel`: isolate changed blocks; never interpret ablations as additive.
- Create a synthetic tiny MLX fixture under `tests/model/`; there is no existing one.
- Preserve `-Werror`; MLX code is headers included by `mlx_backend.cpp`, the TU with relaxed conversion warnings. New TUs must satisfy the full warning regime.
- Compare below-budget QSA to dense; above-budget QSA to transformers; verify causal masks, cache rollback, EOS reset, GDN state, and PLE chunk state.
- Log max absolute and relative deviations, not prose quality.
- Count genuine zero-score top-k ties and exclude only affected query positions from parity.

## 8. Out of scope for this handoff

- Full Flash Aider bakeoff (Sean/benchbot owns it after GO).
- ToolError storm fixes.
- Hardware squeeze or custom Metal beyond QSA/PLE requirements.
- Flash vision tower.
- Publishing or Reddit writeup.
- MTP in v1; all sources treat it as optional and the canonical transformers loader ignores it.

## 9. Reference appendix

Deep references used by this handoff:

- Sidecar map: `/workspace/lmpipe2-sidecar-map.md` (research box only).
- Semantics and tensor names: `/workspace/qwen4exp-reference-semantics.md` (research box only).
- 48GB scouting/runtime status: `/workspace/qwen38-flash-48gb-scouting.md` (research box only).
- In the repo, this handoff is `docs/QWEN4EXP_SIDECAR_HANDOFF.md`.

Critical config snapshot:

```json
{
  "model_type":"qwen4_exp", "text_config":{"model_type":"qwen4_exp_text",
  "hidden_size":2560, "num_hidden_layers":48, "num_attention_heads":24,
  "num_key_value_heads":2, "head_dim":256, "partial_rotary_factor":0.25,
  "full_attention_interval":4, "indexer_n_heads":4, "indexer_kv_heads":1,
  "indexer_head_dim":128, "indexer_budget":2048, "indexer_compress_ratio":4,
  "linear_num_key_heads":16, "linear_num_value_heads":48,
  "linear_key_head_dim":128, "linear_value_head_dim":128,
  "linear_conv_kernel_dim":4, "hc_count":4, "hc_lowrank":320,
  "heads_per_ngram":8, "ngram_size":3, "ngram_vocab_size_base":20000000,
  "make_ngram_vocab_size_divisible_by":128, "split_ngram_parts":128,
  "ple_embed_dim":2560, "ple_conv_kernel_size":4, "ple_layer_ids":[2],
  "num_experts":512, "num_experts_per_tok":10, "moe_intermediate_size":640,
  "shared_expert_intermediate_size":640, "mtp_num_hidden_layers":1,
  "vocab_size":248320, "rms_norm_eps":1e-6
  }
}
```
`layer_types` is 48 entries: `linear_attention` at indices 0,1,2 then `full_attention` at 3, repeating through 47. In this checkpoint “full_attention” is the config label for QSA because indexer tensors are present.

Canonical GDN recurrence:

```text
S~ = alpha*S
error = v - S~^T*k
S = S~ + beta*k*error^T
y = S^T*q
q=L2Norm(SiLU(conv(Wq*x))); k=L2Norm(SiLU(conv(Wk*x))); v=SiLU(conv(Wv*x))
beta=sigmoid(Wb*x)
alpha=exp(-exp(A_log)*softplus(Wa*x+dt_bias))
out=Wo(sigmoid(Wz*x)*RMSNormGated(y))
```
Use 16->48 interleaved head repeat; preserve state in float32.

Canonical QSA mask:

```text
n_blocks = floor(kv_len/4)
score[q,b] = sum_h ReLU(dot(q_rope[q,h], k_rope[block_start,b]))/sqrt(128)
score=-inf unless block_start+3 <= q_position
keep = expand(topk(score, min(512,n_blocks)))
own_tail = key_pos >= ((q_position+1)//4)*4 AND key_pos <= q_position
keep = (keep OR own_tail) AND (key_pos <= q_position)
```
Below 2048 cached tokens use plain causal attention. The returned sparse mask is combined with any existing mask, never substituted for causality.

PLE hash/apply rules:

```text
ngram_heads=16 (8 bigram + 8 trigram); row_width=160; output=concat(16 rows)=2560
hash_bigram=(id[i-1]*m0) XOR (id[i]*m1)
hash_trigram=(id[i-2]*m0) XOR (id[i-1]*m1) XOR (id[i]*m2)
gid=(hash mod per_head_prime)+per_head_offset
shard=gid//rows_per_shard; row=gid%rows_per_shard
```
History shifts are EOS-aware. Use checkpoint multipliers or seed 1234. At 0-indexed layer 1:

```text
key=norm_key(key_proj(ple)); value=value_proj(ple); query=norm_query(widened_hidden)
g=(key*query).sum(-1,keepdims=True)/sqrt(2560)
g=sign(g)*sqrt(max(abs(g),1e-6))
ple_out=sigmoid(g)*value[... ,None,:]
ple_out=ple_out + dilated_causal_conv(norm_conv(ple_out), dilation=3, kernel=4)
h = h + ple_out
```

Bf16-floor tensor list (never below bf16; broader mixed-precision protection may be added):

```text
layers.*.mlp.gate.weight
layers.*.mlp.shared_expert_gate.weight
layers.*.attn_hyper_connection.block_inject_weight.weight
layers.*.mlp_hyper_connection.block_inject_weight.weight
layers.*.linear_attn.in_proj_a.weight
layers.*.linear_attn.in_proj_b.weight
layers.*.self_attn.indexer.index_qk_proj.weight
```
The strict floor is ~36M routing/gating parameters. Quantization-sensitive read gates and attention/GDN/PLE projections may merit 8-bit based on PipeNetwork's mixed-4/8 result, but do not sacrifice correctness to quantization first.

Tensor naming:

- MLX converted backbone prefix: `language_model.model.layers.{i}.*`; logits: `language_model.lm_head.weight`.
- GDN keys: `linear_attn.{A_log,dt_bias,conv1d,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,norm,out_proj}`.
- QSA keys: `self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}` plus `self_attn.indexer.{index_qk_proj,q_layernorm,k_layernorm}`.
- GR keys: `attn_hyper_connection.*`, `mlp_hyper_connection.*`; final `hyper_connection_mixer.{hc_norm,input_mix_weight_down,input_mix_weight_up}` only.
- PLE keys exist only under `layers.1.ple.*`, including `ple_embedding.layer_multipliers`, `ngram_heads_offsets`, `ngram_heads_vocab_sizes`, and `ngram_embedding.shard_0`…`shard_127`.
- Raw HF has `model.language_model.*`, fused `experts.gate_up_proj`, top-level `lm_head`, vision, and `mtp.*`; map/split/drop as stated above.

Reference-source citations used above:

- Sidecar-map report, `lmpipe2-sidecar-map.md:124-153`: current allowlist/refusal path.
- Sidecar-map report, `:209-228`: existing GDN and Metal scan.
- Sidecar-map report, `:230-295`: existing MoE and gated GQA attention.
- Sidecar-map report, `:297-351`: KV/reuse/MTP facilities.
- Sidecar-map report, `:365-410`: eager affine quant store and gather primitives.
- Sidecar-map report, `:412-450`: modulo schedule and ignored `layer_types`.
- Sidecar-map report, `:454-467`: missing QSA, GR and PLE inventory.
- Sidecar-map report, `:489-494`: resident-memory accounting and wired-limit conflict.
- Sidecar-map report, `:498-532`: minimum touch set.
- Sidecar-map report, `:534-578`: diagnostics and lack of tiny fixture.
- Sidecar-map report, `:582-657`: thirteen codebase-specific risks.
- Semantics report, `qwen4exp-reference-semantics.md:48-158`: canonical release config.
- Semantics report, `:162-180`: `full_attention` label remaps to QSA.
- Semantics report, `:194-269`: QSA pooling, scoring, budget and tail semantics.
- Semantics report, `:300-401`: exact sparse-prefill bug and mask fix.
- Semantics report, `:419-438`: fused indexer projection and attention gate.
- Semantics report, `:443-498`: GDN recurrence and output.
- Semantics report, `:500-573`: head repeat and persistent state.
- Semantics report, `:584-696`: conv/projection layout and exact L2 norm.
- Semantics report, `:701-850`: GR equations, widening and final mixer.
- Semantics report, `:864-902`: PLE placement/off-by-one.
- Semantics report, `:927-997`: bigram/trigram heads, prime tables, hash and EOS reset.
- Semantics report, `:998-1052`: physical sharding and mmap guidance.
- Semantics report, `:1054-1131`: PLE concatenate/apply/conv semantics.
- Semantics report, `:1133-1185`: observed random access and streaming requirements.
- Semantics report, `:1188-1222`: seed-1234 bug and checkpoint multipliers.
- Semantics report, `:1295-1337`: evidence for omitting MTP.
- Semantics report, `:1342-1473`: tensor names and zero-centered norm loading.
- Semantics report, `:1475-1532`: bf16 floor and quantization-sensitive groups.
- Semantics report, `:1537-1676`: tiny-scale correctness strategy.
- Scouting report, `qwen38-flash-48gb-scouting.md:5-11`: 48GB recommendation and full-expert NO-FIT.
- Scouting report, `:23-40`: artifact size/runtime comparison.
- Scouting report, `:54-75`: llama.cpp/mlx-lm/mlx-vlm status.
- Scouting report, `:77-96`: exact download/server commands.
- Scouting report, `:105-121`: runtime preconditions and caveats.

Primary upstream references behind those reports:

- Canonical architecture/report: https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf
- Canonical config: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json
- Canonical transformers model: https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen4_exp
- Shared canonical GDN semantics: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- Fixed MLX reference: https://github.com/PipeNetwork/qwen38-flash-next-mlx
- Fixed reference implementation: https://raw.githubusercontent.com/PipeNetwork/qwen38-flash-next-mlx/main/qwen38_flash_next_mlx/qwen4_exp.py
- PipeNetwork fix notes: https://raw.githubusercontent.com/PipeNetwork/qwen38-flash-next-mlx/main/docs/upstream-notes.md
- Upstream MLX PR (open; commit-specific, not an oracle): https://github.com/ml-explore/mlx-lm/pull/1788
- Merged llama.cpp implementation: https://github.com/ggml-org/llama.cpp/pull/27742
- Follow-up llama.cpp fixes: https://github.com/ggml-org/llama.cpp/pull/27941
- Recommended 48GB artifact: https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF

Final execution checklist:

- [ ] Confirm branch and dirty status; record but do not alter unrelated files.
- [ ] Run M0; stop on NO-GO.
- [ ] Add config classification tests before graph code.
- [ ] Make GR match in isolation.
- [ ] Make QSA below-budget dense oracle pass.
- [ ] Make QSA above-budget prefill and decode pass separately.
- [ ] Prove the causal/tail regression test fails when deliberately broken.
- [ ] Make PLE hash/EOS behavior pass with a tiny table.
- [ ] Prove seed-0 and missing `(1+w)` controls fail.
- [ ] Verify incremental, single-shot and chunked module outputs.
- [ ] Verify old Qwen3.5/Qwen3.8 paths remain green.
- [ ] Verify PLE pages are not wired/resident before real-scale loading.
- [ ] Obtain Sean's approval before any large download.

Do not infer correctness from fluent output. The reference reports parity around float32 noise below budget, explicit failure when each bug is reintroduced, and exact incremental/chunked checks at tiny scale. Implement the smallest extension, preserve the dirty tree, and stop at every gate when the evidence says stop.
