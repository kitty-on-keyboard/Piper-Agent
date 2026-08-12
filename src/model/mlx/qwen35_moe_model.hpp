// Adapted from v1 src/llm/models/qwen35_moe_model.hpp -- debugged numerics for the
// exact checkpoint this machine runs (Qwen3.6-35B-A3B-MLX-4bit). The forward pass is
// model math, which passes S2.2's asset test; only the seam around it was rebuilt.
#ifndef LLM_MODELS_QWEN35_MOE_MODEL_HPP
#define LLM_MODELS_QWEN35_MOE_MODEL_HPP

#if LMP_HAVE_MLX

#include "qwen35_moe_config.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "activations.hpp"
#include "gated_delta.hpp"
#include "kv_cache.hpp"
#include "moe_trace.hpp"
#include "switch_glu.hpp"
#include "weight_store.hpp"

#include "mlx/fast.h"
#include "mlx/ops.h"

namespace lmp::model::mlxl {

namespace mx = mlx::core;
using lmp::model::mlxl::KVCache;
using lmp::model::mlxl::SsmCache;
using lmp::model::mlxl::WeightStore;

class Qwen35MoeModel {
public:
    bool load(const std::string& model_dir) {
        prefix_ = "language_model.model.";
        load_error_.clear();
        if (!load_qwen35_moe_config(model_dir, cfg_)) {
            return false;
        }
        // Refuse an unrecognised architecture BEFORE reading ~16 GB of weights. The
        // alternative is what this path used to do: build the MoE graph regardless, then
        // fail on a missing expert tensor at the first token with the whole checkpoint
        // already resident and wired.
        ffn_ = cfg_.ffn_kind();
        if (ffn_ == FfnKind::Unknown) {
            load_error_ = "unsupported architecture: model_type=\"" + cfg_.model_type +
                          "\" (this build knows qwen3_5 / qwen3_5_text dense and "
                          "qwen3_5_moe / qwen3_5_moe_text)";
            return false;
        }
        if (!weights_.load_directory(model_dir)) {
            return false;
        }
        sanitize_weights();
        reset_cache();
        cfg_.vocab_size = cfg_.vocab_size > 0 ? cfg_.vocab_size : 248320;
        return true;
    }

    void reset_cache() {
        kv_caches_.assign(static_cast<std::size_t>(cfg_.num_hidden_layers), KVCache{});
        ssm_caches_.assign(static_cast<std::size_t>(cfg_.num_hidden_layers), SsmCache{});
    }

    [[nodiscard]] int cache_seq_len() const noexcept {
        // kv_caches_.front() is layer 0, which is a linear (gated-delta/SSM) layer
        // under the full_attention_interval hybrid schedule -- forward_linear_layer
        // never touches kv_caches_[layer], only ssm_caches_[layer].
        // Reading front() here always reported 0 regardless of how many tokens
        // were actually prefilled, which made the caller's "empty cache but
        // baked_turns != 0" invariant check fire every iteration once baked_turns
        // was nonzero and force-reset session state on top of a KV cache that was
        // never actually cleared -- silently re-prefilling the whole conversation
        // on every turn (see inference_engine.cpp's cache_seq_len()==0 guard).
        // Read the offset from the first real full-attention layer instead.
        for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
            if (!cfg_.is_linear_layer(layer) &&
                static_cast<std::size_t>(layer) < kv_caches_.size()) {
                return kv_caches_[static_cast<std::size_t>(layer)].offset;
            }
        }
        return 0;
    }

    // Returns the graph, unevaluated. The caller decides what to force and when, which
    // matters twice: it lets the decode loop fold the host-ready float32 conversion into
    // the same evaluation instead of paying a second GPU round-trip for it, and it lets
    // prefill chunks whose logits nobody reads skip the lm_head projection entirely.
    // It also lets `lmp_diag step` time graph construction apart from execution.
    mx::array forward_logits(const mx::array& input_ids) {
        const int seq_len = static_cast<int>(input_ids.shape()[1]);
        mx::array h = forward_hidden(input_ids, seq_len);
        // Contract: only the final position's
        // logits are ever consumed downstream. No-op when seq_len == 1 (decode).
        const int hidden = static_cast<int>(h.shape()[2]);
        mx::array h_last = mx::slice(h, {0, seq_len - 1, 0}, {1, seq_len, hidden});
        return logits_from_hidden(h_last);
    }

    // The opt-in path out of that contract: verifying a k-token speculative draft needs the
    // model's distribution at EVERY drafted position, not just the last one. Returns
    // [1, seq_len, vocab] -- one row per input position -- unevaluated, on the same terms as
    // forward_logits.
    //
    // A second entry point rather than a flag on the first, deliberately. The decode path is
    // tuned and its graph is load-bearing: HANDOFF_PERF.md records that folding the float32
    // cast into this graph measured WORSE (84.8 -> 83.9 tok/s) despite removing a GPU
    // round-trip, because it extended the step's critical path. So forward_logits still
    // builds the graph it built before this function existed. What the two share is the
    // layer stack, which they need verbatim and identically; the only divergence is the
    // final-position slice that one does and the other skips.
    //
    // Not free at the vocabulary this model has: the lm_head projection runs on seq_len rows
    // instead of one, and each row is 248,320 wide. That is the right trade only when the
    // extra rows are actually verified against -- never call this from the single-token
    // decode path, where it would pay for a slice it then throws away.
    mx::array forward_logits_all(const mx::array& input_ids) {
        const int seq_len = static_cast<int>(input_ids.shape()[1]);
        return logits_from_hidden(forward_hidden(input_ids, seq_len));
    }

    // A restore point for speculative decoding: everything needed to put the caches back the
    // way they were before a block of drafted tokens was forwarded.
    //
    // Qwen 3.6 is a HYBRID and the two kinds of layer roll back differently.
    // Full-attention layers keep a per-token history, so any earlier position is reachable
    // by moving an index. Linear (gated-delta) layers keep a recurrence with no per-token
    // history, so only positions that were snapshotted in advance are reachable at all.
    // That asymmetry is why this is checkpoint/restore rather than the `rollback_to(int n)`
    // the plan sketched: a rollback to an arbitrary n is not implementable for 30 of this
    // model's 40 layers, and an API promising it would be a lie whose only symptom is drift
    // in the output.
    struct CacheCheckpoint {
        int seq_len{0};
        // Indexed by layer, including the full-attention layers, whose SsmCache is empty.
        // Keeping the vector layer-aligned rather than packing the linear layers means
        // restore() cannot silently mis-pair a snapshot with a layer if the hybrid schedule
        // ever changes.
        std::vector<SsmCache::Snapshot> ssm;
    };

    [[nodiscard]] CacheCheckpoint checkpoint() const {
        CacheCheckpoint cp;
        cp.seq_len = cache_seq_len();
        cp.ssm.reserve(ssm_caches_.size());
        for (const auto& c : ssm_caches_) {
            cp.ssm.push_back(c.snapshot());
        }
        return cp;
    }

    void restore(const CacheCheckpoint& cp) {
        // truncate_to clamps upward-to-current, so the linear layers' unused KVCaches --
        // which sit at offset 0 forever -- are left alone rather than being grown to
        // cp.seq_len tokens of scratch.
        for (auto& c : kv_caches_) {
            c.truncate_to(cp.seq_len);
        }
        const std::size_t n = std::min(ssm_caches_.size(), cp.ssm.size());
        for (std::size_t i = 0; i < n; ++i) {
            ssm_caches_[i].restore(cp.ssm[i]);
        }
    }

    void eval_caches() {
        for (auto& c : kv_caches_) {
            c.sync();
        }
        for (auto& c : ssm_caches_) {
            c.sync();
        }
    }

    [[nodiscard]] const Qwen35MoeConfig& qwen_config() const noexcept { return cfg_; }

    // Ablation, for attribution only. Timing a block in isolation does not measure what
    // it costs inside a real step: `layers 1` charges the MoE 0.673 ms/layer, and the
    // same chained microbenchmark applied to mlx-lm's own block charges 0.304 -- yet
    // mlx-lm's entire 40-layer step is 9.66 ms, so both numbers are dominated by the
    // measurement, not the work. Deleting a block from a full generation run and reading
    // the end-to-end rate has no such artifact. Set LMP_ABLATE=routed|mlp|delta.
    // Output is garbage when this is on; only tok/s means anything.
    enum class Ablate : std::uint8_t { none, routed, mlp, delta, deltakernel };
    static Ablate ablation() {
        // Read once. A getenv per layer would itself be a per-step cost.
        static const Ablate mode = [] {
            const char* v = std::getenv("LMP_ABLATE");
            if (v == nullptr) return Ablate::none;
            const std::string s(v);
            if (s == "routed") return Ablate::routed;
            if (s == "mlp") return Ablate::mlp;
            if (s == "delta") return Ablate::delta;
            if (s == "deltakernel") return Ablate::deltakernel;
            return Ablate::none;
        }();
        return mode;
    }

    // The three blocks a forward pass is made of, public so tests/model/diag_main.cpp
    // can time them individually against the real weights. Attribution has to run the
    // SAME code the model runs; a copy of these bodies in the driver would drift, and
    // the profile would then describe a forward pass nobody executes (S19.3).
    mx::array forward_moe(int layer, const mx::array& x) const {
        const std::string p = prefix_ + "layers." + std::to_string(layer) + ".mlp.";
        if (ablation() == Ablate::mlp) {
            return mx::zeros_like(x);
        }
        const mx::array gate_logits = weights_.linear(x, p + "gate");
        auto [inds, scores] = lmp::model::mlxl::moe_topk(gate_logits, cfg_.num_experts_per_tok, cfg_.norm_topk_prob);
        // Diagnostic capture (S19.3). Off unless LMP_MOE_TRACE is set; see moe_trace.hpp
        // for why it is a runtime branch rather than a compile-time one, and why a traced
        // run's throughput figures must be discarded.
        if (MoeTrace::instance().enabled() && x.shape()[1] == 1) {
            mx::array ids = mx::astype(mx::reshape(inds, {-1}), mx::int32);
            mx::eval(ids);
            MoeTrace::instance().record(layer, ids.data<int>(),
                                        static_cast<std::size_t>(ids.size()));
        }
        mx::array y = mx::zeros_like(x);
        if (ablation() != Ablate::routed) {
            y = lmp::model::mlxl::switch_glu(
                x, weights_, p + "switch_mlp.gate_proj", p + "switch_mlp.up_proj", p + "switch_mlp.down_proj", inds);
            y = mx::sum(mx::multiply(y, mx::expand_dims(scores, -1)), -2);
        }

        const mx::array shared = weights_.linear(
            lmp::model::mlxl::swiglu(
                weights_.linear(x, p + "shared_expert.gate_proj"),
                weights_.linear(x, p + "shared_expert.up_proj")),
            p + "shared_expert.down_proj");
        const mx::array shared_gate = mx::sigmoid(weights_.linear(x, p + "shared_expert_gate"));
        return mx::add(y, mx::multiply(shared_gate, shared));
    }

    // The dense generation's FFN: one gated MLP where the MoE has 256 routed experts plus
    // a shared one. Same SwiGLU and same quantized `linear` the shared expert already
    // uses, so nothing new reaches the weight store or the kernels.
    mx::array forward_dense_mlp(int layer, const mx::array& x) const {
        const std::string p = prefix_ + "layers." + std::to_string(layer) + ".mlp.";
        if (ablation() == Ablate::mlp) {
            return mx::zeros_like(x);
        }
        return weights_.linear(lmp::model::mlxl::swiglu(weights_.linear(x, p + "gate_proj"),
                                                        weights_.linear(x, p + "up_proj")),
                               p + "down_proj");
    }

    // The single axis the two checkpoints differ on. Classified once at load(), which
    // refuses anything it does not recognise, so this cannot silently pick a graph.
    mx::array forward_ffn(int layer, const mx::array& x) const {
        return ffn_ == FfnKind::Dense ? forward_dense_mlp(layer, x) : forward_moe(layer, x);
    }

    mx::array forward_linear_layer(int layer, const mx::array& x, int seq_len) {
        const std::string p = prefix_ + "layers." + std::to_string(layer) + ".";
        mx::array h = rms_norm(x, p + "input_layernorm.weight");
        if (ablation() == Ablate::delta) {
            h = x;
        } else {
            mx::array attn_out = forward_gated_delta(
                p + "linear_attn.", h, ssm_caches_[static_cast<std::size_t>(layer)], seq_len);
            h = mx::add(x, attn_out);
        }
        mx::array mlp_in = rms_norm(h, p + "post_attention_layernorm.weight");
        return mx::add(h, forward_ffn(layer, mlp_in));
    }

    mx::array forward_full_attn_layer(int layer, const mx::array& x, int seq_len) {
        const std::string p = prefix_ + "layers." + std::to_string(layer) + ".";
        mx::array h = rms_norm(x, p + "input_layernorm.weight");
        mx::array attn_out = forward_self_attn(p + "self_attn.", h, kv_caches_[static_cast<std::size_t>(layer)], seq_len);
        h = mx::add(x, attn_out);
        mx::array mlp_in = rms_norm(h, p + "post_attention_layernorm.weight");
        return mx::add(h, forward_ffn(layer, mlp_in));
    }

    // Set when load() refuses; empty otherwise. MlxBackend surfaces it verbatim so the
    // operator is told WHICH of the load preconditions failed.
    [[nodiscard]] const std::string& load_error() const noexcept { return load_error_; }

private:
    std::string prefix_;
    Qwen35MoeConfig cfg_{};
    FfnKind ffn_{FfnKind::Moe};
    std::string load_error_;
    WeightStore weights_;
    std::vector<KVCache> kv_caches_;
    std::vector<SsmCache> ssm_caches_;

    void sanitize_weights() {
        auto& w = weights_.mutable_weights();

        // Mirrors mlx-lm's Qwen3.5-MoE sanitize(): the RMSNorm "+1" shift and the
        // conv1d axis fixup are only needed for raw (unconverted) HF checkpoints.
        // LM Studio's pre-converted MLX checkpoints already have conv1d weights in
        // MLX layout (last dim == 1) and never ship MTP weights, so applying the
        // shift unconditionally would corrupt every norm weight in the model.
        bool has_mtp_weights = false;
        bool has_unsanitized_conv1d = false;
        for (const auto& [key, val] : w) {
            if (key.find("mtp.") != std::string::npos) {
                has_mtp_weights = true;
            }
            if (key.find("conv1d.weight") != std::string::npos && val.ndim() == 3 &&
                val.shape()[2] != 1) {
                has_unsanitized_conv1d = true;
            }
        }
        const bool should_shift_norm_weights = has_mtp_weights || has_unsanitized_conv1d;

        std::vector<std::pair<std::string, mx::array>> updated;
        std::vector<std::string> to_erase;

        for (auto& [key, val] : w) {
            if (key.find("mtp.") != std::string::npos ||
                key.find("vision_tower") != std::string::npos) {
                to_erase.push_back(key);
                continue;
            }
            if (key.find("conv1d.weight") != std::string::npos && val.ndim() == 3 &&
                val.shape()[2] != 1) {
                updated.emplace_back(key, mx::moveaxis(val, 2, 1));
                to_erase.push_back(key);
            }
            if (should_shift_norm_weights &&
                (key.ends_with(".input_layernorm.weight") ||
                 key.ends_with(".post_attention_layernorm.weight") ||
                 key.ends_with("model.norm.weight") ||
                 key.ends_with(".q_norm.weight") ||
                 key.ends_with(".k_norm.weight"))) {
                if (val.ndim() == 1) {
                    // astype, not a bare mx::array(1.0f). mlx-lm writes `v + 1.0`, and a
                    // Python float is weakly typed, so the weight stays bf16; the C++
                    // scalar is a strongly typed float32 array and would promote it.
                    // These are norm weights -- they multiply into rms_norm on every
                    // step, including model.norm before lm_head -- so a float32 one puts
                    // float32 back into the residual stream, which is the same fault that
                    // cost 3x in forward_gated_delta. No checkpoint here trips
                    // should_shift_norm_weights (it needs mtp weights or an unsanitized
                    // conv1d), so this is latent rather than measured: it costs nothing
                    // today and is wrong for the first checkpoint that does trip it.
                    updated.emplace_back(key, mx::add(val, mx::astype(mx::array(1.0f),
                                                                      val.dtype())));
                    to_erase.push_back(key);
                }
            }
        }
        for (const auto& k : to_erase) {
            w.erase(k);
        }
        for (auto& [k, v] : updated) {
            w.insert_or_assign(std::move(k), std::move(v));
        }

    }

    // The layer stack, shared verbatim by forward_logits and forward_logits_all so the two
    // cannot drift apart. Returns the final-normed hidden state, [1, seq_len, hidden].
    mx::array forward_hidden(const mx::array& input_ids, int seq_len) {
        mx::array h = embed_tokens(input_ids);
        if (MoeTrace::instance().enabled() && seq_len == 1) {
            mx::array t = mx::astype(mx::reshape(input_ids, {-1}), mx::int32);
            mx::eval(t);
            MoeTrace::instance().set_token(t.data<int>()[0]);
        }

        for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
            if (cfg_.is_linear_layer(layer)) {
                h = forward_linear_layer(layer, h, seq_len);
            } else {
                h = forward_full_attn_layer(layer, h, seq_len);
            }
        }

        return rms_norm(h, prefix_ + "norm.weight");
    }

    [[nodiscard]] mx::array embed_tokens(const mx::array& ids) const {
        return weights_.embed_lookup(ids, prefix_ + "embed_tokens");
    }

    [[nodiscard]] mx::array rms_norm(const mx::array& x, const std::string& wkey) const {
        return mx::fast::rms_norm(x, weights_.get(wkey), cfg_.rms_norm_eps);
    }

    [[nodiscard]] mx::array logits_from_hidden(const mx::array& h) const {
        if (cfg_.tie_word_embeddings) {
            return weights_.tied_logits(h, prefix_ + "embed_tokens");
        }
        return weights_.linear(h, "language_model.lm_head");
    }

    mx::array forward_gated_delta(const std::string& p, const mx::array& inputs, SsmCache& cache, int /*seq_len*/) {
        const int B = static_cast<int>(inputs.shape()[0]);
        const int S = static_cast<int>(inputs.shape()[1]);
        const int key_dim = cfg_.linear_num_key_heads * cfg_.linear_key_head_dim;
        const int value_dim = cfg_.linear_num_value_heads * cfg_.linear_value_head_dim;
        const int conv_dim = key_dim * 2 + value_dim;

        mx::array qkv = weights_.linear(inputs, p + "in_proj_qkv");
        mx::array z = weights_.linear(inputs, p + "in_proj_z");
        mx::array b = weights_.linear(inputs, p + "in_proj_b");
        mx::array a = weights_.linear(inputs, p + "in_proj_a");

        mx::array conv_state = cache.conv_state.value_or(
            mx::zeros({B, cfg_.linear_conv_kernel_dim - 1, conv_dim}, inputs.dtype()));
        mx::array conv_input = mx::concatenate({conv_state, qkv}, 1);
        // mx::contiguous, as mlx-lm does: a bare slice is a strided view that both keeps
        // the whole parent conv_input buffer alive across the step boundary and makes
        // next step's concatenate read from a strided source.
        cache.conv_state = mx::contiguous(
            mx::slice(conv_input, {0, conv_input.shape()[1] - (cfg_.linear_conv_kernel_dim - 1), 0},
                      {B, conv_input.shape()[1], conv_dim}));

        mx::array conv_w = weights_.get(p + "conv1d.weight");
        mx::array conv_out = lmp::model::mlxl::silu(
            mx::conv1d(conv_input, conv_w, /*stride=*/1, /*padding=*/0, /*dilation=*/1, conv_dim));

        // One split, not three slices. mlx-lm splits here and `lmp_diag graph` counted
        // the difference: 110 Slice nodes against the reference's 40 Split, for the same
        // partition of the same buffer.
        const std::vector<mx::array> qkv_parts =
            mx::split(conv_out, mx::Shape{key_dim, 2 * key_dim}, 2);

        mx::array q = mx::reshape(qkv_parts[0], {B, S, cfg_.linear_num_key_heads, cfg_.linear_key_head_dim});
        mx::array k = mx::reshape(qkv_parts[1], {B, S, cfg_.linear_num_key_heads, cfg_.linear_key_head_dim});
        mx::array v = mx::reshape(qkv_parts[2], {B, S, cfg_.linear_num_value_heads, cfg_.linear_value_head_dim});
        z = mx::reshape(z, {B, S, cfg_.linear_num_value_heads, cfg_.linear_value_head_dim});

        const float inv_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.linear_key_head_dim));
        // In the dtype of q/k, not float32. mlx-lm writes `(inv_scale**2) * rms_norm(q)`
        // with a Python float, which is weakly typed and leaves q in bf16; the obvious
        // C++ transcription, mx::array(inv_scale), is a strongly typed float32 scalar and
        // promotes instead. That promotion does not stay local -- q and k go float32 into
        // the delta kernel, y comes back float32, the gated norm returns float32, out_proj
        // emits float32, and `h + attn_out` makes the residual stream float32 from layer 1
        // to the end of the model. Every quantized matmul downstream then runs its float32
        // activation path against bf16 weights.
        //
        // Built in the target dtype rather than cast into it: mx::array(v, dtype) is the
        // same scalar as astype(mx::array(v), dtype) but is a constant, where the cast was
        // an AsType node in the graph. Sixty of them a step, per `lmp_diag graph`.
        const mx::array inv2 = mx::array(inv_scale * inv_scale, qkv.dtype());
        const mx::array inv1 = mx::array(inv_scale, qkv.dtype());
        // std::nullopt, not ones(): mlx-lm passes no weight here. A literal ones vector is
        // arithmetically the same, but it allocates and evaluates a fresh array on every
        // one of these calls (twice per layer, 30 linear layers per token) and takes
        // rms_norm's weighted path instead of its unweighted one.
        q = mx::multiply(inv2, mx::fast::rms_norm(q, std::nullopt, 1e-6f));
        k = mx::multiply(inv1, mx::fast::rms_norm(k, std::nullopt, 1e-6f));

        mx::array a_log = weights_.get(p + "A_log");
        mx::array dt_bias = weights_.get(p + "dt_bias");
        // deltakernel keeps the projections, the conv and the norms and removes only the
        // fused recurrence, to split "the custom Metal kernel" from "everything else in
        // this block". Removing the whole block (LMP_ABLATE=delta) cannot tell them apart.
        // inputs.dtype(), not float32. A float32 stand-in here re-promotes the residual
        // stream exactly the way the real bug did and costs 11.8 -> 32.8 ms/token, which
        // makes the ablation measure the promotion instead of the kernel. Kept as a
        // comment rather than a footnote because it is a live trap: any substitute value
        // spliced into this block has to carry the block's own dtype.
        mx::array out = mx::zeros({B, S, cfg_.linear_num_value_heads, cfg_.linear_value_head_dim},
                                  inputs.dtype());
        if (ablation() != Ablate::deltakernel) {
            auto [o, state] =
                lmp::model::mlxl::gated_delta_update(q, k, v, a, b, a_log, dt_bias, cache.delta_state);
            out = o;
            cache.delta_state = state;
        }
        cache.offset += S;

        mx::array norm_w = weights_.get(p + "norm.weight");
        mx::array gated = lmp::model::mlxl::precise_rms_norm_gated(out, z, norm_w, cfg_.rms_norm_eps);
        mx::array flat = mx::reshape(gated, {B, S, value_dim});
        return weights_.linear(flat, p + "out_proj");
    }

    mx::array forward_self_attn(const std::string& p, const mx::array& h, KVCache& cache, int /*seq_len*/) {
        namespace ff = mx::fast;
        const int B = static_cast<int>(h.shape()[0]);
        const int L = static_cast<int>(h.shape()[1]);
        const int n_heads = cfg_.num_attention_heads;
        const int n_kv = cfg_.num_key_value_heads;
        const int head_dim = cfg_.head_dim;
        const int rotary_dim = static_cast<int>(head_dim * cfg_.partial_rotary_factor);

        mx::array q_out = weights_.linear(h, p + "q_proj");
        mx::array k = weights_.linear(h, p + "k_proj");
        mx::array v = weights_.linear(h, p + "v_proj");

        // q_proj emits [q_h0, g_h0, q_h1, g_h1, ...] per head — not [all_q, all_g].
        const mx::array q_heads = mx::reshape(q_out, {B, L, n_heads, head_dim * 2});
        const std::vector<mx::array> qg = mx::split(q_heads, 2, -1);
        mx::array queries = mx::reshape(qg[0], {B, L, n_heads * head_dim});
        mx::array gate = mx::reshape(qg[1], {B, L, n_heads * head_dim});

        queries = mx::transpose(
            mx::reshape(ff::rms_norm(
                            mx::reshape(queries, {B, L, n_heads, head_dim}),
                            weights_.get(p + "q_norm.weight"), cfg_.rms_norm_eps),
                        {B, L, n_heads, head_dim}),
            {0, 2, 1, 3});
        k = mx::transpose(
            mx::reshape(ff::rms_norm(
                            mx::reshape(k, {B, L, n_kv, head_dim}),
                            weights_.get(p + "k_norm.weight"), cfg_.rms_norm_eps),
                        {B, L, n_kv, head_dim}),
            {0, 2, 1, 3});
        v = mx::transpose(mx::reshape(v, {B, L, n_kv, head_dim}), {0, 2, 1, 3});

        const int offset = cache.offset;
        queries = ff::rope(queries, rotary_dim, false, cfg_.rope_theta, 1.0f, offset);
        k = ff::rope(k, rotary_dim, false, cfg_.rope_theta, 1.0f, offset);

        auto [k_cat, v_cat] = cache.update_and_fetch(k, v);
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        // A single query attends to the whole cache, so decode wants no mask at all.
        // Say that with mask_mode rather than an empty mask argument: the type of that
        // argument changed between MLX 0.29 and 0.31 (vector<array> -> optional<array>),
        // and naming it here is what made the build version-specific.
        mx::array attn = ff::scaled_dot_product_attention(
            queries, k_cat, v_cat, scale, (L > 1) ? "causal" : "");
        attn = mx::reshape(mx::transpose(attn, {0, 2, 1, 3}), {B, L, n_heads * head_dim});
        attn = mx::multiply(attn, mx::sigmoid(gate));
        return weights_.linear(attn, p + "o_proj");
    }
};

} // namespace lmp::model::mlxl

#endif // LMP_HAVE_MLX

#endif // LLM_MODELS_QWEN35_MOE_MODEL_HPP
