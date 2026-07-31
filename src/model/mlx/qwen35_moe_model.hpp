// Adapted from v1 src/llm/models/qwen35_moe_model.hpp -- debugged numerics for the
// exact checkpoint this machine runs (Qwen3.6-35B-A3B-MLX-4bit). The forward pass is
// model math, which passes S2.2's asset test; only the seam around it was rebuilt.
#ifndef LLM_MODELS_QWEN35_MOE_MODEL_HPP
#define LLM_MODELS_QWEN35_MOE_MODEL_HPP

#if LMP_HAVE_MLX

#include "qwen35_moe_config.hpp"

#include "activations.hpp"
#include "gated_delta.hpp"
#include "kv_cache.hpp"
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
        if (!load_qwen35_moe_config(model_dir, cfg_)) {
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
        // never touches kv_caches_[layer], only ssm_caches_[layer] (whose own
        // offset field is likewise never incremented; see forward_gated_delta).
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

    mx::array forward_logits(const mx::array& input_ids) {
        mx::array h = embed_tokens(input_ids);
        const int seq_len = static_cast<int>(input_ids.shape()[1]);

        for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
            if (cfg_.is_linear_layer(layer)) {
                h = forward_linear_layer(layer, h, seq_len);
            } else {
                h = forward_full_attn_layer(layer, h, seq_len);
            }
        }

        h = rms_norm(h, prefix_ + "norm.weight");
        // Contract: only the final position's
        // logits are ever consumed downstream. No-op when seq_len == 1 (decode).
        const int hidden = static_cast<int>(h.shape()[2]);
        mx::array h_last = mx::slice(h, {0, seq_len - 1, 0}, {1, seq_len, hidden});
        mx::array logits = logits_from_hidden(h_last);
        mx::eval(logits);
        return logits;
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

    // The three blocks a forward pass is made of, public so tests/model/diag_main.cpp
    // can time them individually against the real weights. Attribution has to run the
    // SAME code the model runs; a copy of these bodies in the driver would drift, and
    // the profile would then describe a forward pass nobody executes (S19.3).
    mx::array forward_moe(int layer, const mx::array& x) const {
        const std::string p = prefix_ + "layers." + std::to_string(layer) + ".mlp.";
        const mx::array gate_logits = weights_.linear(x, p + "gate");
        auto [inds, scores] = lmp::model::mlxl::moe_topk(gate_logits, cfg_.num_experts_per_tok, cfg_.norm_topk_prob);
        mx::array y = lmp::model::mlxl::switch_glu(
            x, weights_, p + "switch_mlp.gate_proj", p + "switch_mlp.up_proj", p + "switch_mlp.down_proj", inds);
        y = mx::sum(mx::multiply(y, mx::expand_dims(scores, -1)), -2);

        const mx::array shared = weights_.linear(
            lmp::model::mlxl::swiglu(
                weights_.linear(x, p + "shared_expert.gate_proj"),
                weights_.linear(x, p + "shared_expert.up_proj")),
            p + "shared_expert.down_proj");
        const mx::array shared_gate = mx::sigmoid(weights_.linear(x, p + "shared_expert_gate"));
        return mx::add(y, mx::multiply(shared_gate, shared));
    }

    mx::array forward_linear_layer(int layer, const mx::array& x, int seq_len) {
        const std::string p = prefix_ + "layers." + std::to_string(layer) + ".";
        mx::array h = rms_norm(x, p + "input_layernorm.weight");
        mx::array attn_out = forward_gated_delta(p + "linear_attn.", h, ssm_caches_[static_cast<std::size_t>(layer)], seq_len);
        h = mx::add(x, attn_out);
        mx::array mlp_in = rms_norm(h, p + "post_attention_layernorm.weight");
        return mx::add(h, forward_moe(layer, mlp_in));
    }

    mx::array forward_full_attn_layer(int layer, const mx::array& x, int seq_len) {
        const std::string p = prefix_ + "layers." + std::to_string(layer) + ".";
        mx::array h = rms_norm(x, p + "input_layernorm.weight");
        mx::array attn_out = forward_self_attn(p + "self_attn.", h, kv_caches_[static_cast<std::size_t>(layer)], seq_len);
        h = mx::add(x, attn_out);
        mx::array mlp_in = rms_norm(h, p + "post_attention_layernorm.weight");
        return mx::add(h, forward_moe(layer, mlp_in));
    }

private:
    std::string prefix_;
    Qwen35MoeConfig cfg_{};
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
                    updated.emplace_back(key, mx::add(val, mx::array(1.0f)));
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
        cache.conv_state = mx::slice(conv_input, {0, conv_input.shape()[1] - (cfg_.linear_conv_kernel_dim - 1), 0},
                                     {B, conv_input.shape()[1], conv_dim});

        mx::array conv_w = weights_.get(p + "conv1d.weight");
        mx::array conv_out = lmp::model::mlxl::silu(
            mx::conv1d(conv_input, conv_w, /*stride=*/1, /*padding=*/0, /*dilation=*/1, conv_dim));

        auto q_part = mx::slice(conv_out, {0, 0, 0}, {B, S, key_dim});
        auto k_part = mx::slice(conv_out, {0, 0, key_dim}, {B, S, 2 * key_dim});
        auto v_part = mx::slice(conv_out, {0, 0, 2 * key_dim}, {B, S, conv_dim});

        mx::array q = mx::reshape(q_part, {B, S, cfg_.linear_num_key_heads, cfg_.linear_key_head_dim});
        mx::array k = mx::reshape(k_part, {B, S, cfg_.linear_num_key_heads, cfg_.linear_key_head_dim});
        mx::array v = mx::reshape(v_part, {B, S, cfg_.linear_num_value_heads, cfg_.linear_value_head_dim});
        z = mx::reshape(z, {B, S, cfg_.linear_num_value_heads, cfg_.linear_value_head_dim});

        const float inv_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.linear_key_head_dim));
        const mx::array inv2 = mx::array(inv_scale * inv_scale);
        const mx::array inv1 = mx::array(inv_scale);
        q = mx::multiply(inv2, mx::fast::rms_norm(q, mx::ones({cfg_.linear_key_head_dim}), 1e-6f));
        k = mx::multiply(inv1, mx::fast::rms_norm(k, mx::ones({cfg_.linear_key_head_dim}), 1e-6f));

        mx::array a_log = weights_.get(p + "A_log");
        mx::array dt_bias = weights_.get(p + "dt_bias");
        auto [out, state] = lmp::model::mlxl::gated_delta_update(q, k, v, a, b, a_log, dt_bias, cache.delta_state);
        cache.delta_state = state;
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
        mx::array queries = mx::reshape(
            mx::slice(q_heads, {0, 0, 0, 0}, {B, L, n_heads, head_dim}),
            {B, L, n_heads * head_dim});
        mx::array gate = mx::reshape(
            mx::slice(q_heads, {0, 0, 0, head_dim}, {B, L, n_heads, head_dim * 2}),
            {B, L, n_heads * head_dim});

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
        mx::array attn = (L > 1)
                             ? ff::scaled_dot_product_attention(queries, k_cat, v_cat, scale, "causal")
                             : ff::scaled_dot_product_attention(
                                   queries, k_cat, v_cat, scale, std::string{}, std::vector<mx::array>{});
        attn = mx::reshape(mx::transpose(attn, {0, 2, 1, 3}), {B, L, n_heads * head_dim});
        attn = mx::multiply(attn, mx::sigmoid(gate));
        return weights_.linear(attn, p + "o_proj");
    }
};

} // namespace lmp::model::mlxl

#endif // LMP_HAVE_MLX

#endif // LLM_MODELS_QWEN35_MOE_MODEL_HPP
