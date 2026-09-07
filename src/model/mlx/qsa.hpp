#ifndef LLM_MLX_QSA_HPP
#define LLM_MLX_QSA_HPP

#if LMP_HAVE_MLX

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include "weight_store.hpp"
#include "src/model/qwen4_exp_ops.hpp"

#include "mlx/fast.h"
#include "mlx/ops.h"

namespace lmp::model::mlxl {

namespace mx = mlx::core;

// Partial RoPE at explicit positions (block starts are not contiguous).
inline mx::array rope_at_positions(const mx::array& x, const mx::array& positions,
                                   int rotary_dim, float theta) {
    const int ndim = static_cast<int>(x.ndim());
    mx::array inv_freq = mx::power(
        mx::array(theta, mx::float32),
        mx::divide(mx::negative(mx::arange(0, rotary_dim, 2, mx::float32)),
                   mx::array(static_cast<float>(rotary_dim), mx::float32)));
    mx::array pos = mx::astype(positions, mx::float32);
    while (static_cast<int>(pos.ndim()) < 2) {
        pos = mx::expand_dims(pos, 0);
    }
    mx::array freqs = mx::multiply(mx::expand_dims(pos, -1), inv_freq);
    mx::array emb = mx::concatenate({freqs, freqs}, -1);
    mx::array cos = mx::astype(mx::cos(emb), x.dtype());
    mx::array sin = mx::astype(mx::sin(emb), x.dtype());
    if (ndim == 4) {
        cos = mx::expand_dims(cos, 2);
        sin = mx::expand_dims(sin, 2);
    }
    mx::Shape lo(ndim, 0);
    mx::Shape hi(x.shape().begin(), x.shape().end());
    hi.back() = rotary_dim;
    mx::array xr = mx::slice(x, lo, hi);
    lo.back() = rotary_dim;
    hi.back() = static_cast<int>(x.shape().back());
    mx::array xp = mx::slice(x, lo, hi);
    const int half = rotary_dim / 2;
    mx::Shape hlo(ndim, 0);
    mx::Shape hhi(xr.shape().begin(), xr.shape().end());
    hhi.back() = half;
    mx::array x1 = mx::slice(xr, hlo, hhi);
    hlo.back() = half;
    hhi.back() = rotary_dim;
    mx::array x2 = mx::slice(xr, hlo, hhi);
    mx::array rot = mx::concatenate({mx::negative(x2), x1}, -1);
    xr = mx::add(mx::multiply(xr, cos), mx::multiply(rot, sin));
    if (static_cast<int>(x.shape().back()) == rotary_dim) {
        return xr;
    }
    return mx::concatenate({xr, xp}, -1);
}

struct IndexerCache {
    std::optional<mx::array> keys; // [B, T, D]
    int offset{0};

    void clear() noexcept {
        keys.reset();
        offset = 0;
    }

    mx::array update(const mx::array& k) {
        if (!keys.has_value()) {
            keys = k;
        } else {
            keys = mx::concatenate({*keys, k}, 1);
        }
        offset = static_cast<int>(keys->shape()[1]);
        return *keys;
    }

    void truncate_to(int n) noexcept {
        n = std::clamp(n, 0, offset);
        if (keys.has_value() && n < offset) {
            const int B = static_cast<int>(keys->shape()[0]);
            const int D = static_cast<int>(keys->shape()[2]);
            keys = mx::slice(*keys, {0, 0, 0}, {B, n, D});
        }
        offset = n;
    }
};

// Boolean mask [B, 1, S, kv_len], or nullopt to use dense causal attention.
inline std::optional<mx::array> qsa_indexer_mask(
    const mx::array& x, const WeightStore& w, const std::string& p, IndexerCache& cache,
    int offset, int n_heads, int kv_heads, int head_dim, int budget, int compress,
    int rotary_dim, float theta, float eps) {
    const int B = static_cast<int>(x.shape()[0]);
    const int S = static_cast<int>(x.shape()[1]);
    mx::array qk = w.linear(x, p + "index_qk_proj");
    const int q_dim = n_heads * head_dim;
    const int k_dim = kv_heads * head_dim;
    mx::array q = mx::reshape(mx::slice(qk, {0, 0, 0}, {B, S, q_dim}), {B, S, n_heads, head_dim});
    mx::array raw_k =
        mx::reshape(mx::slice(qk, {0, 0, q_dim}, {B, S, q_dim + k_dim}), {B, S, head_dim});
    raw_k = cache.update(raw_k);
    const int kv_len = static_cast<int>(raw_k.shape()[1]);
    if (kv_len <= budget) {
        return std::nullopt;
    }

    q = mx::fast::rms_norm(q, w.get(p + "q_layernorm.weight"), eps);
    mx::array q_pos = mx::add(mx::arange(S), mx::array(offset));
    q = rope_at_positions(q, q_pos, rotary_dim, theta);

    const int n_blocks = kv_len / compress;
    mx::array pooled = mx::slice(raw_k, {0, 0, 0}, {B, n_blocks * compress, head_dim});
    pooled = mx::reshape(pooled, {B, n_blocks, compress, head_dim});
    pooled = mx::mean(pooled, 2);
    pooled = mx::fast::rms_norm(pooled, w.get(p + "k_layernorm.weight"), eps);
    mx::array block_starts = mx::multiply(mx::arange(n_blocks), mx::array(compress));
    pooled = rope_at_positions(pooled, block_starts, rotary_dim, theta);

    mx::array qf = mx::astype(q, mx::float32);
    mx::array kf = mx::astype(pooled, mx::float32);
    mx::array k_bt = mx::swapaxes(kf, 1, 2); // [B, D, n_blocks]
    mx::array dots = mx::matmul(qf, k_bt);  // [B, S, H, n_blocks]
    mx::array scores =
        mx::divide(mx::sum(mx::maximum(dots, mx::array(0.0f)), 2),
                   mx::array(std::sqrt(static_cast<float>(head_dim)), mx::float32));
    mx::eval(scores);

    const int block_topk = budget / compress;
    const float* sp = scores.data<float>();
    std::vector<bool> keep(static_cast<std::size_t>(B * S * kv_len), false);
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            const int qp = offset + s;
            const float* row = sp + (b * S + s) * n_blocks;
            auto m = qwen4::qsa_keep_mask(qp, kv_len, compress, block_topk, row);
            for (int t = 0; t < kv_len; ++t) {
                keep[static_cast<std::size_t>((b * S + s) * kv_len + t)] =
                    m[static_cast<std::size_t>(t)] != 0;
            }
        }
    }
    // bool vector is not a contiguous buffer; pack to uint8.
    std::vector<std::uint8_t> packed(keep.size());
    for (std::size_t i = 0; i < keep.size(); ++i) {
        packed[i] = keep[i] ? 1 : 0;
    }
    mx::array mask(packed.data(), {B, 1, S, kv_len}, mx::uint8);
    return mx::astype(mask, mx::bool_);
}


} // namespace lmp::model::mlxl

#endif
#endif
