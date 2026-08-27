#ifndef LLM_MLX_QUANT_ATTENTION_HPP
#define LLM_MLX_QUANT_ATTENTION_HPP

#if LMP_HAVE_MLX

// Attention over a quantized KV cache.
//
// MLX HAS NO QUANTIZED SDPA. Neither the 0.31.2 that LM Studio ships nor the 0.32.0 this
// builds against declares one in `mlx/fast.h` -- `scaled_dot_product_attention` is the
// only entry point, and it takes dense keys and values. mlx-lm does not have a private
// kernel either: it composes the whole thing out of two `quantized_matmul` calls in
// Python (`mlx_lm/models/base.py`, quantized_scaled_dot_product_attention). This is that
// function, in C++, and it is deliberately a transliteration -- the reference is the only
// implementation anyone has validated against these checkpoints, so where a choice looked
// arbitrary it was kept.
//
// The shape trick worth understanding: the cache holds [B, H_kv, S, D] quantized along
// the LAST axis. `quantized_matmul(q, K, transpose=true)` therefore reads K as a weight
// matrix of S output rows contracting over D -- giving scores [.., L, S] -- while
// `quantized_matmul(scores, V, transpose=false)` reads V as S input rows by D outputs,
// contracting over S. One stored layout serves both, which is why the cache does not need
// a transposed copy of V.

#include <limits>
#include <vector>

#include "kv_cache.hpp"

#include "mlx/array.h"
#include "mlx/ops.h"

namespace lmp::model::mlxl {

namespace mx = mlx::core;

namespace detail {

// Insert an axis into all three members at once. Grouped-query attention broadcasts one
// KV head across n_rep query heads, and the reference does this with a tree_map over the
// triple; getting one member out of step here would pair a key's words with another's
// scales, which decodes to plausible noise rather than failing.
inline QTensor expand_kv(const QTensor& t, int axis) {
    return QTensor{mx::expand_dims(t.w, axis), mx::expand_dims(t.scales, axis),
                   mx::expand_dims(t.biases, axis)};
}

} // namespace detail

// `causal` builds the same triangular mask the fused path gets from mask_mode "causal".
// At L == 1 there is nothing to mask -- one query attends to the entire cache -- so the
// caller passes false and the mask is skipped entirely.
[[nodiscard]] inline mx::array quantized_sdpa(const mx::array& queries, const QTensor& keys,
                                              const QTensor& values, float scale, bool causal,
                                              int group_size, int bits) {
    const auto qs = queries.shape();
    const int B = static_cast<int>(qs[0]);
    const int n_q_heads = static_cast<int>(qs[1]);
    const int L = static_cast<int>(qs[2]);
    const int D = static_cast<int>(qs[3]);
    const int n_kv_heads = static_cast<int>(keys.w.shape()[1]);
    const int n_repeats = n_q_heads / n_kv_heads;

    // THE DTYPE TRAP, and the reason this line is not `mx::multiply(queries,
    // mx::array(scale))`. A bare mx::array(float) is a strongly typed float32 array and
    // promotes whatever it touches; the identical mistake in forward_gated_delta put the
    // residual stream in float32 from layer 1 onward and cost 3x decode
    // ("The cause of the decode gap: one strongly-typed scalar"). Python's weakly typed
    // float does not do this, which is why the reference can write `queries *= scale`.
    mx::array q = mx::multiply(queries, mx::astype(mx::array(scale), queries.dtype()));

    QTensor k = keys;
    QTensor v = values;
    if (n_repeats > 1) {
        q = mx::reshape(q, {B, n_kv_heads, n_repeats, L, D});
        k = detail::expand_kv(k, -3);
        v = detail::expand_kv(v, -3);
    }

    mx::array scores = mx::quantized_matmul(q, k.w, k.scales, k.biases, /*transpose=*/true,
                                            group_size, bits, "affine");
    if (causal) {
        const int kL = static_cast<int>(scores.shape().back());
        const mx::array q_idx = mx::arange(kL - L, kL, mx::int32);
        const mx::array k_idx = mx::arange(0, kL, mx::int32);
        const mx::array keep = mx::greater_equal(mx::expand_dims(q_idx, -1),
                                                 mx::expand_dims(k_idx, 0));
        // finfo.min, not -inf: a fully masked row of -inf softmaxes to NaN, and the
        // reference is explicit about using the most negative FINITE value instead.
        const mx::array neg =
            mx::astype(mx::array(std::numeric_limits<float>::lowest()), scores.dtype());
        scores = mx::where(keep, scores, neg);
    }
    // precise=true accumulates the softmax in float32. The reference passes it, and the
    // router softmax in this model already does for the same reason.
    scores = mx::softmax(scores, std::vector<int>{-1}, /*precise=*/true);

    mx::array out = mx::quantized_matmul(scores, v.w, v.scales, v.biases, /*transpose=*/false,
                                         group_size, bits, "affine");
    if (n_repeats > 1) {
        out = mx::reshape(out, {B, n_q_heads, L, D});
    }
    return out;
}

} // namespace lmp::model::mlxl

#endif // LMP_HAVE_MLX

#endif // LLM_MLX_QUANT_ATTENTION_HPP
