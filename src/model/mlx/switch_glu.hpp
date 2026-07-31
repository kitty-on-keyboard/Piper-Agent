#ifndef LLM_MLX_SWITCH_GLU_HPP
#define LLM_MLX_SWITCH_GLU_HPP

#if LMP_HAVE_MLX

#include "activations.hpp"
#include "weight_store.hpp"


#include <vector>

#include "mlx/ops.h"

namespace lmp::model::mlxl {

namespace mx = mlx::core;

inline mx::array switch_linear(const mx::array& x,
                               const WeightStore& ws,
                               const std::string& base_key,
                               const mx::array& indices,
                               bool sorted_indices = false) {
    if (ws.is_quantized(base_key)) {
        const QuantWeight qw = ws.quant(base_key);
        return mx::gather_qmm(
            x, qw.weight, qw.scales, qw.biases, std::nullopt, indices,
            /*transpose=*/true, qw.group_size, qw.bits, qw.mode, sorted_indices);
    }
    const mx::array& w = ws.get(base_key + ".weight");
    return mx::gather_mm(x, mx::swapaxes(w, -1, -2), std::nullopt, indices, sorted_indices);
}

// Mirrors mlx-lm switch_layers._gather_sort / _scatter_unsort.
inline std::tuple<mx::array, mx::array, mx::array> gather_sort(const mx::array& x,
                                                                 const mx::array& indices) {
    const auto idx_shape = indices.shape();
    const int top_k = static_cast<int>(idx_shape.back());
    mx::array flat_idx = mx::reshape(indices, {-1});
    mx::array order = mx::argsort(flat_idx);
    mx::array inv_order = mx::argsort(order);

    mx::array flat_x = mx::flatten(x, 0, -3);
    mx::array row_idx = mx::floor_divide(order, mx::array(static_cast<int32_t>(top_k)));
    mx::array sorted_x = mx::take(flat_x, row_idx, 0);
    // mlx-lm keeps sorted expert indices flat (not reshaped) for gather_mm/qmm.
    mx::array sorted_idx = mx::take(flat_idx, order, 0);
    return {sorted_x, sorted_idx, inv_order};
}

inline mx::array scatter_unsort(const mx::array& x,
                                const mx::array& inv_order,
                                const std::vector<int>& idx_shape) {
    mx::array out = mx::take(x, inv_order, 0);
    const mx::Shape shape(idx_shape.begin(), idx_shape.end());
    return mx::unflatten(out, 0, shape);
}

inline mx::array switch_glu_impl(const mx::array& x,
                                 const WeightStore& ws,
                                 const std::string& gate_base,
                                 const std::string& up_base,
                                 const std::string& down_base,
                                 const mx::array& indices) {
    // Both axes in one op, as mlx-lm's x[:, None, None, :] does. Two chained
    // expand_dims built two nodes where one will do. Note the axes are {-3, -2}, not
    // {-2, -2}: multi-axis expand_dims indexes into the OUTPUT shape, so the two
    // insertion points are distinct there even though chaining inserts at -2 twice
    // (and {-2, -2} is rejected outright as a duplicate axis).
    mx::array expanded = mx::expand_dims(x, std::vector<int>{-3, -2});
    const bool do_sort = static_cast<int>(indices.size()) >= 64;

    mx::array idx = indices;
    mx::array work = expanded;
    std::vector<int> idx_shape;
    mx::array inv_order = mx::zeros({0}, mx::int32);
    if (do_sort) {
        std::tie(work, idx, inv_order) = gather_sort(expanded, indices);
        const auto s = indices.shape();
        idx_shape.assign(s.begin(), s.end());
    }

    const mx::array x_up = switch_linear(work, ws, up_base, idx, do_sort);
    const mx::array x_gate = switch_linear(work, ws, gate_base, idx, do_sort);
    const mx::array activated = swiglu(x_gate, x_up);
    mx::array out = switch_linear(activated, ws, down_base, idx, do_sort);

    if (do_sort) {
        out = scatter_unsort(out, inv_order, idx_shape);
    }
    return mx::squeeze(out, -2);
}

inline mx::array switch_glu(const mx::array& x,
                            const WeightStore& ws,
                            const std::string& gate_base,
                            const std::string& up_base,
                            const std::string& down_base,
                            const mx::array& indices) {
    return switch_glu_impl(x, ws, gate_base, up_base, down_base, indices);
}

inline std::pair<mx::array, mx::array> moe_topk(const mx::array& gate_logits, int top_k, bool norm_topk) {
    // precise=true accumulates in float32, which is what mlx-lm's SparseMoeBlock passes.
    // Left at the bf16 default, 256 router logits land close enough together that the
    // top-8 argpartition picks a different expert set than the reference does.
    const mx::array gates = mx::softmax(gate_logits, -1, /*precise=*/true);
    const int experts = static_cast<int>(gates.shape()[2]);
    mx::array part_inds = mx::argpartition(gates, experts - top_k, 2);
    mx::array inds = mx::slice(part_inds, {0, 0, experts - top_k}, {gates.shape()[0], gates.shape()[1], experts});
    mx::array scores = mx::take_along_axis(gates, inds, 2);
    if (norm_topk) {
        scores = mx::divide(scores, mx::sum(scores, -1, /*keepdims=*/true));
    }
    return {inds, scores};
}

} // namespace lmp::model::mlxl

#endif // LMP_HAVE_MLX

#endif // LLM_MLX_SWITCH_GLU_HPP
