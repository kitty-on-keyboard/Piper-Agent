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
                                 const mx::array& indices,
                                 const std::string& gate_up_base) {
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

    // ONE gather_qmm where there were two. gate_proj and up_proj consume the same
    // `work` rows and the same expert indices and differ only in their weights, so a
    // checkpoint whose two matrices have been stacked along the output axis answers both
    // in a single dispatch -- see fuse_expert_gate_up() in qwen35_moe_model.hpp, which
    // builds the stacked entry at load and erases the halves.
    //
    // This reads the same bytes and computes the same products; the only thing that
    // changes is how many kernel launches the step costs. That matters here and not in
    // prefill: at decode this model gathers 8 of 256 experts for a single row, so
    // gather_qmm is launch-latency-bound, and 40 layers x 1 saved launch is 40 fewer
    // per token. Numerically it is a no-op -- each output row is an independent dot
    // product and affine grouping runs along the INPUT axis, which concatenating output
    // rows does not touch -- so the fused and unfused paths must agree bit for bit.
    // tests/model/test_switch_glu_fusion.cpp asserts exactly that.
    const bool fused = !gate_up_base.empty() && ws.is_quantized(gate_up_base);
    const mx::array activated = [&] {
        if (fused) {
            const mx::array gu = switch_linear(work, ws, gate_up_base, idx, do_sort);
            // gate first, then up -- the order fuse_expert_gate_up() concatenated in.
            const std::vector<mx::array> half = mx::split(gu, 2, -1);
            return swiglu(half[0], half[1]);
        }
        const mx::array x_up = switch_linear(work, ws, up_base, idx, do_sort);
        const mx::array x_gate = switch_linear(work, ws, gate_base, idx, do_sort);
        return swiglu(x_gate, x_up);
    }();
    mx::array out = switch_linear(activated, ws, down_base, idx, do_sort);

    if (do_sort) {
        out = scatter_unsort(out, inv_order, idx_shape);
    }
    return mx::squeeze(out, -2);
}

// `gate_up_base` is optional: empty (the default) is the two-dispatch path, which is
// what the diag drivers and any checkpoint that was not fused at load still take.
inline mx::array switch_glu(const mx::array& x,
                            const WeightStore& ws,
                            const std::string& gate_base,
                            const std::string& up_base,
                            const std::string& down_base,
                            const mx::array& indices,
                            const std::string& gate_up_base = {}) {
    return switch_glu_impl(x, ws, gate_base, up_base, down_base, indices, gate_up_base);
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
