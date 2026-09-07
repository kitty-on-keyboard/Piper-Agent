#ifndef LLM_MLX_GATED_RESIDUAL_HPP
#define LLM_MLX_GATED_RESIDUAL_HPP

#if LMP_HAVE_MLX

#include "activations.hpp"
#include "weight_store.hpp"

#include "mlx/fast.h"
#include "mlx/ops.h"

namespace lmp::model::mlxl {

namespace mx = mlx::core;

// Group RMSNorm: one statistic per `group` trailing features, then multiply by w.
// w is the already-folded (1+raw) scale.
inline mx::array group_rms_norm(const mx::array& x, const mx::array& w, int group, float eps) {
    mx::Shape shape(x.shape().begin(), x.shape().end());
    const int last = static_cast<int>(shape.back());
    shape.back() = last / group;
    shape.push_back(group);
    mx::array g = mx::reshape(x, mx::Shape(shape.begin(), shape.end()));
    g = mx::fast::rms_norm(g, std::nullopt, eps);
    g = mx::reshape(g, x.shape());
    return mx::multiply(g, w);
}

struct GrTensors {
    mx::array mixed;
    mx::array hyper;
    mx::array inject; // [..., hc]; empty when use_combine is false
    bool combine{true};
};

inline GrTensors gated_residual_read(const mx::array& hyper, const WeightStore& w,
                                     const std::string& prefix, int hc, int d, int rank,
                                     float eps, bool use_combine) {
    (void)rank;
    const mx::array& hc_w = w.get(prefix + "hc_norm.weight");
    mx::array normed = group_rms_norm(hyper, hc_w, d, eps);
    const mx::array inv = mx::array(1.0f / static_cast<float>(hc), hyper.dtype());
    mx::array low = w.linear(normed, prefix + "input_mix_weight_down");
    low = silu(mx::multiply(low, inv));
    mx::array gate = mx::sigmoid(w.linear(low, prefix + "input_mix_weight_up"));
    mx::Shape gs(gate.shape().begin(), gate.shape().end());
    gs.back() = hc;
    gs.push_back(d);
    gate = mx::reshape(gate, mx::Shape(gs.begin(), gs.end()));
    mx::array rbar = mx::reshape(normed, gate.shape());
    mx::array mixed = mx::mean(mx::multiply(gate, rbar), /*axis=*/-2);
    GrTensors out{std::move(mixed), hyper, mx::array(0.0f, hyper.dtype()), use_combine};
    if (!use_combine) {
        return out;
    }
    mx::array inj = w.linear(normed, prefix + "block_inject_weight");
    inj = mx::multiply(mx::array(2.0f, hyper.dtype()), mx::sigmoid(mx::multiply(inj, inv)));
    out.inject = std::move(inj);
    return out;
}

inline mx::array gated_residual_write(const GrTensors& gr, const mx::array& y) {
    // y: [..., d]  inject: [..., hc]  hyper: [..., hc*d]
    mx::array scaled = mx::multiply(mx::expand_dims(y, -2), mx::expand_dims(gr.inject, -1));
    mx::Shape flat(scaled.shape().begin(), scaled.shape().end() - 2);
    const int hc = static_cast<int>(scaled.shape()[scaled.ndim() - 2]);
    const int d = static_cast<int>(scaled.shape()[scaled.ndim() - 1]);
    flat.push_back(hc * d);
    return mx::add(gr.hyper, mx::reshape(scaled, mx::Shape(flat.begin(), flat.end())));
}

inline mx::array tile_hyper(const mx::array& h, int hc) {
    std::vector<mx::array> copies(static_cast<std::size_t>(hc), h);
    return mx::concatenate(copies, -1);
}

// Exact L2, not RMS: x * rsqrt(sum(x*x)+eps) in f32. Flash GDN q/k only.
inline mx::array exact_l2_norm(const mx::array& x, float eps = 1e-6f) {
    mx::array xf = mx::astype(x, mx::float32);
    mx::array acc = mx::sum(mx::multiply(xf, xf), -1, true);
    mx::array y = mx::multiply(xf, mx::rsqrt(mx::add(acc, mx::array(eps, mx::float32))));
    return mx::astype(y, x.dtype());
}

} // namespace lmp::model::mlxl

#endif
#endif
