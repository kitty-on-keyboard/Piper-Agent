#ifndef LLM_MLX_GATED_DELTA_HPP
#define LLM_MLX_GATED_DELTA_HPP

#if LMP_HAVE_MLX

#include <optional>
#include <vector>

#include "activations.hpp"

#include "mlx/ops.h"

namespace lmp::model::mlxl {

namespace mx = mlx::core;

inline mx::array softplus(const mx::array& x) {
    return mx::log(mx::add(mx::array(1.0f), mx::exp(x)));
}

inline mx::array compute_g(const mx::array& a_log, const mx::array& a, const mx::array& dt_bias) {
    const mx::array sp = softplus(mx::add(a, dt_bias));
    return mx::exp(mx::multiply(mx::multiply(mx::array(-1.0f), mx::exp(a_log)), sp));
}

inline std::pair<mx::array, mx::array> gated_delta_step(const mx::array& q,
                                                        const mx::array& k,
                                                        const mx::array& v,
                                                        const mx::array& g,
                                                        const mx::array& beta,
                                                        const mx::array& state) {
    // g, beta: [B, H] — broadcast over Dv and Dk like mlx-lm's g[..., None, None]
    mx::array decay = mx::expand_dims(mx::expand_dims(g, -1), -1);
    mx::array new_state = mx::multiply(state, decay);
    mx::array kv_mem = mx::sum(mx::multiply(new_state, mx::expand_dims(k, -2)), -1);
    mx::array delta = mx::multiply(mx::subtract(v, kv_mem), mx::expand_dims(beta, -1));
    new_state = mx::add(new_state, mx::multiply(mx::expand_dims(k, -2), mx::expand_dims(delta, -1)));
    mx::array y = mx::sum(mx::multiply(new_state, mx::expand_dims(q, -2)), -1);
    return {y, new_state};
}

inline std::pair<mx::array, mx::array> gated_delta_update(const mx::array& q,
                                                          const mx::array& k,
                                                          const mx::array& v,
                                                          const mx::array& a,
                                                          const mx::array& b,
                                                          const mx::array& a_log,
                                                          const mx::array& dt_bias,
                                                          std::optional<mx::array> state) {
    const int B = static_cast<int>(q.shape()[0]);
    const int T = static_cast<int>(q.shape()[1]);
    const int Hk = static_cast<int>(k.shape()[2]);
    const int Hv = static_cast<int>(v.shape()[2]);
    const int Dv = static_cast<int>(v.shape()[3]);
    const int Dk = static_cast<int>(k.shape()[3]);

    mx::array beta = mx::sigmoid(b);
    mx::array g = compute_g(a_log, a, dt_bias);

    // GQA-style linear attention: key/query heads may be fewer than value heads
    // (e.g. Qwen 3.6: 16 key heads vs 32 value heads). Repeat along the head axis
    // so all per-timestep tensors line up with the [B, Hv, Dv, Dk] state.
    mx::array qr = q;
    mx::array kr = k;
    if (Hv != Hk) {
        const int repeat_factor = Hv / Hk;
        qr = mx::repeat(q, repeat_factor, 2);
        kr = mx::repeat(k, repeat_factor, 2);
    }

    if (!state.has_value()) {
        state = mx::zeros({B, Hv, Dv, Dk}, mx::float32);
    }

    std::vector<mx::array> ys;
    ys.reserve(static_cast<std::size_t>(T));
    mx::array st = *state;
    for (int t = 0; t < T; ++t) {
        mx::array qt = mx::squeeze(mx::slice(qr, {0, t, 0, 0}, {B, t + 1, Hv, Dk}), 1);
        mx::array kt = mx::squeeze(mx::slice(kr, {0, t, 0, 0}, {B, t + 1, Hv, Dk}), 1);
        mx::array vt = mx::squeeze(mx::slice(v, {0, t, 0, 0}, {B, t + 1, Hv, Dv}), 1);
        mx::array gt = mx::squeeze(mx::slice(g, {0, t, 0}, {B, t + 1, Hv}), 1);
        mx::array bt = mx::squeeze(mx::slice(beta, {0, t, 0}, {B, t + 1, Hv}), 1);
        auto [yt, st_out] = gated_delta_step(qt, kt, vt, gt, bt, st);
        st = st_out;
        ys.push_back(mx::expand_dims(yt, 1));
    }
    return {mx::concatenate(ys, 1), st};
}

} // namespace lmp::model::mlxl

#endif // LMP_HAVE_MLX

#endif // LLM_MLX_GATED_DELTA_HPP
