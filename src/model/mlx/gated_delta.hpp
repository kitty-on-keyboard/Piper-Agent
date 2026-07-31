#ifndef LLM_MLX_GATED_DELTA_HPP
#define LLM_MLX_GATED_DELTA_HPP

#if LMP_HAVE_MLX

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "activations.hpp"

#include "mlx/backend/metal/metal.h"
#include "mlx/fast.h"
#include "mlx/ops.h"

namespace lmp::model::mlxl {

namespace mx = mlx::core;

inline mx::array softplus(const mx::array& x) {
    return mx::log(mx::add(mx::array(1.0f), mx::exp(x)));
}

// Left uncompiled deliberately. mx::compile fuses this chain and, in isolation,
// measures 18.4 us -> 3.3 us per call (diag_main.cpp `chain`). End to end it moved
// decode by nothing: 28.1 -> 27.9 tok/s, inside run-to-run noise. The MoE block's
// cost is not in its elementwise ops, so the fusion was reverted rather than shipped
// on the strength of a micro-benchmark (S5.11: no performance change without a
// number, and this one's number was zero).
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

// --- reference path: one MLX op-set per timestep -----------------------------
//
// Kept as the definition of the recurrence, and as the thing the fused kernel is
// tested against (tests/model/diag_main.cpp `scan`). NOT the production path: at
// T timesteps x 30 linear layers this issues thousands of sequential kernel
// launches per prefill, which is what made prefill 70x slower than LM Studio.
inline std::pair<mx::array, mx::array> gated_delta_update_ops(const mx::array& q,
                                                              const mx::array& k,
                                                              const mx::array& v,
                                                              const mx::array& g,
                                                              const mx::array& beta,
                                                              const mx::array& state) {
    const int B = static_cast<int>(q.shape()[0]);
    const int T = static_cast<int>(q.shape()[1]);
    const int Hk = static_cast<int>(k.shape()[2]);
    const int Hv = static_cast<int>(v.shape()[2]);
    const int Dv = static_cast<int>(v.shape()[3]);
    const int Dk = static_cast<int>(k.shape()[3]);

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

    std::vector<mx::array> ys;
    ys.reserve(static_cast<std::size_t>(T));
    mx::array st = state;
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

// --- production path: the whole recurrence in ONE kernel launch --------------
//
// Same arithmetic, same order, same float32 accumulation as the loop above --
// the difference is where it runs. The [Dv, Dk] state lives in registers for the
// entire sequence instead of being written to and re-read from a 2 MB array T
// times, and the T-step loop is inside the kernel rather than being T rounds of
// MLX graph construction plus dispatch.
//
// This is the same kernel mlx-lm uses (mlx_lm/models/gated_delta.py), which is
// what LM Studio runs on this machine; the loop above is the path mlx-lm labels
// its reference implementation. Porting it is how the 70x prefill gap closes.
//
// Thread layout: one simdgroup (32 lanes along x) owns one row of the state,
// each lane holding Dk/32 columns in registers. grid.y walks Dv rows, grid.z
// walks (batch, value head). Requires Dk % 32 == 0.
inline const char* gated_delta_kernel_source() {
    return R"METAL(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g, beta: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }
          // Advance to the next timestep.
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
    )METAL";
}

inline std::pair<mx::array, mx::array> gated_delta_update_kernel(const mx::array& q,
                                                                 const mx::array& k,
                                                                 const mx::array& v,
                                                                 const mx::array& g,
                                                                 const mx::array& beta,
                                                                 const mx::array& state) {
    const int B = static_cast<int>(q.shape()[0]);
    const int T = static_cast<int>(q.shape()[1]);
    const int Hk = static_cast<int>(k.shape()[2]);
    const int Dk = static_cast<int>(k.shape()[3]);
    const int Hv = static_cast<int>(v.shape()[2]);
    const int Dv = static_cast<int>(v.shape()[3]);

    // Built once: metal_kernel() assembles and caches the compiled variant, but the
    // source string and name vectors are rebuilt on every call otherwise.
    static const mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
        "lmp_gated_delta_scan",
        {"q", "k", "v", "g", "beta", "state_in", "T"},
        {"y", "state_out"},
        gated_delta_kernel_source());

    // Activations enter in their own dtype and only the recurrent state is float32 --
    // mlx-lm's ("InT", input_type) / ("StT", state_type) split, and its
    // output_dtypes=[input_type, state_type].
    //
    // This used to cast q,k,v,g,beta to float32 first, on the reasoning that the
    // recurrence runs in float32 anyway. It does: the accumulators below are float and
    // every read is converted on load, so the arithmetic is identical either way. What
    // the casts actually bought was five materialised float32 copies of the inputs per
    // call -- five extra dispatches and twice the bytes to read, thirty times a token --
    // for values the kernel was going to widen for free. InT applies to y; the state is
    // written as StT so keeping it float32 no longer forces the activations to match it.
    std::vector<mx::array> outs = kernel(
        {q, k, v, g, beta, state, mx::array(T, mx::int32)},
        {mx::Shape{B, T, Hv, Dv}, mx::Shape{B, Hv, Dv, Dk}},
        {q.dtype(), state.dtype()},
        {32, Dv, B * Hv},
        {32, 4, 1},
        {{"InT", q.dtype()}, {"StT", state.dtype()}, {"Dk", Dk}, {"Dv", Dv}, {"Hk", Hk},
         {"Hv", Hv}},
        std::nullopt,
        false,
        {});
    return {outs[0], outs[1]};
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
    const int Hv = static_cast<int>(v.shape()[2]);
    const int Dv = static_cast<int>(v.shape()[3]);
    const int Dk = static_cast<int>(k.shape()[3]);

    mx::array beta = mx::sigmoid(b);
    mx::array g = compute_g(a_log, a, dt_bias);

    if (!state.has_value()) {
        state = mx::zeros({B, Hv, Dv, Dk}, mx::float32);
    }

    // Dk % 32 is the kernel's thread-layout precondition; no checkpoint this project
    // loads violates it, but a head dim that did would silently index out of bounds.
    // y leaves in q's dtype; only the recurrent state stays float32. This mirrors
    // mlx-lm's `return y.astype(q.dtype), state` and its output_dtypes=[input_type,
    // state_type], and it is load-bearing for speed, not just for tidiness: the block's
    // result feeds out_proj and then the residual stream, so a float32 y silently
    // promotes the hidden state for the WHOLE REST OF THE MODEL. Every quantized matmul
    // downstream -- all 40 MoE blocks included -- then runs its float32 activation path
    // against bf16 weights. That single promotion was the bulk of a 34.9 ms decode step.
    auto [y, new_state] = (mx::metal::is_available() && Dk % 32 == 0)
                              ? gated_delta_update_kernel(q, k, v, g, beta, *state)
                              : gated_delta_update_ops(q, k, v, g, beta, *state);
    return {mx::astype(y, q.dtype()), new_state};
}

} // namespace lmp::model::mlxl

#endif // LMP_HAVE_MLX

#endif // LLM_MLX_GATED_DELTA_HPP
