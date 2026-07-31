#ifndef LLM_MLX_ACTIVATIONS_HPP
#define LLM_MLX_ACTIVATIONS_HPP

#if LMP_HAVE_MLX

#include "mlx/ops.h"
#include "mlx/fast.h"

namespace lmp::model::mlxl {

namespace mx = mlx::core;

inline mx::array silu(const mx::array& x) {
    return mx::multiply(x, mx::sigmoid(x));
}

inline mx::array swiglu(const mx::array& gate, const mx::array& up) {
    return mx::multiply(silu(gate), up);
}

// Matches mlx-lm Qwen3NextRMSNormGated: silu(gate) * rms_norm(hidden) in float32.
inline mx::array precise_rms_norm_gated(const mx::array& hidden,
                                        const mx::array& gate,
                                        const mx::array& weight,
                                        float eps) {
    const mx::array normed = mx::fast::rms_norm(hidden, weight, eps);
    const mx::array gate_f = silu(mx::astype(gate, mx::float32));
    const mx::array norm_f = mx::astype(normed, mx::float32);
    return mx::astype(mx::multiply(gate_f, norm_f), hidden.dtype());
}

} // namespace lmp::model::mlxl

#endif // LMP_HAVE_MLX

#endif // LLM_MLX_ACTIVATIONS_HPP
