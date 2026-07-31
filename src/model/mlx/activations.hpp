#ifndef LLM_MLX_ACTIVATIONS_HPP
#define LLM_MLX_ACTIVATIONS_HPP

#if LMP_HAVE_MLX

#include <vector>

#include "mlx/compile.h"
#include "mlx/ops.h"
#include "mlx/fast.h"

namespace lmp::model::mlxl {

namespace mx = mlx::core;

// --- why these are compiled -------------------------------------------------
//
// mlx-lm runs these chains under mx.compile, and `lmp_diag graph` shows exactly what
// that is worth. One decode step, primitive histogram, ours vs the reference:
//
//   QuantizedMatmul   391 vs 391      RMSNorm   191 vs 191
//   GatherQMM         120 vs 120      CustomKernel 30 vs 30
//
// -- the heavy ops are issued in identical counts. The whole difference was 919 extra
// elementwise and shape primitives, because the reference fuses four chains into 170
// Compiled* kernels and we were issuing ~1090 separate dispatches for the same
// arithmetic. That is what open item "small quantized matmuls are ~2x" actually was:
// not the matmuls, which are the same op with the same arguments, but the un-fused
// elementwise work sitting between them.
//
// An earlier pass measured mx::compile on compute_g and recorded "28.1 -> 27.9 tok/s,
// nothing" -- and that measurement was honest but is no longer in force. It was taken
// while the float32-residual bug made a decode step 34.9 ms, where ~0.5 ms of saved
// dispatch is 1.3% and invisible. The step is now ~11.9 ms and the same absolute saving
// is worth several percent. A null result measured under a since-fixed 3x bug has to be
// re-run, not inherited.
//
// Each site holds its compiled function in a function-local static, so the trace is
// built once and amortised over the run. mx::compile re-traces per distinct input
// shape/dtype, which for us means one trace for decode (S=1) and one per prefill chunk
// size -- the same retrace behaviour the reference has.
namespace detail {

inline const std::function<std::vector<mx::array>(const std::vector<mx::array>&)>&
compiled_silu() {
    static const auto f = mx::compile(+[](const std::vector<mx::array>& in) {
        return std::vector<mx::array>{mx::multiply(in[0], mx::sigmoid(in[0]))};
    });
    return f;
}

inline const std::function<std::vector<mx::array>(const std::vector<mx::array>&)>&
compiled_swiglu() {
    static const auto f = mx::compile(+[](const std::vector<mx::array>& in) {
        const mx::array& gate = in[0];
        const mx::array& up = in[1];
        return std::vector<mx::array>{
            mx::multiply(mx::multiply(gate, mx::sigmoid(gate)), up)};
    });
    return f;
}

inline const std::function<std::vector<mx::array>(const std::vector<mx::array>&)>&
compiled_rms_norm_gated_tail() {
    static const auto f = mx::compile(+[](const std::vector<mx::array>& in) {
        const mx::array& normed = in[0];
        const mx::array& gate = in[1];
        const mx::array gate_f = mx::astype(gate, mx::float32);
        const mx::array silu_f = mx::multiply(gate_f, mx::sigmoid(gate_f));
        // normed.dtype() rather than a captured hidden.dtype(): a capture-less lambda is
        // what mx::compile can turn into a function pointer, and the two are the same
        // dtype anyway -- rms_norm preserves its input's. Doing the cast inside the
        // compiled region rather than after it is what lets the fused kernel write the
        // final bf16 result directly instead of materialising a float32 intermediate.
        return std::vector<mx::array>{
            mx::astype(mx::multiply(silu_f, mx::astype(normed, mx::float32)),
                       normed.dtype())};
    });
    return f;
}

} // namespace detail

inline mx::array silu(const mx::array& x) {
    return detail::compiled_silu()({x})[0];
}

inline mx::array swiglu(const mx::array& gate, const mx::array& up) {
    return detail::compiled_swiglu()({gate, up})[0];
}

// Matches mlx-lm Qwen3NextRMSNormGated: silu(gate) * rms_norm(hidden) in float32.
// rms_norm stays outside the compiled region, as it does in the reference -- it is
// already one fused fast op, and the reference's compiled node likewise contains no
// RMSNorm.
inline mx::array precise_rms_norm_gated(const mx::array& hidden,
                                        const mx::array& gate,
                                        const mx::array& weight,
                                        float eps) {
    const mx::array normed = mx::fast::rms_norm(hidden, weight, eps);
    return detail::compiled_rms_norm_gated_tail()({normed, gate})[0];
}

} // namespace lmp::model::mlxl

#endif // LMP_HAVE_MLX

#endif // LLM_MLX_ACTIVATIONS_HPP
