#ifndef LLM_MLX_QGEMM_HPP
#define LLM_MLX_QGEMM_HPP

#if LMP_HAVE_MLX

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mlx/backend/metal/metal.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace lmp::model::mlxl {

namespace mx = mlx::core;

// Affine 4-bit, group 64, transpose=true: y = x @ dequant(W)^T.
// Default is stock mx::quantized_matmul. LMP_QGEMM=1 opts into a Metal kernel
// after a session self-check; mismatch or missing Metal stays on MLX.
// LMP_FUSE_GLU=1: fused dense gate+up+SwiGLU. LMP_FUSE_RMS=1: RMSNorm into QMV.
// Both default off. MoE expert qmm is not this path.
//
// v3: half-group QMV. One output row per simdgroup, KT=1024, each lane owns 32
// contiguous 4-bit weights (half of group-64). Weights stream as uint4 from
// device; x sits in threadgroup and is reused across M. Not v1/v2.

inline mx::array q4_mlx_qmm(const mx::array& x, const mx::array& w, const mx::array& scales,
                            const std::optional<mx::array>& biases, int group_size, int bits,
                            const std::string& mode) {
    return mx::quantized_matmul(x, w, scales, biases, /*transpose=*/true, group_size, bits,
                                mode);
}

// G0: materialise dequant then matmul in float32. Same contract as WeightStore::linear.
inline mx::array q4_ref_linear(const mx::array& x, const mx::array& w, const mx::array& scales,
                               const std::optional<mx::array>& biases, int group_size, int bits,
                               const std::string& mode) {
    const mx::array full = mx::astype(
        mx::dequantize(w, scales, biases, group_size, bits, mode), mx::float32);
    const mx::array xf = mx::astype(x, mx::float32);
    return mx::matmul(xf, mx::transpose(full, {-1, -2}));
}

inline int q4_rows(const mx::array& x) {
    int M = 1;
    const int nd = static_cast<int>(x.ndim());
    for (int i = 0; i < nd - 1; ++i) {
        M *= static_cast<int>(x.shape(i));
    }
    return M;
}

struct Q4Tune {
    int kt{1024};
    int nsg{8};
};

inline Q4Tune& q4_tune() {
    static Q4Tune t;
    return t;
}

inline void q4_set_tune(int kt, int nsg) {
    q4_tune() = Q4Tune{kt, nsg};
}

inline const char* q4_kernel_header() {
    return R"METAL(
        inline void q4_unpack32(uint4 p, thread float* q) {
          uint u[4] = {p.x, p.y, p.z, p.w};
          int t = 0;
          for (int i = 0; i < 4; ++i) {
            uint v = u[i];
            for (int b = 0; b < 8; ++b) {
              q[t++] = float(v & 15u);
              v >>= 4;
            }
          }
        }
    )METAL";
}

inline const char* q4_qmv_source() {
    return R"METAL(
        const int lane = int(thread_index_in_simdgroup);
        const int sg = int(thread_position_in_threadgroup.y);
        const int tg_y = int(threadgroup_position_in_grid.y);
        const int tig = int(thread_index_in_threadgroup);
        const int K = k_dim;
        const int N = n_dim;
        const int n = tg_y * NSG + sg;
        const int groups = K / 64;

        threadgroup float Xs[3][1024];
        float acc[3] = {0.0f, 0.0f, 0.0f};

        for (int kb = 0; kb < K; kb += KT) {
          for (int m = 0; m < M; ++m) {
            for (int i = tig; i < KT; i += (32 * NSG)) {
              const int k = kb + i;
              Xs[m][i] = (k < K) ? float(x[m * K + k]) : 0.0f;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          const int k0 = kb + lane * 32;
          if (n < N && k0 < K) {
            const device uint* wrow =
                ((const device uint*)w) + n * (K / 8) + (kb / 8) + lane * 4;
            const uint4 packed = *(const device uint4*)wrow;
            float qn[32];
            q4_unpack32(packed, qn);
            const int g = k0 / 64;
            const float s = float(scales[n * groups + g]);
            const float b = float(biases[n * groups + g]);
            const int off = lane * 32;
            for (int m = 0; m < M; ++m) {
              float qdot = 0.0f;
              float xsum = 0.0f;
              for (int i = 0; i < 32; ++i) {
                const float xv = Xs[m][off + i];
                xsum += xv;
                qdot += qn[i] * xv;
              }
              acc[m] += s * qdot + b * xsum;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (int m = 0; m < M; ++m) {
          acc[m] = simd_sum(acc[m]);
          if (lane == 0 && n < N) {
            y[m * N + n] = T(acc[m]);
          }
        }
    )METAL";
}

inline mx::array q4_metal_linear(const mx::array& x, const mx::array& w, const mx::array& scales,
                                 const mx::array& biases) {
    const int K = static_cast<int>(x.shape(-1));
    const int M = q4_rows(x);
    const int N = static_cast<int>(scales.shape(0));
    if ((M != 1 && M != 3) || K <= 0 || (K % 64) != 0 || N <= 0) {
        throw std::invalid_argument("q4_metal_linear: unsupported shape");
    }

    mx::array x2 = (x.ndim() == 2) ? x : mx::reshape(x, {M, K});
    const int kt = q4_tune().kt;
    const int nsg = q4_tune().nsg;
    const int ntg = (N + nsg - 1) / nsg;
    const int gy = nsg * ntg;

    static const mx::fast::CustomKernelFunction kn = mx::fast::metal_kernel(
        "lmp_q4_hg", {"x", "w", "scales", "biases", "k_dim", "n_dim"}, {"y"},
        q4_qmv_source(), q4_kernel_header());

    std::vector<mx::array> outs = kn(
        {x2, w, scales, biases, mx::array(K, mx::int32), mx::array(N, mx::int32)},
        {mx::Shape{M, N}}, {x.dtype()}, {32, gy, 1}, {32, nsg, 1},
        {{"T", x.dtype()}, {"M", M}, {"KT", kt}, {"NSG", nsg}}, std::nullopt, false, {});
    mx::array y = outs[0];
    if (x.ndim() == 2) {
        return y;
    }
    mx::Shape out_shape = x.shape();
    out_shape.back() = N;
    return mx::reshape(y, out_shape);
}

inline const char* q4_swiglu_source() {
    return R"METAL(
        const int lane = int(thread_index_in_simdgroup);
        const int sg = int(thread_position_in_threadgroup.y);
        const int tg_y = int(threadgroup_position_in_grid.y);
        const int tig = int(thread_index_in_threadgroup);
        const int K = k_dim;
        const int N = n_dim;
        const int n = tg_y * NSG + sg;
        const int groups = K / 64;

        threadgroup float Xs[3][1024];
        float acc_g[3] = {0.0f, 0.0f, 0.0f};
        float acc_u[3] = {0.0f, 0.0f, 0.0f};

        for (int kb = 0; kb < K; kb += KT) {
          for (int m = 0; m < M; ++m) {
            for (int i = tig; i < KT; i += (32 * NSG)) {
              const int k = kb + i;
              Xs[m][i] = (k < K) ? float(x[m * K + k]) : 0.0f;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          const int k0 = kb + lane * 32;
          if (n < N && k0 < K) {
            const int woff = n * (K / 8) + (kb / 8) + lane * 4;
            const uint4 pg = *(const device uint4*)(((const device uint*)wg) + woff);
            const uint4 pu = *(const device uint4*)(((const device uint*)wu) + woff);
            float qg[32];
            float qu[32];
            q4_unpack32(pg, qg);
            q4_unpack32(pu, qu);
            const int g = k0 / 64;
            const float sgv = float(sgate[n * groups + g]);
            const float bgv = float(bgate[n * groups + g]);
            const float suv = float(sup[n * groups + g]);
            const float buv = float(bup[n * groups + g]);
            const int off = lane * 32;
            for (int m = 0; m < M; ++m) {
              float dg = 0.0f;
              float du = 0.0f;
              float xsum = 0.0f;
              for (int i = 0; i < 32; ++i) {
                const float xv = Xs[m][off + i];
                xsum += xv;
                dg += qg[i] * xv;
                du += qu[i] * xv;
              }
              acc_g[m] += sgv * dg + bgv * xsum;
              acc_u[m] += suv * du + buv * xsum;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (int m = 0; m < M; ++m) {
          const float gv = simd_sum(acc_g[m]);
          const float uv = simd_sum(acc_u[m]);
          if (lane == 0 && n < N) {
            const float silu = gv / (1.0f + exp(-gv));
            y[m * N + n] = T(silu * uv);
          }
        }
    )METAL";
}

inline mx::array q4_metal_swiglu(const mx::array& x, const mx::array& wg, const mx::array& sgate,
                                 const mx::array& bgate, const mx::array& wu, const mx::array& sup,
                                 const mx::array& bup) {
    const int K = static_cast<int>(x.shape(-1));
    const int M = q4_rows(x);
    const int N = static_cast<int>(sgate.shape(0));
    if ((M != 1 && M != 3) || K <= 0 || (K % 64) != 0 || N <= 0) {
        throw std::invalid_argument("q4_metal_swiglu: unsupported shape");
    }
    mx::array x2 = (x.ndim() == 2) ? x : mx::reshape(x, {M, K});
    const int kt = q4_tune().kt;
    const int nsg = q4_tune().nsg;
    const int gy = nsg * ((N + nsg - 1) / nsg);

    static const mx::fast::CustomKernelFunction kn = mx::fast::metal_kernel(
        "lmp_q4_swiglu",
        {"x", "wg", "sgate", "bgate", "wu", "sup", "bup", "k_dim", "n_dim"}, {"y"},
        q4_swiglu_source(), q4_kernel_header());

    std::vector<mx::array> outs = kn(
        {x2, wg, sgate, bgate, wu, sup, bup, mx::array(K, mx::int32), mx::array(N, mx::int32)},
        {mx::Shape{M, N}}, {x.dtype()}, {32, gy, 1}, {32, nsg, 1},
        {{"T", x.dtype()}, {"M", M}, {"KT", kt}, {"NSG", nsg}}, std::nullopt, false, {});
    mx::array y = outs[0];
    if (x.ndim() == 2) {
        return y;
    }
    mx::Shape out_shape = x.shape();
    out_shape.back() = N;
    return mx::reshape(y, out_shape);
}

inline const char* q4_rms_qmv_source() {
    return R"METAL(
        const int lane = int(thread_index_in_simdgroup);
        const int sg = int(thread_position_in_threadgroup.y);
        const int tg_y = int(threadgroup_position_in_grid.y);
        const int tig = int(thread_index_in_threadgroup);
        const int K = k_dim;
        const int N = n_dim;
        const int n = tg_y * NSG + sg;
        const int groups = K / 64;
        const float eps = float(eps_s);

        threadgroup float Xs[3][1024];
        threadgroup float rstd[3];
        threadgroup float red[3][256];
        float acc[3] = {0.0f, 0.0f, 0.0f};

        float local[3] = {0.0f, 0.0f, 0.0f};
        for (int k = tig; k < K; k += (32 * NSG)) {
          for (int m = 0; m < M; ++m) {
            const float xv = float(x[m * K + k]);
            local[m] += xv * xv;
          }
        }
        for (int m = 0; m < M; ++m) {
          red[m][tig] = local[m];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tig == 0) {
          for (int m = 0; m < M; ++m) {
            float s = 0.0f;
            const int nthr = 32 * NSG;
            for (int i = 0; i < nthr; ++i) {
              s += red[m][i];
            }
            rstd[m] = rsqrt(s / float(K) + eps);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kb = 0; kb < K; kb += KT) {
          for (int m = 0; m < M; ++m) {
            for (int i = tig; i < KT; i += (32 * NSG)) {
              const int k = kb + i;
              const float xv = (k < K) ? float(x[m * K + k]) : 0.0f;
              const float gm = (k < K) ? float(gamma[k]) : 0.0f;
              Xs[m][i] = rstd[m] * gm * xv;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          const int k0 = kb + lane * 32;
          if (n < N && k0 < K) {
            const device uint* wrow =
                ((const device uint*)w) + n * (K / 8) + (kb / 8) + lane * 4;
            const uint4 packed = *(const device uint4*)wrow;
            float qn[32];
            q4_unpack32(packed, qn);
            const int g = k0 / 64;
            const float s = float(scales[n * groups + g]);
            const float b = float(biases[n * groups + g]);
            const int off = lane * 32;
            for (int m = 0; m < M; ++m) {
              float qdot = 0.0f;
              float xsum = 0.0f;
              for (int i = 0; i < 32; ++i) {
                const float xv = Xs[m][off + i];
                xsum += xv;
                qdot += qn[i] * xv;
              }
              acc[m] += s * qdot + b * xsum;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (int m = 0; m < M; ++m) {
          acc[m] = simd_sum(acc[m]);
          if (lane == 0 && n < N) {
            y[m * N + n] = T(acc[m]);
          }
        }
    )METAL";
}

inline mx::array q4_metal_rms_linear(const mx::array& x, const mx::array& gamma, float eps,
                                     const mx::array& w, const mx::array& scales,
                                     const mx::array& biases) {
    const int K = static_cast<int>(x.shape(-1));
    const int M = q4_rows(x);
    const int N = static_cast<int>(scales.shape(0));
    if ((M != 1 && M != 3) || K <= 0 || (K % 64) != 0 || N <= 0) {
        throw std::invalid_argument("q4_metal_rms_linear: unsupported shape");
    }
    mx::array x2 = (x.ndim() == 2) ? x : mx::reshape(x, {M, K});
    const int kt = q4_tune().kt;
    const int nsg = q4_tune().nsg;
    const int gy = nsg * ((N + nsg - 1) / nsg);

    static const mx::fast::CustomKernelFunction kn = mx::fast::metal_kernel(
        "lmp_q4_rms",
        {"x", "gamma", "w", "scales", "biases", "k_dim", "n_dim", "eps_s"}, {"y"},
        q4_rms_qmv_source(), q4_kernel_header());

    std::vector<mx::array> outs = kn(
        {x2, gamma, w, scales, biases, mx::array(K, mx::int32), mx::array(N, mx::int32),
         mx::array(eps, mx::float32)},
        {mx::Shape{M, N}}, {x.dtype()}, {32, gy, 1}, {32, nsg, 1},
        {{"T", x.dtype()}, {"M", M}, {"KT", kt}, {"NSG", nsg}}, std::nullopt, false, {});
    mx::array y = outs[0];
    if (x.ndim() == 2) {
        return y;
    }
    mx::Shape out_shape = x.shape();
    out_shape.back() = N;
    return mx::reshape(y, out_shape);
}

inline bool q4_fuse_glu_on() {
    const char* v = std::getenv("LMP_FUSE_GLU");
    return v != nullptr && v[0] != '\0' && v[0] != '0' && std::string(v) != "off";
}

inline bool q4_fuse_rms_on() {
    const char* v = std::getenv("LMP_FUSE_RMS");
    return v != nullptr && v[0] != '\0' && v[0] != '0' && std::string(v) != "off";
}

inline bool q4_env_on() {
    const char* v = std::getenv("LMP_QGEMM");
    return v != nullptr && v[0] != '\0' && v[0] != '0' && std::string(v) != "off";
}

inline float q4_max_abs(const mx::array& a) {
    const mx::array worst = mx::astype(mx::max(mx::abs(a)), mx::float32);
    mx::eval(worst);
    return worst.item<float>();
}

inline float q4_max_abs_diff(const mx::array& a, const mx::array& b) {
    const mx::array worst = mx::astype(mx::max(mx::abs(mx::subtract(a, b))), mx::float32);
    mx::eval(worst);
    return worst.item<float>();
}

inline float q4_rel_err(const mx::array& a, const mx::array& b) {
    const float num = q4_max_abs_diff(a, b);
    const float den = std::max(q4_max_abs(a), std::max(q4_max_abs(b), 1e-6f));
    return num / den;
}

inline bool q4_self_check_ok() {
    constexpr int kGroup = 64;
    constexpr int kBits = 4;
    const int K = 128;
    const int N = 64;
    std::vector<float> wv(static_cast<std::size_t>(N * K));
    std::vector<float> xv(static_cast<std::size_t>(3 * K));
    for (int i = 0; i < N * K; ++i) {
        wv[static_cast<std::size_t>(i)] = 0.15f * std::sin(0.031f * static_cast<float>(i));
    }
    for (int i = 0; i < 3 * K; ++i) {
        xv[static_cast<std::size_t>(i)] = 0.07f * std::cos(0.017f * static_cast<float>(i));
    }
    const mx::array W = mx::astype(mx::array(wv.data(), {N, K}, mx::float32), mx::bfloat16);
    const std::vector<mx::array> q = mx::quantize(W, kGroup, kBits, "affine");
    const mx::array& qw = q[0];
    const mx::array& scales = q[1];
    const mx::array biases = q.size() > 2 ? q[2] : mx::zeros_like(scales);
    const mx::array X1 = mx::astype(mx::array(xv.data(), {1, K}, mx::float32), mx::bfloat16);
    const mx::array X3 = mx::astype(mx::array(xv.data(), {3, K}, mx::float32), mx::bfloat16);
    const mx::array ref1 = q4_mlx_qmm(X1, qw, scales, biases, kGroup, kBits, "affine");
    const mx::array ref3 = q4_mlx_qmm(X3, qw, scales, biases, kGroup, kBits, "affine");
    mx::array got1 = q4_metal_linear(X1, qw, scales, biases);
    mx::array got3 = q4_metal_linear(X3, qw, scales, biases);
    mx::eval({ref1, got1, ref3, got3});
    return q4_rel_err(ref1, got1) < 2e-2f && q4_rel_err(ref3, got3) < 2e-2f;
}

enum class Q4Lane : std::uint8_t { Unset, Off, On };

inline Q4Lane& q4_lane() {
    static Q4Lane lane = Q4Lane::Unset;
    return lane;
}

inline void q4_testing_reset_lane() { q4_lane() = Q4Lane::Unset; }

inline mx::array q4_linear(const mx::array& x, const mx::array& w, const mx::array& scales,
                           const std::optional<mx::array>& biases, int group_size, int bits,
                           const std::string& mode) {
    auto mlx = [&] {
        return q4_mlx_qmm(x, w, scales, biases, group_size, bits, mode);
    };
    if (bits != 4 || group_size != 64 || mode != "affine") {
        return mlx();
    }
    Q4Lane& lane = q4_lane();
    if (lane == Q4Lane::Unset) {
        lane = (!q4_env_on() || !mx::metal::is_available()) ? Q4Lane::Off : Q4Lane::On;
        if (lane == Q4Lane::On && !q4_self_check_ok()) {
            std::fprintf(stderr,
                         "LMP_QGEMM: self-check failed, staying on mx::quantized_matmul\n");
            lane = Q4Lane::Off;
        } else if (lane == Q4Lane::On) {
            std::fprintf(stderr, "LMP_QGEMM: custom q4 GEMM enabled after self-check\n");
        }
    }
    if (lane != Q4Lane::On) {
        return mlx();
    }
    const int K = static_cast<int>(x.shape(-1));
    const int M = q4_rows(x);
    if ((M != 1 && M != 3) || (K % 64) != 0) {
        return mlx();
    }
    try {
        const mx::array b = biases.has_value() ? *biases : mx::zeros_like(scales);
        return q4_metal_linear(x, w, scales, b);
    } catch (...) {
        lane = Q4Lane::Off;
        std::fprintf(stderr, "LMP_QGEMM: kernel threw, falling back to MLX\n");
        return mlx();
    }
}

} // namespace lmp::model::mlxl

#endif // LMP_HAVE_MLX
#endif // LLM_MLX_QGEMM_HPP
