// Custom affine-4 GEMM vs stock mx::quantized_matmul.
//
// Needs Metal, not a checkpoint. Labelled realmodel for the same reason as
// test_switch_glu_fusion: CI has no GPU.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "tests/check.hpp"

#if LMP_HAVE_MLX

#include "src/model/mlx/activations.hpp"
#include "src/model/mlx/qgemm.hpp"
#include "src/model/mlx/weight_store.hpp"

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace mx = mlx::core;
using lmp::model::mlxl::WeightStore;

namespace {

constexpr int kGroup = 64;
constexpr int kBits = 4;

mx::array ramp(const std::vector<int>& shape, float step, float phase) {
    int n = 1;
    for (int d : shape) {
        n *= d;
    }
    std::vector<float> v(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        v[static_cast<std::size_t>(i)] =
            0.5f * std::sin(static_cast<float>(i) * step + phase);
    }
    mx::Shape s(shape.begin(), shape.end());
    return mx::astype(mx::array(v.data(), s, mx::float32), mx::bfloat16);
}

void put_quantized(WeightStore& ws, const std::string& base, const mx::array& w) {
    const std::vector<mx::array> q = mx::quantize(w, kGroup, kBits, "affine");
    ws.set(base + ".weight", q[0]);
    ws.set(base + ".scales", q[1]);
    if (q.size() > 2) {
        ws.set(base + ".biases", q[2]);
    }
}

float max_abs(const mx::array& a, const mx::array& b) {
    return lmp::model::mlxl::q4_max_abs_diff(a, b);
}

double ms_eval(const std::function<mx::array()>& fn, int iters) {
    using Clock = std::chrono::steady_clock;
    mx::eval(fn());
    const auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        mx::eval(fn());
    }
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() /
           static_cast<double>(iters);
}

} // namespace

TEST(q4_dequant_matmul_matches_quantized_matmul) {
    const int M = 2;
    const int K = 128;
    const int N = 64;
    const mx::array W = ramp({N, K}, 0.019f, 0.2f);
    const mx::array x = ramp({M, K}, 0.011f, 0.7f);
    const std::vector<mx::array> q = mx::quantize(W, kGroup, kBits, "affine");
    const mx::array& w = q[0];
    const mx::array& scales = q[1];
    std::optional<mx::array> biases;
    if (q.size() > 2) {
        biases = q[2];
    }

    const mx::array ref = lmp::model::mlxl::q4_ref_linear(x, w, scales, biases, kGroup,
                                                          kBits, "affine");
    const mx::array qmm = lmp::model::mlxl::q4_mlx_qmm(x, w, scales, biases, kGroup, kBits,
                                                       "affine");
    const mx::array qmm_f = mx::astype(qmm, mx::float32);
    mx::eval({ref, qmm_f});
    const float d = max_abs(ref, qmm_f);
    const float rel = lmp::model::mlxl::q4_rel_err(ref, qmm_f);
    std::printf("G0 dequant+matmul vs qmm: max|diff|=%.6g  rel=%.6g  |qmm|=%.6g\n",
                static_cast<double>(d), static_cast<double>(rel),
                static_cast<double>(lmp::model::mlxl::q4_max_abs(qmm_f)));
    CHECK(rel < 2e-2f);
}

TEST(q4_metal_kernel_matches_qmm_on_tiny_shapes) {
    if (!mx::metal::is_available()) {
        std::printf("G1 skipped: no Metal\n");
        CHECK(true);
        return;
    }
    const int K = 128;
    const int N = 64;
    const mx::array W = ramp({N, K}, 0.023f, 1.1f);
    const std::vector<mx::array> q = mx::quantize(W, kGroup, kBits, "affine");
    const mx::array biases = q.size() > 2 ? q[2] : mx::zeros_like(q[1]);
    for (int M : {1, 3}) {
        const mx::array x = ramp({M, K}, 0.013f, 0.4f);
        const mx::array qmm =
            lmp::model::mlxl::q4_mlx_qmm(x, q[0], q[1], biases, kGroup, kBits, "affine");
        const mx::array got = lmp::model::mlxl::q4_metal_linear(x, q[0], q[1], biases);
        mx::eval({qmm, got});
        const float d = max_abs(qmm, got);
        const float rel = lmp::model::mlxl::q4_rel_err(qmm, got);
        std::printf("G1 metal vs qmm (M=%d K=%d N=%d): max|diff|=%.6g  rel=%.6g  |qmm|=%.6g\n",
                    M, K, N, static_cast<double>(d), static_cast<double>(rel),
                    static_cast<double>(lmp::model::mlxl::q4_max_abs(qmm)));
        CHECK(rel < 2e-2f);
    }
}

TEST(q4_metal_parity_and_microbench_at_decode_shapes) {
    if (!mx::metal::is_available()) {
        std::printf("G2 skipped: no Metal\n");
        CHECK(true);
        return;
    }
    const int K = 5120;
    const int N = 5120;
    const mx::array W = ramp({N, K}, 0.007f, 0.3f);
    const std::vector<mx::array> q = mx::quantize(W, kGroup, kBits, "affine");
    const mx::array biases = q.size() > 2 ? q[2] : mx::zeros_like(q[1]);

    auto run_g2 = [&](int kt, int nsg, double* m1, double* m3) -> bool {
        lmp::model::mlxl::q4_set_tune(kt, nsg);
        bool ok = true;
        for (int M : {1, 3}) {
            const mx::array x = ramp({M, K}, 0.009f, 0.5f);
            const mx::array qmm =
                lmp::model::mlxl::q4_mlx_qmm(x, q[0], q[1], biases, kGroup, kBits,
                                             "affine");
            const mx::array got = lmp::model::mlxl::q4_metal_linear(x, q[0], q[1], biases);
            mx::eval({qmm, got});
            const float rel = lmp::model::mlxl::q4_rel_err(qmm, got);
            std::printf("G2 parity KT=%d NSG=%d M=%d: rel=%.6g  |qmm|=%.6g\n", kt, nsg, M,
                        static_cast<double>(rel),
                        static_cast<double>(lmp::model::mlxl::q4_max_abs(qmm)));
            CHECK(rel < 2e-2f);
            const double custom_ms = ms_eval(
                [&] { return lmp::model::mlxl::q4_metal_linear(x, q[0], q[1], biases); },
                32);
            const double mlx_ms = ms_eval(
                [&] {
                    return lmp::model::mlxl::q4_mlx_qmm(x, q[0], q[1], biases, kGroup,
                                                        kBits, "affine");
                },
                32);
            std::printf("G2 bench KT=%d NSG=%d M=%d: mlx=%.3f ms  custom=%.3f ms  "
                        "ratio=%.3f\n",
                        kt, nsg, M, mlx_ms, custom_ms, custom_ms / mlx_ms);
            if (M == 1) {
                *m1 = custom_ms;
            } else {
                *m3 = custom_ms;
            }
            (void)ok;
        }
        const double ratio = *m3 / *m1;
        std::printf("G2 M3/M1=%.3f  (need M1<0.25 M3<0.30 M3/M1<=1.20)\n", ratio);
        return *m1 < 0.25 && *m3 < 0.30 && ratio <= 1.20;
    };

    double m1 = 0;
    double m3 = 0;
    bool hit = run_g2(1024, 8, &m1, &m3);
    if (!hit) {
        std::printf("G2 default missed; autotune KT=512 NSG=4\n");
        hit = run_g2(512, 4, &m1, &m3);
    }
    if (!hit) {
        std::printf("G2 STOP: half-group QMV missed M1<0.25ms M3<0.30ms M3/M1<=1.20; "
                    "default remains stock MLX, no 27B G4/G5\n");
        lmp::model::mlxl::q4_set_tune(1024, 8);
    } else {
        std::printf("G2 HIT: M1=%.3f ms M3=%.3f ms\n", m1, m3);
    }
}

TEST(q4_linear_opt_in_self_check_matches_weight_store) {
    if (!mx::metal::is_available()) {
        std::printf("G3 skipped: no Metal\n");
        CHECK(true);
        return;
    }
    lmp::model::mlxl::q4_testing_reset_lane();
    lmp::model::mlxl::q4_set_tune(1024, 8);
    ::setenv("LMP_QGEMM", "1", 1);

    WeightStore ws;
    const int K = 128;
    const int N = 64;
    put_quantized(ws, "blk.0.q_proj", ramp({N, K}, 0.021f, 0.9f));
    const mx::array x = ramp({1, K}, 0.015f, 0.2f);
    const mx::array custom = ws.linear(x, "blk.0.q_proj");
    lmp::model::mlxl::q4_testing_reset_lane();
    ::setenv("LMP_QGEMM", "0", 1);
    const mx::array stock = ws.linear(x, "blk.0.q_proj");
    mx::eval({custom, stock});
    const float d = max_abs(custom, stock);
    std::printf("G3 WeightStore::linear LMP_QGEMM=1 vs 0: max|diff|=%.6g  rel=%.6g\n",
                static_cast<double>(d),
                static_cast<double>(lmp::model::mlxl::q4_rel_err(custom, stock)));
    CHECK(lmp::model::mlxl::q4_rel_err(custom, stock) < 2e-2f);
    ::unsetenv("LMP_QGEMM");
    lmp::model::mlxl::q4_testing_reset_lane();
}

TEST(q4_fused_swiglu_matches_two_qmm_and_microbench) {
    if (!mx::metal::is_available()) {
        std::printf("GF skipped: no Metal\n");
        CHECK(true);
        return;
    }
    lmp::model::mlxl::q4_set_tune(1024, 8);

    auto two_qmm_swiglu = [&](const mx::array& x, const mx::array& wg, const mx::array& sg,
                              const mx::array& bg, const mx::array& wu, const mx::array& su,
                              const mx::array& bu) {
        const mx::array gate =
            lmp::model::mlxl::q4_mlx_qmm(x, wg, sg, bg, kGroup, kBits, "affine");
        const mx::array up =
            lmp::model::mlxl::q4_mlx_qmm(x, wu, su, bu, kGroup, kBits, "affine");
        return lmp::model::mlxl::swiglu(gate, up);
    };

    {
        const int K = 128;
        const int N = 64;
        const mx::array Wg = ramp({N, K}, 0.023f, 1.1f);
        const mx::array Wu = ramp({N, K}, 0.017f, 0.6f);
        const auto qg = mx::quantize(Wg, kGroup, kBits, "affine");
        const auto qu = mx::quantize(Wu, kGroup, kBits, "affine");
        const mx::array bg = qg.size() > 2 ? qg[2] : mx::zeros_like(qg[1]);
        const mx::array bu = qu.size() > 2 ? qu[2] : mx::zeros_like(qu[1]);
        for (int M : {1, 3}) {
            const mx::array x = ramp({M, K}, 0.013f, 0.4f);
            const mx::array ref = two_qmm_swiglu(x, qg[0], qg[1], bg, qu[0], qu[1], bu);
            const mx::array got = lmp::model::mlxl::q4_metal_swiglu(
                x, qg[0], qg[1], bg, qu[0], qu[1], bu);
            mx::eval({ref, got});
            const float rel = lmp::model::mlxl::q4_rel_err(ref, got);
            std::printf("GF parity M=%d K=%d N=%d: rel=%.6g\n", M, K, N,
                        static_cast<double>(rel));
            CHECK(rel < 2e-2f);
        }
    }

    const int K = 5120;
    const int N = 5120;
    const mx::array Wg = ramp({N, K}, 0.007f, 0.3f);
    const mx::array Wu = ramp({N, K}, 0.009f, 1.4f);
    const auto qg = mx::quantize(Wg, kGroup, kBits, "affine");
    const auto qu = mx::quantize(Wu, kGroup, kBits, "affine");
    const mx::array bg = qg.size() > 2 ? qg[2] : mx::zeros_like(qg[1]);
    const mx::array bu = qu.size() > 2 ? qu[2] : mx::zeros_like(qu[1]);
    bool beat = true;
    for (int M : {1, 3}) {
        const mx::array x = ramp({M, K}, 0.009f, 0.5f);
        const mx::array ref = two_qmm_swiglu(x, qg[0], qg[1], bg, qu[0], qu[1], bu);
        const mx::array got =
            lmp::model::mlxl::q4_metal_swiglu(x, qg[0], qg[1], bg, qu[0], qu[1], bu);
        mx::eval({ref, got});
        const float rel = lmp::model::mlxl::q4_rel_err(ref, got);
        std::printf("GF parity M=%d K=N=5120: rel=%.6g\n", M, static_cast<double>(rel));
        CHECK(rel < 2e-2f);
        const double fused_ms = ms_eval(
            [&] {
                return lmp::model::mlxl::q4_metal_swiglu(x, qg[0], qg[1], bg, qu[0], qu[1],
                                                         bu);
            },
            32);
        const double stock_ms = ms_eval(
            [&] { return two_qmm_swiglu(x, qg[0], qg[1], bg, qu[0], qu[1], bu); }, 32);
        std::printf("GF bench M=%d: two-qmm+swiglu=%.3f ms  fused=%.3f ms  ratio=%.3f\n",
                    M, stock_ms, fused_ms, fused_ms / stock_ms);
        if (fused_ms >= stock_ms) {
            beat = false;
        }
    }
    if (beat) {
        std::printf("GF HIT: fused gate+up+SwiGLU beats two qmm; enable LMP_FUSE_GLU=1\n");
    } else {
        std::printf("GF MISS: fused gate+up+SwiGLU did not beat two-qmm+swiglu; "
                    "default remains two linears\n");
    }

    ::setenv("LMP_FUSE_GLU", "1", 1);
    WeightStore ws;
    put_quantized(ws, "mlp.gate_proj", ramp({64, 128}, 0.021f, 0.9f));
    put_quantized(ws, "mlp.up_proj", ramp({64, 128}, 0.018f, 0.3f));
    const mx::array x = ramp({1, 128}, 0.015f, 0.2f);
    const mx::array fused = ws.swiglu_gate_up(x, "mlp.gate_proj", "mlp.up_proj");
    ::setenv("LMP_FUSE_GLU", "0", 1);
    const mx::array stock = ws.swiglu_gate_up(x, "mlp.gate_proj", "mlp.up_proj");
    mx::eval({fused, stock});
    std::printf("GF WeightStore LMP_FUSE_GLU=1 vs 0: rel=%.6g\n",
                static_cast<double>(lmp::model::mlxl::q4_rel_err(fused, stock)));
    CHECK(lmp::model::mlxl::q4_rel_err(fused, stock) < 2e-2f);
    ::unsetenv("LMP_FUSE_GLU");
}

TEST(q4_fused_rms_qmv_matches_rms_then_qmm) {
    if (!mx::metal::is_available()) {
        std::printf("GR skipped: no Metal\n");
        CHECK(true);
        return;
    }
    lmp::model::mlxl::q4_set_tune(1024, 8);
    constexpr float kEps = 1e-6f;

    auto rms_then_qmm = [&](const mx::array& x, const mx::array& gamma, const mx::array& w,
                            const mx::array& s, const mx::array& b) {
        const mx::array xn = mx::fast::rms_norm(x, gamma, kEps);
        return lmp::model::mlxl::q4_mlx_qmm(xn, w, s, b, kGroup, kBits, "affine");
    };

    {
        const int K = 128;
        const int N = 64;
        const mx::array W = ramp({N, K}, 0.023f, 1.1f);
        const mx::array gamma = ramp({K}, 0.004f, 0.8f);
        const auto q = mx::quantize(W, kGroup, kBits, "affine");
        const mx::array b = q.size() > 2 ? q[2] : mx::zeros_like(q[1]);
        for (int M : {1, 3}) {
            const mx::array x = ramp({M, K}, 0.013f, 0.4f);
            const mx::array ref = rms_then_qmm(x, gamma, q[0], q[1], b);
            const mx::array got =
                lmp::model::mlxl::q4_metal_rms_linear(x, gamma, kEps, q[0], q[1], b);
            mx::eval({ref, got});
            const float rel = lmp::model::mlxl::q4_rel_err(ref, got);
            std::printf("GR parity M=%d K=%d N=%d: rel=%.6g\n", M, K, N,
                        static_cast<double>(rel));
            CHECK(rel < 2e-2f);
        }
    }

    const int K = 5120;
    const int N = 5120;
    const mx::array W = ramp({N, K}, 0.007f, 0.3f);
    const mx::array gamma = ramp({K}, 0.003f, 0.2f);
    const auto q = mx::quantize(W, kGroup, kBits, "affine");
    const mx::array b = q.size() > 2 ? q[2] : mx::zeros_like(q[1]);
    bool beat = true;
    for (int M : {1, 3}) {
        const mx::array x = ramp({M, K}, 0.009f, 0.5f);
        const mx::array ref = rms_then_qmm(x, gamma, q[0], q[1], b);
        const mx::array got =
            lmp::model::mlxl::q4_metal_rms_linear(x, gamma, kEps, q[0], q[1], b);
        mx::eval({ref, got});
        const float rel = lmp::model::mlxl::q4_rel_err(ref, got);
        std::printf("GR parity M=%d K=N=5120: rel=%.6g\n", M, static_cast<double>(rel));
        CHECK(rel < 2e-2f);
        const double fused_ms = ms_eval(
            [&] {
                return lmp::model::mlxl::q4_metal_rms_linear(x, gamma, kEps, q[0], q[1], b);
            },
            32);
        const double stock_ms =
            ms_eval([&] { return rms_then_qmm(x, gamma, q[0], q[1], b); }, 32);
        std::printf("GR bench M=%d: rms+qmm=%.3f ms  fused=%.3f ms  ratio=%.3f\n", M,
                    stock_ms, fused_ms, fused_ms / stock_ms);
        if (fused_ms >= stock_ms) {
            beat = false;
        }
    }
    if (beat) {
        std::printf("GR HIT: RMSNorm-into-QMV beats rms_norm then qmm; enable LMP_FUSE_RMS=1\n");
    } else {
        std::printf("GR MISS: RMSNorm-into-QMV did not beat rms+qmm; default remains "
                    "rms_norm then linear\n");
    }

    ::setenv("LMP_FUSE_RMS", "1", 1);
    WeightStore ws;
    put_quantized(ws, "q_proj", ramp({64, 128}, 0.021f, 0.9f));
    const mx::array x_ws = ramp({1, 128}, 0.015f, 0.2f);
    const mx::array g_ws = ramp({128}, 0.004f, 0.8f);
    const mx::array fused = ws.rms_linear(x_ws, g_ws, kEps, "q_proj");
    ::setenv("LMP_FUSE_RMS", "0", 1);
    const mx::array stock = ws.rms_linear(x_ws, g_ws, kEps, "q_proj");
    mx::eval({fused, stock});
    std::printf("GR WeightStore LMP_FUSE_RMS=1 vs 0: rel=%.6g\n",
                static_cast<double>(lmp::model::mlxl::q4_rel_err(fused, stock)));
    CHECK(lmp::model::mlxl::q4_rel_err(fused, stock) < 2e-2f);
    ::unsetenv("LMP_FUSE_RMS");
}

TEST(q4_stream_probe_vs_qmm_bandwidth) {
    if (!mx::metal::is_available()) {
        std::printf("GP skipped: no Metal\n");
        CHECK(true);
        return;
    }
    const int K = 5120;
    const int N = 5120;
    const mx::array W = ramp({N, K}, 0.007f, 0.3f);
    const std::vector<mx::array> q = mx::quantize(W, kGroup, kBits, "affine");
    const mx::array biases = q.size() > 2 ? q[2] : mx::zeros_like(q[1]);
    const mx::array x = ramp({1, K}, 0.009f, 0.5f);

    const double bytes =
        static_cast<double>(q[0].nbytes() + q[1].nbytes() + biases.nbytes());

    static const mx::fast::CustomKernelFunction probe = mx::fast::metal_kernel(
        "lmp_q4_stream", {"w", "n_words"}, {"y"},
        R"METAL(
        const int gid = int(thread_position_in_grid.x);
        const int n = n_words;
        uint acc = 0;
        for (int i = gid; i < n; i += 8192) {
          acc += ((const device uint*)w)[i];
        }
        acc = simd_sum(acc);
        if (int(thread_index_in_simdgroup) == 0) {
          y[gid / 32] = acc;
        }
        )METAL");

    auto stream = [&] {
        const int n_words = static_cast<int>(q[0].size());
        return probe({q[0], mx::array(n_words, mx::int32)}, {mx::Shape{256}},
                     {mx::uint32}, {8192, 1, 1}, {32, 1, 1}, {}, std::nullopt, false,
                     {})[0];
    };
    const double stream_ms = ms_eval(
        [&] {
            mx::array s = mx::sum(mx::astype(q[1], mx::float32));
            mx::array b = mx::sum(mx::astype(biases, mx::float32));
            return mx::add(mx::astype(stream(), mx::float32), mx::add(s, b));
        },
        32);
    const double qmm_ms = ms_eval(
        [&] {
            return lmp::model::mlxl::q4_mlx_qmm(x, q[0], q[1], biases, kGroup, kBits,
                                                "affine");
        },
        32);
    const double stream_gbs = (bytes / 1e6) / stream_ms;
    const double qmm_gbs = (bytes / 1e6) / qmm_ms;
    std::printf("GP stream=%.3f ms (%.1f GB/s)  qmm=%.3f ms (%.1f GB/s)  "
                "qmm/stream=%.3f  bytes=%.1f MB\n",
                stream_ms, stream_gbs, qmm_ms, qmm_gbs, qmm_ms / stream_ms,
                bytes / 1e6);
    if (qmm_gbs + 1e-9 >= 0.85 * stream_gbs) {
        std::printf("GP GEMM closed: MLX qmm within 15%% of packed-W stream; do not "
                    "reopen custom QMV\n");
    } else {
        std::printf("GP qmm below 85%% of stream; layout/ALU still has room\n");
    }
    CHECK(stream_ms > 0.0);
    CHECK(qmm_ms > 0.0);
}

TEST(q4_row_concat_matches_separate_qmm_and_microbench) {
    if (!mx::metal::is_available()) {
        std::printf("GC skipped: no Metal\n");
        CHECK(true);
        return;
    }

    auto pack = [&](const mx::array& W) {
        return mx::quantize(W, kGroup, kBits, "affine");
    };
    auto qmm = [&](const mx::array& x, const std::vector<mx::array>& q) {
        const mx::array b = q.size() > 2 ? q[2] : mx::zeros_like(q[1]);
        return lmp::model::mlxl::q4_mlx_qmm(x, q[0], q[1], b, kGroup, kBits, "affine");
    };
    auto cat_pack = [&](const std::vector<std::vector<mx::array>>& qs) {
        std::vector<mx::array> ws;
        std::vector<mx::array> ss;
        std::vector<mx::array> bs;
        for (const auto& q : qs) {
            ws.push_back(q[0]);
            ss.push_back(q[1]);
            bs.push_back(q.size() > 2 ? q[2] : mx::zeros_like(q[1]));
        }
        return std::vector<mx::array>{mx::concatenate(ws, 0), mx::concatenate(ss, 0),
                                      mx::concatenate(bs, 0)};
    };

    auto run_pair = [&](const char* tag, int K, const std::vector<int>& Ns, int M,
                        double min_save_ms, bool* hit) {
        std::vector<std::vector<mx::array>> qs;
        std::vector<mx::array> ys;
        const mx::array x = ramp({M, K}, 0.009f, 0.5f);
        for (size_t i = 0; i < Ns.size(); ++i) {
            const mx::array W =
                ramp({Ns[i], K}, 0.007f + 0.001f * static_cast<float>(i), 0.3f * static_cast<float>(i + 1));
            qs.push_back(pack(W));
        }
        for (const auto& q : qs) {
            ys.push_back(qmm(x, q));
        }
        const mx::array y_sep = mx::concatenate(ys, -1);
        const auto qc = cat_pack(qs);
        const mx::array y_cat = qmm(x, qc);
        mx::eval({y_sep, y_cat});
        const float rel = lmp::model::mlxl::q4_rel_err(y_sep, y_cat);
        std::printf("GC parity %s M=%d: rel=%.6g\n", tag, M, static_cast<double>(rel));
        CHECK(rel < 2e-2f);

        const double sep_ms = ms_eval(
            [&] {
                std::vector<mx::array> parts;
                for (const auto& q : qs) {
                    parts.push_back(qmm(x, q));
                }
                return mx::concatenate(parts, -1);
            },
            32);
        const double cat_ms = ms_eval([&] { return qmm(x, qc); }, 32);
        const double save = sep_ms - cat_ms;
        std::printf("GC bench %s M=%d: separate=%.3f ms  concat=%.3f ms  save=%.3f ms\n",
                    tag, M, sep_ms, cat_ms, save);
        if (save < min_save_ms) {
            *hit = false;
        }
    };

    bool qkv_hit = true;
    bool glu_hit = true;
    for (int M : {1, 3}) {
        run_pair("qkv-5120", 5120, {5120, 5120, 5120}, M, 0.03, &qkv_hit);
        run_pair("gateup-17408", 5120, {17408, 17408}, M, 0.02, &glu_hit);
    }
    if (glu_hit) {
        std::printf("GC HIT gate/up concat; load-time fuse is in play\n");
    } else {
        std::printf("GC STOP gate/up: concat did not beat two qmm by 0.02 ms; do not fuse\n");
    }
    if (qkv_hit) {
        std::printf("GC HIT qkv-like concat\n");
    } else {
        std::printf("GC STOP qkv-like: concat did not beat three qmm by 0.03 ms "
                    "(27B q/k/v last-dims already mismatch; skip regardless)\n");
    }
}

#else

TEST(qgemm_needs_mlx) {
    std::printf("skipped: built without MLX\n");
}

#endif
