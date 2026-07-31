// MoE subcommands of the attribution driver (S19.3). Split from diag_main.cpp for the
// per-file line ratchet.
//
// A standing warning about the first two. `blocks` and `moe` are ISOLATION benchmarks:
// they call one block repeatedly on a fixed input. That is not what a decode step does,
// and both have given confidently wrong answers -- `moe` attributed 0.23 ms/call to
// "gather_qmm fed unevaluated indices", which mlx-lm's own block does not reproduce and
// which vanished once the real cause (a float32 residual stream) was fixed. Prefer
// `moestream` for the MoE and `step` for a whole forward pass; treat these two as
// hypothesis generators only.

#include "tests/model/diag_common.hpp"

#include <cstdio>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#if LMP_HAVE_MLX
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/transforms.h"

#include "src/model/mlx/qwen35_moe_model.hpp"
#include "src/model/mlx/switch_glu.hpp"
#include "src/model/mlx/weight_store.hpp"
#endif

namespace lmp::diag {

#if LMP_HAVE_MLX
namespace mlxl = lmp::model::mlxl;
#endif

// --- blocks ----------------------------------------------------------------
//
// Where does one forward pass actually go? Loads the real weights and times each
// block of one layer against them, so "decode is slow" resolves to a named block
// instead of a hypothesis. Uses the same calls forward_* makes.

#if LMP_HAVE_MLX
namespace mx = mlx::core;

// Two numbers, because they answer different questions. `latency` evaluates each call
// on its own, so it includes one GPU round-trip -- on this machine that floor is about
// 0.18 ms and it swamps every block, which is why the first version of this driver
// reported every block as costing the same. `marginal` queues all the calls and
// evaluates once, so it reports what the block adds to a step that is already running.
double time_block(const char* label, int reps, int layers,
                  const std::function<mx::array()>& make) {
    mx::array warm = make();
    mx::eval(warm);

    auto t0 = Clock::now();
    for (int i = 0; i < reps; ++i) {
        mx::array out = make();
        mx::eval(out);
    }
    const double latency = ms(t0, Clock::now()) / reps;

    std::vector<mx::array> batch;
    batch.reserve(static_cast<std::size_t>(reps));
    auto t1 = Clock::now();
    for (int i = 0; i < reps; ++i) {
        batch.push_back(make());
    }
    // MLX is lazy: nothing has run yet. This is pure CPU -- graph construction, weight
    // lookups, shape math -- and in a decode step it is serialized ahead of the GPU.
    const double construct = ms(t1, Clock::now()) / reps;
    mx::eval(batch);
    const double marginal = ms(t1, Clock::now()) / reps;

    std::printf("  %-26s latency %6.3f  cpu %6.3f  marginal %6.3f ms  x%2d = %6.2f ms\n", label,
                latency, construct, marginal, layers, marginal * layers);
    return marginal * layers;
}

int cmd_blocks(int T) {
    mlxl::WeightStore ws;
    if (!ws.load_directory(qwen_dir())) {
        std::printf("weight load failed\n");
        return 1;
    }
    const std::string p = "language_model.model.layers.0.";
    const int hidden = 2048, key_dim = 16 * 128, value_dim = 32 * 128;
    const int conv_dim = key_dim * 2 + value_dim;

    // Match the real activation dtype rather than assuming it: it comes out of the
    // quantized embedding table, and guessing wrong would benchmark a cast.
    const mx::array ids = mx::zeros({1, T}, mx::int32);
    mx::array x = ws.embed_lookup(ids, "language_model.model.embed_tokens");
    mx::eval(x);
    std::printf("T=%d  activation dtype size=%d bytes  hidden=%d\n", T,
                static_cast<int>(x.itemsize()), hidden);

    const int reps = T == 1 ? 50 : 5;
    double total = 0;
    total += time_block("moe gate", reps, 40, [&] { return ws.linear(x, p + "mlp.gate"); });

    const mx::array gate_logits = ws.linear(x, p + "mlp.gate");
    auto [inds, scores] = mlxl::moe_topk(gate_logits, 8, true);
    mx::eval({inds, scores});
    // The pieces the first version of this driver never timed, because it hoisted
    // moe_topk and the combine out of the loop. forward_moe minus everything `blocks`
    // measured is 0.56 of its 0.65 ms, so it is entirely in here.
    total += time_block("moe topk (argpartition)", reps, 40,
                        [&] { return mlxl::moe_topk(gate_logits, 8, true).first; });
    total += time_block("  softmax only", reps, 0,
                        [&] { return mx::softmax(gate_logits, -1); });
    const mx::array gates = mx::softmax(gate_logits, -1);
    mx::eval(gates);
    total += time_block("  argpartition only", reps, 0,
                        [&] { return mx::argpartition(gates, 256 - 8, 2); });
    const mx::array y8 = mx::zeros({1, T, 8, hidden}, x.dtype());
    mx::eval(y8);
    total += time_block("moe combine (sum over 8)", reps, 40, [&] {
        return mx::sum(mx::multiply(y8, mx::expand_dims(scores, -1)), -2);
    });
    total += time_block("moe switch_glu (8/256)", reps, 40, [&] {
        return mlxl::switch_glu(x, ws, p + "mlp.switch_mlp.gate_proj",
                                p + "mlp.switch_mlp.up_proj", p + "mlp.switch_mlp.down_proj",
                                inds);
    });
    total += time_block("moe shared expert", reps, 40, [&] {
        return ws.linear(mlxl::swiglu(ws.linear(x, p + "mlp.shared_expert.gate_proj"),
                                      ws.linear(x, p + "mlp.shared_expert.up_proj")),
                         p + "mlp.shared_expert.down_proj");
    });

    const std::string lp = p + "linear_attn.";
    total += time_block("linear in_proj_qkv", reps, 30,
                        [&] { return ws.linear(x, lp + "in_proj_qkv"); });
    const mx::array qkv = ws.linear(x, lp + "in_proj_qkv");
    const mx::array conv_in =
        mx::concatenate({mx::zeros({1, 3, conv_dim}, qkv.dtype()), qkv}, 1);
    mx::eval(conv_in);
    total += time_block("linear conv1d (depthwise)", reps, 30, [&] {
        return mlxl::silu(mx::conv1d(conv_in, ws.get(lp + "conv1d.weight"), 1, 0, 1, conv_dim));
    });

    const mx::array q = mx::random::normal({1, T, 16, 128}, mx::float32);
    const mx::array k = mx::random::normal({1, T, 16, 128}, mx::float32);
    const mx::array v = mx::random::normal({1, T, 32, 128}, mx::float32);
    const mx::array a = mx::random::normal({1, T, 32}, mx::float32);
    const mx::array bb = mx::random::normal({1, T, 32}, mx::float32);
    const mx::array st = mx::zeros({1, 32, 128, 128}, mx::float32);
    mx::eval({q, k, v, a, bb, st});
    total += time_block("linear gated-delta scan", reps, 30, [&] {
        return mlxl::gated_delta_update(q, k, v, a, bb, ws.get(lp + "A_log"),
                                        ws.get(lp + "dt_bias"), st)
            .first;
    });
    total += time_block("linear out_proj", reps, 30, [&] {
        return ws.linear(mx::zeros({1, T, value_dim}, x.dtype()), lp + "out_proj");
    });

    const std::string ap = "language_model.model.layers.3.self_attn.";
    total += time_block("attn q/k/v/o proj", reps, 10, [&] {
        return ws.linear(mx::zeros({1, T, 16 * 256}, x.dtype()), ap + "o_proj");
    });
    total += time_block("lm_head", reps, 1, [&] {
        return ws.linear(mx::slice(x, {0, T - 1, 0}, {1, T, hidden}),
                         "language_model.lm_head");
    });
    std::printf("  %-26s %8s               %7.2f ms  (measured pieces only)\n", "TOTAL", "",
                total);
    return 0;
}
#endif

// --- moe -------------------------------------------------------------------
//
// Bisect forward_moe. The sum of its pieces timed in isolation is 0.109 ms; the block
// itself is 0.647. Something about composing them costs 6x, and the only way to find
// out which addition it is, is to build the block up one stage at a time and time each
// prefix exactly the way forward_moe builds it -- unevaluated intermediates included,
// since `blocks` hoisted an evaluated `inds` out of the loop and that is precisely the
// difference it hid.

#if LMP_HAVE_MLX
int cmd_moe(int T) {
    mlxl::WeightStore ws;
    if (!ws.load_directory(qwen_dir())) {
        std::printf("weight load failed\n");
        return 1;
    }
    const std::string p = "language_model.model.layers.0.mlp.";
    const mx::array ids = mx::zeros({1, T}, mx::int32);
    mx::array x = ws.embed_lookup(ids, "language_model.model.embed_tokens");
    mx::eval(x);
    const int reps = T == 1 ? 30 : 4;

    // The checkpoint is BF16 and mlx-lm carries BF16 activations. Feeding this MoE
    // FP16 instead costs 6x in MLX 0.31.2 (measured against mlx_lm on this machine:
    // 0.304 ms/layer in BF16, 0.701 in FP16), so print what we actually carry rather
    // than assume it -- a silent dtype promotion here would look exactly like a slow
    // kernel.
    const auto dt = [](const mx::array& a) -> const char* {
        const mx::Dtype d = a.dtype();
        if (d == mx::bfloat16) return "bf16";
        if (d == mx::float16) return "f16";
        if (d == mx::float32) return "f32";
        if (d == mx::int32) return "i32";
        if (d == mx::uint32) return "u32";
        return "other";
    };
    {
        auto [i0, s0] = mlxl::moe_topk(ws.linear(x, p + "gate"), 8, true);
        const mx::array y0 = mlxl::switch_glu(x, ws, p + "switch_mlp.gate_proj",
                                              p + "switch_mlp.up_proj",
                                              p + "switch_mlp.down_proj", i0);
        std::printf("  dtypes: x=%s gate=%s inds=%s scores=%s switch_glu=%s\n", dt(x),
                    dt(ws.linear(x, p + "gate")), dt(i0), dt(s0), dt(y0));
    }

    const auto stage = [&](const char* label, const std::function<mx::array()>& make) {
        mx::array warm = make();
        mx::eval(warm);
        auto t0 = Clock::now();
        std::vector<mx::array> outs;
        outs.reserve(static_cast<std::size_t>(reps));
        for (int i = 0; i < reps; ++i) {
            outs.push_back(make());
        }
        mx::eval(outs);
        const double per = ms(t0, Clock::now()) / reps;
        std::printf("  %-34s %7.3f ms/call  x40 = %6.2f ms\n", label, per, per * 40);
        return per;
    };

    double prev = 0;
    const auto step = [&](const char* label, const std::function<mx::array()>& make) {
        const double now = stage(label, make);
        std::printf("      (+%.3f ms over previous stage)\n", now - prev);
        prev = now;
    };

    step("1 gate", [&] { return ws.linear(x, p + "gate"); });
    step("2 + moe_topk", [&] {
        return mlxl::moe_topk(ws.linear(x, p + "gate"), 8, true).second;
    });
    step("3 + switch_glu", [&] {
        auto [inds, scores] = mlxl::moe_topk(ws.linear(x, p + "gate"), 8, true);
        return mlxl::switch_glu(x, ws, p + "switch_mlp.gate_proj", p + "switch_mlp.up_proj",
                                p + "switch_mlp.down_proj", inds);
    });
    step("4 + combine", [&] {
        auto [inds, scores] = mlxl::moe_topk(ws.linear(x, p + "gate"), 8, true);
        mx::array y = mlxl::switch_glu(x, ws, p + "switch_mlp.gate_proj",
                                       p + "switch_mlp.up_proj", p + "switch_mlp.down_proj",
                                       inds);
        return mx::sum(mx::multiply(y, mx::expand_dims(scores, -1)), -2);
    });
    // Control: the same switch_glu, but fed indices that are already evaluated. This is
    // the only thing `blocks` did differently.
    auto [pre_inds, pre_scores] = mlxl::moe_topk(ws.linear(x, p + "gate"), 8, true);
    mx::eval({pre_inds, pre_scores});
    stage("control: switch_glu, evaluated inds", [&] {
        return mlxl::switch_glu(x, ws, p + "switch_mlp.gate_proj", p + "switch_mlp.up_proj",
                                p + "switch_mlp.down_proj", pre_inds);
    });
    return 0;
}

// --- moestream -------------------------------------------------------------
//
// The routed experts under the access pattern a real decode step actually has: all 40
// layers in one eval, each picking a DIFFERENT 8 of its 256 experts. Every isolation
// benchmark in this driver has instead reused one index set, which keeps 14 MB hot in
// cache and measures a kernel no decode step runs -- that is why `moe` reports 0.06 ms
// for a call that costs 0.64 ms in situ. Ablating the routed experts out of a full run
// charges them 25.8 ms/token; this is the same work with nothing else around it.
//
// Bytes are exact: 8 experts x 3 projections x 512x2048 weights x (4 bits + a bf16
// scale and bias per 64) = 14.16 MB/layer, 566 MB for the 40-layer step.
int cmd_moe_stream(int iters) {
    mlxl::WeightStore ws;
    if (!ws.load_directory(qwen_dir())) {
        std::printf("weight load failed\n");
        return 1;
    }
    const int layers = 40;
    const mx::array ids = mx::zeros({1, 1}, mx::int32);
    const mx::array x = ws.embed_lookup(ids, "language_model.model.embed_tokens");
    mx::eval(x);

    const auto key = [](int layer, const char* proj) {
        return "language_model.model.layers." + std::to_string(layer) + ".mlp.switch_mlp." +
               proj;
    };

    // Index sets built and evaluated up front so neither generation nor the dependency
    // on argpartition is inside the timed region.
    std::vector<mx::array> idx;
    idx.reserve(static_cast<std::size_t>(iters * layers));
    for (int i = 0; i < iters * layers; ++i) {
        idx.push_back(mx::astype(mx::random::randint(0, 256, {1, 1, 8}), mx::uint32));
    }
    mx::eval(idx);
    const mx::array fixed = mx::astype(mx::random::randint(0, 256, {1, 1, 8}), mx::uint32);
    mx::eval({fixed});

    const auto sweep = [&](const char* label, bool vary) {
        // Warm every layer's weights once, so no run pays another run's first touch.
        std::vector<mx::array> warm;
        warm.reserve(static_cast<std::size_t>(layers));
        for (int l = 0; l < layers; ++l) {
            warm.push_back(mlxl::switch_glu(x, ws, key(l, "gate_proj"), key(l, "up_proj"),
                                            key(l, "down_proj"), fixed));
        }
        mx::eval(warm);

        const auto t0 = Clock::now();
        for (int it = 0; it < iters; ++it) {
            std::vector<mx::array> outs;
            outs.reserve(static_cast<std::size_t>(layers));
            for (int l = 0; l < layers; ++l) {
                const mx::array& ix =
                    vary ? idx[static_cast<std::size_t>(it * layers + l)] : fixed;
                outs.push_back(mlxl::switch_glu(x, ws, key(l, "gate_proj"),
                                                key(l, "up_proj"), key(l, "down_proj"), ix));
            }
            mx::eval(outs);
        }
        const double per = ms(t0, Clock::now()) / iters;
        const double bytes = 566231040.0;
        std::printf("  %-40s %7.2f ms/token   %6.1f GB/s\n", label, per,
                    bytes / (per / 1000.0) / 1e9);
    };

    sweep("40 layers, same 8 experts (cached)", false);
    sweep("40 layers, different 8 each (real)", true);

    // The two things the sweeps above still do NOT reproduce from a real step: layer
    // l+1's input is layer l's output (a serial chain, so nothing overlaps), and the
    // expert indices are computed from that input rather than handed over ready-made
    // (so the gather cannot start until its own argpartition lands). Add them one at a
    // time -- whichever one moves the number is the thing that costs 25.8 ms/token.
    const auto chain = [&](const char* label, bool route_from_x) {
        std::vector<mx::array> warm;
        for (int l = 0; l < layers; ++l) {
            warm.push_back(mlxl::switch_glu(x, ws, key(l, "gate_proj"), key(l, "up_proj"),
                                            key(l, "down_proj"), fixed));
        }
        mx::eval(warm);

        const auto t0 = Clock::now();
        for (int it = 0; it < iters; ++it) {
            mx::array h = x;
            for (int l = 0; l < layers; ++l) {
                mx::array ix = route_from_x
                                   ? mlxl::moe_topk(ws.linear(h, "language_model.model.layers." +
                                                                     std::to_string(l) + ".mlp.gate"),
                                                    8, true)
                                         .first
                                   : idx[static_cast<std::size_t>(it * layers + l)];
                const mx::array y = mlxl::switch_glu(h, ws, key(l, "gate_proj"),
                                                     key(l, "up_proj"), key(l, "down_proj"), ix);
                // Reduce [1,1,8,2048] back to [1,1,2048] so the next layer can consume it.
                h = mx::sum(y, -2);
            }
            mx::eval(h);
        }
        const double per = ms(t0, Clock::now()) / iters;
        std::printf("  %-40s %7.2f ms/token   %6.1f GB/s\n", label, per,
                    566231040.0 / (per / 1000.0) / 1e9);
    };

    chain("chained, indices ready-made", false);
    chain("chained, indices routed from x (real)", true);
    return 0;
}
#endif


} // namespace lmp::diag
