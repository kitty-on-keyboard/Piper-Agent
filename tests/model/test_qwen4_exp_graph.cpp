// Tiny-scale Flash-Next modules on MLX vs the GPU-free spec in qwen4_exp_ops.hpp.
// Needs Metal, not a checkpoint. Labelled realmodel for the same reason as
// test_switch_glu_fusion: CI has no GPU.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>
#include <vector>

#include "src/model/qwen4_exp_ops.hpp"
#include "tests/check.hpp"

#if LMP_HAVE_MLX

#include "src/model/mlx/activations.hpp"
#include "src/model/mlx/gated_residual.hpp"
#include "src/model/mlx/ple.hpp"
#include "src/model/mlx/qsa.hpp"
#include "src/model/mlx/qwen35_moe_config.hpp"
#include "src/model/mlx/weight_store.hpp"

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace mx = mlx::core;
using lmp::model::mlxl::WeightStore;
namespace q4 = lmp::model::qwen4;
namespace mlxl = lmp::model::mlxl;

namespace {

mx::array f32(const std::vector<float>& v, std::vector<int> shape) {
    mx::Shape s(shape.begin(), shape.end());
    return mx::array(v.data(), mx::Shape(s.begin(), s.end()), mx::float32);
}

float max_abs_diff(const mx::array& a, const std::vector<float>& b) {
    mx::array x = mx::astype(mx::reshape(a, {-1}), mx::float32);
    mx::eval(x);
    const int n = static_cast<int>(x.size());
    if (n != static_cast<int>(b.size())) {
        return 1e30f;
    }
    const float* p = x.data<float>();
    float m = 0;
    for (int i = 0; i < n; ++i) {
        m = std::max(m, std::fabs(p[i] - b[static_cast<std::size_t>(i)]));
    }
    return m;
}

} // namespace

TEST(gated_residual_mlx_matches_cpu_spec) {
    const int hc = 2, d = 4, rank = 2;
    std::vector<float> hyper = {1, 2, 3, 4, 1, 2, 3, 4};
    std::vector<float> hc_w(8, 1.0f);
    std::vector<float> w_down(rank * hc * d, 0.1f);
    std::vector<float> w_up(hc * d * rank, 0.2f);
    std::vector<float> w_inj(hc * hc * d, 0.0f);
    for (int b = 0; b < hc; ++b) {
        w_inj[static_cast<std::size_t>(b * hc * d + b * d)] = 0.5f;
    }

    const auto cpu = q4::gated_residual_read(hyper.data(), hc_w.data(), w_down.data(),
                                             w_up.data(), w_inj.data(), hc, d, rank, 1e-6f,
                                             true);

    WeightStore w;
    w.set("hc_norm.weight", f32(hc_w, {hc * d}));
    w.set("input_mix_weight_down.weight", f32(w_down, {rank, hc * d}));
    w.set("input_mix_weight_up.weight", f32(w_up, {hc * d, rank}));
    w.set("block_inject_weight.weight", f32(w_inj, {hc, hc * d}));

    mx::array h = f32(hyper, {1, 1, hc * d});
    auto gr = mlxl::gated_residual_read(h, w, "", hc, d, rank, 1e-6f, true);
    mx::eval({gr.mixed, gr.inject});
    CHECK(max_abs_diff(gr.mixed, cpu.mixed) < 1e-5f);
    CHECK(max_abs_diff(gr.inject, cpu.inject) < 1e-5f);

    std::vector<float> y = {0.5f, -0.25f, 1.0f, 0.0f};
    std::vector<float> written(static_cast<std::size_t>(hc * d));
    q4::gated_residual_write(hyper.data(), y.data(), cpu.inject.data(), written.data(), hc, d);
    mx::array y_mx = f32(y, {1, 1, d});
    mx::array out = mlxl::gated_residual_write(gr, y_mx);
    mx::eval(out);
    CHECK(max_abs_diff(out, written) < 1e-5f);

    auto mix_only = mlxl::gated_residual_read(h, w, "", hc, d, rank, 1e-6f, false);
    mx::eval(mix_only.mixed);
    CHECK(max_abs_diff(mix_only.mixed, cpu.mixed) < 1e-5f);
}

TEST(flash_gdn_gated_norm_is_sigmoid_not_silu) {
    // silu(g) = g * sigmoid(g). For |g| != 1 the two GDN output gates disagree.
    // 36 Flash layers using the Qwen3.5 silu tail is what printed `!`.
    std::vector<float> h = {1.0f, -2.0f, 0.5f, 3.0f};
    std::vector<float> g = {0.5f, -1.0f, 2.0f, 0.0f};
    std::vector<float> w = {1.0f, 1.0f, 1.0f, 1.0f};
    mx::array hidden = f32(h, {1, 1, 4});
    mx::array gate = f32(g, {1, 1, 4});
    mx::array weight = f32(w, {4});
    mx::array silu_y =
        mlxl::precise_rms_norm_gated(hidden, gate, weight, 1e-6f, mlxl::GatedNormAct::Silu);
    mx::array sig_y =
        mlxl::precise_rms_norm_gated(hidden, gate, weight, 1e-6f, mlxl::GatedNormAct::Sigmoid);
    mx::eval({silu_y, sig_y});
    mx::array d = mx::max(mx::abs(mx::subtract(silu_y, sig_y)));
    mx::eval(d);
    CHECK(d.item<float>() > 0.05f);

    mx::array nf = mx::astype(mx::reshape(mx::fast::rms_norm(hidden, weight, 1e-6f), {-1}),
                             mx::float32);
    mx::eval(nf);
    const float* npp = nf.data<float>();
    std::vector<float> expect(4);
    for (int i = 0; i < 4; ++i) {
        const float sig = 1.0f / (1.0f + std::exp(-g[static_cast<std::size_t>(i)]));
        expect[static_cast<std::size_t>(i)] = sig * npp[i];
    }
    CHECK(max_abs_diff(sig_y, expect) < 1e-5f);
}

TEST(exact_l2_norm_is_not_rms_norm) {
    std::vector<float> x = {3.0f, 4.0f};
    float cpu[2];
    q4::l2norm(x.data(), cpu, 2, 1e-6f);
    mx::array y = mlxl::exact_l2_norm(f32(x, {1, 2}));
    mx::eval(y);
    CHECK(max_abs_diff(y, {cpu[0], cpu[1]}) < 1e-5f);
    mx::array rms = mx::fast::rms_norm(f32(x, {1, 2}), std::nullopt, 1e-6f);
    mx::eval(rms);
    mx::array d = mx::max(mx::abs(mx::subtract(y, rms)));
    mx::eval(d);
    CHECK(d.item<float>() > 0.05f);
}

TEST(qsa_indexer_below_budget_is_nullopt) {
    WeightStore w;
    const int H = 2, D = 4, hidden = 8, T = 8, budget = 16;
    w.set("index_qk_proj.weight", mx::ones({(H + 1) * D, hidden}, mx::float32));
    w.set("q_layernorm.weight", mx::ones({D}, mx::float32));
    w.set("k_layernorm.weight", mx::ones({D}, mx::float32));
    mlxl::IndexerCache cache;
    mx::array x = mx::ones({1, T, hidden}, mx::float32);
    auto mask = mlxl::qsa_indexer_mask(x, w, "", cache, 0, H, 1, D, budget, 4, 2, 10000.f,
                                       1e-6f);
    CHECK(!mask.has_value());
    CHECK_EQ(cache.offset, T);
}

TEST(qsa_indexer_mask_matches_cpu_keep) {
    WeightStore w;
    const int H = 2, D = 4, hidden = 8, T = 24, budget = 16, compress = 4, rotary = 2;
    std::vector<float> proj(static_cast<std::size_t>((H + 1) * D * hidden));
    for (std::size_t i = 0; i < proj.size(); ++i) {
        proj[i] = 0.05f * std::sin(static_cast<float>(i) * 0.17f);
    }
    w.set("index_qk_proj.weight", f32(proj, {(H + 1) * D, hidden}));
    w.set("q_layernorm.weight", mx::ones({D}, mx::float32));
    w.set("k_layernorm.weight", mx::ones({D}, mx::float32));
    mlxl::IndexerCache cache;
    std::vector<float> xv(static_cast<std::size_t>(T * hidden));
    for (int i = 0; i < T * hidden; ++i) {
        xv[static_cast<std::size_t>(i)] = 0.1f * std::cos(static_cast<float>(i) * 0.09f);
    }
    mx::array x = f32(xv, {1, T, hidden});
    auto mask = mlxl::qsa_indexer_mask(x, w, "", cache, 0, H, 1, D, budget, compress, rotary,
                                       10000.f, 1e-6f);
    REQUIRE(mask.has_value());
    mx::array m = mx::astype(mx::reshape(*mask, {-1}), mx::uint8);
    mx::eval(m);
    const auto* mp = m.data<uint8_t>();

    mx::array qk = w.linear(x, "index_qk_proj");
    const int q_dim = H * D;
    mx::array q = mx::reshape(mx::slice(qk, {0, 0, 0}, {1, T, q_dim}), {1, T, H, D});
    mx::array raw_k = mx::reshape(mx::slice(qk, {0, 0, q_dim}, {1, T, q_dim + D}), {1, T, D});
    const int n_blocks = T / compress;
    mx::array pooled =
        mx::reshape(mx::slice(raw_k, {0, 0, 0}, {1, n_blocks * compress, D}),
                    {1, n_blocks, compress, D});
    pooled = mx::mean(pooled, 2);
    pooled = mx::fast::rms_norm(pooled, w.get("k_layernorm.weight"), 1e-6f);
    pooled = mlxl::rope_at_positions(pooled, mx::multiply(mx::arange(n_blocks), mx::array(compress)),
                                     rotary, 10000.f);
    q = mx::fast::rms_norm(q, w.get("q_layernorm.weight"), 1e-6f);
    q = mlxl::rope_at_positions(q, mx::arange(T), rotary, 10000.f);
    mx::array dots =
        mx::matmul(mx::astype(q, mx::float32), mx::swapaxes(mx::astype(pooled, mx::float32), 1, 2));
    mx::array scores = mx::divide(mx::sum(mx::maximum(dots, mx::array(0.0f)), 2),
                                  mx::array(std::sqrt(static_cast<float>(D))));
    mx::eval(scores);
    const float* sp = scores.data<float>();
    const int topk = budget / compress;
    int mismatches = 0;
    for (int s = 0; s < T; ++s) {
        auto keep = q4::qsa_keep_mask(s, T, compress, topk, sp + s * n_blocks);
        for (int t = 0; t < T; ++t) {
            if ((mp[s * T + t] != 0) != (keep[static_cast<std::size_t>(t)] != 0)) {
                ++mismatches;
            }
        }
    }
    CHECK_EQ(mismatches, 0);
}

TEST(ple_tiny_table_chunked_matches_single_shot) {
    mlxl::Qwen35MoeConfig cfg;
    cfg.hidden_size = 4;
    cfg.hc_count = 2;
    cfg.ngram_size = 3;
    cfg.heads_per_ngram = 1;
    cfg.ple_embed_dim = 4;
    cfg.ple_conv_kernel_size = 2;
    cfg.split_ngram_parts = 1;
    cfg.make_ngram_vocab_size_divisible_by = 1;
    cfg.eos_token_id = 9;
    cfg.rms_norm_eps = 1e-6f;

    const int row_w = 2;
    std::vector<std::int64_t> sizes = {11, 13};
    std::vector<std::int64_t> offsets = {0, 11};
    const auto mults = q4::default_layer_multipliers(q4::kDefaultHashSeed, 3, 50, 0);
    REQUIRE(mults.size() >= 3);

    const int rows = 11 + 13;
    std::vector<float> table(static_cast<std::size_t>(rows * row_w));
    for (int i = 0; i < rows * row_w; ++i) {
        table[static_cast<std::size_t>(i)] = 0.01f * static_cast<float>(i % 7);
    }
    WeightStore w;
    w.set("ple_embedding.ngram_embedding.shard_0", f32(table, {rows, row_w}));
    w.set("key_proj.weight", mx::ones({cfg.hc_count * cfg.hidden_size, cfg.ple_embed_dim}, mx::float32));
    w.set("value_proj.weight", mx::ones({cfg.hidden_size, cfg.ple_embed_dim}, mx::float32));
    w.set("norm_key.weight", mx::ones({cfg.hc_count * cfg.hidden_size}, mx::float32));
    w.set("norm_query.weight", mx::ones({cfg.hc_count * cfg.hidden_size}, mx::float32));
    w.set("norm_conv.weight", mx::ones({cfg.hc_count * cfg.hidden_size}, mx::float32));
    const int k = cfg.ple_conv_kernel_size;
    const int c = cfg.hc_count * cfg.hidden_size;
    w.set("conv1d.weight", mx::ones({c, k, 1}, mx::float32));

    std::vector<int> ids_i = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    mx::array ids = mx::array(ids_i.data(), {1, 12}, mx::int32);
    mx::array hidden = mx::ones({1, 12, c}, mx::float32);

    mlxl::PleCache one;
    mx::array full = mlxl::ple_forward(hidden, ids, w, "", one, cfg, mults.data(), sizes.data(),
                                       offsets.data());
    mx::eval(full);

    mlxl::PleCache chunked;
    mx::array a = mlxl::ple_forward(mx::slice(hidden, {0, 0, 0}, {1, 5, c}),
                                    mx::slice(ids, {0, 0}, {1, 5}), w, "", chunked, cfg,
                                    mults.data(), sizes.data(), offsets.data());
    mx::array b = mlxl::ple_forward(mx::slice(hidden, {0, 5, 0}, {1, 12, c}),
                                    mx::slice(ids, {0, 5}, {1, 12}), w, "", chunked, cfg,
                                    mults.data(), sizes.data(), offsets.data());
    mx::array cat = mx::concatenate({a, b}, 1);
    mx::eval(cat);
    mx::array diff = mx::max(mx::abs(mx::subtract(full, cat)));
    mx::eval(diff);
    CHECK(diff.item<float>() < 1e-4f);

    const auto m0 = q4::default_layer_multipliers(0, 3, 50, 0);
    CHECK(m0[0] != mults[0]);
}

TEST(weight_store_skips_eval_of_ple_shards) {
    CHECK(WeightStore::is_offloaded_key("layers.1.ple.ple_embedding.ngram_embedding.shard_0"));
    CHECK(WeightStore::is_offloaded_key(
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight"));
    CHECK(WeightStore::is_offloaded_key(
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.127.scales"));
    CHECK(!WeightStore::is_offloaded_key("layers.1.ple.key_proj.weight"));
    CHECK(mlxl::is_ple_shard_key("ple_embedding.ngram_embedding.shards.3.biases"));
    CHECK(WeightStore::is_lazy_expert_key(
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"));
    CHECK(WeightStore::is_lazy_expert_key(
        "language_model.model.layers.7.mlp.switch_mlp.down_proj.scales"));
    CHECK(!WeightStore::is_lazy_expert_key(
        "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight"));
}

TEST(ple_gather_accepts_shards_dot_naming) {
    WeightStore w;
    const std::vector<float> table = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    w.set("ple_embedding.ngram_embedding.shards.0", f32(table, {3, 2}));
    const std::vector<std::int64_t> gids = {0, 2};
    mx::array rows =
        mlxl::ple_gather_rows(w, "ple_embedding.ngram_embedding.", gids, 1, 2, 1, 3);
    mx::eval(rows);
    CHECK_EQ(static_cast<int>(rows.size()), 4);
    const float* p = rows.data<float>();
    CHECK_EQ(p[0], 1.f);
    CHECK_EQ(p[1], 2.f);
    CHECK_EQ(p[2], 5.f);
    CHECK_EQ(p[3], 6.f);
}

#else

TEST(qwen4_exp_graph_needs_mlx) {
    std::printf("skipped: built without MLX\n");
}

#endif
