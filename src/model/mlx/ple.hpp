#ifndef LLM_MLX_PLE_HPP
#define LLM_MLX_PLE_HPP

#if LMP_HAVE_MLX

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "activations.hpp"
#include "gated_residual.hpp"
#include "qwen35_moe_config.hpp"
#include "weight_store.hpp"
#include "src/model/qwen4_exp_ops.hpp"

#include "mlx/fast.h"
#include "mlx/io.h"
#include "mlx/ops.h"

namespace lmp::model::mlxl {

namespace mx = mlx::core;

struct PleCache {
    std::optional<mx::array> conv_state;
    std::vector<int> ngram_ctx;
    void clear() noexcept {
        conv_state.reset();
        ngram_ctx.clear();
    }
};

inline bool is_ple_shard_key(std::string_view key) {
    return WeightStore::is_offloaded_key(key);
}

inline std::string ple_shard_base(const WeightStore& w, const std::string& prefix, int shard) {
    const std::string dotted = prefix + "shards." + std::to_string(shard);
    const std::string underscored = prefix + "shard_" + std::to_string(shard);
    auto present = [&](const std::string& base) {
        return w.is_quantized(base) || w.has(base + ".weight") || w.has(base);
    };
    if (present(dotted)) {
        return dotted;
    }
    if (present(underscored)) {
        return underscored;
    }
    return {};
}

// Gather n-gram rows. Tiny tables live in WeightStore; large shards are never eval'd.
// Groups by shard so a decode/prefill step does one take per touched shard, not per gid.
inline mx::array ple_gather_rows(const WeightStore& w, const std::string& shard_prefix,
                                 const std::vector<std::int64_t>& gids, int n_heads,
                                 int row_width, int n_shards, std::int64_t rows_per_shard) {
    if (w.has_ple_disk()) {
        (void)n_heads;
        (void)shard_prefix;
        return w.ple_disk().gather_rows(gids, row_width, n_shards, rows_per_shard);
    }
    const int n = static_cast<int>(gids.size());
    std::vector<float> out(static_cast<std::size_t>(n * row_width), 0.0f);
    if (n == 0 || n_shards <= 0) {
        (void)n_heads;
        return mx::array(out.data(), {n, row_width});
    }

    struct Hit {
        int out_i;
        int row;
    };
    std::vector<std::vector<Hit>> by_shard(static_cast<std::size_t>(n_shards));
    for (int i = 0; i < n; ++i) {
        int shard = 0;
        std::int64_t row64 = 0;
        qwen4::shard_row(gids[static_cast<std::size_t>(i)], rows_per_shard, &shard, &row64);
        if (shard < 0 || shard >= n_shards) {
            continue;
        }
        by_shard[static_cast<std::size_t>(shard)].push_back(
            {i, static_cast<int>(row64)});
    }

    for (int shard = 0; shard < n_shards; ++shard) {
        const auto& hits = by_shard[static_cast<std::size_t>(shard)];
        if (hits.empty()) {
            continue;
        }
        const std::string base = ple_shard_base(w, shard_prefix, shard);
        if (base.empty()) {
            continue;
        }
        std::vector<int> rows;
        rows.reserve(hits.size());
        for (const Hit& h : hits) {
            rows.push_back(h.row);
        }
        mx::array idx(rows.data(), {static_cast<int>(rows.size())}, mx::int32);
        mx::array gathered =
            (w.is_quantized(base) || w.has(base + ".weight"))
                ? w.embed_lookup(idx, base)
                : mx::take(w.get(base), idx, 0);
        gathered = mx::contiguous(mx::astype(gathered, mx::float32));
        mx::eval(gathered);
        const int dim = static_cast<int>(gathered.shape().back());
        const int copy = std::min(dim, row_width);
        const float* src = gathered.data<float>();
        for (std::size_t j = 0; j < hits.size(); ++j) {
            std::copy(src + static_cast<std::size_t>(j) * static_cast<std::size_t>(dim),
                      src + static_cast<std::size_t>(j) * static_cast<std::size_t>(dim) +
                          static_cast<std::size_t>(copy),
                      out.data() + static_cast<std::size_t>(hits[j].out_i * row_width));
        }
    }
    (void)n_heads;
    return mx::array(out.data(), {n, row_width});
}

inline mx::array ple_forward(const mx::array& hidden, const mx::array& input_ids,
                             WeightStore& w, const std::string& prefix, PleCache& cache,
                             const Qwen35MoeConfig& cfg, const std::uint64_t* mults,
                             const std::int64_t* sizes, const std::int64_t* offsets) {
    const int B = static_cast<int>(hidden.shape()[0]);
    const int S = static_cast<int>(hidden.shape()[1]);
    const int d = cfg.hidden_size;
    const int hc = cfg.hc_count;
    const int ngram = cfg.ngram_size;
    const int hpn = cfg.heads_per_ngram;
    const int n_heads = (ngram - 1) * hpn;
    const int row_width = cfg.ple_embed_dim / n_heads;
    const int eos = cfg.eos_token_id;

    mx::array ids_e = mx::astype(mx::reshape(input_ids, {-1}), mx::int32);
    mx::eval(ids_e);
    const int ntok = static_cast<int>(ids_e.size());
    std::vector<int> ids(static_cast<std::size_t>(ntok));
    const int* ip = ids_e.data<int>();
    std::copy(ip, ip + ntok, ids.begin());

    const int ctx_len = ngram - 1;
    if (cache.ngram_ctx.empty()) {
        cache.ngram_ctx.assign(static_cast<std::size_t>(ctx_len), eos);
    }
    std::vector<int> history = cache.ngram_ctx;
    history.insert(history.end(), ids.begin(), ids.end());
    auto g = qwen4::ngram_gids(history, mults, sizes, offsets, ngram, hpn, eos);
    // Keep only the new positions.
    std::vector<std::int64_t> new_gids;
    new_gids.reserve(static_cast<std::size_t>(S * n_heads));
    const int hist_t = static_cast<int>(history.size());
    for (int t = hist_t - S; t < hist_t; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            new_gids.push_back(g.gid[static_cast<std::size_t>(t * n_heads + h)]);
        }
    }
    cache.ngram_ctx.assign(history.end() - ctx_len, history.end());

    const int n_shards = cfg.split_ngram_parts > 0 ? cfg.split_ngram_parts : 1;
    std::int64_t total_rows = 0;
    for (int h = 0; h < n_heads; ++h) {
        total_rows += sizes[h];
    }
    const int div = cfg.make_ngram_vocab_size_divisible_by > 0
                        ? cfg.make_ngram_vocab_size_divisible_by
                        : 1;
    const std::int64_t padded = ((total_rows + div - 1) / div) * div;
    const std::int64_t rows_per_shard = (padded + n_shards - 1) / n_shards;

    mx::array rows = ple_gather_rows(w, prefix + "ple_embedding.ngram_embedding.", new_gids,
                                     n_heads, row_width, n_shards, rows_per_shard);
    mx::array emb = mx::reshape(rows, {B, S, n_heads * row_width});
    emb = mx::astype(emb, hidden.dtype());

    mx::array key = group_rms_norm(w.linear(emb, prefix + "key_proj"),
                                   w.get(prefix + "norm_key.weight"), d, cfg.rms_norm_eps);
    mx::array value = w.linear(emb, prefix + "value_proj");
    mx::array query = group_rms_norm(hidden, w.get(prefix + "norm_query.weight"), d,
                                     cfg.rms_norm_eps);
    key = mx::reshape(key, {B, S, hc, d});
    query = mx::reshape(query, {B, S, hc, d});
    mx::array gate = mx::divide(mx::sum(mx::multiply(key, query), -1, true),
                                mx::array(std::sqrt(static_cast<float>(d)), hidden.dtype()));
    mx::array abs_g = mx::maximum(mx::abs(gate), mx::array(1e-6f, hidden.dtype()));
    gate = mx::multiply(mx::sqrt(abs_g), mx::sign(gate));
    mx::array gated = mx::multiply(mx::sigmoid(gate), mx::expand_dims(value, -2));
    gated = mx::reshape(gated, {B, S, hc * d});

    const int k = cfg.ple_conv_kernel_size;
    const int dil = cfg.ngram_size;
    const int state_len = (k - 1) * dil;
    mx::array conv_in = group_rms_norm(gated, w.get(prefix + "norm_conv.weight"), d,
                                       cfg.rms_norm_eps);
    mx::array state = cache.conv_state.value_or(
        mx::zeros({B, state_len, hc * d}, hidden.dtype()));
    mx::array full = mx::concatenate({state, conv_in}, 1);
    cache.conv_state = mx::contiguous(mx::slice(full, {0, full.shape()[1] - state_len, 0},
                                                {B, full.shape()[1], hc * d}));
    mx::array conv_w = w.get(prefix + "conv1d.weight");
    mx::array conv_out = silu(mx::conv1d(full, conv_w, 1, 0, dil, hc * d));
    const int out_t = static_cast<int>(conv_out.shape()[1]);
    if (out_t != S) {
        conv_out = mx::slice(conv_out, {0, out_t - S, 0}, {B, out_t, hc * d});
    }
    return mx::add(gated, conv_out);
}

} // namespace lmp::model::mlxl

#endif
#endif
