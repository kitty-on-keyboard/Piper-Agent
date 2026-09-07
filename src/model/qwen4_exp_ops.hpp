#ifndef LMP_MODEL_QWEN4_EXP_OPS_HPP
#define LMP_MODEL_QWEN4_EXP_OPS_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace lmp::model::qwen4 {

// GPU-free reference arithmetic for Flash-Next. Used by gate tests and as the
// specification the MLX kernels must match. Do not "simplify" the QSA mask or the
// n-gram seed; both are documented bugs that still generate fluent text.

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline float rms(const float* x, int n, float eps) {
    double acc = 0;
    for (int i = 0; i < n; ++i) {
        acc += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    return std::sqrt(static_cast<float>(acc / n) + eps);
}

// Group RMSNorm: one statistic per branch of width `d`, then x/rms * w.
// `w` is assumed already folded as (1+raw) at sanitize.
inline void group_rms_norm(const float* x, const float* w, float* out, int hc, int d,
                           float eps) {
    for (int b = 0; b < hc; ++b) {
        const float* xb = x + b * d;
        const float r = rms(xb, d, eps);
        for (int i = 0; i < d; ++i) {
            out[b * d + i] = (xb[i] / r) * w[b * d + i];
        }
    }
}

// Gated-residual read. hyper is [hc*d], returns mixed [d] and inject [hc].
// use_combine=false (final mixer) skips inject and only returns mixed.
struct GrRead {
    std::vector<float> mixed;
    std::vector<float> inject; // empty when !use_combine
};

inline GrRead gated_residual_read(const float* hyper, const float* hc_w,
                                  const float* w_down, const float* w_up,
                                  const float* w_inject, int hc, int d, int rank,
                                  float eps, bool use_combine) {
    const int hc_dim = hc * d;
    std::vector<float> rbar(static_cast<std::size_t>(hc_dim));
    group_rms_norm(hyper, hc_w, rbar.data(), hc, d, eps);

    std::vector<float> down(static_cast<std::size_t>(rank), 0.0f);
    const float inv_hc = 1.0f / static_cast<float>(hc);
    for (int r = 0; r < rank; ++r) {
        double acc = 0;
        for (int i = 0; i < hc_dim; ++i) {
            acc += static_cast<double>(w_down[r * hc_dim + i]) *
                   static_cast<double>(rbar[static_cast<std::size_t>(i)]);
        }
        down[static_cast<std::size_t>(r)] = silu(static_cast<float>(acc) * inv_hc);
    }
    std::vector<float> g(static_cast<std::size_t>(hc_dim), 0.0f);
    for (int i = 0; i < hc_dim; ++i) {
        double acc = 0;
        for (int r = 0; r < rank; ++r) {
            acc += static_cast<double>(w_up[i * rank + r]) *
                   static_cast<double>(down[static_cast<std::size_t>(r)]);
        }
        g[static_cast<std::size_t>(i)] = sigmoid(static_cast<float>(acc));
    }

    GrRead out;
    out.mixed.assign(static_cast<std::size_t>(d), 0.0f);
    for (int b = 0; b < hc; ++b) {
        for (int i = 0; i < d; ++i) {
            out.mixed[static_cast<std::size_t>(i)] +=
                g[static_cast<std::size_t>(b * d + i)] * rbar[static_cast<std::size_t>(b * d + i)];
        }
    }
    for (int i = 0; i < d; ++i) {
        out.mixed[static_cast<std::size_t>(i)] *= inv_hc;
    }
    if (!use_combine || w_inject == nullptr) {
        return out;
    }
    out.inject.assign(static_cast<std::size_t>(hc), 0.0f);
    for (int b = 0; b < hc; ++b) {
        double acc = 0;
        for (int i = 0; i < hc_dim; ++i) {
            acc += static_cast<double>(w_inject[b * hc_dim + i]) *
                   static_cast<double>(rbar[static_cast<std::size_t>(i)]);
        }
        out.inject[static_cast<std::size_t>(b)] = 2.0f * sigmoid(static_cast<float>(acc) * inv_hc);
    }
    return out;
}

inline void gated_residual_write(const float* hyper, const float* y, const float* inject,
                                 float* out, int hc, int d) {
    for (int b = 0; b < hc; ++b) {
        for (int i = 0; i < d; ++i) {
            out[b * d + i] = hyper[b * d + i] + inject[b] * y[i];
        }
    }
}

// Exact L2 (not RMS): x * rsqrt(sum(x*x)+eps) in f32.
inline void l2norm(const float* x, float* out, int n, float eps = 1e-6f) {
    double acc = 0;
    for (int i = 0; i < n; ++i) {
        acc += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(acc) + eps);
    for (int i = 0; i < n; ++i) {
        out[i] = x[i] * scale;
    }
}

// QSA keep mask for one query. keep[key_pos] is true iff that key may attend.
// scores_per_block: n_blocks floats (already ReLU-sum / sqrt(d)); ineligible blocks
// should be -inf BEFORE this call, or we apply eligibility here.
inline std::vector<char> qsa_keep_mask(int q_pos, int kv_len, int compress, int block_topk,
                                       const float* scores) {
    std::vector<char> keep(static_cast<std::size_t>(kv_len), 0);
    if (kv_len <= 0) {
        return keep;
    }
    const int n_blocks = kv_len / compress;
    std::vector<int> eligible;
    eligible.reserve(static_cast<std::size_t>(n_blocks));
    for (int b = 0; b < n_blocks; ++b) {
        const int block_end = b * compress + compress - 1;
        if (block_end <= q_pos) {
            eligible.push_back(b);
        }
    }
    const int k = std::min(block_topk, static_cast<int>(eligible.size()));
    std::vector<int> chosen = eligible;
    if (k > 0 && static_cast<int>(chosen.size()) > k) {
        std::partial_sort(chosen.begin(), chosen.begin() + k, chosen.end(),
                          [&](int a, int b) { return scores[a] > scores[b]; });
        chosen.resize(static_cast<std::size_t>(k));
    }
    for (int b : chosen) {
        for (int t = 0; t < compress; ++t) {
            keep[static_cast<std::size_t>(b * compress + t)] = 1;
        }
    }
    const int own_start = ((q_pos + 1) / compress) * compress;
    for (int p = 0; p < kv_len; ++p) {
        const bool own_tail = p >= own_start && p <= q_pos;
        const bool causal = p <= q_pos;
        keep[static_cast<std::size_t>(p)] =
            static_cast<char>((keep[static_cast<std::size_t>(p)] || own_tail) && causal);
    }
    return keep;
}

inline bool qsa_short_circuit(int kv_len, int budget) { return kv_len <= budget; }

// --- PLE hash ----------------------------------------------------------------

constexpr std::uint64_t kMask64 = 0xFFFFFFFFFFFFFFFFULL;
constexpr std::uint64_t kGamma = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t kM1 = 0xBF58476D1CE4E5B9ULL;
constexpr std::uint64_t kM2 = 0x94D049BB133111EBULL;
constexpr int kPrime1 = 10007;
constexpr int kDefaultHashSeed = 1234;

inline std::uint64_t splitmix64(std::uint64_t v) {
    v = (v + kGamma) & kMask64;
    v = ((v ^ (v >> 30)) * kM1) & kMask64;
    v = ((v ^ (v >> 27)) * kM2) & kMask64;
    return (v ^ (v >> 31)) & kMask64;
}

inline bool is_prime(int v) {
    if (v < 2) {
        return false;
    }
    if (v % 2 == 0) {
        return v == 2;
    }
    for (int d = 3; d * d <= v; d += 2) {
        if (v % d == 0) {
            return false;
        }
    }
    return true;
}

inline int nth_prime_after(int start, int count) {
    int p = start;
    for (int i = 0; i < count; ++i) {
        ++p;
        while (!is_prime(p)) {
            ++p;
        }
    }
    return p;
}

// Transformers / checkpoint default multipliers for seed 1234, ple_layer_index 0.
inline std::vector<std::uint64_t> default_layer_multipliers(int seed, int ngram_size,
                                                            int vocab_size,
                                                            int ple_layer_index = 0) {
    std::vector<std::uint64_t> mults;
    const std::uint64_t max_long = (1ULL << 63) - 1;
    const std::uint64_t half =
        std::max<std::uint64_t>(1, (max_long / static_cast<std::uint64_t>(std::max(vocab_size, 1))) / 2);
    const std::uint64_t base_seed =
        static_cast<std::uint64_t>(seed) + static_cast<std::uint64_t>(kPrime1) * static_cast<std::uint64_t>(ple_layer_index);
    for (int i = 0; i < ngram_size; ++i) {
        const std::uint64_t sm =
            splitmix64((base_seed + kGamma * static_cast<std::uint64_t>(i + 1)) & kMask64);
        mults.push_back(2 * (sm % half) + 1);
    }
    return mults;
}

// EOS-aware right shift of a token history. Positions that would cross EOS become eos.
inline std::vector<int> shift_right_eos(const std::vector<int>& ids, int shift, int eos) {
    if (shift == 0) {
        return ids;
    }
    const int t = static_cast<int>(ids.size());
    std::vector<int> prev(static_cast<std::size_t>(t), -1);
    int last = -1;
    for (int i = 0; i < t; ++i) {
        prev[static_cast<std::size_t>(i)] = last;
        if (ids[static_cast<std::size_t>(i)] == eos) {
            last = i;
        }
    }
    std::vector<int> out(static_cast<std::size_t>(t), eos);
    for (int i = 0; i < t; ++i) {
        const int in_segment = i - (prev[static_cast<std::size_t>(i)] + 1);
        const int src = i - shift;
        if (in_segment >= shift && src >= 0) {
            out[static_cast<std::size_t>(i)] = ids[static_cast<std::size_t>(src)];
        }
    }
    return out;
}

struct NgramGid {
    std::vector<std::int64_t> gid; // [T, ngram_heads]
    int ngram_heads = 0;
};

inline NgramGid ngram_gids(const std::vector<int>& ids, const std::uint64_t* mults,
                           const std::int64_t* sizes, const std::int64_t* offsets,
                           int ngram_size, int heads_per_ngram, int eos) {
    NgramGid out;
    out.ngram_heads = (ngram_size - 1) * heads_per_ngram;
    const int t = static_cast<int>(ids.size());
    out.gid.assign(static_cast<std::size_t>(t * out.ngram_heads), 0);
    std::vector<std::vector<int>> shifted;
    shifted.reserve(static_cast<std::size_t>(ngram_size));
    for (int s = 0; s < ngram_size; ++s) {
        shifted.push_back(shift_right_eos(ids, s, eos));
    }
    int head = 0;
    for (int ngram = 2; ngram <= ngram_size; ++ngram) {
        for (int h = 0; h < heads_per_ngram; ++h) {
            const int gi = head;
            for (int i = 0; i < t; ++i) {
                std::uint64_t mixed =
                    static_cast<std::uint64_t>(static_cast<std::int64_t>(shifted[0][static_cast<std::size_t>(i)])) *
                    mults[0];
                for (int p = 1; p < ngram; ++p) {
                    mixed ^= static_cast<std::uint64_t>(
                                 static_cast<std::int64_t>(shifted[static_cast<std::size_t>(p)]
                                                               [static_cast<std::size_t>(i)])) *
                             mults[p];
                }
                const std::int64_t sz = sizes[gi];
                const std::int64_t off = offsets[gi];
                out.gid[static_cast<std::size_t>(i * out.ngram_heads + gi)] =
                    static_cast<std::int64_t>(mixed % static_cast<std::uint64_t>(sz)) + off;
            }
            ++head;
        }
    }
    return out;
}

inline void shard_row(std::int64_t gid, std::int64_t rows_per_shard, int* shard, std::int64_t* row) {
    *shard = static_cast<int>(gid / rows_per_shard);
    *row = gid % rows_per_shard;
}

} // namespace lmp::model::qwen4

#endif
