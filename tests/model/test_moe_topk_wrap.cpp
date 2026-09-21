// G1 activated-expert wrapping: k1 activate / k2 renormalize.
//
// GATE: pure CPU, no MLX, no checkpoint. Locks the routing algebra and the
// LMP_MOE_K1 / LMP_MOE_K2 kill-switch resolve rules so CI (LMP_WITH_MLX=OFF) catches
// regressions without Metal.

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "src/model/mlx/moe_topk_wrap.hpp"
#include "tests/check.hpp"

using lmp::model::mlxl::MoeTopkPair;
using lmp::model::mlxl::MoeTopkResult;
using lmp::model::mlxl::moe_topk_cpu;
using lmp::model::mlxl::resolve_moe_topk_wrap;

namespace {

// Fixed fixture: 16 distinct logits so top-k ties cannot hide a wrong partition.
constexpr int kExperts = 16;
constexpr float kLogits[kExperts] = {
    0.10f, 2.40f, -0.50f, 1.80f, 0.90f, 3.10f, -1.20f, 1.10f,
    0.30f, 2.00f, -0.10f, 1.50f, 0.70f, 2.70f, -0.80f, 1.30f,
};

float sum_scores(const MoeTopkResult& r) {
    float s = 0.0f;
    for (float v : r.scores) {
        s += v;
    }
    return s;
}

// Sort by descending score for stable comparison (argpartition leaves order free).
std::vector<std::pair<int, float>> ranked(const MoeTopkResult& r) {
    std::vector<std::pair<int, float>> out;
    out.reserve(r.indices.size());
    for (std::size_t i = 0; i < r.indices.size(); ++i) {
        out.emplace_back(r.indices[i], r.scores[i]);
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first < b.first;
    });
    return out;
}

} // namespace

// (a) k1==k2 matches the single-k baseline on a fixed logit fixture.
TEST(moe_wrap_k1_eq_k2_matches_baseline) {
    constexpr int k = 8;
    const MoeTopkResult baseline = moe_topk_cpu(kLogits, kExperts, k, k, true);
    const MoeTopkResult wrap = moe_topk_cpu(kLogits, kExperts, k, k, true);
    CHECK(baseline.indices.size() == static_cast<std::size_t>(k));
    CHECK(wrap.indices.size() == static_cast<std::size_t>(k));
    const auto a = ranked(baseline);
    const auto b = ranked(wrap);
    CHECK(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].first == b[i].first);
        CHECK(std::fabs(a[i].second - b[i].second) < 1e-6f);
    }
    // Single-k renormalize: scores sum to 1.
    CHECK(std::fabs(sum_scores(baseline) - 1.0f) < 1e-5f);
}

// (b) k1<k2 renormalizes over top-k2 mass, not just top-k1.
TEST(moe_wrap_k1_lt_k2_renorms_over_k2_mass) {
    constexpr int k1 = 4;
    constexpr int k2 = 8;
    const MoeTopkResult wrapped = moe_topk_cpu(kLogits, kExperts, k1, k2, true);
    const MoeTopkResult only_k1 = moe_topk_cpu(kLogits, kExperts, k1, k1, true);

    CHECK(wrapped.indices.size() == static_cast<std::size_t>(k1));
    CHECK(only_k1.indices.size() == static_cast<std::size_t>(k1));

    // Activated expert *set* matches top-k1 (same highest softmax experts).
    auto w = ranked(wrapped);
    auto b = ranked(only_k1);
    for (std::size_t i = 0; i < w.size(); ++i) {
        CHECK(w[i].first == b[i].first);
    }

    // Weights differ: wrap keeps k2-normalized mass on the k1 slots, so they sum to
    // less than 1 (the discarded k2-k1 experts hold the rest). Plain k1 renorms to 1.
    const float wrap_sum = sum_scores(wrapped);
    const float k1_sum = sum_scores(only_k1);
    CHECK(std::fabs(k1_sum - 1.0f) < 1e-5f);
    CHECK(wrap_sum < 0.999f);
    CHECK(wrap_sum > 0.0f);

    // Per-expert scores must not match the k1-only renormalization.
    bool any_score_diff = false;
    for (std::size_t i = 0; i < w.size(); ++i) {
        if (std::fabs(w[i].second - b[i].second) > 1e-6f) {
            any_score_diff = true;
        }
    }
    CHECK(any_score_diff);

    // Explicit check: wrap scores == (raw softmax mass of top-k1) / (mass of top-k2).
    const MoeTopkResult top2 = moe_topk_cpu(kLogits, kExperts, k2, k2, false);
    float mass_k2 = 0.0f;
    for (float s : top2.scores) {
        mass_k2 += s;
    }
    float mass_k1 = 0.0f;
    for (const auto& p : w) {
        // raw softmax for this expert from the unnormalized-over-topk path with k=k2
        for (std::size_t i = 0; i < top2.indices.size(); ++i) {
            if (top2.indices[i] == p.first) {
                mass_k1 += top2.scores[i];
            }
        }
    }
    CHECK(std::fabs(wrap_sum - (mass_k1 / mass_k2)) < 1e-5f);
}

// (c) invalid k1/k2 rejected or clamped safely.
TEST(moe_wrap_invalid_k_rejected_or_clamped) {
    MoeTopkPair out{};

    // Kill switch off: unset → defaults.
    CHECK(resolve_moe_topk_wrap(8, 256, nullptr, nullptr, out));
    CHECK(out.k1 == 8);
    CHECK(out.k2 == 8);

    // Prove-it arm.
    CHECK(resolve_moe_topk_wrap(8, 256, "4", "16", out));
    CHECK(out.k1 == 4);
    CHECK(out.k2 == 16);

    // k2 < k1 → clamp k2 up to k1.
    CHECK(resolve_moe_topk_wrap(8, 256, "8", "4", out));
    CHECK(out.k1 == 8);
    CHECK(out.k2 == 8);

    // Non-positive / garbage → reject, out restored to defaults.
    CHECK(!resolve_moe_topk_wrap(8, 256, "0", "16", out));
    CHECK(out.k1 == 8);
    CHECK(out.k2 == 8);
    CHECK(!resolve_moe_topk_wrap(8, 256, "4", "-1", out));
    CHECK(out.k1 == 8);
    CHECK(out.k2 == 8);
    CHECK(!resolve_moe_topk_wrap(8, 256, "nope", "16", out));
    CHECK(out.k1 == 8);
    CHECK(out.k2 == 8);

    // Over n_experts → clamp into range.
    CHECK(resolve_moe_topk_wrap(8, 16, "20", "30", out));
    CHECK(out.k1 == 16);
    CHECK(out.k2 == 16);

    // Empty strings count as unset for that side.
    CHECK(resolve_moe_topk_wrap(8, 256, "", "", out));
    CHECK(out.k1 == 8);
    CHECK(out.k2 == 8);
}
