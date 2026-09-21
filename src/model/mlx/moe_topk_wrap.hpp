#ifndef LLM_MLX_MOE_TOPK_WRAP_HPP
#define LLM_MLX_MOE_TOPK_WRAP_HPP

// Training-free activated-expert wrapping (G1): k1 experts run (matmul), k2 scores
// enter renormalization, with k2 >= k1. See arXiv 2609.04575.
//
// Kill switch (default OFF — product path stays cfg.num_experts_per_tok, usually 8):
//   LMP_MOE_K1  — activated expert count
//   LMP_MOE_K2  — renormalization expert count
// Unset both, or resolve to k1==k2==default_k, → identical to today's single-k path.
//
// This header is CPU-only so gate CI (LMP_WITH_MLX=OFF) can lock the routing algebra
// without Metal. The MLX moe_topk overloads in switch_glu.hpp mirror the same rules.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lmp::model::mlxl {

struct MoeTopkPair {
    int k1{0};
    int k2{0};
};

struct MoeTopkResult {
    std::vector<int> indices;
    std::vector<float> scores;
};

namespace detail {

[[nodiscard]] inline std::optional<int> parse_positive_int(const char* s) noexcept {
    if (s == nullptr || s[0] == '\0') {
        return std::nullopt;
    }
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || (end != nullptr && *end != '\0') || v < 1 || v > 1000000L) {
        return std::nullopt;
    }
    return static_cast<int>(v);
}

// Apply k2 >= k1 and 1..n_experts clamps. Returns false → caller keeps defaults.
[[nodiscard]] inline bool finalize_moe_topk_pair(int default_k,
                                                 int n_experts,
                                                 int k1,
                                                 int k2,
                                                 MoeTopkPair& out) noexcept {
    if (default_k < 1 || n_experts < 1) {
        return false;
    }
    if (k1 < 1 || k2 < 1) {
        return false;
    }
    const int max_k = n_experts;
    if (k1 > max_k) {
        k1 = max_k;
    }
    if (k2 > max_k) {
        k2 = max_k;
    }
    if (k2 < k1) {
        k2 = k1;
    }
    out.k1 = k1;
    out.k2 = k2;
    return out.k1 >= 1 && out.k2 >= out.k1;
}

} // namespace detail

// Resolve optional env (or test-injected) strings against the product default_k.
// Both nullptr/empty → (default_k, default_k) and returns true (kill switch off).
// Malformed / non-positive → false and out = (default_k, default_k).
[[nodiscard]] inline bool resolve_moe_topk_wrap(int default_k,
                                                int n_experts,
                                                const char* k1_cstr,
                                                const char* k2_cstr,
                                                MoeTopkPair& out) noexcept {
    out = MoeTopkPair{default_k, default_k};
    const bool k1_set = k1_cstr != nullptr && k1_cstr[0] != '\0';
    const bool k2_set = k2_cstr != nullptr && k2_cstr[0] != '\0';
    if (!k1_set && !k2_set) {
        return true;
    }
    // One set without the other: treat the missing side as default_k (not an error),
    // then finalize/clamp. Prove-it arms set both explicitly (e.g. 4/16).
    const auto p1 = k1_set ? detail::parse_positive_int(k1_cstr) : std::optional<int>{default_k};
    const auto p2 = k2_set ? detail::parse_positive_int(k2_cstr) : std::optional<int>{default_k};
    if (!p1.has_value() || !p2.has_value()) {
        return false;
    }
    if (!detail::finalize_moe_topk_pair(default_k, n_experts, *p1, *p2, out)) {
        out = MoeTopkPair{default_k, default_k};
        return false;
    }
    return true;
}

// Cached getenv read. Parsed against cfg at the call site (default_k / n_experts vary).
struct MoeTopkEnvRaw {
    bool any_set{false};
    std::optional<std::string> k1;
    std::optional<std::string> k2;
};

[[nodiscard]] inline const MoeTopkEnvRaw& moe_topk_env_raw() {
    // Read once per process. A getenv per MoE layer would itself be a per-step cost.
    static const MoeTopkEnvRaw cached = [] {
        MoeTopkEnvRaw r;
        const char* a = std::getenv("LMP_MOE_K1");
        const char* b = std::getenv("LMP_MOE_K2");
        if ((a == nullptr || a[0] == '\0') && (b == nullptr || b[0] == '\0')) {
            return r;
        }
        r.any_set = true;
        if (a != nullptr && a[0] != '\0') {
            r.k1 = std::string(a);
        }
        if (b != nullptr && b[0] != '\0') {
            r.k2 = std::string(b);
        }
        return r;
    }();
    return cached;
}

[[nodiscard]] inline MoeTopkPair moe_topk_wrap_from_env(int default_k, int n_experts) {
    MoeTopkPair out{default_k, default_k};
    const MoeTopkEnvRaw& env = moe_topk_env_raw();
    if (!env.any_set) {
        return out;
    }
    const char* a = env.k1 ? env.k1->c_str() : nullptr;
    const char* b = env.k2 ? env.k2->c_str() : nullptr;
    (void)resolve_moe_topk_wrap(default_k, n_experts, a, b, out);
    return out;
}

// Pure-CPU reference of the wrap rule on one router logit row.
// When k1==k2 this matches today's single-k renormalize-over-top-k algebra.
[[nodiscard]] inline MoeTopkResult moe_topk_cpu(const float* logits,
                                                int n_experts,
                                                int k1,
                                                int k2,
                                                bool norm_topk) {
    MoeTopkResult r;
    if (logits == nullptr || n_experts < 1 || k1 < 1 || k2 < 1) {
        return r;
    }
    if (k1 > n_experts) {
        k1 = n_experts;
    }
    if (k2 > n_experts) {
        k2 = n_experts;
    }
    if (k2 < k1) {
        k2 = k1;
    }

    // precise softmax in float32 (mlx-lm SparseMoeBlock / moe_topk use precise=true).
    float max_logit = logits[0];
    for (int i = 1; i < n_experts; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    std::vector<float> probs(static_cast<std::size_t>(n_experts));
    double sum = 0.0;
    for (int i = 0; i < n_experts; ++i) {
        const float e = std::exp(logits[i] - max_logit);
        probs[static_cast<std::size_t>(i)] = e;
        sum += static_cast<double>(e);
    }
    const float inv = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
    for (float& p : probs) {
        p *= inv;
    }

    std::vector<int> order(static_cast<std::size_t>(n_experts));
    for (int i = 0; i < n_experts; ++i) {
        order[static_cast<std::size_t>(i)] = i;
    }
    // Unsorted top-k2 (same contract as argpartition + slice of the largest k2).
    std::nth_element(order.begin(), order.end() - k2, order.end(),
                     [&](int a, int b) { return probs[static_cast<std::size_t>(a)] <
                                               probs[static_cast<std::size_t>(b)]; });
    std::vector<int> top2(order.end() - k2, order.end());
    std::vector<float> scores2(static_cast<std::size_t>(k2));
    for (int i = 0; i < k2; ++i) {
        scores2[static_cast<std::size_t>(i)] =
            probs[static_cast<std::size_t>(top2[static_cast<std::size_t>(i)])];
    }
    if (norm_topk) {
        float s = 0.0f;
        for (float v : scores2) {
            s += v;
        }
        if (s > 0.0f) {
            const float scale = 1.0f / s;
            for (float& v : scores2) {
                v *= scale;
            }
        }
    }

    // Activated set = top-k1 among the k2 by (possibly k2-normalized) score.
    std::vector<int> local(static_cast<std::size_t>(k2));
    for (int i = 0; i < k2; ++i) {
        local[static_cast<std::size_t>(i)] = i;
    }
    std::nth_element(local.begin(), local.end() - k1, local.end(),
                     [&](int a, int b) {
                         return scores2[static_cast<std::size_t>(a)] <
                                scores2[static_cast<std::size_t>(b)];
                     });
    r.indices.resize(static_cast<std::size_t>(k1));
    r.scores.resize(static_cast<std::size_t>(k1));
    for (int i = 0; i < k1; ++i) {
        const int li = local[static_cast<std::size_t>(k2 - k1 + i)];
        r.indices[static_cast<std::size_t>(i)] = top2[static_cast<std::size_t>(li)];
        r.scores[static_cast<std::size_t>(i)] = scores2[static_cast<std::size_t>(li)];
    }
    return r;
}

} // namespace lmp::model::mlxl

#endif // LLM_MLX_MOE_TOPK_WRAP_HPP
