#include "src/model/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lmp::model {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

void apply_repetition_penalty(std::vector<float>& logits, const std::vector<TokenId>& recent,
                              float penalty) {
    if (penalty == 1.0F) {
        return;
    }
    for (TokenId id : recent) {
        if (id < 0 || static_cast<std::size_t>(id) >= logits.size()) {
            continue;
        }
        float& l = logits[static_cast<std::size_t>(id)];
        l = l > 0 ? l / penalty : l * penalty;
    }
}

// Keeps the k highest logits, -inf elsewhere.
void apply_top_k(std::vector<float>& logits, std::int32_t k) {
    if (k <= 0 || static_cast<std::size_t>(k) >= logits.size()) {
        return;
    }
    std::vector<float> copy = logits;
    std::nth_element(copy.begin(), copy.begin() + (k - 1), copy.end(), std::greater<>());
    const float cutoff = copy[static_cast<std::size_t>(k - 1)];
    std::int32_t kept = 0;
    for (float& l : logits) {
        if (l > cutoff) {
            ++kept;
        }
    }
    // Ties at the cutoff are kept left-to-right until k is full, so the kept count is
    // exactly k rather than "k plus however many tied".
    for (float& l : logits) {
        if (l < cutoff) {
            l = kNegInf;
        } else if (l == cutoff) {
            l = (kept < k) ? (++kept, l) : kNegInf;
        }
    }
}

struct Dist {
    std::vector<float> probs;   // over all ids; zero where excluded
    float total = 0.0F;
};

Dist softmax(const std::vector<float>& logits, float temperature) {
    Dist d;
    d.probs.assign(logits.size(), 0.0F);
    const float t = temperature <= 0.0F ? 1.0F : temperature;
    float max_logit = kNegInf;
    for (float l : logits) {
        max_logit = std::max(max_logit, l);
    }
    if (max_logit == kNegInf) {
        return d; // everything masked
    }
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (logits[i] != kNegInf) {
            d.probs[i] = std::exp((logits[i] - max_logit) / t);
            d.total += d.probs[i];
        }
    }
    return d;
}

void apply_min_p(Dist& d, float min_p) {
    if (min_p <= 0.0F || d.total <= 0.0F) {
        return;
    }
    float peak = 0.0F;
    for (float p : d.probs) {
        peak = std::max(peak, p);
    }
    const float floor = peak * min_p;
    for (float& p : d.probs) {
        if (p < floor) {
            d.total -= p;
            p = 0.0F;
        }
    }
}

void apply_top_p(Dist& d, float top_p) {
    if (top_p >= 1.0F || d.total <= 0.0F) {
        return;
    }
    std::vector<std::size_t> order(d.probs.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(),
              [&d](std::size_t a, std::size_t b) { return d.probs[a] > d.probs[b]; });
    float cum = 0.0F;
    bool cut = false;
    for (std::size_t idx : order) {
        if (cut) {
            d.total -= d.probs[idx];
            d.probs[idx] = 0.0F;
            continue;
        }
        cum += d.probs[idx] / d.total;
        // The nucleus includes the prob that crosses the threshold, per the paper.
        cut = cum >= top_p;
    }
    d.total = 0.0F;
    for (float p : d.probs) {
        d.total += p;
    }
}

} // namespace

std::uint64_t Sampler::next_u64() noexcept {
    // splitmix64: tiny, seedable, and plenty for sampling. Determinism under a fixed
    // seed is what the tests and the replay path rely on.
    rng_ += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = rng_;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

SampleResult Sampler::sample(std::vector<float>& logits,
                             const std::function<bool(TokenId)>& mask,
                             const std::vector<TokenId>& recent) {
    // Mask first (see header).
    if (mask) {
        for (std::size_t i = 0; i < logits.size(); ++i) {
            if (!mask(static_cast<TokenId>(i))) {
                logits[i] = kNegInf;
            }
        }
    }
    apply_repetition_penalty(logits, recent, params_.repetition_penalty);
    apply_top_k(logits, params_.top_k);

    Dist d = softmax(logits, params_.temperature);
    apply_min_p(d, params_.min_p);
    apply_top_p(d, params_.top_p);

    if (d.total <= 0.0F) {
        return {kInvalidToken, true};
    }

    const double u = static_cast<double>(next_u64() >> 11U) * 0x1.0p-53;
    double target = u * static_cast<double>(d.total);
    for (std::size_t i = 0; i < d.probs.size(); ++i) {
        target -= static_cast<double>(d.probs[i]);
        if (target <= 0.0 && d.probs[i] > 0.0F) {
            return {static_cast<TokenId>(i), false};
        }
    }
    // Rounding fell off the end; return the last nonzero.
    for (std::size_t i = d.probs.size(); i > 0; --i) {
        if (d.probs[i - 1] > 0.0F) {
            return {static_cast<TokenId>(i - 1), false};
        }
    }
    return {kInvalidToken, true};
}

} // namespace lmp::model
