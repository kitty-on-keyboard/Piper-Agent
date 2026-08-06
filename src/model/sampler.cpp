#include "src/model/sampler.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace lmp::model {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// One candidate that survived top-k. Everything after selection works on this list --
// at k=20 that is twenty entries instead of five full passes over 248,320 floats.
struct Cand {
    std::size_t id = 0;
    float logit = kNegInf;
};

// Word-at-a-time: where the mask is dense (Think and Text allow all but eight ids) a
// fully-allowed word is one compare for sixty-four tokens.
void apply_mask(std::vector<float>& logits, const TokenMask* mask) {
    if (mask == nullptr) {
        return;
    }
    const std::vector<std::uint64_t>& words = mask->words();
    for (std::size_t wi = 0; wi < words.size(); ++wi) {
        std::uint64_t denied = ~words[wi];
        while (denied != 0) {
            const auto bit = static_cast<std::size_t>(std::countr_zero(denied));
            denied &= denied - 1;
            const std::size_t id = (wi << 6U) + bit;
            if (id < logits.size()) {
                logits[id] = kNegInf;
            }
        }
    }
    // The logits row is wider than the vocabulary the mask covers (248,320 vs 248,077
    // on this checkpoint). Those rows decode to nothing, so they are not emittable.
    for (std::size_t i = mask->size(); i < logits.size(); ++i) {
        logits[i] = kNegInf;
    }
}

// Once per UNIQUE id, never per occurrence. `recent` is a sliding window that keeps
// duplicates, and generated code repeats identifier subtokens densely -- compounding the
// division (penalty^n) sank a subtoken the model was copying below its space-prefixed
// twin after ~8 repeats, which wrote "idlePer cent" into a real agent's file. The pinned
// Qwen defaults (1.05) assume the HF semantics, which are per-id.
void apply_repetition_penalty(std::vector<float>& logits, const std::vector<TokenId>& recent,
                              float penalty) {
    if (penalty == 1.0F) {
        return;
    }
    std::vector<TokenId> unique_ids(recent);
    std::sort(unique_ids.begin(), unique_ids.end());
    unique_ids.erase(std::unique(unique_ids.begin(), unique_ids.end()), unique_ids.end());
    for (TokenId id : unique_ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= logits.size()) {
            continue;
        }
        float& l = logits[static_cast<std::size_t>(id)];
        l = l > 0 ? l / penalty : l * penalty;
    }
}

// The k highest finite logits, returned in ascending id order. A masked-out or
// otherwise -inf logit is never a candidate: softmax gave it zero weight anyway, so
// selecting over the finite entries only is the same distribution.
std::vector<Cand> select_top_k(const std::vector<float>& logits, std::int32_t k) {
    const bool bounded = k > 0 && static_cast<std::size_t>(k) < logits.size();
    const auto cmp = [](const Cand& a, const Cand& b) { return a.logit > b.logit; };
    std::vector<Cand> heap;
    if (bounded) {
        heap.reserve(static_cast<std::size_t>(k) + 1);
    }
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (logits[i] == kNegInf) {
            continue;
        }
        if (!bounded) {
            heap.push_back({i, logits[i]});
            continue;
        }
        // Strict >: on a tie the id encountered first keeps the slot, matching the
        // left-to-right tie-break the previous full-sort selection used.
        if (heap.size() == static_cast<std::size_t>(k) && !(logits[i] > heap.front().logit)) {
            continue;
        }
        heap.push_back({i, logits[i]});
        std::push_heap(heap.begin(), heap.end(), cmp);
        if (heap.size() > static_cast<std::size_t>(k)) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            heap.pop_back();
        }
    }
    std::sort(heap.begin(), heap.end(),
              [](const Cand& a, const Cand& b) { return a.id < b.id; });
    return heap;
}

struct Dist {
    std::vector<float> probs; // parallel to the candidate list
    float total = 0.0F;
};

Dist softmax(const std::vector<Cand>& cands, float temperature) {
    Dist d;
    d.probs.assign(cands.size(), 0.0F);
    float max_logit = kNegInf;
    std::size_t max_index = 0;
    for (std::size_t i = 0; i < cands.size(); ++i) {
        if (cands[i].logit > max_logit) {
            max_logit = cands[i].logit;
            max_index = i;
        }
    }
    if (max_logit == kNegInf) {
        return d; // everything masked
    }
    // Temperature zero means greedy decoding, not "sample at temperature one". Besides
    // matching the conventional API contract, this is what makes evaluator smoke runs
    // deterministic independently of seed. Cands are in ascending id order and strict >
    // above keeps the lowest id on an exact tie.
    if (temperature <= 0.0F) {
        d.probs[max_index] = 1.0F;
        d.total = 1.0F;
        return d;
    }
    for (std::size_t i = 0; i < cands.size(); ++i) {
        d.probs[i] = std::exp((cands[i].logit - max_logit) / temperature);
        d.total += d.probs[i];
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
    // Ordering the survivors, not the vocabulary: top-k already ran, so this sorts
    // twenty entries rather than std::sort over 248,320 indices per token.
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

float TokenDist::prob_of(TokenId id) const noexcept {
    if (total <= 0.0F) {
        return 0.0F;
    }
    const auto it = std::lower_bound(ids.begin(), ids.end(), id);
    if (it == ids.end() || *it != id) {
        return 0.0F;
    }
    return probs[static_cast<std::size_t>(it - ids.begin())] / total;
}

TokenDist Sampler::distribution(std::vector<float>& logits, const TokenMask* mask,
                                const std::vector<TokenId>& recent) const {
    // Mask first (see header): an id the grammar forbids must have probability zero
    // regardless of how the distribution is shaped afterwards.
    apply_mask(logits, mask);
    apply_repetition_penalty(logits, recent, params_.repetition_penalty);

    const std::vector<Cand> cands = select_top_k(logits, params_.top_k);
    Dist d = softmax(cands, params_.temperature);
    apply_min_p(d, params_.min_p);
    apply_top_p(d, params_.top_p);

    TokenDist out;
    out.ids.reserve(cands.size());
    out.probs.reserve(cands.size());
    for (std::size_t i = 0; i < cands.size(); ++i) {
        out.ids.push_back(static_cast<TokenId>(cands[i].id));
        out.probs.push_back(d.probs[i]);
    }
    out.total = d.total;
    return out;
}

SampleResult Sampler::draw(const TokenDist& dist) {
    if (dist.total <= 0.0F) {
        return {kInvalidToken, true};
    }
    const double u = static_cast<double>(next_u64() >> 11U) * 0x1.0p-53;
    double target = u * static_cast<double>(dist.total);
    for (std::size_t i = 0; i < dist.probs.size(); ++i) {
        target -= static_cast<double>(dist.probs[i]);
        if (target <= 0.0 && dist.probs[i] > 0.0F) {
            return {dist.ids[i], false};
        }
    }
    // Rounding fell off the end; return the last nonzero.
    for (std::size_t i = dist.probs.size(); i > 0; --i) {
        if (dist.probs[i - 1] > 0.0F) {
            return {dist.ids[i - 1], false};
        }
    }
    return {kInvalidToken, true};
}

SampleResult Sampler::sample(std::vector<float>& logits, const TokenMask* mask,
                             const std::vector<TokenId>& recent) {
    // Split into distribution() + draw() so speculative decoding can verify against the
    // same shaped row this path samples from. The order of operations, and the point at
    // which randomness is consumed, are unchanged.
    return draw(distribution(logits, mask, recent));
}

} // namespace lmp::model
