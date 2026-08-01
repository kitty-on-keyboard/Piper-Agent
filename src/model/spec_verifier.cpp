#include "src/model/spec_verifier.hpp"

#include <algorithm>
#include <cmath>

namespace lmp::model {
namespace {

std::uint64_t next_u64(std::uint64_t& state) noexcept {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

double next_uniform(std::uint64_t& state) noexcept {
    return static_cast<double>(next_u64(state) >> 11U) * 0x1.0p-53;
}

// Double accumulation, not float. The cook-off entrants all summed in float, which was
// fine at the brief's vocabulary sizes; the candidate rows here are small too, so this
// costs nothing and removes the question.
TokenId sample_row(std::span<const float> probs, std::uint64_t& state) noexcept {
    double sum = 0.0;
    for (float v : probs) {
        sum += static_cast<double>(v);
    }
    if (sum <= 0.0) {
        return 0;
    }
    double target = next_uniform(state) * sum;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        target -= static_cast<double>(probs[i]);
        if (target <= 0.0 && probs[i] > 0.0F) {
            return static_cast<TokenId>(i);
        }
    }
    for (std::size_t i = probs.size(); i > 0; --i) {
        if (probs[i - 1] > 0.0F) {
            return static_cast<TokenId>(i - 1);
        }
    }
    return 0;
}

} // namespace

SpecResult SpecVerifier::verify(std::span<const TokenId> draft,
                                std::span<const float> draft_probs,
                                std::span<const std::span<const float>> target_rows) {
    SpecResult res;
    if (target_rows.empty()) {
        return res;
    }
    const std::size_t k = std::min(draft.size(), target_rows.size() - 1);

    for (std::size_t i = 0; i < k; ++i) {
        const std::span<const float> row = target_rows[i];
        double row_sum = 0.0;
        for (float v : row) {
            row_sum += static_cast<double>(v);
        }
        const auto t = static_cast<std::size_t>(draft[i]);
        const double p = (draft[i] >= 0 && t < row.size() && row_sum > 0.0)
                             ? static_cast<double>(row[t]) / row_sum
                             : 0.0;
        const double q = i < draft_probs.size() ? static_cast<double>(draft_probs[i]) : 1.0;

        // accept with probability min(1, p/q); q == 0 cannot be divided by, and a drafter
        // that assigned zero mass to what it proposed has told us nothing, so the target's
        // own opinion decides.
        const double accept = (q > 0.0) ? std::min(1.0, p / q) : (p > 0.0 ? 1.0 : 0.0);

        if (next_uniform(rng_) < accept) {
            res.accepted.push_back(draft[i]);
            ++res.accepted_drafts;
            continue;
        }

        // Rejected. Sample the replacement from the residual: the target row with the
        // rejected token's mass reduced by q, clamped at zero, renormalised. With q == 1
        // (this project's regime) that is exactly "the row with that token removed", and
        // the whole procedure is exactly distribution-preserving.
        std::vector<float> residual(row.begin(), row.end());
        if (draft[i] >= 0 && t < residual.size()) {
            residual[t] = static_cast<float>(
                std::max(0.0, static_cast<double>(residual[t]) - q * row_sum));
        }
        double resid_sum = 0.0;
        for (float v : residual) {
            resid_sum += static_cast<double>(v);
        }
        if (resid_sum <= 0.0) {
            // The whole row was the rejected token. Fall back to the row itself rather
            // than returning nothing -- the floor (accepted is never empty) is what stops
            // a decode loop from hanging.
            residual.assign(row.begin(), row.end());
        }
        res.accepted.push_back(sample_row(residual, rng_));
        res.ended_on_rejection = true;
        return res;
    }

    res.accepted.push_back(sample_row(target_rows[k], rng_));
    return res;
}

} // namespace lmp::model
