// Planted defect: the ACCEPTANCE test is correct (u < min(1, p/q)), but on rejection the
// replacement is drawn from target_rows[i] directly instead of from the residual. This is
// the subtler of the two mistakes and the one a code review is most likely to wave through,
// because the line it is missing is a correction that looks like a normalisation detail.
#include <spec_verifier.hpp>
#include <algorithm>
namespace spec {
namespace {
std::uint64_t next(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
float uni(std::uint64_t& s) { return static_cast<float>((next(s) >> 11) * 0x1.0p-53); }
TokenId sample(std::span<const float> row, std::uint64_t& s) {
    float u = uni(s), acc = 0.0F;
    for (std::size_t i = 0; i < row.size(); ++i) { acc += row[i]; if (u < acc) return static_cast<TokenId>(i); }
    return static_cast<TokenId>(row.size() - 1);
}
} // namespace
SpecVerifier::SpecVerifier(std::uint64_t seed) : rng_(seed) {}
Result SpecVerifier::verify(std::span<const TokenId> draft, std::span<const float> draft_probs,
                            std::span<const std::span<const float>> target_rows) {
    Result r;
    std::size_t i = 0;
    for (; i < draft.size(); ++i) {
        const std::span<const float> row = target_rows[i];
        const auto t = static_cast<std::size_t>(draft[i]);
        const float p = t < row.size() ? row[t] : 0.0F;
        const float q = draft_probs[i];
        const float thresh = (q <= 0.0F) ? 1.0F : std::min(1.0F, p / q);
        if (uni(rng_) >= thresh) break;
        r.accepted.push_back(draft[i]);
        ++r.accepted_drafts;
    }
    if (i < draft.size()) {
        r.accepted.push_back(sample(target_rows[i], rng_));   // <-- should be the residual
        r.ended_on_rejection = true;
    } else {
        r.accepted.push_back(sample(target_rows[draft.size()], rng_));
    }
    return r;
}
} // namespace spec
