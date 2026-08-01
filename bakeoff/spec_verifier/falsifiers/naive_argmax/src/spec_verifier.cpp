// Planted defect: "accept the draft token if it is the target's top choice". This is the
// exact failure Brief C says the task exists to catch -- it biases the output toward
// confident tokens while passing every structural test.
//
// Note which drafter exposes it. With a BAD drafter this never accepts, falls through to
// sampling target_rows[0], and the first-token histogram converges to the target perfectly:
// tv_det_bad comes out clean. It is tv_det_GOOD that goes red. The brief asks for the
// bad-drafter histogram first; on its own that test would pass this implementation.
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
    (void)draft_probs;
    std::size_t i = 0;
    for (; i < draft.size(); ++i) {
        const std::span<const float> row = target_rows[i];
        const auto best = static_cast<TokenId>(std::max_element(row.begin(), row.end()) - row.begin());
        if (draft[i] != best) break;                       // <-- the defect
        r.accepted.push_back(draft[i]);
        ++r.accepted_drafts;
    }
    if (i < draft.size()) { r.accepted.push_back(sample(target_rows[i], rng_)); r.ended_on_rejection = true; }
    else { r.accepted.push_back(sample(target_rows[draft.size()], rng_)); }
    return r;
}
} // namespace spec
