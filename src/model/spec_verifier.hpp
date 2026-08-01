#pragma once
//
// SpecVerifier -- the acceptance rule for speculative decoding.
//
// ADOPTED from the Brief C cook-off (bakeoff/spec_verifier/), entrant e3, which was the
// most compact of five implementations that all scored identically at the sampling-noise
// floor. The procedure is unchanged; only the namespace and the row type differ.
//
// What it is for: keep the drafted tokens the model agrees with, so the committed output
// is a sample from the model's own distribution rather than one biased toward confident
// tokens. Accepting a draft token because it happens to be the argmax is the failure this
// exists to prevent, and it is invisible to every other test -- the falsifier that does it
// scores 0.86 total-variation from the target while passing the floor, determinism,
// perfect-drafter and degenerate cases.
//
// TWO THINGS THIS PROJECT DOES DIFFERENTLY FROM THE BRIEF, both load-bearing:
//
// 1. `draft_probs` is always 1.0 here. The brief's residual reduction -- subtract the
//    scalar q from the rejected token's mass, clamp, renormalise -- is NOT
//    distribution-preserving in general (worst total-variation 0.23, by exact enumeration
//    over 20,000 random (p,q) pairs). It becomes exact precisely when the drafter is
//    deterministic, q = 1. SuffixProposer proposes a concrete continuation from matched
//    history; it is not a sampling model and has no calibrated probability to offer. So
//    acceptance collapses to `u < p(t)` and the procedure is exact. Feeding a confidence
//    score into that slot instead buys measurable bias in exchange for nothing.
//
// 2. The rows are POST-SAMPLER distributions over a small candidate set, not raw softmax
//    over the 248,320-wide vocabulary. See speculative.hpp: verifying against raw softmax
//    while the loop samples from the masked/tempered/top-p distribution means the
//    committed tokens follow neither law, and lets speculation commit a token the grammar
//    forbids.
//
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "src/model/qwen_tokenizer.hpp"

namespace lmp::model {

struct SpecResult {
    // Tokens to commit, in order. Always contains at least one: even when every draft
    // token is rejected the verification pass yields one fresh token, which is where
    // speculative decoding's guaranteed floor comes from.
    std::vector<TokenId> accepted;
    // How many DRAFT tokens survived. accepted.size() == accepted_drafts + 1.
    std::size_t accepted_drafts = 0;
    // True when the final token came from the corrected distribution after a rejection
    // rather than from the model's own next-position row.
    bool ended_on_rejection = false;
};

class SpecVerifier {
  public:
    explicit SpecVerifier(std::uint64_t seed) : rng_(seed) {}

    // `draft` holds k proposed tokens, `draft_probs[i]` the drafter's probability for
    // draft[i], and `target_rows` the model's k+1 rows (one per drafted position, plus
    // the position after the last). Rows need not sum to 1; they are renormalised.
    [[nodiscard]] SpecResult verify(std::span<const TokenId> draft,
                                    std::span<const float> draft_probs,
                                    std::span<const std::span<const float>> target_rows);

  private:
    std::uint64_t rng_;
};

} // namespace lmp::model
