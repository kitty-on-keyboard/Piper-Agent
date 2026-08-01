#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace spec {

using TokenId = std::int32_t;

struct Result {
    // Tokens to commit, in order. Always contains at least one token: even when every
    // draft token is rejected, the verification pass yields one fresh token, which is
    // where speculative decoding's guaranteed floor comes from.
    std::vector<TokenId> accepted;
    // How many DRAFT tokens survived. accepted.size() == accepted_drafts + 1 normally.
    std::size_t accepted_drafts = 0;
    // True when the final token came from the corrected distribution after a rejection,
    // rather than from the model's own next-position row.
    bool ended_on_rejection = false;
};

class SpecVerifier {
  public:
    // `seed` fixes the RNG. Two verifiers with the same seed and inputs agree exactly.
    explicit SpecVerifier(std::uint64_t seed);

    // `draft` holds k proposed tokens.
    // `draft_probs[i]` is the DRAFTER's probability for draft[i] (one scalar per position).
    // `target_rows[i]` is the TARGET model's full probability row for position i --
    //   target_rows.size() == draft.size() + 1, because verification also produces the
    //   distribution for the position after the last draft token.
    // Every row is a proper distribution: non-negative, sums to ~1.
    [[nodiscard]] Result verify(std::span<const TokenId> draft,
                                std::span<const float> draft_probs,
                                std::span<const std::span<const float>> target_rows);

  private:
    std::uint64_t rng_;
};

} // namespace spec
