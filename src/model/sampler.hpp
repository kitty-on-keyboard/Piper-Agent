#pragma once
//
// Sampler -- pure function over logits (spec S5.9, S5.11).
//
// Pure and CPU-side so it is testable to the exact token without a GPU. The MLX backend
// hands it one row of logits; everything after that -- penalty, mask, temperature,
// top-k, top-p, min-p, the draw -- happens here, deterministically under a seeded RNG.
//
// The grammar mask is applied FIRST, before any renormalisation, because an id the
// grammar forbids must have probability zero regardless of how the distribution is
// shaped afterwards (S5.6). Masking after top-k could leave zero legal candidates.
//
#include <cstdint>
#include <vector>

#include "src/model/backend.hpp"
#include "src/model/qwen_tokenizer.hpp"
#include "src/model/token_mask.hpp"

namespace lmp::model {

struct SampleResult {
    TokenId id = kInvalidToken;
    // True when every logit was masked out. The caller treats this as a hard error:
    // it means the grammar and the vocabulary disagree, which is a build defect, not a
    // sampling accident.
    bool no_legal_token = false;
};

// The step's distribution AFTER everything that shapes it: repetition penalty, the
// grammar mask, temperature, top-k, min-p, top-p. Small -- top-k survivors, not 248,320
// entries -- and in ascending id order.
//
// Exposed because speculative decoding has to VERIFY against this row rather than against
// raw softmax. Verifying against the unshaped distribution would make the committed tokens
// follow neither law, and -- because the mask is applied here -- would let speculation
// commit a token the grammar forbids (S5.6 makes that a build defect, not a stylistic
// preference). See speculative.hpp.
struct TokenDist {
    std::vector<TokenId> ids;  // ascending
    std::vector<float> probs;  // parallel to ids, UNNORMALISED
    float total = 0.0F;

    [[nodiscard]] bool empty() const noexcept { return total <= 0.0F || ids.empty(); }
    // Normalised probability of `id`, or 0 when it did not survive shaping -- which
    // includes every token the grammar masked out.
    [[nodiscard]] float prob_of(TokenId id) const noexcept;
};

class Sampler {
  public:
    explicit Sampler(const SamplingParams& params) : params_(params), rng_(params.seed) {}

    // `mask` is the legal-token set for this step; pass nullptr for unconstrained. Ids
    // at or beyond mask->size() are denied -- the logits row is wider than the
    // tokenizer's vocabulary and the trailing rows decode to nothing.
    // `recent` feeds repetition penalty (applied to ids present in it).
    [[nodiscard]] SampleResult sample(std::vector<float>& logits, const TokenMask* mask,
                                      const std::vector<TokenId>& recent);

    // sample(), split at the point where the distribution is known. Consumes no
    // randomness, so a caller may build a row without perturbing the sampling stream.
    // `logits` is modified in place, exactly as sample() modifies it.
    [[nodiscard]] TokenDist distribution(std::vector<float>& logits, const TokenMask* mask,
                                         const std::vector<TokenId>& recent) const;
    // The draw half. sample() is exactly draw(distribution(...)).
    [[nodiscard]] SampleResult draw(const TokenDist& dist);

  private:
    [[nodiscard]] std::uint64_t next_u64() noexcept;

    SamplingParams params_;
    std::uint64_t rng_;
};

} // namespace lmp::model
