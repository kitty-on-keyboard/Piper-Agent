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

class Sampler {
  public:
    explicit Sampler(const SamplingParams& params) : params_(params), rng_(params.seed) {}

    // `mask` is the legal-token set for this step; pass nullptr for unconstrained. Ids
    // at or beyond mask->size() are denied -- the logits row is wider than the
    // tokenizer's vocabulary and the trailing rows decode to nothing.
    // `recent` feeds repetition penalty (applied to ids present in it).
    [[nodiscard]] SampleResult sample(std::vector<float>& logits, const TokenMask* mask,
                                      const std::vector<TokenId>& recent);

  private:
    [[nodiscard]] std::uint64_t next_u64() noexcept;

    SamplingParams params_;
    std::uint64_t rng_;
};

} // namespace lmp::model
