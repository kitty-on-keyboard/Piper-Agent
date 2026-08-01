#pragma once
//
// Speculative decoding: propose k tokens cheaply, verify them all in one forward pass,
// keep the ones the model agrees with.
//
// Deliberately model-free. Everything MLX-shaped is behind SpecForward, so the block
// algebra -- which positions are committed, what the cache must be rolled back to, which
// row feeds the next step -- is testable in the gate with no GPU and no 19 GB checkpoint.
// That is where an off-by-one in speculative decoding actually hides: it does not crash,
// it silently shifts the distribution, and the text stays fluent.
//
// ------------------------------------------------------------------------------------
// THE CONSTRAINTS, all measured, none negotiable (docs/MOE_ROUTING_FINDINGS.md):
//
//   * DRAFT LENGTH MUST BE ADAPTIVE. Fixed-length drafting is a NET LOSS on this MoE:
//     0.678x at k=4, 0.454x at k=8, because verifying k tokens drags in the union of
//     their experts and expert scattering eats the entire speculative gain. Only very
//     short drafts and cumulative-probability stopping win. SuffixProposer's stop rule
//     (extend while the running probability the draft is still correct exceeds
//     draft_cost_ratio) is exactly that rule, so the draft length is its output, never a
//     constant here.
//
//   * NO EXPERT-AWARE DRAFTING. Tested three ways on a 16,914-step trace; the best-looking
//     variant turned out byte-identical to fixed-length-1. Predicting routing is real and
//     still not actionable. Do not add it back.
//
// ------------------------------------------------------------------------------------
// WHY SPECULATION IS GATED ON THE GRAMMAR PHASE.
//
// Verification needs the target distribution at every drafted position, and those
// distributions are masked -- an id the grammar forbids has probability zero (S5.6). But
// TurnGrammar cannot roll back and cannot be copied, so the mask for position n+i, which
// depends on tokens the model has not committed yet, is not directly available.
//
// The way out is that in the Think and Text phases the mask is a cached per-phase bitset:
// "outside a tool call the legal set is everything except a handful of structural ids".
// It depends on the PHASE, not on the token history. So as long as no drafted token can
// change the phase, every position in the block shares one mask, and the grammar advance
// over the committed tokens is exactly what it would have been.
//
// Hence: speculate only in Think/Text, and truncate the draft before the first special
// token (specials are what change phase). Inside a tool call -- where parsephony's mask is
// state-dependent per token -- fall back to ordinary decoding. That is not much of a loss:
// tool-call bodies are short and the bulk of a turn is Think and Text.
//
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "src/model/backend.hpp"
#include "src/model/qwen_tokenizer.hpp"
#include "src/model/sampler.hpp"
#include "src/model/spec_verifier.hpp"
#include "src/model/token_mask.hpp"

namespace draft {
class SuffixProposer;
}

namespace lmp::model {

struct SpecConfig {
    // Off by default. Speculation is a throughput bet whose payoff depends on the
    // workload repeating itself; the plain path stays the reference.
    bool enabled = false;
    // Upper bound only -- the proposer decides the actual length, and the measurements
    // above say a constant would ship a regression.
    std::size_t max_draft = 8;
};

struct SpecStats {
    std::uint64_t blocks = 0;          // speculative blocks attempted
    std::uint64_t drafted = 0;         // tokens proposed
    std::uint64_t accepted_drafts = 0; // proposed tokens that survived verification
    std::uint64_t committed = 0;       // tokens emitted by speculative blocks
    std::uint64_t fallbacks = 0;       // steps that decoded normally instead

    [[nodiscard]] double acceptance_rate() const noexcept {
        return drafted > 0 ? static_cast<double>(accepted_drafts) / static_cast<double>(drafted)
                           : 0.0;
    }
};

// The model, as speculative decoding needs to see it.
class SpecForward {
  public:
    virtual ~SpecForward() = default;

    // Forward `tokens` and return ONE LOGITS ROW PER POSITION. Row i is the distribution
    // for the position after tokens[i].
    virtual void forward_all(std::span<const TokenId> tokens,
                             std::vector<std::vector<float>>& rows) = 0;

    // Forward `tokens` and return only the final position's row -- the tuned decode path.
    virtual void forward_last(std::span<const TokenId> tokens, std::vector<float>& row) = 0;

    // Mark the current cache state, and return to it. Restoring must be exact: the
    // full-attention layers move an index, the gated-delta layers restore a snapshot.
    virtual void checkpoint() = 0;
    virtual void restore() = 0;
};

struct SpecStep {
    // Tokens to emit, in order. Never empty on success -- the verification pass always
    // yields at least one token, which is speculative decoding's guaranteed floor.
    std::vector<TokenId> committed;
    // The logits row for the position after the last committed token, ready for the next
    // step. The cache is left holding every committed token.
    std::vector<float> next_logits;
    bool no_legal_token = false;
    bool speculated = false;
};

class SpeculativeDecoder {
  public:
    SpeculativeDecoder(const SamplingParams& params, SpecConfig config);
    ~SpeculativeDecoder();
    SpeculativeDecoder(const SpeculativeDecoder&) = delete;
    SpeculativeDecoder& operator=(const SpeculativeDecoder&) = delete;

    // Feed the proposer everything the model has actually seen. Cheap; call it with the
    // prompt once and with committed tokens as they land.
    void observe(std::span<const TokenId> tokens);

    // Advance one block. `logits` is the row for the next position and is consumed
    // (shaped in place, as the plain sampler consumes it). `context` is the token history
    // the proposer matches against. `may_speculate` is the caller's grammar-phase gate;
    // `is_special` truncates the draft before anything that could change the phase.
    SpecStep step(std::vector<float>& logits, const TokenMask* mask,
                  const std::vector<TokenId>& recent, std::span<const TokenId> context,
                  bool may_speculate, const std::function<bool(TokenId)>& is_special,
                  SpecForward& fwd);

    [[nodiscard]] const SpecStats& stats() const noexcept { return stats_; }

  private:
    [[nodiscard]] std::vector<TokenId> propose(std::span<const TokenId> context,
                                               const std::function<bool(TokenId)>& is_special,
                                               const TokenMask* mask) const;

    SamplingParams params_;
    SpecConfig config_;
    Sampler sampler_;
    SpecVerifier verifier_;
    std::unique_ptr<draft::SuffixProposer> proposer_;
    SpecStats stats_;
};

} // namespace lmp::model
