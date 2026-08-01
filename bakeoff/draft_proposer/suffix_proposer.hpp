#pragma once
//
// SuffixProposer -- model-free draft proposer for speculative decoding.
//
// Amalgamated from the 5-entrant cook-off. A1's storage is kept almost verbatim (compact
// array-backed trie, 32-bit node indices, hash-mapped roots, first-child/next-sibling) --
// it was the only entrant fast enough to be free at 0.04 us p50, and structure was not
// where the entrants differed.
//
// The DECISION is rebuilt, because that is where every entrant left value on the table:
//
//   1. SUPPORT. A1 trusts a context seen ONCE: count/count = 1.0, so it proposes eight
//      tokens at "confidence 1.0" off a single observation. In a repetitive corpus most
//      long contexts occur exactly once, so this is the main source of waste. A match must
//      now clear `min_support` before it is evidence of anything.
//
//   2. CUMULATIVE, NOT STEP, PROBABILITY. A1 stops when the STEP probability drops below
//      0.1 -- i.e. it keeps extending while there is a 90% chance of being wrong at that
//      position. But verification stops at the FIRST rejection, so every token after a bad
//      one is wasted regardless. The quantity that matters is the probability the draft is
//      still correct AT position k, which is the running product. That is what gates
//      extension here.
//
//   3. BACKOFF ON SUPPORT, NOT JUST LENGTH. Every entrant took the longest match. A
//      32-token match seen once is weaker evidence than a 6-token match seen fifty times.
//      This walks back from the longest match to the longest one that also has support.
//
// The stop rule follows from what a draft token is worth. Verifying one more position
// costs `draft_cost_ratio` (in units of a saved token) and returns a token with
// probability equal to the running product, so extending pays exactly while
// `cumulative >= draft_cost_ratio`. On an MoE at batch 1 a wasted position is not free --
// it drags in its own experts -- which is why the default is not near-zero like A1's.
//
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace draft {

using TokenId = std::int32_t;

struct Config {
    std::size_t min_match_len = 3;
    std::size_t max_match_len = 32;
    std::size_t max_indexed_tokens = 1u << 20;

    // --- additions over the cook-off interface (defaults chosen by measurement) ---

    // A context observed fewer times than this is not evidence. 1 disables the check and
    // reproduces the cook-off behaviour.
    std::uint32_t min_support = 2;

    // Stop extending when the running probability that the draft is still correct falls
    // below this. Equivalently: the cost of verifying one more position, in units of the
    // value of one saved token.
    //
    // Theory says lambda/(1+lambda) = 0.5 when a wasted position costs about what a saved
    // token is worth, which is roughly the MoE batch-1 case. The sweep says accepted
    // tokens PLATEAU above 0.3 while waste keeps falling -- on the synthetic corpus,
    // 0.80 reaches 4.160 accepted at 0.000 waste, against 4.231 at 0.358 for 0.50. So the
    // threshold is almost free: it truncates tokens that were going to be rejected anyway.
    //
    // Defaulted to 0.60 rather than the measured optimum of 0.80 deliberately. The corpus
    // that produced these numbers is synthetic and its continuations are near-deterministic
    // once a supported match exists; real agent text is messier, and 0.60 keeps recall in
    // reserve for it while still capturing 4.114 of the 4.160 available net. RE-SWEEP THIS
    // ON REAL TRACES before trusting either number.
    float draft_cost_ratio = 0.60f;
};

struct Proposal {
    std::vector<TokenId> tokens;
    // The running product: a calibrated estimate that the WHOLE proposal is correct, not a
    // per-step figure. Callers can threshold it directly.
    float confidence = 0.0f;
    std::size_t matched_len = 0;
};

class SuffixProposer {
  public:
    explicit SuffixProposer(Config config = {});
    ~SuffixProposer();
    SuffixProposer(const SuffixProposer&) = delete;
    SuffixProposer& operator=(const SuffixProposer&) = delete;

    void ingest(std::span<const TokenId> sequence);
    [[nodiscard]] Proposal propose(std::span<const TokenId> context,
                                   std::size_t max_tokens) const;

    [[nodiscard]] std::size_t indexed_tokens() const noexcept;
    [[nodiscard]] std::size_t sequence_count() const noexcept;
    void clear() noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace draft
