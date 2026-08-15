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
    // Non-zero when the target carries an MTP head, carrying that head's block size.
    // Drafting then goes through the head instead of history matching, and the fixed
    // length is correct rather than a regression: the "adaptive or nothing" measurements
    // above describe expert scattering on the MoE, and a dense target verifies k positions
    // for about one weight read regardless of k.
    std::size_t mtp_block_size = 0;
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

    // ---- the grafted-drafter path (MTP) ---------------------------------------------
    //
    // An MTP head is not a second model: it is one layer plus an fc that consumes
    // concat(embedding, hidden), and it borrows the TARGET's embedding table and LM head.
    // So it needs hidden states, which the logits-only methods above cannot carry.
    //
    // These are primitives on purpose. Everything stateful about MTP -- carrying the seed
    // token and hidden across rounds, trimming its cache when only part of a draft is
    // accepted, keeping its positions straight -- lives in MtpProposer, ABOVE this seam,
    // because that bookkeeping is where the bug lands and it does not announce itself: get
    // it wrong and nothing crashes, the acceptance rate just quietly sags and looks like a
    // model that drafts poorly. Above the seam a scripted fake can prove it in the gate.
    //
    // Defaulted so a target without an MTP head, and the gate's own fakes, need not care.

    // Does this target have an MTP head loaded and bound?
    [[nodiscard]] virtual bool has_mtp() const { return false; }

    // Hidden rows produced by the most recent forward_all / forward_last, one per
    // position forwarded. Row i is the final-normed hidden state AFTER tokens[i] -- the
    // same state the LM head consumes, which is what the reference feeds the MTP head.
    virtual void last_hidden(std::vector<std::vector<float>>& rows) { rows.clear(); }

    // One MTP step: (token, hidden) -> next hidden, appending one position to the MTP
    // head's own KV cache. `hidden` is a row from last_hidden or a previous mtp_step.
    virtual void mtp_step(TokenId /*tok*/, std::span<const float> /*hidden*/,
                          std::vector<float>& out_hidden) {
        out_hidden.clear();
    }

    // The TARGET's LM head applied to an MTP hidden row. The MTP checkpoint ships no
    // lm_head of its own, which is also why a drafted token is not free: this is the
    // single most expensive tensor either model touches.
    virtual void mtp_logits(std::span<const float> /*hidden*/, std::vector<float>& row) {
        row.clear();
    }

    // Drop the last n positions from the MTP head's cache, and clear it entirely.
    virtual void mtp_trim(std::size_t /*n*/) {}
    virtual void mtp_reset() {}
};

// What a draft proposer has to do, so the history-matching proposer and the MTP head are
// interchangeable and the gate exercises both through one path.
class DraftProposer {
  public:
    virtual ~DraftProposer() = default;

    // Everything the model has actually seen. Cheap; the prompt once, then committed
    // tokens as they land. A grafted drafter may ignore it -- its state comes from hidden.
    virtual void ingest(std::span<const TokenId> tokens) = 0;

    // May the target's forward be DEFERRED underneath this proposer -- that is, can it
    // draft the next round without the target having consumed the tokens committed since
    // the last verification pass?
    //
    // Only a proposer carrying its own forward state can. The MTP head can: settle()
    // leaves it holding a seed token and hidden, one prediction ahead, so it needs
    // nothing from the target to start the next round. A history matcher cannot: it
    // matches against the ledger, and the ledger only holds what the target consumed, so
    // a deferred prefix would leave it proposing the continuation of a stale suffix.
    //
    // False by default, which keeps the deferral off for anything that has not thought
    // about it. See SpeculativeDecoder's `pending_` for what it buys.
    [[nodiscard]] virtual bool can_draft_deferred() const { return false; }

    // Propose up to max_draft tokens. `fwd` is the target: a grafted drafter runs through
    // its hidden-state primitives, a history drafter never touches it.
    [[nodiscard]] virtual std::vector<TokenId> propose(std::span<const TokenId> context,
                                                       std::size_t max_draft,
                                                       SpecForward& fwd) = 0;

    // How many of the proposed tokens survived, and what the block actually emits (those
    // accepted drafts followed by the bonus token). Called after EVERY speculative block,
    // including full acceptance and total rejection (where `committed` is empty).
    //
    // The bonus token is in here because a grafted drafter needs it: to seed the next
    // round it must forward the tokens it did not already have, and the bonus is the last
    // of them. Passing only a count would leave it seeding from a token it never saw.
    //
    // `prefix` is how many leading positions of the verification forward were
    // already-committed tokens riding along rather than drafts (see `pending_`). It
    // shifts where this round's hidden rows start: with a prefix, the row that predicted
    // drafted[0] is rows[prefix - 1], and with none it is the row captured at propose().
    // Getting it wrong does not throw and does not corrupt the output -- the verifier
    // still guarantees the distribution -- it just pairs the head against states the
    // target never reached, and reads as a drafter that has quietly gone mediocre.
    virtual void settle(std::size_t /*accepted*/, std::span<const TokenId> /*committed*/,
                        std::size_t /*prefix*/, SpecForward&) {}

    virtual void reset() {}
};

struct SpecStep {
    // Tokens to emit, in order. Never empty on success -- the verification pass always
    // yields at least one token, which is speculative decoding's guaranteed floor.
    std::vector<TokenId> committed;
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

    // The row for the first position, from prefill. Call once before the first step().
    //
    // The decoder owns the row from then on, and does NOT hand it back between steps.
    // It cannot: with a deferred prefix the row for the next position does not exist yet
    // -- producing it is exactly the forward pass being deferred. A caller that kept
    // passing a row in would be asserting the target had consumed something it had not.
    void seed(std::vector<float> row);

    // Advance one block. `src` is the constrained-decoding source, taken whole rather
    // than as a single mask: where its mask is not block-stable the decoder walks it over
    // the draft to derive the mask at each position, which is what lets a block run
    // inside a tool call. May be null (unconstrained decode). `context` is the token
    // history the proposer matches against; `may_speculate` is the caller's gate; and
    // `is_special` truncates the draft before anything that could change the phase.
    SpecStep step(MaskSource* src, const std::vector<TokenId>& recent,
                  std::span<const TokenId> context, bool may_speculate,
                  const std::function<bool(TokenId)>& is_special, SpecForward& fwd);

    [[nodiscard]] const SpecStats& stats() const noexcept { return stats_; }

  private:
    [[nodiscard]] std::vector<TokenId> propose(std::span<const TokenId> context,
                                               const std::function<bool(TokenId)>& is_special,
                                               const TokenMask* mask, SpecForward& fwd);

    // Consume the deferred prefix: forward it, leaving `row_` valid and `pending_` empty.
    void flush(SpecForward& fwd);

    // Walk `src` over `drafted`, capturing the mask at each drafted position plus the
    // bonus position after them, and TRUNCATING `drafted` at the first token the grammar
    // refuses. Leaves `src` exactly where it found it. Returns the number of masks
    // captured, which is drafted.size() + 1 unless the walk truncated.
    std::size_t walk_masks(MaskSource& src, std::vector<TokenId>& drafted);

    // The mask for position i of the block, or `fallback` when the walk did not run
    // (a block-stable source needs only one).
    [[nodiscard]] const TokenMask* mask_at(std::size_t i, const TokenMask* fallback) const;

    SamplingParams params_;
    SpecConfig config_;
    Sampler sampler_;
    SpecVerifier verifier_;
    std::unique_ptr<DraftProposer> proposer_;
    SpecStats stats_;

    // The row for the current position. Valid only while `pending_` is empty.
    std::vector<float> row_;

    // COMMITTED TOKENS THE TARGET HAS NOT FORWARDED YET, and the reason this class holds
    // state across steps at all.
    //
    // A block used to end by forwarding its own tail purely to obtain the next row. On a
    // dense target that is a SECOND FULL WEIGHT READ for one logits row: measured at 62.6
    // ms against the 60.9 ms the whole 3-position verification pass cost, i.e. 43% of the
    // block, which is why speculation benchmarked BELOW plain decode (16.5 vs 17.4 tok/s)
    // at a healthy 65% acceptance. A 1-token forward is not cheaper than a 3-token one --
    // both read all 15 GB -- so the fix is to not do it at all: carry the tail here and
    // prepend it to the NEXT block's verification pass, which was going to run anyway.
    //
    // Bounded by kMaxDeferred: it resets to one token whenever a block is fully accepted,
    // and grows only on partial acceptance, but a long unlucky streak must not widen the
    // verification pass without limit.
    std::vector<TokenId> pending_;

    // One mask per position of the current block, when the source had to be walked.
    // A member rather than a local so the bitsets keep their capacity across blocks:
    // each is a vocabulary-wide bitset (~31 KB here) and a block needs up to five.
    // `probe_n_` is how many are live; the vector itself is never shrunk.
    std::vector<TokenMask> probe_masks_;
    std::size_t probe_n_ = 0;
};

} // namespace lmp::model
