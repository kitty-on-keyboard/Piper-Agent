#include "src/model/speculative.hpp"

#include <algorithm>
#include <array>

#include "bakeoff/draft_proposer/suffix_proposer.hpp"
#include "src/model/mtp_proposer.hpp"

namespace lmp::model {

namespace {

// The drafter is deterministic -- it proposes a concrete continuation from matched
// history, it is not a sampling model -- so every drafted token carries q = 1.
//
// This is not a shortcut, it is the condition under which the acceptance rule is EXACT.
// The scalar residual reduction is not distribution-preserving for a soft drafter (worst
// total-variation 0.23 by exact enumeration); at q = 1 it is exact to machine precision
// (8.3e-17). Acceptance then reduces to `u < p(t)`: keep the drafted token with the
// target's own probability for it. See bakeoff/spec_verifier/README.md.
constexpr float kDeterministicDrafter = 1.0F;

// The repetition-penalty window MlxBackend::generate keeps. Speculative rows must be
// shaped against the same window or they are not the rows sequential decoding would use.
constexpr std::size_t kRecentWindow = 64;

// The most committed-but-unforwarded tokens a block may carry in (see `pending_`).
//
// A safety valve, not a tuning knob. The prefix resets to one token on every fully
// accepted block, so in practice it sits at one to three; this only bounds the tail of an
// unlucky streak, because the prefix widens the verification pass and that pass is the
// one thing in the block that is not free. Past this, pay the forward and start over.
constexpr std::size_t kMaxDeferred = 16;

// The history-matching proposer, behind DraftProposer. It is stateless across a block --
// it matches a suffix and returns a continuation -- so settle() has nothing to undo.
class SuffixDraftProposer final : public DraftProposer {
  public:
    void ingest(std::span<const TokenId> tokens) override {
        impl_.ingest(std::span<const draft::TokenId>(tokens.data(), tokens.size()));
    }

    [[nodiscard]] std::vector<TokenId> propose(std::span<const TokenId> context,
                                               std::size_t max_draft, SpecForward&) override {
        // The LENGTH is the proposer's decision, not the caller's. Its stop rule is
        // cumulative acceptance probability against draft_cost_ratio; max_draft is only a
        // ceiling. NOTE that ratio was fitted to the MoE's expert-bandwidth cost model
        // (docs/MOE_ROUTING_FINDINGS.md) and does not describe a dense target, where
        // verifying k positions costs about one weight read regardless of k.
        const draft::Proposal pr = impl_.propose(
            std::span<const draft::TokenId>(context.data(), context.size()), max_draft);
        std::vector<TokenId> out;
        out.reserve(pr.tokens.size());
        for (draft::TokenId t : pr.tokens) {
            out.push_back(static_cast<TokenId>(t));
        }
        return out;
    }

  private:
    draft::SuffixProposer impl_;
};

} // namespace

SpeculativeDecoder::SpeculativeDecoder(const SamplingParams& params, SpecConfig config)
    : params_(params), config_(config), sampler_(params),
      // A stream of its own, seeded off the sampling seed. The speculative path draws
      // from the verifier rather than the sampler, so the two must not be the same
      // stream -- and a fixed seed still makes a run reproducible.
      verifier_(params.seed ^ 0x5EC0DE5EC0DE5EC0ULL),
      proposer_(config.mtp_block_size >= 2
                    ? std::unique_ptr<DraftProposer>(
                          std::make_unique<MtpProposer>(config.mtp_block_size))
                    : std::unique_ptr<DraftProposer>(
                          std::make_unique<SuffixDraftProposer>())) {}

SpeculativeDecoder::~SpeculativeDecoder() = default;

void SpeculativeDecoder::observe(std::span<const TokenId> tokens) {
    if (!config_.enabled || tokens.empty()) {
        return;
    }
    proposer_->ingest(tokens);
}

std::vector<TokenId> SpeculativeDecoder::propose(
    std::span<const TokenId> context, const std::function<bool(TokenId)>& is_special,
    const TokenMask* mask, SpecForward& fwd) {
    // A grafted drafter proposes from the target's hidden state, so an empty context is
    // not by itself a reason to skip -- only a history drafter needs one.
    if (context.empty() && !fwd.has_mtp()) {
        return {};
    }
    const std::vector<TokenId> raw = proposer_->propose(context, config_.max_draft, fwd);

    std::vector<TokenId> out;
    out.reserve(raw.size());
    for (const TokenId id : raw) {
        // Truncate, never skip. A draft is a contiguous continuation; dropping a token
        // from the middle would propose a sequence the model was never going to produce.
        //
        // Special tokens are what move the grammar between phases, and the whole block
        // shares one mask on the assumption that the phase does not move (see header).
        if (is_special && is_special(id)) {
            break;
        }
        // A masked-out proposal has p = 0, is rejected with certainty, and would waste a
        // draft slot -- and every slot after it, since verification stops at the first
        // rejection.
        if (mask != nullptr && !mask->allows(id)) {
            break;
        }
        out.push_back(id);
    }
    return out;
}

void SpeculativeDecoder::seed(std::vector<float> row) {
    row_ = std::move(row);
    pending_.clear();
}

void SpeculativeDecoder::flush(SpecForward& fwd) {
    if (pending_.empty()) {
        return;
    }
    fwd.forward_last(std::span<const TokenId>(pending_), row_);
    pending_.clear();
}

SpecStep SpeculativeDecoder::step(const TokenMask* mask, const std::vector<TokenId>& recent,
                                  std::span<const TokenId> context, bool may_speculate,
                                  const std::function<bool(TokenId)>& is_special,
                                  SpecForward& fwd) {
    SpecStep out;

    // A prefix left from the previous block can only ride along under a proposer that
    // drafts from its own state; anything else needs the target caught up first.
    if (!proposer_->can_draft_deferred() || pending_.size() >= kMaxDeferred) {
        flush(fwd);
    }

    // Drafted BEFORE the row is shaped, which the deferred path requires: with a prefix
    // outstanding there is no row yet to shape. Safe to reorder because distribution() is
    // const and consumes no randomness, and a grafted proposer reads only hidden state.
    const std::vector<TokenId> drafted =
        (config_.enabled && may_speculate) ? propose(context, is_special, mask, fwd)
                                           : std::vector<TokenId>{};

    // No block to fold the prefix into, so it has to be paid for on its own.
    if (drafted.empty()) {
        flush(fwd);
        // Ordinary decoding. Draws from the verifier's stream too, so that turning
        // speculation on or off does not silently change which stream the plain steps
        // consume -- the distribution is identical either way, the sequence is not.
        const TokenDist dist0 = sampler_.distribution(row_, mask, recent);
        if (dist0.empty()) {
            out.no_legal_token = true;
            return out;
        }
        const std::span<const float> row(dist0.probs.data(), dist0.probs.size());
        const std::array<std::span<const float>, 1> rows{row};
        const SpecResult r =
            verifier_.verify({}, {}, std::span<const std::span<const float>>(rows));
        if (r.accepted.empty()) {
            out.no_legal_token = true;
            return out;
        }
        const auto idx = static_cast<std::size_t>(r.accepted.front());
        if (idx >= dist0.ids.size()) {
            out.no_legal_token = true;
            return out;
        }
        out.committed.push_back(dist0.ids[idx]);
        fwd.forward_last(std::span<const TokenId>(out.committed), row_);
        ++stats_.fallbacks;
        return out;
    }

    // --- the speculative block ---------------------------------------------------
    ++stats_.blocks;
    stats_.drafted += drafted.size();
    out.speculated = true;

    // The prefix rides in front of the drafts, in ONE pass. Position j-1 of that pass is
    // the row the first draft is verified against -- the row a trailing forward_last used
    // to be run to obtain, at the price of a whole extra read of the weights.
    const std::size_t prefix = pending_.size();
    std::vector<TokenId> input;
    input.reserve(prefix + drafted.size());
    input.insert(input.end(), pending_.begin(), pending_.end());
    input.insert(input.end(), drafted.begin(), drafted.end());

    fwd.checkpoint();
    std::vector<std::vector<float>> rows;
    fwd.forward_all(std::span<const TokenId>(input), rows);

    // One shaped distribution per drafted position, plus the row for the position the
    // first draft sits at. With no prefix that is the row carried from the last block;
    // with one it is rows[prefix - 1], produced by the pass above. The repetition penalty
    // is built from the PROVISIONAL history: verification at position i only happens if
    // drafted[0..i-1] were accepted, so `recent` extended by exactly that prefix is the
    // history the sequential path would have had. Anything else would verify against a
    // row the model would never have produced.
    //
    // `recent` already contains the deferred tokens -- the caller pushes every committed
    // token into its window as it is emitted, whether or not the target has consumed it
    // yet -- so the shaping here is the same either way.
    std::vector<TokenDist> dists;
    dists.reserve(drafted.size() + 1);
    if (prefix == 0) {
        dists.push_back(sampler_.distribution(row_, mask, recent));
    } else if (prefix - 1 < rows.size()) {
        dists.push_back(sampler_.distribution(rows[prefix - 1], mask, recent));
    }
    if (dists.empty() || dists.front().empty()) {
        proposer_->settle(0, {}, prefix, fwd);
        out.no_legal_token = true;
        return out;
    }
    std::vector<TokenId> recent_i = recent;
    for (std::size_t i = 0; i < drafted.size() && prefix + i < rows.size(); ++i) {
        recent_i.push_back(drafted[i]);
        // The SAME bounded window the sequential loop keeps. The repetition penalty is
        // applied once per OCCURRENCE in this list, so an unbounded window would penalise
        // a repeated token geometrically harder than the plain path does -- verifying
        // against a row the model would never have produced. Caught by the gate test,
        // which saw a p=1.0 token verify at p=0.515.
        if (recent_i.size() > kRecentWindow) {
            recent_i.erase(recent_i.begin());
        }
        dists.push_back(sampler_.distribution(rows[prefix + i], mask, recent_i));
    }

    // Map into the verifier's index space: each row is dense over that position's
    // candidate list, and a drafted token that did not survive shaping is given an
    // out-of-range index, which the verifier reads as p = 0 and rejects with certainty.
    std::vector<std::vector<float>> compact;
    std::vector<std::span<const float>> row_spans;
    compact.reserve(dists.size());
    row_spans.reserve(dists.size());
    for (const TokenDist& d : dists) {
        compact.push_back(d.probs);
        row_spans.emplace_back(compact.back().data(), compact.back().size());
    }
    std::vector<TokenId> draft_idx;
    draft_idx.reserve(drafted.size());
    for (std::size_t i = 0; i < drafted.size() && i + 1 < dists.size(); ++i) {
        const std::vector<TokenId>& ids = dists[i].ids;
        const auto it = std::lower_bound(ids.begin(), ids.end(), drafted[i]);
        draft_idx.push_back((it != ids.end() && *it == drafted[i])
                                ? static_cast<TokenId>(it - ids.begin())
                                : static_cast<TokenId>(ids.size()));
    }
    const std::vector<float> ones(draft_idx.size(), kDeterministicDrafter);

    const SpecResult r =
        verifier_.verify(std::span<const TokenId>(draft_idx), std::span<const float>(ones),
                         std::span<const std::span<const float>>(row_spans));
    if (r.accepted.empty()) {
        // Nothing survived. A stateful drafter still appended to its own cache while
        // proposing, so it has to be told, or its cache outruns the target's by exactly
        // the drafts nobody kept -- a drift that never throws and only shows up as
        // steadily worse proposals.
        proposer_->settle(0, {}, prefix, fwd);
        out.no_legal_token = true;
        return out;
    }

    const std::size_t m = std::min(r.accepted_drafts, draft_idx.size());
    for (std::size_t i = 0; i < m; ++i) {
        out.committed.push_back(drafted[i]);
    }
    // The final token is an index into the row at position m, not a token id.
    const auto tail_idx = static_cast<std::size_t>(r.accepted.back());
    if (m >= dists.size() || tail_idx >= dists[m].ids.size()) {
        proposer_->settle(0, {}, prefix, fwd);
        out.no_legal_token = true;
        return out;
    }
    out.committed.push_back(dists[m].ids[tail_idx]);

    // Settle here, and not a line earlier: a grafted drafter seeds the next round by
    // forwarding the tokens it did not already hold, and the last of those is the bonus
    // token, which only exists now. It also reads the hidden rows from the verification
    // pass above, so this must precede the forward_last / restore below -- those replace
    // what fwd.last_hidden() reports.
    proposer_->settle(m, std::span<const TokenId>(out.committed), prefix, fwd);

    stats_.accepted_drafts += m;
    stats_.committed += out.committed.size();

    // Where the cache is left, and who pays for the next row.
    //
    // Full acceptance is the cheap case: the pass consumed the prefix and every draft, so
    // only the bonus token -- sampled, never forwarded -- is outstanding.
    //
    // Partial acceptance cannot simply drop the rejected tail. The full-attention layers
    // could (their rollback is an index), but the gated-delta layers hold a recurrence
    // with no per-token history, so the only reachable earlier state is the checkpoint --
    // which sits BEFORE the prefix as well. Everything that rode in comes back out.
    //
    // DEFERRING is the point of all this. The alternative, taken whenever the proposer
    // cannot draft without the target, is to forward the outstanding tokens now purely to
    // obtain a row -- and on a dense target that is a full read of the weights for one
    // row, as expensive as the verification pass it follows. So when the proposer carries
    // its own seed, nothing is forwarded here at all: the tokens go into `pending_` and
    // ride in front of the next block's verification, which was going to run regardless.
    const bool full = (m == drafted.size());
    const bool defer = proposer_->can_draft_deferred() &&
                       (full || pending_.size() + out.committed.size() <= kMaxDeferred);
    if (defer) {
        if (full) {
            pending_.assign(1, out.committed.back());
        } else {
            fwd.restore();
            pending_.insert(pending_.end(), out.committed.begin(), out.committed.end());
        }
        // No row until the next pass produces one; holding a stale one would let a caller
        // shape a distribution for a position the target has not reached.
        row_.clear();
    } else if (full) {
        const std::span<const TokenId> tail(&out.committed.back(), 1);
        fwd.forward_last(tail, row_);
        pending_.clear();
    } else {
        fwd.restore();
        // The prefix went back out with the rejected tail, so it is replayed too.
        std::vector<TokenId> replay = pending_;
        replay.insert(replay.end(), out.committed.begin(), out.committed.end());
        fwd.forward_last(std::span<const TokenId>(replay), row_);
        pending_.clear();
    }
    return out;
}

} // namespace lmp::model
