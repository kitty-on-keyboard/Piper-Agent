#include "src/model/speculative.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>

#include "src/model/draft_proposer/suffix_proposer.hpp"
#include "src/model/mtp_proposer.hpp"

namespace lmp::model {

TokenId SpecForward::mtp_step_greedy(TokenId tok, std::span<const float> hidden,
                                     std::vector<float>& out_hidden) {
    mtp_step(tok, hidden, out_hidden);
    if (out_hidden.empty()) {
        return 0;
    }
    std::vector<float> row;
    mtp_logits(out_hidden, row);
    if (row.empty()) {
        out_hidden.clear();
        return 0;
    }
    std::size_t best = 0;
    for (std::size_t i = 1; i < row.size(); ++i) {
        if (row[i] > row[best]) {
            best = i;
        }
    }
    return static_cast<TokenId>(best);
}

TokenId SpecForward::mtp_argmax(std::span<const float> hidden) {
    std::vector<float> row;
    mtp_logits(hidden, row);
    if (row.empty()) {
        return 0;
    }
    std::size_t best = 0;
    for (std::size_t i = 1; i < row.size(); ++i) {
        if (row[i] > row[best]) {
            best = i;
        }
    }
    return static_cast<TokenId>(best);
}

namespace {

// The drafter is deterministic -- it proposes a concrete continuation from matched
// history, it is not a sampling model -- so every drafted token carries q = 1.
//
// This is not a shortcut, it is the condition under which the acceptance rule is EXACT.
// The scalar residual reduction is not distribution-preserving for a soft drafter (worst
// total-variation 0.23 by exact enumeration); at q = 1 it is exact to machine precision
// (8.3e-17). Acceptance then reduces to `u < p(t)`: keep the drafted token with the
// target's own probability for it. See src/model/spec_verifier.hpp.
constexpr float kDeterministicDrafter = 1.0F;

// The repetition-penalty window MlxBackend::generate keeps. Speculative rows must be
// shaped against the same window or they are not the rows sequential decoding would use.
constexpr std::size_t kRecentWindow = 64;

// The most committed-but-unforwarded tokens a block may carry in (see `pending_`).
//
// A REAL TUNING KNOB, not the safety valve it was first written as, and the first value
// (16) shipped a regression at long context.
//
// A deferred token is re-forwarded once per block until a fully accepted block clears it:
// partial acceptance rolls the cache back past the prefix, so the prefix returns to
// `pending_` and rides in again next block. At high acceptance that costs nothing because
// blocks clear. At 55% acceptance roughly 70% of blocks are partial, the prefix grows,
// and every verification pass drags it -- 146.5 ms per pass at 28k against the 88.7 ms a
// 3-position pass costs there.
//
// So the cap trades two costs against each other: too low pays a forward per block, too
// high pays width on every pass. Swept on one binary, one session, `bench 2 28000 100`:
//
//   cap    1      2      4      6      8     16
//   tok/s  13.5   16.1   17.4   15.3   14.6  14.1     (plain decode is 14.8)
//
// 4 is the peak, and it costs nothing at short context -- 23.4 tok/s at 547 tokens
// against 16's 23.9, inside the spread of both. 16 was 0.95x of plain at 28k; 4 is 1.18x.
//
// Re-sweep with LMP_MAX_DEFERRED if the draft length or the acceptance rate moves; the
// optimum is a function of both, and of how much a pass widens with context.
constexpr std::size_t kMaxDeferredDefault = 4;

// Overridable so the cap can be swept on one binary in one session, which is how the
// number above was found to be wrong: a deferred token is re-forwarded once per block
// until a fully accepted block clears it, so at low acceptance the prefix grows and every
// verification pass drags it along. Measured at 28k context, 55% acceptance: the pass
// cost 146.5 ms against the 88.7 ms a 3-position pass costs there.
std::size_t max_deferred() {
    static const std::size_t v = [] {
        const char* s = std::getenv("LMP_MAX_DEFERRED");
        if (s == nullptr) {
            return kMaxDeferredDefault;
        }
        const int n = std::atoi(s);
        return n > 0 ? static_cast<std::size_t>(n) : kMaxDeferredDefault;
    }();
    return v;
}

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

std::size_t SpeculativeDecoder::walk_masks(MaskSource& src, std::vector<TokenId>& drafted) {
    probe_n_ = 0;
    const auto capture = [this](const TokenMask& m) {
        if (probe_masks_.size() <= probe_n_) {
            probe_masks_.resize(probe_n_ + 1);
        }
        // Assignment, not push_back: vector::operator= reuses the destination's capacity,
        // so a steady state of blocks allocates nothing.
        probe_masks_[probe_n_] = m;
        ++probe_n_;
    };

    src.checkpoint();
    capture(src.mask());
    for (std::size_t i = 0; i < drafted.size(); ++i) {
        // Refused by the mask, or refused by the automaton -- either way the draft ends
        // here. Truncate rather than skip: a draft is a contiguous continuation, and the
        // verifier stops at the first rejection anyway.
        if (!probe_masks_[i].allows(drafted[i]) || !src.probe_advance(drafted[i])) {
            drafted.resize(i);
            break;
        }
        capture(src.mask());

        // A POSITION WITH NO LEGAL TOKEN CANNOT BE VERIFIED INTO, and this is the bug
        // that took tool-call speculation down.
        //
        // The mask is one token deep: it answers "is this id legal here", not "does this
        // id leave a continuation". Single-token decoding is protected by the model
        // sampling sensibly, but a DRAFTER guesses, and a wrong guess can walk the
        // automaton into a state no token in the vocabulary can leave -- a state the
        // model itself would never have reached, and which no scan of the well-formed
        // stream finds.
        //
        // The block then shapes a distribution there and it comes back empty. That empty
        // row is not caught as "nothing accepted", because SpecVerifier always pushes a
        // token (its documented floor, so a decode loop cannot hang); it surfaces one
        // line later as `tail_idx >= dists[m].ids.size()`, which is vacuously true when
        // ids is empty. The decoder reads that as "the grammar and the vocabulary
        // disagree", reports a build defect, and ends the run.
        //
        // Every position the block may land on -- including the bonus at index
        // drafted.size() -- has to have something legal in it. So stop the draft here and
        // drop the empty mask: shorter draft, same answer, no dead end to fall into.
        if (!probe_masks_[probe_n_ - 1].any()) {
            --probe_n_;
            drafted.resize(i);
            break;
        }
    }
    src.rollback();
    return probe_n_;
}

const TokenMask* SpeculativeDecoder::mask_at(std::size_t i, const TokenMask* fallback) const {
    return i < probe_n_ ? &probe_masks_[i] : fallback;
}

void SpeculativeDecoder::flush(SpecForward& fwd) {
    if (pending_.empty()) {
        return;
    }
    fwd.forward_last(std::span<const TokenId>(pending_), row_);
    pending_.clear();
}

SpecStep SpeculativeDecoder::decode_one(const TokenMask* mask,
                                        const std::vector<TokenId>& recent, SpecForward& fwd) {
    SpecStep out;
    // Draws from the verifier's stream too, so that turning speculation on or off does
    // not silently change which stream the plain steps consume -- the distribution is
    // identical either way, the sequence is not.
    const TokenDist dist0 = sampler_.distribution(row_, mask, recent);
    if (dist0.empty()) {
        // Genuinely nothing legal here. This IS the build defect the caller reports: the
        // ordinary path reached it with no speculation involved.
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

SpecStep SpeculativeDecoder::abandon_block(std::size_t prefix, const TokenMask* mask,
                                           const std::vector<TokenId>& recent, SpecForward& fwd) {
    // ABANDON THE BLOCK, do not fail the run. Speculation is an optimisation; it must
    // not be able to turn a decodable step into a dead one. Undo the verification
    // forward, tell the drafter nothing survived, pay for the deferred prefix, and
    // take one token the ordinary way -- which is the reference path and answers for
    // itself whether there is genuinely nothing legal here.
    fwd.restore();
    proposer_->settle(0, {}, prefix, fwd);
    ++stats_.abandoned;
    flush(fwd);
    return decode_one(mask_at(0, mask), recent, fwd);
}

void SpeculativeDecoder::update_cache_and_forward(std::size_t m, std::size_t draft_count,
                                                  const std::vector<TokenId>& committed,
                                                  SpecForward& fwd) {
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
    const bool full = (m == draft_count);
    const bool defer = proposer_->can_draft_deferred() &&
                       (full || pending_.size() + committed.size() <= max_deferred());
    if (defer) {
        if (full) {
            pending_.assign(1, committed.back());
        } else {
            fwd.restore();
            pending_.insert(pending_.end(), committed.begin(), committed.end());
        }
        // No row until the next pass produces one; holding a stale one would let a caller
        // shape a distribution for a position the target has not reached.
        row_.clear();
    } else if (full) {
        const std::span<const TokenId> tail(&committed.back(), 1);
        fwd.forward_last(tail, row_);
        pending_.clear();
    } else {
        fwd.restore();
        // The prefix went back out with the rejected tail, so it is replayed too.
        std::vector<TokenId> replay = pending_;
        replay.insert(replay.end(), committed.begin(), committed.end());
        fwd.forward_last(std::span<const TokenId>(replay), row_);
        pending_.clear();
    }
}

std::vector<TokenDist> SpeculativeDecoder::shape_distributions(
    std::size_t prefix, std::span<const std::vector<float>> rows,
    std::span<const TokenId> drafted, const TokenMask* mask,
    const std::vector<TokenId>& recent) const {
    std::vector<TokenDist> dists;
    dists.reserve(drafted.size() + 1);
    if (prefix == 0) {
        dists.push_back(sampler_.distribution(row_, mask_at(0, mask), recent));
    } else if (prefix - 1 < rows.size()) {
        dists.push_back(sampler_.distribution(rows[prefix - 1], mask_at(0, mask), recent));
    }
    if (dists.empty() || dists.front().empty()) {
        return dists;
    }
    std::vector<TokenId> recent_i = recent;
    for (std::size_t i = 0; i < drafted.size() && prefix + i < rows.size(); ++i) {
        recent_i.push_back(drafted[i]);
        if (recent_i.size() > kRecentWindow) {
            recent_i.erase(recent_i.begin());
        }
        dists.push_back(sampler_.distribution(rows[prefix + i], mask_at(i + 1, mask), recent_i));
    }
    return dists;
}

std::vector<TokenId> SpeculativeDecoder::map_draft_indices(
    std::span<const TokenId> drafted, std::span<const TokenDist> dists) const {
    std::vector<TokenId> draft_idx;
    draft_idx.reserve(drafted.size());
    for (std::size_t i = 0; i < drafted.size() && i + 1 < dists.size(); ++i) {
        const std::vector<TokenId>& ids = dists[i].ids;
        const auto it = std::lower_bound(ids.begin(), ids.end(), drafted[i]);
        draft_idx.push_back((it != ids.end() && *it == drafted[i])
                                ? static_cast<TokenId>(it - ids.begin())
                                : static_cast<TokenId>(ids.size()));
    }
    return draft_idx;
}

bool SpeculativeDecoder::evaluate_speculative_block(
    std::span<const TokenId> drafted, std::span<const TokenDist> dists,
    const std::vector<TokenId>& draft_idx, std::size_t prefix, SpecForward& fwd,
    SpecStep& out) {
    std::vector<std::vector<float>> compact;
    std::vector<std::span<const float>> row_spans;
    compact.reserve(dists.size());
    row_spans.reserve(dists.size());
    for (const TokenDist& d : dists) {
        compact.push_back(d.probs);
        row_spans.emplace_back(compact.back().data(), compact.back().size());
    }

    const std::vector<float> ones(draft_idx.size(), kDeterministicDrafter);
    const SpecResult r =
        verifier_.verify(std::span<const TokenId>(draft_idx), std::span<const float>(ones),
                         std::span<const std::span<const float>>(row_spans));
    if (r.accepted.empty()) {
        return false;
    }

    const std::size_t m = std::min(r.accepted_drafts, draft_idx.size());
    for (std::size_t i = 0; i < m; ++i) {
        out.committed.push_back(drafted[i]);
    }
    const auto tail_idx = static_cast<std::size_t>(r.accepted.back());
    if (m >= dists.size() || tail_idx >= dists[m].ids.size()) {
        out.committed.clear();
        return false;
    }
    out.committed.push_back(dists[m].ids[tail_idx]);

    proposer_->settle(m, std::span<const TokenId>(out.committed), prefix, fwd);

    stats_.accepted_drafts += m;
    stats_.committed += out.committed.size();

    update_cache_and_forward(m, drafted.size(), out.committed, fwd);
    return true;
}

SpecStep SpeculativeDecoder::step(MaskSource* src, const std::vector<TokenId>& recent,
                                  std::span<const TokenId> context, bool may_speculate,
                                  const std::function<bool(TokenId)>& is_special,
                                  SpecForward& fwd) {
    SpecStep out;
    probe_n_ = 0;
    const TokenMask* mask = src != nullptr ? &src->mask() : nullptr;
    const bool walking = src != nullptr && !src->mask_is_block_stable() && src->can_checkpoint();

    if (!proposer_->can_draft_deferred() || pending_.size() >= max_deferred()) {
        flush(fwd);
    }

    std::vector<TokenId> drafted =
        (config_.enabled && may_speculate)
            ? propose(context, is_special, walking ? nullptr : mask, fwd)
            : std::vector<TokenId>{};

    if (walking && !drafted.empty()) {
        walk_masks(*src, drafted);
        mask = mask_at(0, mask);
    }

    if (drafted.empty()) {
        flush(fwd);
        return decode_one(mask, recent, fwd);
    }

    ++stats_.blocks;
    stats_.drafted += drafted.size();
    out.speculated = true;

    const std::size_t prefix = pending_.size();
    std::vector<TokenId> input;
    input.reserve(prefix + drafted.size());
    input.insert(input.end(), pending_.begin(), pending_.end());
    input.insert(input.end(), drafted.begin(), drafted.end());

    fwd.checkpoint();
    std::vector<std::vector<float>> rows;
    fwd.forward_all(std::span<const TokenId>(input), rows);

    const std::vector<TokenDist> dists =
        shape_distributions(prefix, rows, drafted, mask, recent);
    if (dists.empty() || dists.front().empty()) {
        return abandon_block(prefix, mask, recent, fwd);
    }

    const std::vector<TokenId> draft_idx = map_draft_indices(drafted, dists);
    if (!evaluate_speculative_block(drafted, dists, draft_idx, prefix, fwd, out)) {
        return abandon_block(prefix, mask, recent, fwd);
    }

    return out;
}

} // namespace lmp::model
