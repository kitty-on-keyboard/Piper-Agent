#include "src/model/mtp_proposer.hpp"

#include <algorithm>
#include <cstddef>

namespace lmp::model {

namespace {

// The head drafts greedily. Speculative decoding stays distribution-preserving through the
// VERIFIER, not through the drafter -- and the acceptance rule the verifier implements is
// exact only for a deterministic drafter (q = 1), which is the same assumption the suffix
// proposer already relies on. Sampling here would break that, not improve the draft.
[[nodiscard]] TokenId argmax_of(std::span<const float> row) noexcept {
    std::size_t best = 0;
    for (std::size_t i = 1; i < row.size(); ++i) {
        if (row[i] > row[best]) {
            best = i;
        }
    }
    return static_cast<TokenId>(best);
}

} // namespace

void MtpProposer::reset() {
    round_appended_ = 0;
    has_seed_ = false;
    seed_token_ = 0;
    seed_hidden_.clear();
    h_current_.clear();
    drafted_.clear();
}

std::vector<TokenId> MtpProposer::propose(std::span<const TokenId> context,
                                          std::size_t max_draft, SpecForward& fwd) {
    drafted_.clear();
    round_appended_ = 0;
    const std::size_t want = std::min(max_draft, draft_len());
    if (!fwd.has_mtp() || want == 0) {
        return {};
    }

    // The row the last committed token was sampled from. Held for settle(), which pairs
    // drafted token 0 against it (see the header's note on the pairing).
    std::vector<std::vector<float>> rows;
    fwd.last_hidden(rows);
    if (rows.empty() || rows.back().empty()) {
        return {};
    }
    h_current_ = rows.back();

    TokenId tok = 0;
    std::vector<float> h_prev;
    if (has_seed_) {
        // The previous round already forwarded through the accepted tokens and predicted
        // one more. That prediction is this round's first draft and costs nothing: the
        // head has already paid for it.
        tok = seed_token_;
        h_prev = seed_hidden_;
        drafted_.push_back(tok);
        has_seed_ = false;
    } else {
        if (context.empty()) {
            return {};
        }
        tok = context.back();
        h_prev = h_current_;
    }

    while (drafted_.size() < want) {
        std::vector<float> h_next;
        fwd.mtp_step(tok, h_prev, h_next);
        if (h_next.empty()) {
            break;
        }
        ++round_appended_;
        h_prev = std::move(h_next);

        std::vector<float> row;
        fwd.mtp_logits(h_prev, row);
        if (row.empty()) {
            break;
        }
        tok = argmax_of(row);
        drafted_.push_back(tok);
    }
    return drafted_;
}

void MtpProposer::settle(std::size_t accepted, std::span<const TokenId> committed,
                         std::size_t prefix, SpecForward& fwd) {
    if (!fwd.has_mtp()) {
        return;
    }

    // Only this round's appends are speculative. `accepted` counts drafted tokens the
    // target kept; anything the head appended beyond that describes a continuation that
    // did not happen and has to come back off its cache.
    const std::size_t keep = std::min(accepted, round_appended_);
    const std::size_t trim = round_appended_ - keep;
    if (trim > 0) {
        fwd.mtp_trim(trim);
    }
    round_appended_ = 0;

    if (committed.empty()) {
        // Total rejection: nothing to seed from, and the next round starts cold from the
        // target's own hidden. Dropping the seed matters -- keeping a stale one would
        // draft from a position the target never reached.
        has_seed_ = false;
        seed_hidden_.clear();
        return;
    }

    // Verification hidden rows, indexed so that verify[i] is the target's hidden at the
    // position that PREDICTED drafted[i].
    //
    // Without a prefix the pass covered the drafts alone, so its row i is the hidden
    // AFTER drafted[i] -- one position too late to pair with drafted[i] itself -- and the
    // missing first row is h_current_, captured at propose() before this pass replaced it.
    //
    // With a prefix the pass began with `prefix` already-committed tokens, so the row that
    // predicted drafted[0] is inside this pass at rows[prefix - 1], and h_current_ belongs
    // to an older block. Both forms yield drafted.size() + 1 entries.
    std::vector<std::vector<float>> rows;
    fwd.last_hidden(rows);
    std::vector<const std::vector<float>*> verify;
    verify.reserve(rows.size() + 1);
    if (prefix == 0) {
        verify.push_back(&h_current_);
        for (const std::vector<float>& r : rows) {
            verify.push_back(&r);
        }
    } else {
        for (std::size_t i = prefix - 1; i < rows.size(); ++i) {
            verify.push_back(&rows[i]);
        }
    }

    // Catch the head up on everything the target committed that it has not already
    // forwarded: the accepted drafts past `keep`, then the bonus token. Each pairs with
    // the TARGET's hidden at that position, not with the head's own output -- these are
    // independent steps, not a chain.
    std::vector<float> h_last;
    for (std::size_t i = keep; i < accepted && i < drafted_.size() && i < verify.size(); ++i) {
        std::vector<float> h_next;
        fwd.mtp_step(drafted_[i], *verify[i], h_next);
        if (h_next.empty()) {
            has_seed_ = false;
            return;
        }
        h_last = std::move(h_next);
    }

    const std::size_t bonus_idx = accepted;
    if (bonus_idx < verify.size()) {
        std::vector<float> h_next;
        fwd.mtp_step(committed.back(), *verify[bonus_idx], h_next);
        if (!h_next.empty()) {
            h_last = std::move(h_next);
        }
    }

    if (h_last.empty()) {
        has_seed_ = false;
        return;
    }

    // Seed the next round from the head's own last state, which is one prediction the
    // next propose() gets without paying for a forward.
    std::vector<float> row;
    fwd.mtp_logits(h_last, row);
    if (row.empty()) {
        has_seed_ = false;
        return;
    }
    seed_token_ = argmax_of(row);
    seed_hidden_ = std::move(h_last);
    has_seed_ = true;
}

} // namespace lmp::model
