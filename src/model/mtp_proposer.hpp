#ifndef LMP_MODEL_MTP_PROPOSER_HPP
#define LMP_MODEL_MTP_PROPOSER_HPP
//
// MtpProposer -- the Qwen3.5/3.6 multi-token-prediction head, as a draft proposer.
//
// The MTP head is not a second model. It is one full-attention layer plus an `fc` over
// concat(embedding, hidden), and it ships NO embedding table and NO lm_head: it borrows
// the target's. So it drafts from hidden states rather than from token history, which is
// what SpecForward's mtp_* primitives carry.
//
// EVERYTHING STATEFUL LIVES HERE, above the seam, on purpose. The seed token and hidden
// carried between rounds, the count of positions this round appended to the MTP cache,
// the trim when only part of a draft survives -- that is where the bug lands, and it does
// not announce itself. Get it wrong and nothing throws: the drafter simply proposes from
// a state the target never reached, acceptance sags, and it reads as "MTP is mediocre on
// this model" rather than as a defect. Above the seam a scripted fake proves it in the
// gate; below it, only a GPU could.
//
// ------------------------------------------------------------------------------------
// THE PAIRING, which is the part that is easy to get backwards.
//
// The head consumes (next_token, hidden_at_the_position_that_predicted_it) -- NOT a token
// with its own hidden state. The reference makes this visible in its prefill, which pairs
// input_ids[i+1] with hidden[i]. Two consequences:
//
//   * Drafting starts from (last_committed_token, h_current), where h_current is the
//     hidden row the last committed token was SAMPLED from -- the row the previous
//     forward_last produced, not the one that follows it.
//   * When settling, drafted token i pairs with verification hidden row i, where row 0 is
//     h_current and row i+1 is the target's hidden after drafted[i]. SpecForward's
//     forward_all reports only the latter, so h_current has to be captured at propose()
//     time and prepended here. Off by one in either direction still runs, still decodes
//     fluent text, and just quietly stops being worth doing.
//
// Ported from mlx-vlm 0.6.12, speculative/drafters/qwen3_5_mtp.
//
#include <cstddef>
#include <span>
#include <vector>

#include "src/model/backend.hpp"
#include "src/model/speculative.hpp"

namespace lmp::model {

class MtpProposer final : public DraftProposer {
  public:
    // `block_size` is the checkpoint's own config value. It yields block_size - 1 drafted
    // tokens per round, which is why the Qwen3.6-27B card says "block size 2" while its
    // config.json says 3 -- both are right about different quantities.
    explicit MtpProposer(std::size_t block_size) noexcept : block_size_(block_size) {}

    // The head drafts from hidden state; token history tells it nothing.
    void ingest(std::span<const TokenId>) override {}

    [[nodiscard]] std::vector<TokenId> propose(std::span<const TokenId> context,
                                               std::size_t max_draft,
                                               SpecForward& fwd) override;

    void settle(std::size_t accepted, std::span<const TokenId> committed,
                SpecForward& fwd) override;

    void reset() override;

    // How many tokens a full round proposes. Zero when the head cannot draft at all.
    [[nodiscard]] std::size_t draft_len() const noexcept {
        return block_size_ >= 2 ? block_size_ - 1 : 0;
    }

  private:
    std::size_t block_size_;
    // Positions this round appended to the MTP head's cache. Only these may be trimmed:
    // anything settled in an earlier round is already agreed with the target.
    std::size_t round_appended_ = 0;

    bool has_seed_ = false;
    TokenId seed_token_ = 0;
    std::vector<float> seed_hidden_;

    // The hidden row the last committed token was sampled from. Captured at propose(),
    // because the verification forward replaces what last_hidden() reports.
    std::vector<float> h_current_;

    // What this round proposed, so settle() can pair survivors with hidden rows.
    std::vector<TokenId> drafted_;
};

} // namespace lmp::model

#endif // LMP_MODEL_MTP_PROPOSER_HPP
