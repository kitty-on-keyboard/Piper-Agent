// MtpProposer's bookkeeping, with no GPU and no checkpoint.
//
// This is the whole reason the MTP head sits ABOVE the SpecForward seam. Every failure
// mode here is silent: pair a token with the wrong hidden row, forget to trim the head's
// cache when only part of a draft survives, or carry a stale seed across a rejection, and
// nothing throws. The head just drafts from a state the target never reached, acceptance
// sags, and it reads as "MTP is mediocre on this model" rather than as a defect. A
// scripted fake is the only instrument that can tell those apart.

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "src/model/mtp_proposer.hpp"
#include "src/model/speculative.hpp"

#include "tests/check.hpp"

using namespace lmp::model;

namespace {

// Hidden states are one-element vectors carrying a tag, so a test can say exactly WHICH
// row a step was handed rather than merely that it got one of the right shape.
std::vector<float> tag(float v) { return {v}; }
float tag_of(std::span<const float> h) { return h.empty() ? -1.0F : h[0]; }

struct StepCall {
    TokenId tok;
    float hidden_tag;
};

class FakeMtp final : public SpecForward {
  public:
    // --- the plain path, unused here but required by the interface ---
    void forward_all(std::span<const TokenId>, std::vector<std::vector<float>>&) override {}
    void forward_last(std::span<const TokenId>, std::vector<float>&) override {}
    void checkpoint() override {}
    void restore() override {}

    // --- the grafted-drafter path ---
    [[nodiscard]] bool has_mtp() const override { return true; }

    void last_hidden(std::vector<std::vector<float>>& rows) override { rows = report; }

    void mtp_step(TokenId tok, std::span<const float> hidden,
                  std::vector<float>& out) override {
        steps.push_back({tok, tag_of(hidden)});
        // A distinguishable output tag, so the NEXT call's hidden identifies its source.
        out = tag(1000.0F + static_cast<float>(steps.size()));
    }

    void mtp_logits(std::span<const float>, std::vector<float>& row) override {
        row.assign(16, 0.0F);
        row[static_cast<std::size_t>(next_pred) % 16] = 1.0F;
        ++next_pred;
    }

    void mtp_trim(std::size_t n) override { trims.push_back(n); }

    // What last_hidden() should report on the next call.
    std::vector<std::vector<float>> report;
    std::vector<StepCall> steps;
    std::vector<std::size_t> trims;
    int next_pred = 3; // first drafted token id is 3, then 4, ...
};

} // namespace

TEST(mtp_drafts_block_size_minus_one_tokens) {
    // The checkpoint's config says block_size 3; its model card says "block size 2". Both
    // are right -- the round yields block_size - 1 tokens.
    MtpProposer p(3);
    CHECK_EQ(p.draft_len(), std::size_t{2});

    MtpProposer eight(8);
    CHECK_EQ(eight.draft_len(), std::size_t{7});

    // A degenerate block size must draft nothing rather than underflow to a huge count.
    MtpProposer one(1);
    CHECK_EQ(one.draft_len(), std::size_t{0});
    MtpProposer zero(0);
    CHECK_EQ(zero.draft_len(), std::size_t{0});
}

TEST(first_round_pairs_the_last_committed_token_with_the_row_it_was_sampled_from) {
    FakeMtp fwd;
    fwd.report = {tag(7.0F)}; // the row the last committed token came from
    MtpProposer p(3);

    const std::vector<TokenId> context{11, 42};
    const std::vector<TokenId> got = p.propose(context, 8, fwd);

    REQUIRE(got.size() == 2);
    REQUIRE(fwd.steps.size() == 2);
    // The pairing that is easy to invert: (next_token, hidden_that_predicted_it). Step 0
    // must be the LAST COMMITTED token against h_current -- not the token before it, and
    // not a row from the verification pass.
    CHECK_EQ(fwd.steps[0].tok, TokenId{42});
    CHECK(fwd.steps[0].hidden_tag == 7.0F);
    // Step 1 chains on the head's OWN output, since the target has no opinion yet.
    CHECK_EQ(fwd.steps[1].tok, got[0]);
    CHECK(fwd.steps[1].hidden_tag == 1001.0F);
}

TEST(a_full_round_seeds_the_next_one_so_it_pays_for_one_step_fewer) {
    FakeMtp fwd;
    fwd.report = {tag(7.0F)};
    MtpProposer p(3);

    const std::vector<TokenId> context{42};
    const std::vector<TokenId> first = p.propose(context, 8, fwd);
    REQUIRE(first.size() == 2);
    REQUIRE(fwd.steps.size() == 2);

    // Everything accepted. Verification rows: index 0 is h_current (held by the proposer),
    // 1 and 2 are the target's hidden after each drafted token.
    fwd.report = {tag(20.0F), tag(21.0F)};
    const std::vector<TokenId> committed{first[0], first[1], 99};
    fwd.steps.clear();
    p.settle(2, std::span<const TokenId>(committed), 0, fwd);

    // Nothing to trim on full acceptance, and the head catches up on the bonus token only
    // -- the two accepted drafts it had already forwarded must NOT be replayed.
    CHECK(fwd.trims.empty());
    REQUIRE(fwd.steps.size() == 1);
    CHECK_EQ(fwd.steps[0].tok, TokenId{99});
    // The bonus pairs with verification row index `accepted` == 2, which is report[1].
    CHECK(fwd.steps[0].hidden_tag == 21.0F);

    // The next round opens with that seed: two drafts for the price of ONE step.
    fwd.report = {tag(30.0F)};
    fwd.steps.clear();
    const std::vector<TokenId> second = p.propose(committed, 8, fwd);
    CHECK_EQ(second.size(), std::size_t{2});
    CHECK_EQ(fwd.steps.size(), std::size_t{1});
}

TEST(partial_acceptance_trims_exactly_the_positions_the_target_did_not_keep) {
    FakeMtp fwd;
    fwd.report = {tag(7.0F)};
    MtpProposer p(4); // 3 drafts, so a partial accept has something to trim

    const std::vector<TokenId> context{42};
    const std::vector<TokenId> drafted = p.propose(context, 8, fwd);
    REQUIRE(drafted.size() == 3);
    REQUIRE(fwd.steps.size() == 3);

    // Only the first draft survived.
    fwd.report = {tag(20.0F), tag(21.0F), tag(22.0F)};
    fwd.steps.clear();
    const std::vector<TokenId> committed{drafted[0], 99};
    p.settle(1, std::span<const TokenId>(committed), 0, fwd);

    // Appended 3, kept 1 -> exactly 2 come back off. Trimming the wrong count is the
    // defect that never surfaces: the head stays ahead of the target for the rest of the
    // turn and every later draft is built on a continuation that did not happen.
    REQUIRE(fwd.trims.size() == 1);
    CHECK_EQ(fwd.trims[0], std::size_t{2});

    // Then it catches up on the bonus only -- draft 0 was already forwarded and kept.
    //
    // The bonus pairs with verification row `accepted` == 1, i.e. report[0] == 20.0: the
    // row AFTER draft 0 is the distribution the bonus was sampled from. Not report[1] --
    // that row follows a draft the target rejected, and pairing against it would seed the
    // head from a continuation that never happened.
    REQUIRE(fwd.steps.size() == 1);
    CHECK_EQ(fwd.steps[0].tok, TokenId{99});
    CHECK(fwd.steps[0].hidden_tag == 20.0F);
}

TEST(total_rejection_trims_everything_and_drops_the_seed) {
    FakeMtp fwd;
    fwd.report = {tag(7.0F)};
    MtpProposer p(3);

    const std::vector<TokenId> context{42};
    REQUIRE(p.propose(context, 8, fwd).size() == 2);
    REQUIRE(fwd.steps.size() == 2);

    fwd.steps.clear();
    p.settle(0, {}, 0, fwd); // nothing survived, no bonus

    REQUIRE(fwd.trims.size() == 1);
    CHECK_EQ(fwd.trims[0], std::size_t{2});
    CHECK(fwd.steps.empty());

    // A stale seed here would draft from a position the target never reached, so the next
    // round must start cold: two drafts, two steps, and the first paired with h_current.
    fwd.report = {tag(50.0F)};
    fwd.steps.clear();
    const std::vector<TokenId> next = p.propose(context, 8, fwd);
    CHECK_EQ(next.size(), std::size_t{2});
    REQUIRE(fwd.steps.size() == 2);
    CHECK_EQ(fwd.steps[0].tok, TokenId{42});
    CHECK(fwd.steps[0].hidden_tag == 50.0F);
}

TEST(max_draft_caps_the_round_below_the_block_size) {
    FakeMtp fwd;
    fwd.report = {tag(7.0F)};
    MtpProposer p(8);

    const std::vector<TokenId> context{42};
    const std::vector<TokenId> got = p.propose(context, 2, fwd);
    CHECK_EQ(got.size(), std::size_t{2});
    CHECK_EQ(fwd.steps.size(), std::size_t{2});
}

TEST(a_target_without_an_mtp_head_drafts_nothing) {
    class NoMtp final : public SpecForward {
      public:
        void forward_all(std::span<const TokenId>, std::vector<std::vector<float>>&) override {}
        void forward_last(std::span<const TokenId>, std::vector<float>&) override {}
        void checkpoint() override {}
        void restore() override {}
    };
    NoMtp fwd;
    MtpProposer p(3);
    const std::vector<TokenId> context{42};
    CHECK(p.propose(context, 8, fwd).empty());
    // settle must be inert too, not reach through a head that is not there.
    p.settle(0, {}, 0, fwd);
}

TEST(no_hidden_row_means_no_draft_rather_than_a_guess) {
    FakeMtp fwd;
    fwd.report = {}; // the target has not produced a hidden row yet
    MtpProposer p(3);
    const std::vector<TokenId> context{42};
    CHECK(p.propose(context, 8, fwd).empty());
    CHECK(fwd.steps.empty());
}

TEST(a_deferred_prefix_shifts_which_hidden_rows_the_drafts_pair_against) {
    // When the block carries committed-but-unforwarded tokens, the verification pass
    // begins with THEM, so the row that predicted drafted[0] is inside this pass at
    // rows[prefix - 1] -- not the h_current captured at propose(), which belongs to an
    // earlier block. Pairing against the wrong row does not throw and does not corrupt
    // the output; the head simply drafts from a state the target never reached and
    // acceptance sags, which reads as a mediocre drafter rather than as a defect.
    FakeMtp fwd;
    fwd.report = {tag(7.0F)};
    MtpProposer p(3); // 2 drafts

    const std::vector<TokenId> context{42};
    const std::vector<TokenId> drafted = p.propose(context, 8, fwd);
    REQUIRE(drafted.size() == 2);

    // A pass over [p0, p1, d0, d1]: two deferred tokens in front of the two drafts.
    // Rows 0 and 1 follow the deferred tokens; row 1 is the one that predicted d0.
    fwd.report = {tag(50.0F), tag(51.0F), tag(52.0F), tag(53.0F)};
    fwd.steps.clear();
    const std::vector<TokenId> committed{drafted[0], drafted[1], 99};
    p.settle(2, std::span<const TokenId>(committed), 2, fwd);

    // Full acceptance, so the head only catches up on the bonus. The bonus pairs with
    // verify[accepted] == verify[2], which with prefix 2 is rows[2 - 1 + 2] == rows[3].
    CHECK(fwd.trims.empty());
    REQUIRE(fwd.steps.size() == 1);
    CHECK_EQ(fwd.steps[0].tok, TokenId{99});
    CHECK(fwd.steps[0].hidden_tag == 53.0F);
}

TEST(a_deferred_prefix_pairs_a_partially_accepted_draft_against_its_own_row) {
    // The same shift, on the branch where it is easiest to get wrong: only the first
    // draft survived, so the head has to be trimmed AND caught up, and the row the bonus
    // pairs with sits one past the surviving draft rather than at the end of the pass.
    FakeMtp fwd;
    fwd.report = {tag(7.0F)};
    MtpProposer p(4); // 3 drafts

    const std::vector<TokenId> context{42};
    const std::vector<TokenId> drafted = p.propose(context, 8, fwd);
    REQUIRE(drafted.size() == 3);
    REQUIRE(fwd.steps.size() == 3);

    // A pass over [p0, d0, d1, d2]: one deferred token, then the three drafts.
    fwd.report = {tag(60.0F), tag(61.0F), tag(62.0F), tag(63.0F)};
    fwd.steps.clear();
    const std::vector<TokenId> committed{drafted[0], 99};
    p.settle(1, std::span<const TokenId>(committed), 1, fwd);

    // Appended 3, kept 1 -> 2 come back off, exactly as with no prefix.
    REQUIRE(fwd.trims.size() == 1);
    CHECK_EQ(fwd.trims[0], std::size_t{2});

    // verify[i] == rows[prefix - 1 + i] == rows[i], so the bonus at verify[1] is rows[1]
    // -- the row AFTER the one surviving draft. rows[2] and rows[3] follow drafts the
    // target rejected, and seeding from either would build the next round on a
    // continuation that never happened.
    REQUIRE(fwd.steps.size() == 1);
    CHECK_EQ(fwd.steps[0].tok, TokenId{99});
    CHECK(fwd.steps[0].hidden_tag == 61.0F);
}
