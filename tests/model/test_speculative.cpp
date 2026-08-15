// The speculative block algebra, tested with no GPU and no checkpoint.
//
// This is the whole reason SpecForward is an interface. A wrong speculative decoder does
// not crash and does not produce garbage -- it commits a token from the wrong position, or
// leaves the cache one ahead of the ledger, and the text stays fluent while the
// distribution quietly shifts. The only cheap way to make that loud is to drive the loop
// with a model whose correct output is known exactly.
//
// ScriptForward is that model: a deterministic next-token function. Against it, speculative
// decoding has exactly one correct answer -- the same token sequence ordinary decoding
// would produce -- and any off-by-one in "which positions were committed, what does the
// cache hold now, which row feeds the next step" changes it.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <cstdint>
#include <span>
#include <vector>

#include "src/model/speculative.hpp"
#include "tests/check.hpp"

using namespace lmp::model;

namespace {

constexpr std::size_t kVocab = 64;

// A deterministic "model": the token at position i is always seq[i]. Emits a near-one-hot
// logits row so the shaped distribution is a point mass and verification has no slack.
class ScriptForward final : public SpecForward {
  public:
    explicit ScriptForward(std::vector<TokenId> seq, std::size_t start)
        : seq_(std::move(seq)), pos_(start) {}

    void forward_all(std::span<const TokenId> tokens,
                     std::vector<std::vector<float>>& rows) override {
        rows.clear();
        ++forward_all_calls;
        hidden_.clear();
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            hidden_.push_back({static_cast<float>(pos_)});
            ++pos_;
            consumed_.push_back(tokens[i]);
            rows.push_back(row_for(pos_));
        }
        max_batch = std::max(max_batch, tokens.size());
        verified_since_read_ = true;
    }

    void forward_last(std::span<const TokenId> tokens, std::vector<float>& row) override {
        ++forward_last_calls;
        // An ordinary step breaks the head's chain: the next round has no seed.
        primed_ = false;
        hidden_.clear();
        for (TokenId t : tokens) {
            hidden_.push_back({static_cast<float>(pos_)});
            ++pos_;
            consumed_.push_back(t);
        }
        row = row_for(pos_);
    }

    void checkpoint() override {
        mark_ = pos_;
        mark_consumed_ = consumed_.size();
        ++checkpoints;
    }

    void restore() override {
        pos_ = mark_;
        consumed_.resize(mark_consumed_);
        ++restores;
    }

    // --- the grafted-drafter path, off unless `mtp` is set --------------------------
    //
    // A PERFECT head, on purpose: it drafts exactly the continuation the target would
    // produce, so every block is fully accepted and the test drives the full-acceptance
    // deferral branch -- the one where nothing at all is forwarded after verification.
    //
    // It predicts by position rather than by chaining hidden tags, and the discriminator
    // is `verified_since_read_`: propose() reads last_hidden() BEFORE this round's
    // verification pass, settle() reads it after. The two want different cursors -- the
    // drafts start at the target's current position, while the seed is for the position
    // after the bonus token, which the target has not forwarded. Modelling that with tag
    // arithmetic alone is off by one at one site or the other, which is exactly the
    // hazard MtpProposer's header warns about.
    [[nodiscard]] bool has_mtp() const override { return mtp; }

    void last_hidden(std::vector<std::vector<float>>& rows) override {
        rows = hidden_;
        if (verified_since_read_) {
            // settle(): the seed is for the position after the BONUS token, and the bonus
            // is the one thing the verification pass did not forward.
            draft_cursor_ = pos_ + 1;
            verified_since_read_ = false;
            primed_ = true;
        } else if (!primed_) {
            // A cold round -- no seed to continue from, so drafting opens at the target's
            // own position.
            draft_cursor_ = pos_;
        }
        // Otherwise the cursor runs on from where settle left it. A seeded round's first
        // draft was already produced there, and pos_ cannot be used to recompute it: with
        // a deferred prefix the target deliberately lags the committed tokens.
    }

    void mtp_step(TokenId, std::span<const float> h, std::vector<float>& out) override {
        out = {h.empty() ? 0.0F : h[0]};
        ++mtp_steps;
    }

    void mtp_logits(std::span<const float>, std::vector<float>& row) override {
        row = row_for(draft_cursor_++);
    }

    void mtp_trim(std::size_t n) override { mtp_trimmed += n; }

    // The row the loop starts from: the distribution for the position after `start`.
    [[nodiscard]] std::vector<float> initial_row() const { return row_for(pos_); }
    [[nodiscard]] std::size_t pos() const noexcept { return pos_; }
    [[nodiscard]] const std::vector<TokenId>& consumed() const noexcept { return consumed_; }

    std::size_t forward_all_calls = 0;
    std::size_t forward_last_calls = 0;
    std::size_t checkpoints = 0;
    std::size_t restores = 0;
    std::size_t max_batch = 0;
    std::size_t mtp_steps = 0;
    std::size_t mtp_trimmed = 0;
    bool mtp = false;

  private:
    [[nodiscard]] std::vector<float> row_for(std::size_t position) const {
        std::vector<float> row(kVocab, 0.0F);
        if (position < seq_.size()) {
            row[static_cast<std::size_t>(seq_[position])] = 50.0F;
        } else {
            row[0] = 50.0F;
        }
        return row;
    }

    std::vector<TokenId> seq_;
    std::size_t pos_;
    std::size_t mark_ = 0;
    std::size_t mark_consumed_ = 0;
    std::vector<TokenId> consumed_;
    std::vector<std::vector<float>> hidden_;
    std::size_t draft_cursor_ = 0;
    bool verified_since_read_ = false;
    bool primed_ = false;
};

// Repetitive on purpose: SuffixProposer only proposes where it has matched history with
// support, which is exactly the agent workload this is for (re-emitted tool scaffolding,
// paths, code it just read). A random sequence would correctly propose nothing and the
// test would pass without ever speculating -- which is why every case below asserts that
// speculation actually happened.
std::vector<TokenId> repetitive_sequence(std::size_t n) {
    const std::vector<TokenId> phrase_a{7, 8, 9, 10, 11, 12};
    const std::vector<TokenId> phrase_b{20, 21, 22, 23};
    std::vector<TokenId> out;
    std::uint32_t x = 99;
    while (out.size() < n) {
        x = x * 1664525U + 1013904223U;
        const std::vector<TokenId>& p = (x >> 16U) % 3U == 0U ? phrase_b : phrase_a;
        out.insert(out.end(), p.begin(), p.end());
    }
    out.resize(n);
    return out;
}

// The point of these tests is the BLOCK ALGEBRA, so the shaped distribution has to be a
// genuine point mass. It is not by default: repetition_penalty is 1.05 and is applied once
// per OCCURRENCE in `recent`, so a repeated token's logit is divided geometrically. The
// first version of this file passed the whole history as `recent` and watched a p=1.0
// token verify at p=0.515 -- the decoder was right and the harness was wrong. Left as a
// comment because it is the same trap the production loop avoids only by capping the
// window at 64.
SamplingParams deterministic_params() {
    SamplingParams p;
    p.seed = 1234;
    p.temperature = 1.0F;
    p.repetition_penalty = 1.0F;
    p.top_p = 1.0F;
    return p;
}

struct RunOut {
    std::vector<TokenId> emitted;
    SpecStats stats;
};

// void, with an out-parameter: REQUIRE expands to a bare `return`, so a helper that
// asserts cannot also have a return value.
void run(SpecForward& fwd, std::vector<float> logits, std::size_t want, bool may_speculate,
         const std::function<bool(TokenId)>& is_special, SpeculativeDecoder& dec,
         std::vector<TokenId>& history, RunOut& out) {
    // `recent` is the backend's bounded window of GENERATED tokens, not the whole history.
    std::vector<TokenId> recent;
    dec.seed(std::move(logits));
    while (out.emitted.size() < want) {
        SpecStep st = dec.step(nullptr, recent, std::span<const TokenId>(history),
                               may_speculate, is_special, fwd);
        REQUIRE(!st.no_legal_token);
        REQUIRE(!st.committed.empty()); // the floor
        for (TokenId t : st.committed) {
            out.emitted.push_back(t);
            history.push_back(t);
            recent.push_back(t);
            if (recent.size() > 64) {
                recent.erase(recent.begin());
            }
        }
        dec.observe(std::span<const TokenId>(st.committed));
    }
    out.emitted.resize(want);
    out.stats = dec.stats();
}

} // namespace

TEST(speculation_reproduces_the_deterministic_sequence_exactly) {
    // The headline. Against a deterministic model, the committed sequence must be the
    // model's own continuation -- token for token, at every position. This fails on any
    // off-by-one in the block algebra, and on a rollback that leaves the cache holding a
    // rejected token.
    const std::vector<TokenId> seq = repetitive_sequence(600);
    const std::size_t start = 300;

    ScriptForward fwd(seq, start);
    SpecConfig cfg;
    cfg.enabled = true;
    SpeculativeDecoder dec(deterministic_params(), cfg);

    std::vector<TokenId> history(seq.begin(), seq.begin() + static_cast<long>(start));
    dec.observe(std::span<const TokenId>(history));

    const std::size_t want = 200;
    RunOut got;
    run(fwd, fwd.initial_row(), want, true, [](TokenId) { return false; }, dec, history, got);

    const std::vector<TokenId> expect(seq.begin() + static_cast<long>(start),
                                      seq.begin() + static_cast<long>(start + want));
    CHECK_EQ(got.emitted.size(), expect.size());
    for (std::size_t i = 0; i < expect.size() && i < got.emitted.size(); ++i) {
        if (got.emitted[i] != expect[i]) {
            std::fprintf(stderr, "  [debug] first divergence at %zu: got %d want %d\n", i,
                         got.emitted[i], expect[i]);
            std::fprintf(stderr, "  [debug] got  ...");
            for (std::size_t j = (i > 6 ? i - 6 : 0); j < i + 8 && j < got.emitted.size(); ++j)
                std::fprintf(stderr, " %d", got.emitted[j]);
            std::fprintf(stderr, "\n  [debug] want ...");
            for (std::size_t j = (i > 6 ? i - 6 : 0); j < i + 8 && j < expect.size(); ++j)
                std::fprintf(stderr, " %d", expect[j]);
            std::fprintf(stderr, "\n");
            break;
        }
    }
    CHECK(got.emitted == expect);

    // And it must have actually speculated, or the assertion above proved nothing.
    CHECK(got.stats.blocks > 0);
    CHECK(got.stats.accepted_drafts > 0);
    CHECK(fwd.max_batch > 1);
    std::fprintf(stderr,
                 "  [spec-gate] blocks=%llu drafted=%llu accepted=%llu (%.1f%%) "
                 "forward_all=%zu forward_last=%zu restores=%zu\n",
                 static_cast<unsigned long long>(got.stats.blocks),
                 static_cast<unsigned long long>(got.stats.drafted),
                 static_cast<unsigned long long>(got.stats.accepted_drafts),
                 100.0 * got.stats.acceptance_rate(), fwd.forward_all_calls,
                 fwd.forward_last_calls, fwd.restores);
}

TEST(the_cache_ends_holding_exactly_the_committed_tokens) {
    // The invariant that keeps the ledger and the tensors in step. After every block the
    // model must have consumed the committed prefix and nothing else -- no rejected draft
    // left behind, no committed token missing.
    const std::vector<TokenId> seq = repetitive_sequence(400);
    const std::size_t start = 200;

    ScriptForward fwd(seq, start);
    SpecConfig cfg;
    cfg.enabled = true;
    SpeculativeDecoder dec(deterministic_params(), cfg);

    std::vector<TokenId> history(seq.begin(), seq.begin() + static_cast<long>(start));
    dec.observe(std::span<const TokenId>(history));

    dec.seed(fwd.initial_row());
    std::vector<TokenId> emitted;
    std::vector<TokenId> recent;
    for (int block = 0; block < 40; ++block) {
        SpecStep st = dec.step(nullptr, recent, std::span<const TokenId>(history), true,
                               [](TokenId) { return false; }, fwd);
        REQUIRE(!st.no_legal_token);
        for (TokenId t : st.committed) {
            emitted.push_back(t);
            history.push_back(t);
            recent.push_back(t);
            if (recent.size() > 64) {
                recent.erase(recent.begin());
            }
        }
        dec.observe(std::span<const TokenId>(st.committed));

        // The scripted model's position advances by exactly one per consumed token, so
        // this is the cache offset the real backend would hold. It holds for this fake
        // because it reports no MTP head, so the decoder never defers a forward -- with a
        // deferred prefix the cache legitimately lags the emitted tokens by pending_.
        CHECK_EQ(fwd.pos(), start + emitted.size());
        CHECK_EQ(fwd.consumed().size(), emitted.size());
        CHECK(fwd.consumed() == emitted);
    }
}

TEST(a_block_unstable_mask_falls_back_to_one_token_at_a_time) {
    // Inside a tool call the mask is state-dependent per token, so a block-wide snapshot
    // of it would be wrong. The decoder must decode singly there -- and still produce the
    // same sequence.
    const std::vector<TokenId> seq = repetitive_sequence(400);
    const std::size_t start = 200;

    ScriptForward fwd(seq, start);
    SpecConfig cfg;
    cfg.enabled = true;
    SpeculativeDecoder dec(deterministic_params(), cfg);

    std::vector<TokenId> history(seq.begin(), seq.begin() + static_cast<long>(start));
    dec.observe(std::span<const TokenId>(history));

    const std::size_t want = 60;
    RunOut got;
    run(fwd, fwd.initial_row(), want, /*may_speculate=*/false, [](TokenId) { return false; },
        dec, history, got);

    const std::vector<TokenId> expect(seq.begin() + static_cast<long>(start),
                                      seq.begin() + static_cast<long>(start + want));
    CHECK(got.emitted == expect);
    CHECK_EQ(got.stats.blocks, std::uint64_t{0});
    CHECK_EQ(got.stats.fallbacks, std::uint64_t{want});
    CHECK_EQ(fwd.forward_all_calls, std::size_t{0});
    CHECK_EQ(fwd.max_batch, std::size_t{0});
}

TEST(drafts_are_truncated_before_a_phase_changing_token) {
    // The block shares one mask on the assumption that the phase does not move, and only a
    // structural token moves it. So no drafted token may be one -- and the decoder must
    // still be correct, just with shorter drafts.
    const std::vector<TokenId> seq = repetitive_sequence(400);
    const std::size_t start = 200;

    ScriptForward fwd(seq, start);
    SpecConfig cfg;
    cfg.enabled = true;
    SpeculativeDecoder dec(deterministic_params(), cfg);

    std::vector<TokenId> history(seq.begin(), seq.begin() + static_cast<long>(start));
    dec.observe(std::span<const TokenId>(history));

    // Treat one mid-phrase id as structural. Every draft that would have crossed it must
    // stop short.
    const auto is_special = [](TokenId id) { return id == 10; };
    const std::size_t want = 120;
    RunOut got;
    run(fwd, fwd.initial_row(), want, true, is_special, dec, history, got);

    const std::vector<TokenId> expect(seq.begin() + static_cast<long>(start),
                                      seq.begin() + static_cast<long>(start + want));
    CHECK(got.emitted == expect);
    CHECK(got.stats.blocks > 0);
}

TEST(speculation_is_off_by_default) {
    // SpecConfig::enabled defaults false, and a decoder built that way must never draft --
    // the plain path stays the reference until someone opts in.
    const std::vector<TokenId> seq = repetitive_sequence(300);
    ScriptForward fwd(seq, 150);
    SpeculativeDecoder dec(deterministic_params(), SpecConfig{});

    std::vector<TokenId> history(seq.begin(), seq.begin() + 150);
    dec.observe(std::span<const TokenId>(history));

    RunOut got;
    run(fwd, fwd.initial_row(), 40, true, [](TokenId) { return false; }, dec, history, got);
    const std::vector<TokenId> expect(seq.begin() + 150, seq.begin() + 190);
    CHECK(got.emitted == expect);
    CHECK_EQ(got.stats.blocks, std::uint64_t{0});
    CHECK_EQ(fwd.forward_all_calls, std::size_t{0});
}

// --- the deferred prefix -------------------------------------------------------------
//
// A block used to end by forwarding its own tail purely to obtain the next row. On a
// dense target that is a full read of the weights for ONE row -- measured at 62.6 ms
// against the 60.9 ms the entire 3-position verification pass cost, which is why
// speculation benchmarked below plain decode. The tail is now carried and prepended to
// the next block's verification pass instead.
//
// The risk in that is not a crash. It is committing a token from the wrong position, or
// pairing the drafter against a hidden row from the block before, and both keep producing
// fluent text. So the assertion that matters is the same one the rest of this file makes:
// against a deterministic model there is exactly one correct continuation.

TEST(deferring_the_forward_does_not_change_what_is_committed) {
    const std::vector<TokenId> seq = repetitive_sequence(400);
    const std::size_t start = 100;

    ScriptForward fwd(seq, start);
    fwd.mtp = true;
    SpecConfig cfg;
    cfg.enabled = true;
    cfg.mtp_block_size = 3; // 2 drafts a round

    SpeculativeDecoder dec(deterministic_params(), cfg);
    std::vector<TokenId> history(seq.begin(), seq.begin() + static_cast<long>(start));
    RunOut got;
    run(fwd, fwd.initial_row(), 60, true, [](TokenId) { return false; }, dec, history, got);

    // The headline: identical to what ordinary decoding produces.
    const std::vector<TokenId> expect(seq.begin() + static_cast<long>(start),
                                      seq.begin() + static_cast<long>(start) + 60);
    CHECK(got.emitted == expect);
    CHECK(got.stats.blocks > 0);

    // The head is perfect here, so every block is fully accepted and NOTHING is forwarded
    // after verification. The single forward_last is the cold start: the first step has no
    // hidden state to draft from, so it decodes one token the ordinary way.
    CHECK_EQ(fwd.forward_last_calls, std::size_t{1});
    CHECK_EQ(got.stats.accepted_drafts, got.stats.drafted);

    // And the prefix really did ride along: a round drafts 2, so a batch of 3 can only be
    // a deferred token in front of them.
    CHECK_EQ(fwd.max_batch, std::size_t{3});
}

TEST(a_deferred_prefix_leaves_the_cache_behind_the_emitted_tokens_but_never_ahead) {
    // The invariant that replaces "the cache holds exactly what was emitted". Deferral
    // means the target legitimately lags -- by exactly the tokens not yet forwarded -- but
    // it must never hold a token that was not committed, and never hold them out of order.
    // A cache that ran AHEAD would be the silent-stale-context failure S5.10 exists for.
    const std::vector<TokenId> seq = repetitive_sequence(400);
    const std::size_t start = 100;

    ScriptForward fwd(seq, start);
    fwd.mtp = true;
    SpecConfig cfg;
    cfg.enabled = true;
    cfg.mtp_block_size = 3;

    SpeculativeDecoder dec(deterministic_params(), cfg);
    std::vector<TokenId> history(seq.begin(), seq.begin() + static_cast<long>(start));
    std::vector<TokenId> emitted;
    std::vector<TokenId> recent;
    dec.seed(fwd.initial_row());

    bool lagged = false;
    for (int block = 0; block < 30; ++block) {
        SpecStep st = dec.step(nullptr, recent, std::span<const TokenId>(history), true,
                               [](TokenId) { return false; }, fwd);
        REQUIRE(!st.no_legal_token);
        for (TokenId t : st.committed) {
            emitted.push_back(t);
            history.push_back(t);
            recent.push_back(t);
            if (recent.size() > 64) {
                recent.erase(recent.begin());
            }
        }
        dec.observe(std::span<const TokenId>(st.committed));

        // consumed() is what the target actually forwarded, and it must be a PREFIX of
        // what was emitted -- never longer, never divergent.
        REQUIRE(fwd.consumed().size() <= emitted.size());
        const std::vector<TokenId> prefix(emitted.begin(),
                                          emitted.begin() +
                                              static_cast<long>(fwd.consumed().size()));
        CHECK(fwd.consumed() == prefix);
        CHECK_EQ(fwd.pos(), start + fwd.consumed().size());
        if (fwd.consumed().size() < emitted.size()) {
            lagged = true;
        }
    }
    // If it never lagged, nothing was deferred and this test proved nothing.
    CHECK(lagged);
}
