// plan_turn_reuse -- the cross-turn cache decision, tested with no GPU (G3).
//
// This is where an off-by-one in KV reuse actually hides: it does not crash, it decodes
// against a prefix that is one token out and the text stays fluent. So the algebra is a
// pure function and every branch is asserted here rather than discovered on the model.

#include <string>
#include <vector>

#include "src/model/kv_cache.hpp"
#include "tests/check.hpp"

using namespace lmp::model;

namespace {

std::vector<TokenId> ids(std::initializer_list<int> v) {
    return std::vector<TokenId>(v.begin(), v.end());
}

KvCacheLedger ledger_of(std::initializer_list<int> v) {
    KvCacheLedger l;
    l.append(ids(v));
    return l;
}

} // namespace

// The within-turn fast path, unchanged: the cache is a verified prefix, prefill the tail.
TEST(an_append_only_prompt_extends) {
    const KvCacheLedger l = ledger_of({1, 2, 3});
    const TurnReuse r = plan_turn_reuse(l, ids({1, 2, 3, 4, 5}), 2, true);
    CHECK(r.mode == ReuseMode::Extend);
    CHECK_EQ(r.prefill_from, std::size_t{3});
}

// The case this whole mechanism exists for. Between turns the ledger holds the previous
// prompt PLUS what was generated, and the new prompt inserts a turn record before the
// live-state block -- so they diverge mid-ledger. With a checkpoint at or below the common
// prefix, that is a rollback, not a full re-prefill.
TEST(divergence_past_a_valid_checkpoint_restores) {
    const KvCacheLedger l = ledger_of({1, 2, 3, 9, 9});
    const TurnReuse r = plan_turn_reuse(l, ids({1, 2, 3, 4, 5, 6}), 3, true);
    CHECK(r.mode == ReuseMode::Restore);
    CHECK_EQ(r.prefill_from, std::size_t{3});
}

// The checkpoint is only reachable if the prompt still agrees with the cache up to it.
// Compaction, a steering message and a persona change all land here, and none of them is
// special-cased -- the comparison fails and the answer falls through.
TEST(divergence_before_the_checkpoint_resets) {
    const KvCacheLedger l = ledger_of({1, 2, 3, 4, 5});
    const TurnReuse r = plan_turn_reuse(l, ids({1, 7, 3, 4, 5, 6}), 4, true);
    CHECK(r.mode == ReuseMode::Reset);
    CHECK_EQ(r.prefill_from, std::size_t{0});
}

TEST(no_checkpoint_resets) {
    const KvCacheLedger l = ledger_of({1, 2, 3, 9});
    CHECK(plan_turn_reuse(l, ids({1, 2, 3, 4}), 3, false).mode == ReuseMode::Reset);
    CHECK(plan_turn_reuse(l, ids({1, 2, 3, 4}), 0, true).mode == ReuseMode::Reset);
}

// A checkpoint longer than what either side holds is a caller bug; it must refuse to
// restore rather than read past the end of one of them.
TEST(an_out_of_range_checkpoint_resets) {
    const KvCacheLedger l = ledger_of({1, 2, 3, 9});
    CHECK(plan_turn_reuse(l, ids({1, 2}), 3, true).mode == ReuseMode::Reset);
    CHECK(plan_turn_reuse(l, ids({1, 2, 3, 4, 5}), 99, true).mode == ReuseMode::Reset);
}

// THE off-by-one probe. The cache and the prompt agree for three tokens; a checkpoint
// claiming four is exactly the mistake that decodes fluent wrong text, and it must be
// refused by the id-by-id comparison rather than accepted on a fingerprint.
TEST(a_checkpoint_one_token_too_long_resets) {
    const KvCacheLedger l = ledger_of({1, 2, 3, 8, 8});
    CHECK(plan_turn_reuse(l, ids({1, 2, 3, 4, 5}), 3, true).mode == ReuseMode::Restore);
    CHECK(plan_turn_reuse(l, ids({1, 2, 3, 4, 5}), 4, true).mode == ReuseMode::Reset);
}

// An empty cache has nothing to offer and must not claim a checkpoint it never took.
TEST(an_empty_ledger_resets) {
    const KvCacheLedger l;
    const TurnReuse r = plan_turn_reuse(l, ids({1, 2, 3}), 2, true);
    CHECK(r.mode == ReuseMode::Extend); // empty IS a prefix; prefill everything
    CHECK_EQ(r.prefill_from, std::size_t{0});
}

// --- images -------------------------------------------------------------------
//
// THE COLLISION THIS EXISTS TO STOP. Every image in a prompt is a run of the SAME
// `<|image_pad|>` id, so two different pictures are a byte-identical token run. An
// id-only comparison calls that a verified prefix and hands back a KV cache encoding the
// OTHER image -- and the model then describes a picture it was never shown, fluently,
// with nothing in the log to say so. It is the same shape as the stale-context failure
// S5.10 exists to prevent, arriving through a door that ids alone cannot see.

namespace {

constexpr TokenId kPad = 248056; // <|image_pad|> on this checkpoint

std::vector<ContentTag> tags(std::initializer_list<ContentTag> t) { return {t}; }

} // namespace

TEST(two_different_images_behind_identical_pads_do_not_reuse) {
    KvCacheLedger l;
    // Prompt: text, then four pad tokens carrying image A.
    const std::vector<TokenId> prompt = ids({1, 2, kPad, kPad, kPad, kPad});
    const std::vector<ContentTag> a = tags({0, 0, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA});
    l.append(std::span<const TokenId>(prompt), std::span<const ContentTag>(a));

    // The SAME prompt, same ids, different picture.
    const std::vector<ContentTag> b = tags({0, 0, 0xBBBB, 0xBBBB, 0xBBBB, 0xBBBB});
    const ReuseDecision d = l.plan_reuse(prompt, b);
    // Reuse stops at the first pad, not at the end of the run.
    CHECK_EQ(d.reusable, std::size_t{2});
    CHECK(d.divergent);

    // ...and the reason an id-only comparison could not see it: the two prompts are
    // byte-identical as token sequences. Nothing in `ids` distinguishes the pictures, so
    // the old ledger reported the whole run reusable and served image A's KV for image B.
    CHECK(l.ids() == prompt);

    // Dropping the tags entirely is divergence too, not a free pass: a prompt claiming no
    // image where the cache holds one is exactly as stale.
    CHECK(l.plan_reuse(prompt).divergent);
}

TEST(the_same_image_twice_still_reuses) {
    KvCacheLedger l;
    const std::vector<TokenId> prompt = ids({1, kPad, kPad, 9});
    const std::vector<ContentTag> t = tags({0, 0xC0FFEE, 0xC0FFEE, 0});
    l.append(std::span<const TokenId>(prompt), std::span<const ContentTag>(t));
    const ReuseDecision d = l.plan_reuse(prompt, t);
    CHECK_EQ(d.reusable, prompt.size());
    CHECK(!d.divergent);
}

// An image cannot be restored past either: the checkpoint is only usable when the
// verified prefix reaches it, and the tag walk shortens that prefix.
TEST(a_checkpoint_behind_a_changed_image_falls_through_to_reset) {
    KvCacheLedger l;
    const std::vector<TokenId> prompt = ids({1, kPad, kPad, 9, 10});
    const std::vector<ContentTag> a = tags({0, 0xAAAA, 0xAAAA, 0, 0});
    l.append(std::span<const TokenId>(prompt), std::span<const ContentTag>(a));
    const std::vector<ContentTag> b = tags({0, 0xBBBB, 0xBBBB, 0, 0});
    // Checkpoint at 4 sits AFTER the image, so a changed image invalidates it.
    const TurnReuse r = plan_turn_reuse(l, prompt, 4, true, b);
    CHECK(r.mode == ReuseMode::Reset);
    CHECK_EQ(r.prefill_from, std::size_t{0});
}

// Text-only runs must be untouched: no tags anywhere, and the fast path unchanged.
TEST(a_prompt_with_no_images_behaves_exactly_as_before) {
    KvCacheLedger l;
    l.append(ids({1, 2, 3}));
    const TurnReuse r = plan_turn_reuse(l, ids({1, 2, 3, 4, 5}), 2, true);
    CHECK(r.mode == ReuseMode::Extend);
    CHECK_EQ(r.prefill_from, std::size_t{3});
    for (const ContentTag t : l.tags()) {
        CHECK_EQ(t, ContentTag{0});
    }
}

// Truncation is what the speculative rollback uses, and it must take the tags with it --
// a ledger whose ids and tags had different lengths would compare past the end of one.
TEST(truncation_keeps_ids_and_tags_the_same_length) {
    KvCacheLedger l;
    const std::vector<TokenId> prompt = ids({1, kPad, kPad, 9});
    const std::vector<ContentTag> t = tags({0, 0xD00D, 0xD00D, 0});
    l.append(std::span<const TokenId>(prompt), std::span<const ContentTag>(t));
    l.truncate_last(2);
    CHECK_EQ(l.size(), std::size_t{2});
    CHECK_EQ(l.tags().size(), l.ids().size());
    l.truncate_to(1);
    CHECK_EQ(l.tags().size(), l.ids().size());
    l.clear();
    CHECK(l.tags().empty());
}

TEST(reuse_mode_str_matches_enumerators) {
    CHECK_EQ(std::string(reuse_mode_str(ReuseMode::Extend)), std::string("Extend"));
    CHECK_EQ(std::string(reuse_mode_str(ReuseMode::Restore)), std::string("Restore"));
    CHECK_EQ(std::string(reuse_mode_str(ReuseMode::Reset)), std::string("Reset"));
}

TEST(classify_reuse_reason_names_reset_causes) {
    CHECK_EQ(std::string(classify_reuse_reason(ReuseMode::Extend, 3, 3, 2, true, 5)),
             std::string(""));
    CHECK_EQ(std::string(classify_reuse_reason(ReuseMode::Restore, 5, 3, 3, true, 6)),
             std::string("checkpoint_restore"));
    CHECK_EQ(std::string(classify_reuse_reason(ReuseMode::Reset, 0, 0, 0, false, 4)),
             std::string("first_turn"));
    CHECK_EQ(std::string(classify_reuse_reason(ReuseMode::Reset, 4, 0, 0, true, 4)),
             std::string("no_checkpoint"));
    CHECK_EQ(std::string(classify_reuse_reason(ReuseMode::Reset, 5, 1, 4, true, 6)),
             std::string("ledger_mismatch"));
}

// --- HS1: honest fat-prefix suffix-only reuse --------------------------------
//
// The harness in lmp_diag reuse builds an explicit shared prefix of exactly P tokens,
// checkpoints at P, then appends S on turn 2. After turn 1 the ledger holds P plus the
// generated tail; turn 2's prompt is P+S and diverges after P. That must Restore to P
// so generate() reports prefill_reused_tokens == P (reuse_frac = 1.0 against P_shared).
//
// The 2026-09-03 failure (reused=11 every time) was checkpointing at a chat-template
// message boundary (~system header), not at the end of a fat shared body. The second
// test pins that false-friend so it cannot be mistaken for a pass.

TEST(hs1_fat_prefix_of_P_then_suffix_S_restores_exactly_P) {
    constexpr int P = 64;
    constexpr int S = 16;
    std::vector<TokenId> shared;
    shared.reserve(static_cast<std::size_t>(P));
    for (int i = 0; i < P; ++i) {
        shared.push_back(static_cast<TokenId>(1000 + i));
    }
    // Turn 1 ledger: shared prefix + generated junk past the checkpoint.
    std::vector<TokenId> ledger_ids = shared;
    ledger_ids.push_back(42);
    ledger_ids.push_back(43);
    KvCacheLedger ledger;
    ledger.append(ledger_ids);

    std::vector<TokenId> turn2 = shared;
    for (int i = 0; i < S; ++i) {
        turn2.push_back(static_cast<TokenId>(2000 + i));
    }

    const TurnReuse r = plan_turn_reuse(ledger, turn2, /*checkpoint_len=*/shared.size(),
                                       /*checkpoint_valid=*/true);
    CHECK(r.mode == ReuseMode::Restore);
    CHECK_EQ(r.prefill_from, shared.size());
    // generate() records prefill_reused_tokens = prefill_from, so reuse_frac vs P_shared
    // is exactly 1.0 when the algebra is honest.
    const double reuse_frac =
        static_cast<double>(r.prefill_from) / static_cast<double>(shared.size());
    CHECK(reuse_frac >= 0.95);
}

TEST(hs1_checkpoint_at_chat_header_only_reuses_header_not_fat_body) {
    // False-friend of the 2026-09-03 sweep: a short "header" checkpoint (~11 tokens)
    // with a long body after it. Restore then only reuses the header; the fat body is
    // re-prefilled. That must NOT be reported as a fat-prefix win.
    constexpr int kHeader = 11;
    constexpr int kBody = 64;
    std::vector<TokenId> turn1;
    for (int i = 0; i < kHeader + kBody; ++i) {
        turn1.push_back(static_cast<TokenId>(3000 + i));
    }
    std::vector<TokenId> ledger_ids = turn1;
    ledger_ids.push_back(7); // generated
    KvCacheLedger ledger;
    ledger.append(ledger_ids);

    // Turn 2 shares the header+body, then appends a short suffix -- but checkpoints only
    // at the header, the way a chat-template "start of last message" offset would.
    std::vector<TokenId> turn2 = turn1;
    turn2.push_back(4000);
    turn2.push_back(4001);

    const TurnReuse r = plan_turn_reuse(ledger, turn2, /*checkpoint_len=*/kHeader,
                                       /*checkpoint_valid=*/true);
    CHECK(r.mode == ReuseMode::Restore);
    CHECK_EQ(r.prefill_from, static_cast<std::size_t>(kHeader));
    // Against a claimed P_shared of the fat body+header, this is the ~11/75 failure mode.
    const double false_frac =
        static_cast<double>(r.prefill_from) /
        static_cast<double>(kHeader + kBody);
    CHECK(false_frac < 0.95);
}

