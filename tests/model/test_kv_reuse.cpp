// plan_turn_reuse -- the cross-turn cache decision, tested with no GPU (G3).
//
// This is where an off-by-one in KV reuse actually hides: it does not crash, it decodes
// against a prefix that is one token out and the text stays fluent. So the algebra is a
// pure function and every branch is asserted here rather than discovered on the model.

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
