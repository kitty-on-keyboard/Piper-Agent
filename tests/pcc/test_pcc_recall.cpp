// Budgeted recall, and the rehydration that makes compaction non-destructive.
//
// The budget is the contract. Every check that matters here is a check that recall()
// returned LESS than it was allowed to, because a retrieval that overruns its budget
// does not degrade the prompt gracefully -- it silently displaces the mission.

#include <string>
#include <vector>

#include "src/pcc/recall.hpp"
#include "src/pcc/store.hpp"
#include "tests/check.hpp"

namespace {

using namespace lmp::pcc;

TimeUs g_now = 1000;
TimeUs manual_clock() { return g_now; }

Store fresh_store() {
    g_now = 1000;
    Store store(":memory:");
    store.set_clock(&manual_clock);
    return store;
}

std::int64_t add_fact(Store& store, const std::string& key, const std::string& body) {
    Record r;
    r.kind = kind::kFact;
    r.key = key;
    r.title = key;
    r.body = body;
    return store.remember(std::move(r));
}

std::int64_t add_turn(Store& store, std::uint64_t event, const std::string& body) {
    Record r;
    r.kind = kind::kTurn;
    r.body = body;
    r.first_event = event;
    r.last_event = event;
    return store.append(std::move(r));
}

TEST(recall_returns_nothing_for_an_empty_store) {
    Store store = fresh_store();
    const Recall out = recall(store, {"anything", "", 500, {}, 60, ""});
    CHECK(out.text.empty());
    CHECK_EQ(out.tokens_used, std::size_t{0});
    CHECK_EQ(out.entries.size(), std::size_t{0});
}

TEST(recall_stays_within_its_token_budget) {
    Store store = fresh_store();
    // Thirty facts, each far larger than the budget can hold in aggregate.
    for (int i = 0; i < 30; ++i) {
        add_fact(store, "fact-" + std::to_string(i),
                 "sampler throughput note " + std::to_string(i) + " " +
                     std::string(600, 'x'));
        g_now += 100;
    }

    RecallRequest req;
    req.query = "sampler throughput note";
    req.token_budget = 300;
    const Recall out = recall(store, req);

    CHECK(out.tokens_used <= req.token_budget);
    CHECK(estimate_tokens(out.text) <= req.token_budget);
    // It got something, and it told the caller what it withheld.
    CHECK(out.included > 0);
    CHECK(out.pointers_only > 0);
    CHECK(out.text.find("not included") != std::string::npos);
    CHECK(out.text.find("pcc://item/") != std::string::npos);
}

TEST(recall_keeps_packing_past_an_oversized_entry) {
    // One enormous result in the middle of the ranking must not evict everything behind
    // it. Stopping at the first overflow is the obvious implementation and it wastes
    // most of the budget.
    Store store = fresh_store();
    add_fact(store, "small-1", "widget alpha");
    g_now += 100;
    add_fact(store, "huge", "widget " + std::string(8000, 'y'));
    g_now += 100;
    add_fact(store, "small-2", "widget beta");

    RecallRequest req;
    req.query = "widget";
    req.token_budget = 200;
    const Recall out = recall(store, req);

    CHECK(out.tokens_used <= req.token_budget);
    CHECK_EQ(out.included, std::size_t{2});
    CHECK_EQ(out.pointers_only, std::size_t{1});
    CHECK(out.text.find("widget alpha") != std::string::npos);
    CHECK(out.text.find("widget beta") != std::string::npos);
}

TEST(recall_prefers_the_fresher_of_two_equally_relevant_facts) {
    // Both match the query identically, so BM25 cannot separate them and the recency
    // list decides. This is the check that would have caught the cook-off entrant whose
    // freshness factor was applied with the wrong sign.
    Store store = fresh_store();
    g_now = 1000;
    add_fact(store, "old", "gated delta rule applies");
    g_now = 9000;
    add_fact(store, "new", "gated delta rule applies");

    RecallRequest req;
    req.query = "gated delta rule";
    req.token_budget = 4000;
    const Recall out = recall(store, req);

    REQUIRE(out.entries.size() == 2);
    CHECK_EQ(out.entries[0].item.key, std::string("new"));
    CHECK(out.entries[0].score >= out.entries[1].score);
}

TEST(recall_marks_a_superseded_entry_when_asked_for_the_past) {
    Store store = fresh_store();
    g_now = 1000;
    add_fact(store, "verdict", "speculative decoding is a clear win");
    g_now = 5000;
    add_fact(store, "verdict", "speculative decoding measured slower on this model");

    RecallRequest req;
    req.query = "speculative decoding";
    req.token_budget = 4000;
    req.as_of.valid = 2000;
    const Recall out = recall(store, req);

    REQUIRE(!out.entries.empty());
    CHECK(out.text.find("clear win") != std::string::npos);
    // The header has to say so. A past belief rendered identically to a current one is
    // how a replay convinces itself a stale conclusion is live.
    CHECK(out.text.find("SUPERSEDED") != std::string::npos);
}

TEST(recall_entries_carry_a_resolvable_uri) {
    Store store = fresh_store();
    const std::int64_t id = add_fact(store, "k", "unmistakable marker phrase");

    RecallRequest req;
    req.query = "unmistakable marker";
    const Recall out = recall(store, req);
    REQUIRE(!out.entries.empty());
    CHECK_EQ(out.entries[0].uri, "pcc://item/" + std::to_string(id));

    // The URI resolves back to the item it names -- the round trip the MCP resource
    // handler depends on.
    const std::optional<Item> fetched = store.get(id);
    REQUIRE(fetched.has_value());
    CHECK_EQ(item_uri(*fetched), out.entries[0].uri);
}

TEST(rehydrate_recovers_the_turns_behind_a_compacted_span) {
    // The scenario src/context creates on every trim: turns 1-6 are summarized into one
    // line each and erased from the prompt. Their full text has to still be reachable
    // from the event range the summary printed.
    Store store = fresh_store();
    for (std::uint64_t i = 1; i <= 6; ++i) {
        add_turn(store, i,
                 "turn " + std::to_string(i) +
                     ": read_file src/model/sampler.cpp -> the temperature clamp is at "
                     "line 88 and it rounds toward zero");
        g_now += 100;
    }

    const Recall out = rehydrate(store, 1, 6, 4000);
    CHECK_EQ(out.included, std::size_t{6});
    CHECK_EQ(out.pointers_only, std::size_t{0});
    CHECK(out.text.find("temperature clamp") != std::string::npos);
    // Newest first, so a tight budget keeps the turns nearest the boundary.
    CHECK(out.text.find("turn 6") < out.text.find("turn 1"));
}

TEST(rehydrate_under_a_tight_budget_keeps_the_newest_turns) {
    Store store = fresh_store();
    for (std::uint64_t i = 1; i <= 10; ++i) {
        add_turn(store, i, "turn " + std::to_string(i) + " " + std::string(400, 'z'));
        g_now += 100;
    }

    const Recall out = rehydrate(store, 1, 10, 250);
    CHECK(out.tokens_used <= std::size_t{250});
    CHECK(out.included > 0);
    CHECK(out.included < 10);
    CHECK(out.text.find("turn 10") != std::string::npos);
    CHECK(out.text.find("turn 1 ") == std::string::npos);
}

TEST(rehydrate_of_an_unknown_range_is_empty_not_everything) {
    // The failure that matters: a range query that silently drops its predicate returns
    // the whole store, which under a budget looks like a plausible answer.
    Store store = fresh_store();
    add_turn(store, 5, "the only turn");
    const Recall out = rehydrate(store, 900, 950, 4000);
    CHECK(out.text.empty());
    CHECK_EQ(out.entries.size(), std::size_t{0});
}

TEST(recall_accepts_an_injected_token_counter) {
    // The sidecar passes the real tokenizer. A counter that reports double must halve
    // what fits -- if the budget were being applied to the default estimate regardless,
    // this would not change.
    Store store = fresh_store();
    for (int i = 0; i < 10; ++i) {
        add_fact(store, "f" + std::to_string(i), "marker " + std::string(100, 'q'));
        g_now += 100;
    }

    RecallRequest req;
    req.query = "marker";
    req.token_budget = 300;
    const Recall cheap = recall(store, req, estimate_tokens);
    const Recall dear = recall(store, req, [](std::string_view s) {
        return estimate_tokens(s) * 4;
    });
    CHECK(dear.included < cheap.included);
    CHECK(dear.tokens_used <= req.token_budget);
}

TEST(check_framework_can_still_fail_here) {
    Store store = fresh_store();
    add_fact(store, "k", "v");
    EXPECT_FAILING_CHECKS(2, {
        const Recall out = recall(store, {"v", "", 500, {}, 60, ""});
        CHECK(out.entries.empty());
        CHECK_EQ(out.tokens_used, std::size_t{99999});
    });
}

} // namespace

// The live run's own mission is T0 of its own prompt -- always there, never compacted --
// and it is a near-perfect BM25 match for any query drawn from it, so it ranks first and
// spends a slice of the recall budget quoting the model's instructions back at it.
//
// MEASURED on the first run with a populated store: ~60 tokens of a 1500-token budget.
TEST(recall_does_not_hand_back_the_live_sessions_own_mission) {
    Store store = fresh_store();

    const auto mission_row = [&](const char* session, const char* body) {
        Record r;
        r.session = session;
        r.kind = kind::kTurn;
        r.title = title::kMission;
        r.body = body;
        return store.append(std::move(r));
    };
    mission_row("live", "you were told: build a resource monitor in SwiftUI");
    mission_row("older", "you were told: build a resource monitor in AppKit");
    Record turn;
    turn.session = "live";
    turn.kind = kind::kTurn;
    turn.title = "read_file";
    turn.body = "resource monitor sampling lives in HostStatsService";
    store.append(std::move(turn));

    RecallRequest req;
    req.query = "resource monitor";
    req.token_budget = 500;

    // Unfiltered, all three match and the live mission is among them.
    const Recall all = recall(store, req);
    CHECK(all.text.find("SwiftUI") != std::string::npos);

    req.suppress_mission_of = "live";
    const Recall filtered = recall(store, req);
    // THE ASSERTION: this session's mission is gone...
    CHECK(filtered.text.find("SwiftUI") == std::string::npos);
    // ...while an EARLIER mission in the same workspace is genuinely new information and
    // stays, and so does every other row of the live session -- including the turns a long
    // run has since compacted away, which are the whole reason this store exists.
    CHECK(filtered.text.find("AppKit") != std::string::npos);
    CHECK(filtered.text.find("HostStatsService") != std::string::npos);
    CHECK(filtered.entries.size() == all.entries.size() - 1);
}
