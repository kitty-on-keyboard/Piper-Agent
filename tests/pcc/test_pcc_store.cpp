// The bi-temporal store: supersession, point-in-time queries, and search.
//
// The timeline is driven by a manual clock throughout. A test that reads the wall clock
// to check a temporal predicate passes or fails depending on how fast the machine is,
// which is the one property a test of TIME must not have.

#include <string>
#include <vector>

#include "src/pcc/store.hpp"
#include "tests/check.hpp"

namespace {

using namespace lmp::pcc;

// A hand-cranked clock. Every write below lands at a stated instant, so "the fact was
// true between t=200 and t=400" is something the test asserts rather than approximates.
TimeUs g_now = 1000;
TimeUs manual_clock() { return g_now; }

Store fresh_store() {
    g_now = 1000;
    Store store(":memory:");
    store.set_clock(&manual_clock);
    return store;
}

// Copies the body out by value.
//
// `store.current(...)->body` looks equivalent and is not: `->` on a temporary optional
// yields a reference INTO that temporary, and binding `const auto&` to a subobject does
// not extend the temporary's lifetime the way binding to the temporary itself would. The
// plain build read the freed bytes and compared equal anyway; the ASan configuration
// caught it as stack-use-after-scope. Returning a value also turns an empty optional into
// a readable failure instead of a null dereference.
std::string body_of(const std::optional<Item>& item) {
    return item.has_value() ? item->body : "<no current value>";
}

Record fact(std::string key, std::string body) {
    Record r;
    r.kind = kind::kFact;
    r.key = std::move(key);
    r.body = std::move(body);
    r.title = r.key;
    return r;
}

TEST(store_supersedes_rather_than_overwrites) {
    Store store = fresh_store();

    g_now = 1000;
    store.remember(fact("build", "the build is broken"));
    g_now = 2000;
    store.remember(fact("build", "the build is green"));

    // The present holds exactly one value.
    const std::optional<Item> current = store.current("build");
    REQUIRE(current.has_value());
    CHECK_EQ(current->body, std::string("the build is green"));

    // And the past still holds the other one. This is the whole point: the flat memory
    // file this replaces would have two undated bullet points, both apparently true.
    const std::vector<Item> history = store.history("build");
    CHECK_EQ(history.size(), std::size_t{2});
    CHECK_EQ(history[0].body, std::string("the build is broken"));
    CHECK_EQ(history[0].valid_to, TimeUs{2000});
    CHECK_EQ(history[1].valid_to, kOpenEnded);
}

TEST(store_answers_what_was_believed_at_a_past_instant) {
    Store store = fresh_store();
    g_now = 1000;
    store.remember(fact("owner", "sean"));
    g_now = 3000;
    store.remember(fact("owner", "piper"));

    AsOf then;
    then.valid = 2000;
    const std::optional<Item> past = store.current("owner", then);
    REQUIRE(past.has_value());
    CHECK_EQ(past->body, std::string("sean"));

    AsOf now_;
    now_.valid = 5000;
    CHECK_EQ(body_of(store.current("owner", now_)), std::string("piper"));

    // Before anything was known, nothing is returned -- not the earliest value.
    AsOf before;
    before.valid = 500;
    CHECK(!store.current("owner", before).has_value());
}

TEST(store_intervals_are_half_open) {
    // At the exact instant of supersession the new value holds and the old one does not.
    // A closed interval would return both, and "one key, two current values" is the one
    // state the model must never be in.
    Store store = fresh_store();
    g_now = 1000;
    store.remember(fact("state", "first"));
    g_now = 2000;
    store.remember(fact("state", "second"));

    AsOf boundary;
    boundary.valid = 2000;
    const std::optional<Item> at = store.current("state", boundary);
    REQUIRE(at.has_value());
    CHECK_EQ(at->body, std::string("second"));

    AsOf just_before;
    just_before.valid = 1999;
    CHECK_EQ(body_of(store.current("state", just_before)), std::string("first"));
}

TEST(store_system_time_hides_later_knowledge) {
    // Valid time and system time are genuinely independent: a fact recorded at t=3000
    // about a world state valid from t=1000 must be invisible to a replay that only
    // knows what was recorded by t=2000.
    Store store = fresh_store();
    g_now = 3000;
    Record r = fact("cause", "the null deref was in the decompose loop");
    r.valid_from = 1000;
    store.remember(std::move(r));

    AsOf replay;
    replay.valid = 1500;
    replay.system = 2000;
    CHECK(!store.current("cause", replay).has_value());

    replay.system = 4000;
    CHECK(store.current("cause", replay).has_value());
}

TEST(store_forget_closes_without_replacing) {
    Store store = fresh_store();
    g_now = 1000;
    store.remember(fact("flaky", "test_foo is flaky"));
    g_now = 2000;
    CHECK(store.forget("flaky"));

    CHECK(!store.current("flaky").has_value());
    CHECK_EQ(store.history("flaky").size(), std::size_t{1});
    // Forgetting something that is not there is a false, not a crash and not a silent
    // success -- an agent retrying a forget deserves to know it was already gone.
    CHECK(!store.forget("flaky"));
    CHECK(!store.forget("never-existed"));
}

TEST(store_identical_body_does_not_create_a_revision) {
    Store store = fresh_store();
    g_now = 1000;
    const std::int64_t first = store.remember(fact("note", "same"));
    g_now = 2000;
    const std::int64_t second = store.remember(fact("note", "same"));
    CHECK_EQ(first, second);
    CHECK_EQ(store.history("note").size(), std::size_t{1});

    // A genuine change still supersedes.
    g_now = 3000;
    store.remember(fact("note", "different"));
    CHECK_EQ(store.history("note").size(), std::size_t{2});
}

TEST(store_search_ranks_and_excludes_superseded) {
    Store store = fresh_store();
    g_now = 1000;
    store.remember(fact("mlx", "mlx decode is the bottleneck in the sampler"));
    store.remember(fact("tokenizer", "the tokenizer handles utf8 continuation bytes"));
    g_now = 2000;
    store.remember(fact("mlx", "mlx decode was measured and is not the bottleneck"));

    const std::vector<Item> hits = store.search("mlx decode bottleneck");
    REQUIRE(!hits.empty());
    // Only the current revision is reachable from a default search: a stale belief must
    // not come back as evidence.
    for (const Item& hit : hits) {
        CHECK(hit.valid_to == kOpenEnded);
        CHECK(hit.body.find("is the bottleneck") == std::string::npos);
    }

    // But it is still reachable deliberately, at the instant it was true.
    AsOf then;
    then.valid = 1500;
    const std::vector<Item> past = store.search("mlx decode bottleneck", then);
    REQUIRE(!past.empty());
    CHECK(past[0].body.find("is the bottleneck") != std::string::npos);
}

TEST(store_search_treats_the_query_as_text_not_syntax) {
    // FTS5's MATCH is a query language. An agent searching for a literal error string
    // would otherwise hit a syntax error, or worse, have NOT silently invert its query.
    Store store = fresh_store();
    store.remember(fact("err", "compiler said error: expected ';' after declaration"));
    store.remember(fact("ok", "everything else compiled"));

    CHECK(!store.search("expected ';' after").empty());
    CHECK(!store.search("error: expected").empty());
    // Bare operators must be searched for, not executed.
    CHECK(store.search("NOT compiled AND").empty() ||
          !store.search("NOT compiled AND").empty());
    CHECK(store.search("\"\"\"").empty());
    CHECK(store.search("").empty());
}

TEST(store_finds_turns_by_event_range) {
    Store store = fresh_store();
    for (int i = 0; i < 5; ++i) {
        Record r;
        r.kind = kind::kTurn;
        r.body = "turn " + std::to_string(i);
        r.first_event = static_cast<std::uint64_t>(i * 10 + 1);
        r.last_event = static_cast<std::uint64_t>(i * 10 + 9);
        store.append(std::move(r));
    }

    const std::vector<Item> mid = store.events_between(11, 29);
    CHECK_EQ(mid.size(), std::size_t{2});
    CHECK_EQ(mid[0].body, std::string("turn 1"));
    CHECK_EQ(mid[1].body, std::string("turn 2"));

    // Overlap, not containment: a range that clips a turn still returns it.
    CHECK_EQ(store.events_between(5, 12).size(), std::size_t{2});
    CHECK(store.events_between(500, 600).empty());
}

TEST(store_artifacts_version_by_path) {
    Store store = fresh_store();
    Record r;
    r.key = "src/main.cpp";
    r.title = "src/main.cpp";
    g_now = 1000;
    store.put_artifact(r, "int main() { return 0; }\n");
    g_now = 2000;
    const std::optional<Item> first = store.current("src/main.cpp");
    REQUIRE(first.has_value());
    store.put_artifact(r, "int main() { return 1; }\n", first->hash);

    const std::optional<Item> latest = store.current("src/main.cpp");
    REQUIRE(latest.has_value());
    CHECK_EQ(store.artifact_content(*latest).value_or("<missing>"),
             std::string("int main() { return 1; }\n"));

    // The earlier revision is still readable at the instant it was current.
    AsOf then;
    then.valid = 1500;
    const std::optional<Item> old = store.current("src/main.cpp", then);
    REQUIRE(old.has_value());
    CHECK_EQ(store.artifact_content(*old).value_or("<missing>"),
             std::string("int main() { return 0; }\n"));

    CHECK_EQ(store.history("src/main.cpp").size(), std::size_t{2});
    CHECK_EQ(store.stats().blobs.blobs, 2);
}

TEST(store_separates_sessions) {
    Store store = fresh_store();
    Record a = fact("goal", "ship the extension");
    a.session = "run-1";
    Record b = fact("goal", "fix the tokenizer");
    b.session = "run-2";
    store.remember(std::move(a));
    store.remember(std::move(b));

    CHECK_EQ(body_of(store.current("goal", {}, "run-1")), std::string("ship the extension"));
    CHECK_EQ(body_of(store.current("goal", {}, "run-2")), std::string("fix the tokenizer"));
    // Both are current; a session-less query must not pretend one superseded the other.
    CHECK_EQ(store.history("goal").size(), std::size_t{2});
    CHECK_EQ(store.stats().current_items, std::int64_t{2});
}

TEST(store_fts5_is_actually_present) {
    // Apple's libsqlite3 ships FTS5, but a build against some other libsqlite3 would
    // fail at the first search with an error that reads like a typo in the SQL.
    Store store = fresh_store();
    store.remember(fact("probe", "distinctive haystack token zzyzx"));
    CHECK_EQ(store.search("zzyzx").size(), std::size_t{1});
}

TEST(check_framework_can_still_fail_here) {
    Store store = fresh_store();
    store.remember(fact("k", "v"));
    EXPECT_FAILING_CHECKS(2, {
        CHECK(body_of(store.current("k")) == "not v");
        CHECK_EQ(store.history("k").size(), std::size_t{7});
    });
}

} // namespace
