// A run id has to be an identity, and the one this replaced was a request counter.
//
// The bug these tests exist to prevent is not a crash. It is `run_id="2"` appearing on 63
// unrelated runs in the real event log, which makes resume splice strangers' turns into
// one conversation and makes 63 runs share one PCC session. Nothing about that is visible
// from inside a single run, so the properties have to be asserted directly.

#include <chrono>
#include <set>
#include <string>
#include <vector>

#include "src/platform/clock.hpp"
#include "src/surface/run_id.hpp"
#include "tests/check.hpp"

using lmp::surface::is_minted_run_id;
using lmp::surface::mint_run_id;

TEST(a_minted_run_id_has_the_declared_shape) {
    const lmp::platform::SystemClock clock;
    const std::string id = mint_run_id(clock);
    // `r-` + 16 hex of nanoseconds + `-` + 8 hex of entropy.
    CHECK_EQ(id.size(), std::size_t{27});
    CHECK(id.rfind("r-", 0) == 0);
    CHECK(is_minted_run_id(id));
}

TEST(minted_run_ids_do_not_repeat_even_when_minted_back_to_back) {
    const lmp::platform::SystemClock clock;
    // The failure mode being excluded is a clock whose resolution is coarser than the loop:
    // ids minted inside one tick share their whole time half and are separated only by the
    // random half. 2,000 in a tight loop is well inside one millisecond on this machine.
    std::set<std::string> seen;
    for (int i = 0; i < 2000; ++i) {
        seen.insert(mint_run_id(clock));
    }
    CHECK_EQ(seen.size(), std::size_t{2000});
}

TEST(a_frozen_clock_still_yields_distinct_ids) {
    // The sharpest version of the same question: if wall time cannot move at all, the id
    // must still be unique, because two sidecars can start in the same nanosecond and a
    // fake clock in a test certainly can.
    lmp::platform::ManualClock clock;
    std::set<std::string> seen;
    for (int i = 0; i < 500; ++i) {
        seen.insert(mint_run_id(clock));
    }
    CHECK_EQ(seen.size(), std::size_t{500});
}

TEST(minted_run_ids_sort_by_creation_time) {
    // String ordering IS time ordering, which is what lets "the newest unfinished run" be
    // a max() over ids instead of a join back to the log's timestamps.
    lmp::platform::ManualClock clock;
    const std::string first = mint_run_id(clock);
    clock.advance(std::chrono::nanoseconds{std::chrono::seconds{1}});
    const std::string second = mint_run_id(clock);
    clock.advance(std::chrono::nanoseconds{std::chrono::seconds{1}});
    const std::string third = mint_run_id(clock);
    CHECK(first < second);
    CHECK(second < third);
}

TEST(the_legacy_request_counter_ids_are_not_mistaken_for_minted_ones) {
    // Old logs are full of these, and they are exactly the colliding ones -- so anything
    // offering a resumable run has to be able to refuse them by inspection.
    for (const char* legacy : {"1", "2", "63", "", "r-", "run-2"}) {
        CHECK(!is_minted_run_id(legacy));
    }
    // And a well-formed id with one non-hex character is not one either: the check must
    // be a real parse, not a length test.
    std::string bad = mint_run_id(lmp::platform::SystemClock{});
    bad[5] = 'z';
    CHECK(!is_minted_run_id(bad));
}
