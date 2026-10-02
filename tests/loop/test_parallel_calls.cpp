// run_calls_concurrently (S9.1 amended). Two claims, and both need proving:
//
//   1. The calls actually OVERLAP. A "parallel" dispatch that happens to run things in
//      sequence passes every correctness test ever written for it, which is exactly how a
//      concurrency change ships doing nothing. So overlap is asserted directly rather than
//      inferred from a stopwatch: every call waits at a rendezvous for all the others,
//      which only a dispatch with all of them in flight at once can get through. (This
//      used to compare wall-clock time against a serial baseline. A loaded sanitizer
//      runner turned that comparison red on correct code.)
//
//   2. Results come back in CALL ORDER regardless of completion order. The work below
//      finishes deliberately backwards -- the first call is the slowest -- so an
//      implementation that collects results as they arrive fails here.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "src/loop/parallel_calls.hpp"

#include "tests/check.hpp"

using lmp::loop::run_calls_concurrently;
using lmp::tools::ToolResult;

namespace {

constexpr auto kUnit = std::chrono::milliseconds(60);

// Each call checks in, then waits until every call has checked in. All of them get
// through only if all of them are in flight at once -- the property under test. A serial
// dispatch runs the first call alone, and it waits for calls that cannot start until it
// returns.
//
// The deadline turns that deadlock into a failure instead of a hang. It is not a
// performance budget: a correct dispatch gets through the moment its last thread starts,
// so a slow or loaded machine, or a sanitizer build, makes it later but never wrong.
class Rendezvous {
public:
    Rendezvous(std::size_t expected, std::chrono::steady_clock::duration patience)
        : expected_(expected), deadline_(std::chrono::steady_clock::now() + patience) {}

    // True when this call saw every call arrive before the deadline.
    bool arrive() {
        std::unique_lock<std::mutex> lock(mu_);
        ++arrived_;
        cv_.notify_all();
        return cv_.wait_until(lock, deadline_, [this] { return arrived_ == expected_; });
    }

private:
    const std::size_t expected_;
    const std::chrono::steady_clock::time_point deadline_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::size_t arrived_ = 0;
};

} // namespace

TEST(concurrent_calls_overlap_instead_of_queueing) {
    const std::vector<std::size_t> indices = {0, 1, 2, 3};
    Rendezvous all_four(indices.size(), std::chrono::seconds(10));
    std::atomic<int> met{0};

    const std::vector<ToolResult> results =
        run_calls_concurrently(indices, [&all_four, &met](std::size_t i) {
            if (all_four.arrive()) {
                met.fetch_add(1, std::memory_order_relaxed);
            }
            return ToolResult::okay("call " + std::to_string(i));
        });

    CHECK_EQ(results.size(), indices.size());
    // Every call saw the other three in flight. A pool capped below four lets only its
    // last batch through; a serial loop lets only the final call through.
    CHECK_EQ(met.load(std::memory_order_relaxed), 4);
}

TEST(the_overlap_check_fails_a_serial_dispatch) {
    // The check above is worth something only if sequential execution cannot pass it
    // (S2.1.2), so run the same rendezvous through a plain loop. The first call waits
    // alone until the deadline and every later one finds it already passed; only the
    // last call, which completes the count itself, gets through. The short patience is
    // safe because this outcome does not depend on how long anything takes.
    Rendezvous all_four(4, std::chrono::milliseconds(50));
    int met = 0;
    for (int i = 0; i < 4; ++i) {
        if (all_four.arrive()) {
            ++met;
        }
    }
    CHECK_EQ(met, 1);
}

TEST(results_are_indexed_to_their_call_not_to_completion_order) {
    const std::vector<std::size_t> indices = {0, 1, 2, 3};
    // Backwards: index 0 takes longest, index 3 returns almost immediately.
    const auto work = [](std::size_t i) {
        std::this_thread::sleep_for(kUnit * (4 - i));
        return ToolResult::okay("call " + std::to_string(i));
    };

    const std::vector<ToolResult> results = run_calls_concurrently(indices, work);

    REQUIRE(results.size() == 4);
    for (std::size_t i = 0; i < 4; ++i) {
        CHECK_EQ(results[i].summary, "call " + std::to_string(i));
    }
}

TEST(the_index_list_is_what_is_run_not_a_range) {
    // The caller passes only the ELIGIBLE calls, which are a subset of the turn's calls
    // and need not be contiguous. Results must line up with that subset, not with 0..n.
    const std::vector<std::size_t> indices = {1, 3};
    std::atomic<int> ran{0};
    const std::vector<ToolResult> results =
        run_calls_concurrently(indices, [&ran](std::size_t i) {
            ran.fetch_add(1, std::memory_order_relaxed);
            return ToolResult::okay("call " + std::to_string(i));
        });

    CHECK_EQ(ran.load(std::memory_order_relaxed), 2);
    REQUIRE(results.size() == 2);
    CHECK_EQ(results[0].summary, std::string("call 1"));
    CHECK_EQ(results[1].summary, std::string("call 3"));
}

TEST(a_single_call_needs_no_thread_and_an_empty_list_does_nothing) {
    std::atomic<int> ran{0};
    const std::vector<ToolResult> none =
        run_calls_concurrently({}, [&ran](std::size_t) {
            ran.fetch_add(1, std::memory_order_relaxed);
            return ToolResult::okay("never");
        });
    CHECK(none.empty());
    CHECK_EQ(ran.load(std::memory_order_relaxed), 0);

    const std::vector<ToolResult> one =
        run_calls_concurrently({7}, [](std::size_t i) {
            return ToolResult::okay("call " + std::to_string(i));
        });
    REQUIRE(one.size() == 1);
    CHECK_EQ(one[0].summary, std::string("call 7"));
}
