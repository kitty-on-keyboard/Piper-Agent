// run_calls_concurrently (S9.1 amended). Two claims, and both need proving:
//
//   1. The calls actually OVERLAP. A "parallel" dispatch that happens to run things in
//      sequence passes every correctness test ever written for it, which is exactly how a
//      concurrency change ships doing nothing. So the timing assertion is the point, not
//      decoration -- it is measured against a serial baseline taken in the same test, on
//      the same machine, rather than against a constant somebody guessed.
//
//   2. Results come back in CALL ORDER regardless of completion order. The work below
//      finishes deliberately backwards -- the first call is the slowest -- so an
//      implementation that collects results as they arrive fails here.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

#include "src/loop/parallel_calls.hpp"

#include "tests/check.hpp"

using lmp::loop::run_calls_concurrently;
using lmp::tools::ToolResult;

namespace {

constexpr auto kUnit = std::chrono::milliseconds(60);

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
        .count();
}

} // namespace

TEST(concurrent_calls_overlap_instead_of_queueing) {
    const std::vector<std::size_t> indices = {0, 1, 2, 3};
    const auto sleep_work = [](std::size_t i) {
        std::this_thread::sleep_for(kUnit);
        return ToolResult::okay("call " + std::to_string(i));
    };

    // The serial baseline, measured here so the comparison survives a slow or loaded
    // machine: whatever four sequential sleeps cost, four concurrent ones must cost far
    // less. A hardcoded millisecond threshold would be a flake generator.
    const auto t_serial = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < indices.size(); ++i) {
        (void)sleep_work(i);
    }
    const double serial_ms = ms_since(t_serial);

    const auto t_par = std::chrono::steady_clock::now();
    const std::vector<ToolResult> results = run_calls_concurrently(indices, sleep_work);
    const double parallel_ms = ms_since(t_par);

    CHECK_EQ(results.size(), indices.size());
    // Four at once should land near one unit, not four. Half the serial time is a wide
    // margin that still cannot be met by sequential execution.
    CHECK(parallel_ms < serial_ms / 2.0);
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
