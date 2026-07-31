#include "src/platform/clock.hpp"

#include <chrono>

#include "tests/check.hpp"

using namespace std::chrono_literals;
using lmp::platform::ManualClock;
using lmp::platform::SystemClock;

TEST(manual_clock_advances_exactly) {
    ManualClock c;
    const auto t0 = c.mono();
    c.advance(1500ms);
    const auto t1 = c.mono();
    // Exact, not a tolerance band. This is the whole reason the seam exists: a timing
    // assertion that needs a sleep either flakes or eats the gate's five-minute budget.
    CHECK_EQ(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(),
             static_cast<long long>(1500));
}

TEST(manual_clock_advances_both_faces_together) {
    ManualClock c;
    const auto w0 = lmp::platform::to_ns(c.wall());
    const auto m0 = lmp::platform::to_us(c.mono());
    c.advance(2s);
    CHECK_EQ(lmp::platform::to_ns(c.wall()) - w0, static_cast<long long>(2'000'000'000));
    CHECK_EQ(lmp::platform::to_us(c.mono()) - m0, static_cast<long long>(2'000'000));
}

TEST(wall_clock_can_move_backwards_and_mono_cannot_follow) {
    // Real wall clocks step backwards under NTP correction. Any code that computes a
    // duration from wall() is a bug; this is the double that lets a test prove it.
    ManualClock c;
    c.advance(10s);
    const auto mono_before = c.mono();
    c.set_wall(1s);
    CHECK_EQ(lmp::platform::to_ns(c.wall()), static_cast<long long>(1'000'000'000));
    CHECK(c.mono() == mono_before);
}

TEST(system_clock_is_monotonic_on_mono_face) {
    SystemClock c;
    const auto a = c.mono();
    const auto b = c.mono();
    CHECK(b >= a);
    CHECK(lmp::platform::to_ns(c.wall()) > 0);
}
