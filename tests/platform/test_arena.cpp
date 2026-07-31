#include "src/platform/arena.hpp"

#include <string>
#include <vector>

#include "tests/check.hpp"

using lmp::platform::Arena;

TEST(arena_reports_what_it_handed_out) {
    Arena a(1024, 4096);
    CHECK_EQ(a.used(), std::size_t{0});
    CHECK_EQ(a.cap(), std::size_t{4096});
    CHECK(!a.overflowed());

    void* p = a.resource()->allocate(100, alignof(std::max_align_t));
    CHECK(p != nullptr);
    CHECK_EQ(a.used(), std::size_t{100});
    CHECK_EQ(a.high_water(), std::size_t{100});
}

TEST(arena_reset_rewinds_used_but_keeps_high_water) {
    Arena a(1024, 4096);
    (void)a.resource()->allocate(300, 8);
    CHECK_EQ(a.used(), std::size_t{300});
    a.reset();
    CHECK_EQ(a.used(), std::size_t{0});
    // high_water survives: the peak is a property of the run, not of the current scope.
    CHECK_EQ(a.high_water(), std::size_t{300});
}

TEST(arena_overflow_is_sticky_and_does_not_refuse) {
    Arena a(64, 128);
    void* p = a.resource()->allocate(200, 8);
    // The allocation SUCCEEDS. cap is a high-water line, not a refusal -- an arena that
    // aborted a 300-second run over a mis-set budget would be worse than the allocation.
    CHECK(p != nullptr);
    CHECK(a.overflowed());

    a.reset();
    // Sticky across reset: a run that blew the budget once did blow the budget.
    CHECK(a.overflowed());
    CHECK_EQ(a.used(), std::size_t{0});
}

TEST(arena_backs_a_pmr_container) {
    Arena a(4096, 1 << 20);
    {
        std::pmr::vector<int> v(a.resource());
        for (int i = 0; i < 500; ++i) {
            v.push_back(i);
        }
        CHECK_EQ(v.size(), std::size_t{500});
        CHECK_EQ(v[499], 499);
    }
    CHECK(a.used() > 0);
    CHECK(!a.overflowed());
}

TEST(arena_used_is_a_floor_not_an_estimate) {
    // used() counts REQUESTED bytes, before the monotonic resource's alignment padding.
    // The header promises that direction of error; this pins it, so a future change that
    // makes used() an overstatement fails here rather than quietly inverting the
    // guarantee the header makes.
    Arena a(4096, 1 << 20);
    (void)a.resource()->allocate(3, 64);
    (void)a.resource()->allocate(3, 64);
    CHECK_EQ(a.used(), std::size_t{6});
}
