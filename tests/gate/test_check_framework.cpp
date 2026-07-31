#include <string>

#include "tests/check.hpp"

// Falsifiability of the framework itself (S2.1.2).
//
// Every other test in this repo asserts through CHECK. If CHECK could not fail, all of
// them would report green forever and nobody would learn anything. So before trusting a
// single green elsewhere, prove that red is reachable: run checks that MUST fail and
// assert that the framework counted exactly them.
//
// This is the same discipline the verification choke point will apply to the agent's own
// green results in phase 7, applied here first because the tooling has to earn it before
// the agent does.

TEST(check_can_go_red) {
    EXPECT_FAILING_CHECKS(1, { CHECK(false); });
    EXPECT_FAILING_CHECKS(1, { CHECK_EQ(1, 2); });
    EXPECT_FAILING_CHECKS(3, {
        CHECK(false);
        CHECK_EQ(std::string("a"), std::string("b"));
        CHECK(1 > 2);
    });
}

TEST(check_stays_green_on_truth) {
    EXPECT_FAILING_CHECKS(0, {
        CHECK(true);
        CHECK_EQ(2 + 2, 4);
        CHECK_EQ(std::string("x"), std::string("x"));
    });
}

TEST(require_aborts_its_case_and_counts_once) {
    // REQUIRE returns from the enclosing function, so the lambda body after it must not
    // run. If it did, a REQUIRE that failed would keep executing code whose precondition
    // it just proved false -- which is a crash, not a test failure.
    int reached = 0;
    EXPECT_FAILING_CHECKS(1, {
        REQUIRE(false);
        reached = 1;
    });
    CHECK_EQ(reached, 0);
}

TEST(the_check_counter_actually_counts) {
    // A framework whose CHECK is a no-op reports a perfect suite. test_main.cpp fails any
    // binary reporting zero checks; this pins that the counter moves per check, so the
    // number that guard reads is a real one.
    const auto before = lmp::test::reg().checks;
    CHECK(true);
    CHECK(true);
    CHECK_EQ(lmp::test::reg().checks - before, static_cast<std::int64_t>(3));
}
