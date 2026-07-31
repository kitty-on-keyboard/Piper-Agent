#pragma once
//
// A ~120-line check framework. Deliberately not a dependency.
//
// The one property it must have, and the reason it is not three asserts in a main():
// a green result counts only when the check has been PROVEN capable of going red
// (S2.1.2). So `expect_failure()` below lets a test run a check it knows will fail and
// assert that the framework noticed. tests/gate/test_check_framework.cpp uses it to
// prove that CHECK can fail at all. Without that, every other test in the repo is
// asserting against a framework nobody has falsified, and a framework whose CHECK is
// a no-op reports a perfect suite.
//
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace lmp::test {

using TestFn = void (*)();

struct Case {
    const char* name;
    TestFn fn;
};

struct Registry {
    std::vector<Case> cases;
    std::int64_t checks = 0;
    std::int64_t failures = 0;
    // When >= 0, failures are counted here and suppressed from the report instead of
    // failing the run. Only expect_failure() sets it.
    std::int64_t expected_sink = -1;
};

inline Registry& reg() {
    static Registry r;
    return r;
}

struct Register {
    Register(const char* name, TestFn fn) { reg().cases.push_back({name, fn}); }
};

inline void record_failure(const char* file, int line, const std::string& msg) {
    Registry& r = reg();
    if (r.expected_sink >= 0) {
        ++r.expected_sink;
        return;
    }
    ++r.failures;
    std::fprintf(stderr, "  FAIL %s:%d: %s\n", file, line, msg.c_str());
}

// Runs `fn`, expecting exactly `n` checks inside it to fail. Those failures do not
// fail the run; a mismatch does.
template <class F>
void expect_failure(const char* file, int line, int n, F&& fn) {
    Registry& r = reg();
    const std::int64_t saved = r.expected_sink;
    r.expected_sink = 0;
    fn();
    const std::int64_t seen = r.expected_sink;
    r.expected_sink = saved;
    ++r.checks;
    if (seen != n) {
        record_failure(file, line,
                       "expected " + std::to_string(n) + " failing check(s), saw " +
                           std::to_string(seen));
    }
}

inline std::string show(const std::string& v) { return "\"" + v + "\""; }
inline std::string show(std::string_view v) { return "\"" + std::string(v) + "\""; }
inline std::string show(const char* v) { return v ? "\"" + std::string(v) + "\"" : "(null)"; }
inline std::string show(bool v) { return v ? "true" : "false"; }
template <class T>
std::string show(const T& v) {
    return std::to_string(v);
}

} // namespace lmp::test

#define LMP_CAT_(a, b) a##b
#define LMP_CAT(a, b) LMP_CAT_(a, b)

#define TEST(name)                                                              \
    static void name();                                                         \
    static const ::lmp::test::Register LMP_CAT(lmp_reg_, name){#name, name};    \
    static void name()

#define CHECK(expr)                                                             \
    do {                                                                        \
        ++::lmp::test::reg().checks;                                            \
        if (!(expr)) {                                                          \
            ::lmp::test::record_failure(__FILE__, __LINE__, "CHECK(" #expr ")"); \
        }                                                                       \
    } while (false)

#define CHECK_EQ(a, b)                                                          \
    do {                                                                        \
        ++::lmp::test::reg().checks;                                            \
        const auto& lmp_a_ = (a);                                               \
        const auto& lmp_b_ = (b);                                               \
        if (!(lmp_a_ == lmp_b_)) {                                              \
            ::lmp::test::record_failure(                                        \
                __FILE__, __LINE__,                                             \
                std::string(#a " == " #b " -- got ") + ::lmp::test::show(lmp_a_) + \
                    " vs " + ::lmp::test::show(lmp_b_));                        \
        }                                                                       \
    } while (false)

// Aborts the current test case; use when continuing would crash rather than just fail.
#define REQUIRE(expr)                                                           \
    do {                                                                        \
        ++::lmp::test::reg().checks;                                            \
        if (!(expr)) {                                                          \
            ::lmp::test::record_failure(__FILE__, __LINE__,                     \
                                        "REQUIRE(" #expr ") -- aborting case"); \
            return;                                                             \
        }                                                                       \
    } while (false)

#define EXPECT_FAILING_CHECKS(n, ...) \
    ::lmp::test::expect_failure(__FILE__, __LINE__, (n), [&]() __VA_ARGS__)
