#include <cstdio>

#include "tests/check.hpp"

// Linked into every test binary by lmp_add_test().
//
// Prints the check count, not just pass/fail. A suite that reports "OK" while running
// zero checks is v1's `ctest -E realmodel` failure wearing a different hat: the number
// looked fine and counted nothing. If this line ever prints "0 checks", that is the
// finding.
int main() {
    lmp::test::Registry& r = lmp::test::reg();
    for (const lmp::test::Case& c : r.cases) {
        std::fprintf(stderr, "- %s\n", c.name);
        c.fn();
    }
    std::fprintf(stderr, "%s: %d case(s), %lld check(s), %lld failure(s)\n",
                 r.failures == 0 ? "PASS" : "FAIL", static_cast<int>(r.cases.size()),
                 static_cast<long long>(r.checks), static_cast<long long>(r.failures));
    if (r.cases.empty() || r.checks == 0) {
        std::fprintf(stderr, "FAIL: a test binary that asserts nothing is not a test\n");
        return 2;
    }
    return r.failures == 0 ? 0 : 1;
}
