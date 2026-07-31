// S15.1: "Assertions enabled in test builds -- v1 compiled 317 asserts out with -DNDEBUG
// for two months."
//
// Two independent proofs, because the failure was silent and cost two months:
//
//   1. A compile-time refusal. If NDEBUG is defined, this TU does not build, and the
//      error names cmake/LmpAssertions.cmake so the fix is one hop away.
//   2. A runtime proof by intervention (S19.3). Fork a child, fail an assert with a
//      value the optimiser cannot fold, and require that the child died of SIGABRT.
//      Checking the macro is checking a proxy; killing a process is checking the thing.

#ifdef NDEBUG
#error "NDEBUG is defined in a test build. Assertions are compiled out and every assert() \
in this repo is a comment. cmake/LmpAssertions.cmake strips -DNDEBUG from every \
configuration -- if you are seeing this, that module is not being included, or something \
downstream re-added the define."
#endif

#include <sys/wait.h>
#include <unistd.h>

#include <cassert>
#include <csignal>
#include <cstdlib>

#include "tests/check.hpp"

TEST(assert_actually_aborts_the_process) {
    const pid_t pid = ::fork();
    REQUIRE(pid >= 0);

    if (pid == 0) {
        // Child. Silence the abort message so a passing gate is not full of scary text.
        ::freopen("/dev/null", "w", stderr);
        volatile int one = 1;
        assert(one == 2);
        // Reached only if assert() expanded to nothing. Exit 0 so the parent's SIGABRT
        // assertion fails loudly rather than the child hanging.
        ::_exit(0);
    }

    int status = 0;
    REQUIRE(::waitpid(pid, &status, 0) == pid);
    CHECK(WIFSIGNALED(status));
    if (WIFSIGNALED(status)) {
        CHECK_EQ(WTERMSIG(status), SIGABRT);
    }
    CHECK(!WIFEXITED(status));
}
