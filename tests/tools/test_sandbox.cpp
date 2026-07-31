// Phase 5's exit criterion, verbatim from S17: "A command that writes outside root,
// opens a socket, or spins forever is stopped -- PROVEN BY A TEST THAT ATTEMPTS EACH."
// These tests attack the sandbox and pass only when the attack fails.

#include <csignal>
#include <unistd.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "src/platform/fs.hpp"
#include "src/tools/sandbox.hpp"

#include "tests/check.hpp"

using namespace lmp::tools;

namespace {

std::string temp_dir() {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = std::string(base ? base : "/tmp") + "/lmp_sbx_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const char* made = ::mkdtemp(buf.data());
    return made ? std::string(made) : std::string();
}

ExecLimits limits(int wall_seconds) {
    return ExecLimits{wall_seconds, wall_seconds, 2LL << 30, 256, 64, 1U << 20};
}

} // namespace

TEST(a_write_outside_the_root_is_stopped) {
    const std::string root = temp_dir();
    const std::string outside = temp_dir();
    REQUIRE(!root.empty());
    REQUIRE(!outside.empty());

    const std::string target = outside + "/escape.txt";
    const ExecutionGrant grant = grant_execution(SandboxTier::T1_Seatbelt);
    const ExecOutcome o = run_sandboxed(grant, "echo pwned > '" + target + "'", root,
                                        root, limits(10));

    // The command ran and FAILED -- the jail, not a refusal, stopped it.
    CHECK(o.status == Status::ToolError);
    const lmp::platform::FileContents f = lmp::platform::read_file_whole(target, 1024);
    CHECK(f.status == lmp::platform::FsStatus::NotFound);
}

TEST(a_write_inside_the_root_succeeds) {
    // The falsification half: if this failed too, the previous test would be proving
    // "the sandbox breaks everything", not "the jail holds".
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    const ExecutionGrant grant = grant_execution(SandboxTier::T1_Seatbelt);
    const ExecOutcome o =
        run_sandboxed(grant, "echo ok > inside.txt", root, root, limits(10));
    CHECK(o.status == Status::Ok);
    const lmp::platform::FileContents f =
        lmp::platform::read_file_whole(root + "/inside.txt", 1024);
    CHECK(f.ok());
    CHECK_EQ(f.bytes, std::string("ok\n"));
}

TEST(opening_a_socket_is_stopped) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    const ExecutionGrant grant = grant_execution(SandboxTier::T1_Seatbelt);
    // nc -z probes a port. Under (deny network*) it must fail -- whether or not
    // anything is listening, which is the point: the denial is in the profile, not in
    // the observed effect (S7.4).
    const ExecOutcome o = run_sandboxed(
        grant, "nc -z -G 2 127.0.0.1 22 && echo CONNECTED", root, root, limits(10));
    CHECK(o.status == Status::ToolError);
    CHECK(o.output.find("CONNECTED") == std::string::npos);
}

TEST(a_command_that_spins_forever_is_stopped) {
    // Two independent stoppers, and which one fires depends on whether the runaway
    // burns CPU. Measured rather than assumed: a busy loop trips RLIMIT_CPU first and
    // dies of SIGXCPU (24), never reaching the wall clock. The requirement is that it
    // is STOPPED; naming only one mechanism would have made this test pass for the
    // wrong reason -- or, as it did on the first run, fail for one.
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    const ExecutionGrant grant = grant_execution(SandboxTier::T1_Seatbelt);
    const ExecOutcome o =
        run_sandboxed(grant, "while true; do :; done", root, root, limits(1));
    CHECK(o.status != Status::Ok);
    CHECK(o.signalled);
    CHECK_EQ(o.signal, SIGXCPU);
}

TEST(a_command_that_sleeps_forever_is_killed_at_the_wall_clock) {
    // The independent wall-clock killer (S7.3): this one burns no CPU, so RLIMIT_CPU
    // never fires and only the wall clock can stop it. An unattended run cannot afford
    // a command that never returns even if it damages nothing.
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    ExecLimits lim = limits(2);
    lim.cpu_seconds = 60; // deliberately generous, so the wall clock is the only stopper
    const ExecutionGrant grant = grant_execution(SandboxTier::T1_Seatbelt);
    const ExecOutcome o = run_sandboxed(grant, "sleep 30", root, root, lim);
    CHECK(o.status == Status::Timeout);
    CHECK(o.wall_clock_killed);
}

TEST(t0_refuses_and_t2_refuses_rather_than_downgrades) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());

    const ExecOutcome t0 = run_sandboxed(grant_execution(SandboxTier::T0_NoExec),
                                         "echo hi", root, root, limits(5));
    CHECK(t0.status == Status::Refused);

    // T2 is not wired: it must REFUSE, not silently run in T1 -- a silent downgrade is
    // v1's unsafe_host default wearing a new name (S7.2, S13).
    const ExecOutcome t2 = run_sandboxed(grant_execution(SandboxTier::T2_Container),
                                         "echo hi", root, root, limits(5));
    CHECK(t2.status == Status::Refused);
    CHECK(t2.output.find("container") != std::string::npos);
}

TEST(output_is_capped_not_unbounded) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    ExecLimits lim = limits(20);
    lim.max_output_bytes = 4096;
    const ExecutionGrant grant = grant_execution(SandboxTier::T1_Seatbelt);
    const ExecOutcome o = run_sandboxed(grant, "seq 1 200000", root, root, lim);
    CHECK(o.output.size() <= 4096);
    CHECK(o.output_truncated);
}

TEST(the_classifier_output_cannot_become_a_grant) {
    // The S7.1 structural claim, stated as a compile-time fact: RiskHint and
    // ExecutionGrant are unrelated types, ExecutionGrant's constructor is private, and
    // the only mint is grant_execution(tier). This test documents the negative space --
    // if someone adds a conversion, the comment below is the review flag.
    //
    //   RiskHint hint = classify_command(...);
    //   ExecutionGrant g = hint;              // must never compile
    //   ExecutionGrant g(SandboxTier::T1);    // must never compile (private ctor)
    const RiskHint hint = classify_command("rm -rf /", "/work/repo", "/work/repo");
    CHECK(hint.caps.destroys_data);
    CHECK(hint.caps.writes_outside_workspace);
    static_assert(!std::is_convertible_v<RiskHint, ExecutionGrant>);
    static_assert(!std::is_constructible_v<ExecutionGrant, SandboxTier>);
}

TEST(the_seatbelt_profile_denies_by_default_where_it_matters) {
    const std::string p = seatbelt_profile("/work/repo");
    CHECK(p.find("(deny network*)") != std::string::npos);
    CHECK(p.find("(deny file-write*)") != std::string::npos);
    CHECK(p.find("(subpath \"/work/repo\")") != std::string::npos);
}
