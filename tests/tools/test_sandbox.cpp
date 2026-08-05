// Phase 5's exit criterion, verbatim from S17: "A command that writes outside root,
// opens a socket, or spins forever is stopped -- PROVEN BY A TEST THAT ATTEMPTS EACH."
// These tests attack the sandbox and pass only when the attack fails.

#include <csignal>
#include <climits>
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

    // EITHER stopper satisfies the requirement, and this test used to pin SIGXCPU --
    // contradicting the comment directly above it. limits(1) sets cpu_seconds AND
    // wall_clock_seconds to 1, so which one lands first is a race; on a fast idle machine
    // the busy loop burns its CPU second first and dies of SIGXCPU, and under ASan on a
    // shared CI runner the wall-clock killer (polled every 200 ms) gets there first and
    // the child is SIGKILLed instead, with signalled left false.
    //
    // That is not a defect in the sandbox -- the runaway was stopped both times, which is
    // the entire requirement. It was a defect in the assertion, and it went unnoticed
    // because the sanitizers job had never built far enough to run this suite.
    CHECK(o.wall_clock_killed || o.signalled);
    if (o.signalled) {
        CHECK_EQ(o.signal, SIGXCPU);
    }
}

TEST(a_command_that_must_fork_still_runs) {
    // RLIMIT_NPROC counts every process owned by the REAL UID, so applying
    // max_processes as an absolute number makes fork() return EAGAIN the moment the
    // desktop already owns that many -- which on any machine with an editor open it
    // does. See nproc_ceiling() in sandbox.cpp.
    //
    // Every other test here runs a command simple enough that /bin/sh execs it without
    // forking, so all of them passed while the real agent could not run one shell
    // command: the first end-to-end run got `/bin/sh: fork: Resource temporarily
    // unavailable` on all nine of its shell calls. A pipeline forces the fork that a
    // builtin elides, which is what makes this test see the bug.
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    const ExecutionGrant grant = grant_execution(SandboxTier::T1_Seatbelt);
    const ExecOutcome o =
        run_sandboxed(grant, "echo one | cat && echo two", root, root, limits(20));

    CHECK(o.status == Status::Ok);
    CHECK_EQ(o.exit_code, 0);
    CHECK(o.output.find("one") != std::string::npos);
    CHECK(o.output.find("two") != std::string::npos);
    // The precise symptom, named so a regression is recognised on sight.
    CHECK(o.output.find("Resource temporarily unavailable") == std::string::npos);
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

    // With no usable runtime, T2 must REFUSE rather than silently run in T1 -- a silent
    // downgrade is v1's unsafe_host default wearing a new name (S7.2, S13). The env
    // switch makes "no runtime" reachable on a host that HAS one, so this assertion is
    // about the code path rather than about this machine's software.
    ::setenv("LMP_DISABLE_CONTAINER", "1", 1);
    const ExecOutcome t2 = run_sandboxed(grant_execution(SandboxTier::T2_Container),
                                         "echo hi", root, root, limits(5));
    CHECK(t2.status == Status::Refused);
    CHECK(t2.output.find("container") != std::string::npos);
    // The refusal names WHY, so an operator can act on it instead of guessing.
    CHECK(t2.output.find("refusal, not a") != std::string::npos);
    // And nothing ran: a refused command produces no output of its own.
    CHECK(t2.output.find("hi\n") == std::string::npos);
}

// The container invocation itself, asserted without needing a runtime installed: the
// flags ARE the containment, so they are worth pinning.
TEST(the_container_invocation_carries_its_containment) {
    ContainerRuntime rt;
    rt.available = true;
    rt.binary = "docker";
    rt.image = "img@sha256:deadbeef";
    const std::string cmd =
        container_command(rt, "pytest -q", "/work/space", "/work/space", limits(5));
    CHECK(cmd.find("--network none") != std::string::npos);       // egress denied
    CHECK(cmd.find("--pids-limit") != std::string::npos);         // fork bombs
    CHECK(cmd.find("--memory") != std::string::npos);
    // Mounted at the SAME path it has on the host: every diagnostic the model has already
    // seen names the host path, and remapping would describe a filesystem nothing else in
    // the run knows about.
    CHECK(cmd.find("--volume /work/space:/work/space") != std::string::npos);
    CHECK(cmd.find("img@sha256:deadbeef") != std::string::npos);  // pinned by digest
    CHECK(cmd.find("'pytest -q'") != std::string::npos);          // quoted, not injected
}

// A command carrying a quote must not break out of the -c argument.
TEST(the_container_invocation_quotes_the_command) {
    ContainerRuntime rt;
    rt.available = true;
    rt.binary = "docker";
    rt.image = "i";
    const std::string cmd =
        container_command(rt, "echo 'a'; rm -rf /", "/w", "/w", limits(5));
    CHECK(cmd.find("'echo '\\''a'\\''; rm -rf /'") != std::string::npos);
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

namespace {

// RESOLVED, like the profile's own paths: confstr answers /var/folders/..., /var is a
// symlink to /private/var, and Seatbelt matches subpaths after resolution -- so an
// unresolved rule matches nothing. Comparing the unresolved form here would fail against
// a correct profile, which is how this test first ran.
std::string user_temp_root() {
    char buf[PATH_MAX];
    const std::size_t n = ::confstr(_CS_DARWIN_USER_TEMP_DIR, buf, sizeof(buf));
    if (n == 0 || n > sizeof(buf)) {
        return {};
    }
    char real[PATH_MAX];
    std::string out = ::realpath(buf, real) != nullptr ? std::string(real) : std::string(buf);
    while (out.size() > 1 && out.back() == '/') {
        out.pop_back();
    }
    return out;
}

} // namespace

TEST(atomic_saves_are_allowed_without_opening_the_temp_root) {
    // macOS stages every atomic save in .../T/TemporaryItems and renames it into place,
    // so a jail that denies that directory denies `swift build` writing its own build
    // manifests INSIDE the workspace -- with an error naming the destination file, which
    // is why it read as a permissions problem with the workspace.
    //
    // The narrowness is the assertion: TemporaryItems is allowed, the temp root that
    // contains it is not. Allowing the root instead is the escape the next test attacks.
    const std::string p = seatbelt_profile("/work/repo");
    const std::string temp_root = user_temp_root();
    REQUIRE(!temp_root.empty());

    CHECK(p.find("(subpath \"" + temp_root + "/TemporaryItems\")") != std::string::npos);
    CHECK(p.find("(subpath \"" + temp_root + "\")") == std::string::npos);
}

TEST(a_write_into_the_per_user_temp_root_is_still_stopped) {
    // The break-out test above writes wherever TMPDIR points, which on a machine that
    // does not set it is /tmp. This one names the per-user temp root explicitly, because
    // that is the directory the toolchain allowance sits INSIDE and the one an
    // over-broad rule would have opened.
    const std::string root = temp_dir();
    const std::string temp_root = user_temp_root();
    REQUIRE(!root.empty());
    REQUIRE(!temp_root.empty());

    const std::string target = temp_root + "/lmp_escape_probe.txt";
    ::unlink(target.c_str());
    const ExecutionGrant grant = grant_execution(SandboxTier::T1_Seatbelt);
    const ExecOutcome o =
        run_sandboxed(grant, "echo pwned > '" + target + "'", root, root, limits(10));

    CHECK(o.status == Status::ToolError);
    const lmp::platform::FileContents f = lmp::platform::read_file_whole(target, 1024);
    CHECK(f.status == lmp::platform::FsStatus::NotFound);
}

TEST(an_approved_tier_number_names_the_tier_the_operator_asked_for) {
    // sandbox_tier=3 was accepted by the wire, acknowledged by name in the editor, and
    // then routed to the container by every execution site -- which refused for want of a
    // runtime. The operator's own opt-in was unreachable.
    CHECK(tier_for(0) == SandboxTier::T0_NoExec);
    CHECK(tier_for(1) == SandboxTier::T1_Seatbelt);
    CHECK(tier_for(2) == SandboxTier::T2_Container);
    CHECK(tier_for(3) == SandboxTier::T3_HostUnsandboxed);
    // Out of range clamps DOWN, never onto the least contained tier: an unrecognised
    // number must not be how the jail comes off (S13).
    CHECK(tier_for(-1) == SandboxTier::T0_NoExec);
    CHECK(tier_for(4) == SandboxTier::T2_Container);
    CHECK(tier_for(99) == SandboxTier::T2_Container);
}

TEST(t1_makes_swiftpm_runnable_without_touching_the_tier) {
    // The nesting is not a filesystem problem and no profile allowance fixes it: macOS
    // refuses the inner sandbox_apply outright. Measured against a real package -- T1
    // with the flag and T3 without it produce byte-identical compiler output.
    CHECK(t1_compat_rewrite("swift build") == "swift build --disable-sandbox");
    CHECK(t1_compat_rewrite("swift test") == "swift test --disable-sandbox");
    CHECK(t1_compat_rewrite("swift run") == "swift run --disable-sandbox");

    // The shape an operator's verify contract actually has. This is the case the note
    // could never fix: the contract is run verbatim, so a `swift test` contract was
    // unpassable at T1 however correct the code became.
    CHECK(t1_compat_rewrite("cd /w/p && swift test") ==
          "cd /w/p && swift test --disable-sandbox");
    // Inserted after the SUBCOMMAND, so a caller's own trailing pipeline still applies to
    // the whole invocation rather than being handed to swift as arguments.
    CHECK(t1_compat_rewrite("swift test 2>&1 | tee out.txt") ==
          "swift test --disable-sandbox 2>&1 | tee out.txt");
    CHECK(t1_compat_rewrite("/usr/bin/swift build -v") ==
          "/usr/bin/swift build --disable-sandbox -v");
}

TEST(t1_sees_past_launcher_prefixes_to_the_program) {
    // THE 85-TURN BUG. `xcrun` is how Apple's documentation spells a toolchain invocation
    // and therefore how every model spells it. The old rule required the character before
    // `swift` to be start-of-string or one of `;&|(`; in `xcrun swift build` it is the `n`
    // of `xcrun`, so `swift` read as an argument and the rewrite declined.
    //
    // MEASURED: a run whose declared contract was `xcrun swift build`, at T1, on a tree
    // whose only real defect was a few Mach API type errors. The rewrite fired zero times,
    // every verification it ever took was the nesting EPERM, and the model rewrote correct
    // code for 85 turns chasing a failure the harness was causing.
    CHECK(t1_compat_rewrite("xcrun swift build") == "xcrun swift build --disable-sandbox");
    CHECK(t1_compat_rewrite("xcrun swift test") == "xcrun swift test --disable-sandbox");
    CHECK(t1_compat_rewrite("cd /w/ResMon && xcrun swift build") ==
          "cd /w/ResMon && xcrun swift build --disable-sandbox");
    CHECK(t1_compat_rewrite("/usr/bin/xcrun swift build") ==
          "/usr/bin/xcrun swift build --disable-sandbox");
    // A launcher flag that takes a SEPARATE value token: stopping at `macosx` would find no
    // program and decline, which is the same silent miss one level in.
    CHECK(t1_compat_rewrite("xcrun -sdk macosx swift build") ==
          "xcrun -sdk macosx swift build --disable-sandbox");
    CHECK(t1_compat_rewrite("arch -arm64 swift test") ==
          "arch -arm64 swift test --disable-sandbox");
    CHECK(t1_compat_rewrite("env FOO=1 swift build") ==
          "env FOO=1 swift build --disable-sandbox");
    CHECK(t1_compat_rewrite("nice -n 10 xcrun swift build") ==
          "nice -n 10 xcrun swift build --disable-sandbox");
    // An assignment with no launcher in front of it.
    CHECK(t1_compat_rewrite("FOO=1 swift build") == "FOO=1 swift build --disable-sandbox");
}

TEST(t1_covers_swift_package_and_decides_per_segment) {
    // `swift package resolve` compiles Package.swift exactly as `build` does and dies at T1
    // exactly as `build` does. Excluding it as "out of scope" left dependency resolution --
    // part of every build -- unrunnable at T1.
    CHECK(t1_compat_rewrite("swift package resolve") ==
          "swift package --disable-sandbox resolve");
    CHECK(t1_compat_rewrite("xcrun swift package update") ==
          "xcrun swift package --disable-sandbox update");

    // PER SEGMENT. A whole-string check for the flag let one half of an `&&` opt the other
    // half out of being fixed, which is the silent half of the original miss.
    CHECK(t1_compat_rewrite("swift build --disable-sandbox && swift test") ==
          "swift build --disable-sandbox && swift test --disable-sandbox");
    CHECK(t1_compat_rewrite("swift build && xcrun swift test") ==
          "swift build --disable-sandbox && xcrun swift test --disable-sandbox");
}

TEST(t1_leaves_alone_everything_that_is_not_a_swiftpm_build) {
    // Empty means "unchanged", and the quiet cases matter as much as the loud one: a
    // rewrite that fires on the wrong command line is a command the operator did not
    // write, running as if they had.
    CHECK(t1_compat_rewrite("swift build --disable-sandbox").empty()); // already asked
    CHECK(t1_compat_rewrite("swiftc main.swift").empty());             // different program
    CHECK(t1_compat_rewrite("swift --version").empty());               // rejects the flag
    CHECK(t1_compat_rewrite("xcodebuild test").empty());  // needs T3, not a flag
    CHECK(t1_compat_rewrite("xcodebuild -scheme App build").empty());
    CHECK(t1_compat_rewrite("cmake --build build").empty());
    // The word appearing is not the command running. Looking for the PROGRAM rather than a
    // word at a command position is more permissive about POSITION and no more permissive
    // about IDENTITY: an unrelated program shadows everything after it, so these still find
    // `echo`, `grep` and `rg` and stop there.
    CHECK(t1_compat_rewrite("echo swift build").empty());
    CHECK(t1_compat_rewrite("grep -r 'swift test' .").empty());
    CHECK(t1_compat_rewrite("echo \"swift build is how you build it\"").empty());
    CHECK(t1_compat_rewrite("rg 'swift build' src/").empty());
}

TEST(only_t1_rewrites_because_only_t1_nests) {
    // T3 exists so a toolchain T1 cannot host still has a way to run. If the harness
    // altered the command there too, the two tiers would stop being comparable -- and
    // "it fails identically at T1 and T3" is the evidence that separates a harness
    // problem from a broken build. A real run lost its whole budget for want of it.
    const std::string root = temp_dir();
    REQUIRE(!root.empty());

    const ExecOutcome host = run_sandboxed(grant_execution(SandboxTier::T3_HostUnsandboxed),
                                           "echo swift build", root, root, limits(10));
    CHECK(host.rewritten_command.empty());

    const ExecOutcome jailed = run_sandboxed(grant_execution(SandboxTier::T1_Seatbelt),
                                             "echo hello", root, root, limits(10));
    CHECK(jailed.rewritten_command.empty());
}
