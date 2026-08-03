// The verification choke point, driven against a real temp workspace (S10.1, S10.2).
//
// Added in response to a SURVIVING MUTATION: `return true; // paid for once` in
// Verifier::prove_falsifiable could be flipped and nothing noticed. That is a finding
// about the suite, not about the mutant (S11.4), and this file is the fix.

#include <unistd.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "src/context/context.hpp"
#include "src/loop/verification.hpp"
#include "src/platform/fs.hpp"

#include "tests/check.hpp"

using namespace lmp;

namespace {

std::string temp_dir() {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = std::string(base ? base : "/tmp") + "/lmp_ver_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const char* made = ::mkdtemp(buf.data());
    return made ? std::string(made) : std::string();
}

tools::Registry make_registry(const std::string& root) {
    tools::WorkspaceContext ctx;
    ctx.root = root;
    ctx.max_read_bytes = 1U << 20;
    ctx.max_model_read_bytes = 16384;
    ctx.max_result_bytes = 4096;
    ctx.spool_dir = root;
    ctx.shell_wall_clock_seconds = 20;
    return tools::Registry(std::move(ctx));
}

} // namespace

TEST(the_choke_point_records_every_result_including_refusals) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    tools::Registry reg = make_registry(root);
    context::ContextStore ctx("m");
    loop::Verifier v(reg, ctx);

    CHECK(v.run_and_record("true", 1));
    CHECK(!v.run_and_record("false", 1));
    REQUIRE(ctx.verifications().size() == 2);
    CHECK(ctx.verifications()[0].passed);
    CHECK(!ctx.verifications()[1].passed);

    // A refusal is recorded as NOT PASSED but is labelled as never having run -- it is
    // not evidence in either direction, and calling it a failure would send the agent
    // off fixing a build that was never attempted (S6.2).
    CHECK(!v.run_and_record("true", 0)); // tier 0 refuses execution
    REQUIRE(ctx.verifications().size() == 3);
    CHECK(ctx.verifications()[2].detail.find("REFUSED") == 0);
}

// A red that only means "there is no such command" is not evidence the check can fail,
// and must not buy a falsifiability proof (S10.2, S6.2).
//
// MEASURED: a run declared `python -m pytest ...` on a host whose only interpreter is
// `python3`. The baseline came back red -- exit 127, `command not found` -- and that red
// marked the contract falsifiable. pytest had never been executed once.
TEST(a_command_that_could_not_run_is_not_a_red) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    tools::Registry reg = make_registry(root);
    context::ContextStore ctx("m");
    loop::Verifier v(reg, ctx);

    CHECK(!v.run_and_record("definitely-not-a-real-command-xyz --version", 1));
    REQUIRE(ctx.verifications().size() == 1);
    const context::VerificationRecord& rec = ctx.verifications()[0];
    CHECK(!rec.passed);
    CHECK(!rec.ran); // never executed, so not evidence in either direction
    CHECK(rec.detail.find("NEVER RAN") == 0);

    // And therefore it proves nothing: a later green off the back of it stays unproven.
    CHECK(!v.is_proven("definitely-not-a-real-command-xyz --version"));

    // A command that DOES run and fails is still a real red.
    CHECK(!v.run_and_record("false", 1));
    CHECK(ctx.verifications()[1].ran);
    CHECK(v.is_proven("false"));
}

TEST(an_unproven_check_is_recorded_as_unproven) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    tools::Registry reg = make_registry(root);
    context::ContextStore ctx("m");
    loop::Verifier v(reg, ctx);

    CHECK(!v.is_proven("true"));
    CHECK(v.run_and_record("true", 1));
    // Green, but nobody has shown this check can go red, so it is not evidence yet.
    CHECK(!ctx.verifications()[0].falsifiable);
}

TEST(falsifiability_is_proven_by_intervention_and_then_reused) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    tools::Registry reg = make_registry(root);
    context::ContextStore ctx("m");
    loop::Verifier v(reg, ctx);

    const std::string marker = root + "/ok.flag";
    REQUIRE(platform::write_file_atomic(marker, "1").ok());
    const std::string check = "test -f '" + marker + "'";

    int breaks = 0;
    int restores = 0;
    const bool proven = v.prove_falsifiable(
        check, 1,
        [&] { ++breaks; return ::unlink(marker.c_str()) == 0; },
        [&] { ++restores; return platform::write_file_atomic(marker, "1").ok(); });

    CHECK(proven);
    CHECK_EQ(breaks, 1);
    CHECK_EQ(restores, 1);
    CHECK(v.is_proven(check));
    // Canonicalisation means the proof covers the wrapped form too -- one identity,
    // one proof (S10.2).
    CHECK(v.is_proven(check + " ; echo $?"));

    // Paid for once: a second proof does no work.
    const bool again = v.prove_falsifiable(
        check, 1, [&] { ++breaks; return true; }, [&] { ++restores; return true; });
    CHECK(again);
    CHECK_EQ(breaks, 1);   // the mutation that flipped this early return is now caught
    CHECK_EQ(restores, 1);

    // And a recorded run now carries the proof.
    CHECK(v.run_and_record(check, 1));
    CHECK(ctx.verifications().back().falsifiable);
}

TEST(a_check_that_cannot_go_red_is_not_proven) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    tools::Registry reg = make_registry(root);
    context::ContextStore ctx("m");
    loop::Verifier v(reg, ctx);

    // `true` passes no matter what the breaker does. That is exactly the check that
    // must NOT count as evidence -- it is the shape of a test that grades its own
    // homework.
    const bool proven = v.prove_falsifiable("true", 1, [] { return true; },
                                            [] { return true; });
    CHECK(!proven);
    CHECK(!v.is_proven("true"));
}

TEST(a_check_that_is_already_red_proves_nothing) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    tools::Registry reg = make_registry(root);
    context::ContextStore ctx("m");
    loop::Verifier v(reg, ctx);
    // Breaking something already broken demonstrates nothing about the check.
    CHECK(!v.prove_falsifiable("false", 1, [] { return true; }, [] { return true; }));
}
