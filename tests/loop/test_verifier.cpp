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

// `|| echo "..."` is the most common status swallower a model writes, and it is total: the
// or-list makes the whole command exit 0 whatever happened. A contract that cannot go red
// never becomes evidence, so the run is told its green is unproven every turn forever.
TEST(an_or_list_swallower_comes_off_like_every_other_wrapper) {
    CHECK_EQ(lmp::loop::executable_form("swift build || echo \"Build successful\""),
             std::string("swift build"));
    CHECK_EQ(lmp::loop::executable_form("make test || true"), std::string("make test"));
    CHECK_EQ(lmp::loop::executable_form("make test || :"), std::string("make test"));
    // Nested behind a formatter pipe, which hides it from the pipe stripper until the
    // or-list comes off first.
    CHECK_EQ(lmp::loop::executable_form("pytest -q | tail -5 || echo done"), std::string("pytest -q"));

    // A REAL fallback is not a swallower: its status still means something, and rewriting
    // the model's command would change what is being checked.
    CHECK_EQ(lmp::loop::executable_form("make || make clean"), std::string("make || make clean"));
}

// The exact contract from the run that prompted this:
//   swift build 2>&1 | grep -E "(error:|warning:)" || echo "Build successful"
// Measured on a workspace whose build was full of errors -- it exited 0 every time.
// Stripping the or-list exposes a SECOND problem the strip cannot fix, and naming that is
// worth more to the run than any amount of proving.
TEST(a_trailing_grep_is_reported_as_a_broken_criterion_not_a_broken_build) {
    const std::string contract =
        "swift build 2>&1 | grep -E \"(error:|warning:)\" || echo \"Build successful\"";
    const std::string runnable = lmp::loop::executable_form(contract);
    CHECK_EQ(runnable, std::string("swift build 2>&1 | grep -E \"(error:|warning:)\""));

    // Inverted, not merely unprovable: green while the build is broken, red once clean.
    CHECK(!lmp::loop::unfalsifiable_reason(runnable).empty());
    // A plain invocation is a fine criterion and must not be flagged.
    CHECK(lmp::loop::unfalsifiable_reason("swift build").empty());
    CHECK(lmp::loop::unfalsifiable_reason("swift test --filter HostStatsServiceTests").empty());
    // A pipeline whose last stage is a real command still decides its own status.
    CHECK(lmp::loop::unfalsifiable_reason("swift test | ./summarize").empty());
}

// --- unmoved failures (Gap 1, Gap 2) ----------------------------------------
//
// The whole group is driven by one real run: 45 turns and 2508 seconds against
// `xcodebuild build -scheme ResMon` in a project whose only scheme was `Untitled Project`.
// The check failed identically at the baseline and again nineteen turns and eleven file
// writes later, and the harness read the second red as PROOF the check could fail.

// The comparison the diagnosis asked for -- "byte-identical across attempts" -- would
// never have fired on the run it was derived from. These are the two real failures, and
// they differ in a timestamp, a pid:tid pair and a result-bundle name with the time in it.
TEST(a_failure_signature_ignores_the_parts_that_move_on_their_own) {
    const std::string first =
        "[exit 65]\n2026-08-03 12:56:49.742 xcodebuild[36739:4497951] Writing error result "
        "bundle to /var/folders/n3/T/ResultBundle_2026-03-08_12-56-0049.xcresult\n"
        "xcodebuild: error: The project named \"ResMon\" does not contain a scheme named "
        "\"ResMon\".\n";
    const std::string second =
        "[exit 65]\n2026-08-03 13:04:02.118 xcodebuild[37511:4507330] Writing error result "
        "bundle to /var/folders/n3/T/ResultBundle_2026-03-08_13-04-0002.xcresult\n"
        "xcodebuild: error: The project named \"ResMon\" does not contain a scheme named "
        "\"ResMon\".\n";
    CHECK(first != second); // the naive comparison, and why it is not enough
    CHECK_EQ(lmp::loop::failure_signature(first), lmp::loop::failure_signature(second));

    // It must still tell two DIFFERENT failures apart, or every red in a run collapses
    // into one and no check is ever falsifiable again.
    CHECK(lmp::loop::failure_signature("error: cannot find 'foo' in scope") !=
          lmp::loop::failure_signature("error: cannot find 'bar' in scope"));
}

namespace {

// A reading of `contract`, as the ledger would hold it.
context::VerificationRecord red(std::string contract, std::string detail,
                                std::size_t writes) {
    context::VerificationRecord v;
    v.contract = std::move(contract);
    v.detail = std::move(detail);
    v.workspace_writes = writes;
    v.ran = true;
    v.passed = false;
    return v;
}

} // namespace

TEST(a_red_that_survives_a_changed_workspace_is_not_about_the_code) {
    const std::string c = "xcodebuild build -scheme ResMon";
    std::vector<context::VerificationRecord> ledger{
        red(c, "does not contain a scheme named \"ResMon\"", 0),
        red(c, "does not contain a scheme named \"ResMon\"", 11)};
    // BOTH ends, not just the later one. The asymmetric rule leaves the most recent red
    // standing, and the most recent red is the one is_proven() reaches first.
    CHECK(lmp::loop::failure_is_unmoved(ledger, 0));
    CHECK(lmp::loop::failure_is_unmoved(ledger, 1));

    // Re-running a check without touching anything in between is expected to say the same
    // thing and says nothing about the check.
    std::vector<context::VerificationRecord> rerun{red(c, "same output", 4),
                                                   red(c, "same output", 4)};
    CHECK(!lmp::loop::failure_is_unmoved(rerun, 0));
    CHECK(!lmp::loop::failure_is_unmoved(rerun, 1));

    // A failure the work DID move is ordinary evidence, which is the case that must keep
    // working -- this is what a run fixing compile errors looks like.
    std::vector<context::VerificationRecord> moved{red(c, "4 errors in Stats.swift", 0),
                                                   red(c, "1 error in View.swift", 6)};
    CHECK(!lmp::loop::failure_is_unmoved(moved, 0));
    CHECK(!lmp::loop::failure_is_unmoved(moved, 1));
}

// The bug this closes: `xcodebuild -scheme ResMon` was certified `falsifiable: 1` on the
// strength of "scheme not found". Same shape as the hole never_executed() was added for --
// a red that proves nothing because it was never about the thing being checked.
TEST(an_unmoved_red_does_not_certify_a_contract_as_falsifiable) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    tools::Registry reg = make_registry(root);
    context::ContextStore ctx("m");
    loop::Verifier v(reg, ctx);

    // A check whose failure never mentions the workspace, run either side of real work.
    const std::string check = "echo not-a-scheme >&2; exit 65";
    CHECK(!v.run_and_record(check, 1));
    ctx.record_deliverable("Stats.swift");
    ctx.record_deliverable("View.swift");
    CHECK(!v.run_and_record(check, 1));

    REQUIRE(ctx.verifications().size() == 2);
    CHECK(!ctx.verifications()[0].falsifiable);
    // Before this rule the second reading was certified by the first.
    CHECK(!ctx.verifications()[1].falsifiable);
    CHECK(ctx.verifications()[1].workspace_writes == 2);

    // And the run is told, in the one place that decides whether it can finish.
    ctx.set_verify_contract(check);
    const loop::UnmovedContract stuck = loop::unmoved_contract(ctx);
    CHECK(stuck.unmoved);
    CHECK_EQ(stuck.contract, lmp::loop::canonicalize_check(check));
}

// The FAIL_TO_PASS case, which is the one that must not regress: red before the work,
// green after it, and the red is the proof. Nothing above may cost this.
TEST(an_ordinary_red_then_green_still_proves_the_check) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    tools::Registry reg = make_registry(root);
    context::ContextStore ctx("m");
    loop::Verifier v(reg, ctx);

    const std::string check = "test -f " + root + "/built";
    CHECK(!v.run_and_record(check, 1));
    ctx.record_deliverable("built");
    CHECK(v.run_and_record("touch " + root + "/built", 1));
    CHECK(v.run_and_record(check, 1));

    const auto& vs = ctx.verifications();
    REQUIRE(vs.size() == 3);
    CHECK(vs[2].passed);
    CHECK(vs[2].falsifiable); // the earlier red is still the proof
    ctx.set_verify_contract(check);
    CHECK(!loop::unmoved_contract(ctx).unmoved);
}

// An operator contract is not the model's to re-derive, so the finding must not be raised
// against one -- `plan` is forbidden to replace it and the run would be pinned to a tool
// that cannot help it.
TEST(the_finding_is_about_the_latest_reading_of_the_declared_contract) {
    context::ContextStore ctx("m");
    ctx.set_verify_contract("make check");
    CHECK(!loop::unmoved_contract(ctx).unmoved); // no readings at all

    ctx.record_verification(red("make check", "no rule to make target", 0));
    CHECK(!loop::unmoved_contract(ctx).unmoved); // one reading proves nothing

    ctx.record_deliverable("a.c");
    ctx.record_verification(red("make check", "no rule to make target", 1));
    CHECK(loop::unmoved_contract(ctx).unmoved);

    // A later reading that DID move clears it: the criterion is reading the workspace again.
    ctx.record_deliverable("b.c");
    ctx.record_verification(red("make check", "a.c:3: undefined reference", 2));
    CHECK(!loop::unmoved_contract(ctx).unmoved);

    // A contract nobody declared cannot be unmoved.
    context::ContextStore bare("m");
    bare.record_verification(red("make check", "x", 0));
    bare.record_verification(red("make check", "x", 1));
    CHECK(!loop::unmoved_contract(bare).unmoved);
}

// A run that CORRECTS its own build command stops being watched at the moment it starts
// being right: the contract is matched by containment, so the fixed command is not the
// check and never reaches the ledger.
//
// MEASURED: a 45-turn run declared `xcodebuild -scheme ResMon`, found mid-run that the only
// scheme was `Untitled Project`, rebuilt correctly with it, and recorded ZERO verifications
// for its last 31 minutes.
TEST(a_command_running_the_contracts_program_a_different_way_is_a_near_miss) {
    const std::string contract = "xcodebuild -scheme ResMon -destination 'platform=macOS'";
    // The real fixed command, with the `cd` prefix every model writes.
    CHECK(lmp::loop::is_near_miss(
        "cd /Users/dev/ResMon && xcodebuild -scheme \"Untitled Project\" build", contract));
    // Being the check wins over being a near miss.
    CHECK(!lmp::loop::is_near_miss("cd /Users/dev/ResMon && " + contract, contract));
    // A different program is not a near miss at all.
    CHECK(!lmp::loop::is_near_miss("swift build", contract));
    CHECK(!lmp::loop::is_near_miss("ls -la", contract));

    // The program is found past the noise every real invocation carries.
    CHECK_EQ(lmp::loop::check_program("cd /a/b && make test"), std::string_view("make"));
    CHECK_EQ(lmp::loop::check_program("PYTHONPATH=. python3 -m pytest spec/"),
             std::string_view("python3"));
    CHECK_EQ(lmp::loop::check_program("make test"), std::string_view("make"));
}
