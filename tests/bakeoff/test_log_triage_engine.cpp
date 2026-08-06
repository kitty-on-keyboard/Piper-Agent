// Pins `log_triage::compact` -- the engine at src/tools/log_triage.hpp that runs on every
// shell tool result (src/tools/registry.cpp:488).
//
// WHY THIS FILE EXISTS. Until 2026-07-31 nothing tested the engine at all. The bakeoff
// README asserted that `src/testing/test_log_triage_{corpus,engine}.cpp` validated and
// pinned it "in the standard gate"; that directory has never existed, and the gate manifest
// has never named such a test. The round-2 patch then changed line selection on nearly
// every case in the corpus and the whole gate stayed green, because there was nothing for
// it to turn red. That is the gap this closes.
//
// Two kinds of check, deliberately separated:
//
//  1. THE CONTRACT, on synthetic input. Never exceed the budget; return a fitting log byte
//     for byte. These hold for any input and are what the caller relies on.
//  2. THE SCORE, ratcheted against the real corpus. The engine is a pile of tuned
//     constants; the only statement of what it is FOR is the weighted total, so that total
//     is pinned here rather than living only in a scoreboard binary nobody runs. The pins
//     are `<=` on losses so an improvement never fails the gate, and `==` on the exact
//     count so a regression cannot hide behind a compensating win elsewhere.
//
// This does NOT re-check the corpus itself -- that is test_corpus_loads, which opens every
// referenced log and asserts every key entry is findable in it.

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <log_triage/corpus.hpp>

#include "src/tools/log_triage.hpp"
#include "tests/check.hpp"

using LtCase = log_triage_corpus::Case;

namespace {

std::vector<LtCase> load(const std::string& manifest) {
    std::vector<LtCase> cases = log_triage_corpus::load(LMP_BAKEOFF_LT_DIR, manifest);
    if (cases.empty()) {
        lmp::test::record_failure(__FILE__, __LINE__, manifest + ": loaded no cases");
    }
    ++lmp::test::reg().checks;
    return cases;
}

log_triage_corpus::Tally run(const std::vector<LtCase>& cases) {
    log_triage_corpus::Tally t;
    for (const LtCase& c : cases) {
        for (std::size_t budget : log_triage_corpus::kBudgets) {
            const std::string out = log_triage::compact(c.log, budget);
            log_triage_corpus::accumulate(t, log_triage_corpus::score_case(c, budget, out));
        }
    }
    return t;
}

} // namespace

// --- the contract ----------------------------------------------------------

TEST(a_log_that_already_fits_comes_back_byte_for_byte) {
    // The caller compares. Rewriting a log that fits destroys information for no gain, and
    // it is scored as a weight-3 violation for that reason.
    const std::string log = "clang: \x1b[31merror:\x1b[0m no such file\nmake: *** Error 1\n";
    CHECK_EQ(log_triage::compact(log, 8192), log);
    // Exactly at the cap is still a fit.
    CHECK_EQ(log_triage::compact(log, log.size()), log);
}

TEST(output_never_exceeds_the_budget) {
    // The budget check is an incremental upper bound, not a rebuild (that is what keeps the
    // engine at hundredths of a second on a 12 MB log), so it is the part most able to be
    // quietly wrong. Swept one byte at a time across the range where markers, gaps and
    // truncation all interact.
    std::string log;
    for (int i = 0; i < 200; ++i) {
        log += "/usr/bin/clang++ -DNDEBUG -c src/f" + std::to_string(i) + ".cpp -o f.o\n";
        log += "src/f" + std::to_string(i) + ".cpp:" + std::to_string(i + 3) +
               ":9: error: use of undeclared identifier 'widget'\n";
        log += "    return widget(x);\n           ^\n";
    }
    for (std::size_t budget = 0; budget <= 600; ++budget) {
        if (log_triage::compact(log, budget).size() > budget) {
            lmp::test::record_failure(__FILE__, __LINE__,
                                      "over budget at " + std::to_string(budget));
            break;
        }
    }
    ++lmp::test::reg().checks;
    CHECK(log_triage::compact(log, 0).empty());
}

TEST(the_locator_survives_a_budget_that_fits_almost_nothing) {
    // The failure this engine exists to prevent: the 40 bytes saying which line of its own
    // code to edit did not reach the model, and it guessed.
    std::string log = "[ 12%] Building CXX object CMakeFiles/app.dir/main.cpp.o\n";
    for (int i = 0; i < 400; ++i) {
        log += "/opt/homebrew/bin/cmake -E cmake_progress_report /tmp/build " +
               std::to_string(i) + "\n";
    }
    log += "src/widget.cpp:88:5: error: no member named 'draw' in 'Widget'\n";
    log += "    w.draw();\n      ^\n";
    log += "1 error generated.\n";

    const std::string out = log_triage::compact(log, 256);
    CHECK(out.size() <= 256);
    CHECK(out.find("src/widget.cpp:88:5") != std::string::npos);
}

// --- round-2 regressions ---------------------------------------------------
//
// Each of these was a real defect in the shipped round-1 engine, found by reading output the
// scorer had already gone quiet on. They are pinned individually because the corpus totals
// below would let any one of them come back masked by a win somewhere else.

TEST(a_clang_include_stack_does_not_outrank_the_diagnostic_it_precedes) {
    // "In file included from ..." carries `path:line:`, so the locator matcher fires on it
    // and it scored as a full anchor -- 150-200 bytes of SDK path each, dozens of them in a
    // template blow-up. They announce nothing wrong; the diagnostic they precede does.
    std::string log;
    for (int i = 0; i < 40; ++i) {
        log += "In file included from /Applications/Xcode.app/Contents/Developer/Toolchains/"
               "XcodeDefault.xctoolchain/usr/include/c++/v1/__algorithm/sort.h:" +
               std::to_string(i + 10) + ":\n";
    }
    log += "src/render.cpp:41:17: error: no matching function for call to 'blend'\n";
    log += "    auto c = blend(a, b);\n             ^~~~~\n";

    const std::string out = log_triage::compact(log, 512);
    CHECK(out.size() <= 512);
    CHECK(out.find("src/render.cpp:41:17") != std::string::npos);
    CHECK(out.find("no matching function") != std::string::npos);
}

TEST(a_flood_of_warning_addresses_does_not_crowd_out_the_errors) {
    // rustc prints the same ` --> path:L:C` under a `warning:` as under an `error:`. Round 1
    // demoted those bare locators to kWarning, but phase 1 packs by ANCHOR-ness rather than
    // by score, so the demotion never bit and 240 warning addresses filled the budget ahead
    // of the real errors. `diagnostic` -- anchor AND score >= kAnchor -- is what makes it
    // bite.
    std::string log;
    for (int i = 0; i < 240; ++i) {
        log += "warning: unused variable: `tmp" + std::to_string(i) + "`\n";
        log += "  --> src/shard.rs:" + std::to_string(i + 1) + ":9\n";
    }
    log += "error[E0308]: mismatched types\n";
    log += "  --> src/lib.rs:77:22\n";
    log += "   |\n77 |     let n: u32 = name;\n   |                  ^^^^ expected `u32`\n";

    const std::string out = log_triage::compact(log, 2048);
    CHECK(out.size() <= 2048);
    CHECK(out.find("src/lib.rs:77:22") != std::string::npos);
    CHECK(out.find("mismatched types") != std::string::npos);
}

TEST(a_system_note_ranks_below_a_real_diagnostic_but_a_local_one_does_not) {
    // Principle 2 ranks by locality; within the system tier round 1 did not rank by severity
    // at all, so libc++'s instantiation backtrace tied with the one error and crowded it out.
    // The fix must not touch LOCAL notes -- they are the only actionable line in a template
    // blow-up, and keeping them is what solves build_no_matching_ctor.
    std::string sys;
    for (int i = 0; i < 20; ++i) {
        sys += "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault."
               "xctoolchain/usr/include/c++/v1/__algorithm/sort.h:" +
               std::to_string(300 + i) +
               ":5: note: in instantiation of function template specialization "
               "'std::__sort<Cmp &, Item *>' requested here\n";
    }
    const std::string err =
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/"
        "include/c++/v1/__algorithm/comp.h:22:21: error: invalid operands to binary "
        "expression\n";
    const std::string out = log_triage::compact(sys + err, 1024);
    CHECK(out.size() <= 1024);
    CHECK(out.find("comp.h:22:21") != std::string::npos);

    // The local note, same shape, must still be kept ahead of the system lines.
    const std::string local_note =
        "src/sorting.cpp:19:5: note: in instantiation of function template specialization "
        "'sort_items<Cmp>' requested here\n";
    const std::string out2 = log_triage::compact(sys + local_note + err, 512);
    CHECK(out2.size() <= 512);
    CHECK(out2.find("src/sorting.cpp:19:5") != std::string::npos);
}

// --- the score -------------------------------------------------------------

TEST(the_engine_holds_its_corpus_score) {
    const std::vector<LtCase> cases = load("corpus.jsonl");
    CHECK_EQ(cases.size(), std::size_t{25});
    const log_triage_corpus::Tally t = run(cases);

    CHECK_EQ(t.points, 75);
    // Round 2, 2026-07-31: 34 -> 15 weighted, 71 -> 73 exact. The residual 15 is almost
    // entirely bare_error_limit at 2048, which is capacity-bound (19 diagnostics and 38
    // context lines against a 2048-byte cap), not a triage failure.
    CHECK(t.weighted <= 15);
    CHECK_EQ(t.exact, 73);
    // A locator or a message lost is the failure mode the engine exists to prevent, and
    // both are currently zero. Anything above zero here is a regression whatever the total
    // does.
    CHECK_EQ(t.locator_miss, 0);
    CHECK_EQ(t.message_miss, 0);
    CHECK_EQ(t.over_budget, 0);
    CHECK_EQ(t.passthrough_violation, 0);
    // Denominators, so a shrinking key cannot be read as an improving engine.
    CHECK_EQ(t.locator_total, 177);
    CHECK_EQ(t.message_total, 195);
    CHECK_EQ(t.context_total, 384);
}

TEST(the_engine_holds_its_holdout_score) {
    const std::vector<LtCase> cases = load("holdout.jsonl");
    CHECK_EQ(cases.size(), std::size_t{7});
    const log_triage_corpus::Tally t = run(cases);

    CHECK_EQ(t.points, 21);
    // Round 2: 20 -> 0. NOT a blind result -- ho_rustc_no_cargo was already burned in round
    // 1 and the warning-address fix above was found by re-reading its output. The defensible
    // claim is narrower: the other six cases were perfect before and remain perfect.
    CHECK(t.weighted <= 0);
    CHECK_EQ(t.exact, 21);
    CHECK_EQ(t.over_budget, 0);
    CHECK_EQ(t.passthrough_violation, 0);
    CHECK_EQ(t.locator_total, 39);
    CHECK_EQ(t.message_total, 48);
    CHECK_EQ(t.context_total, 117);
}

TEST(structured_analyze_extracts_pytest_and_ctest_fields) {
    const std::string pytest =
        "============================= test session starts ==============================\n"
        "collected 3 items\n"
        "tests/test_mod.py::test_ok PASSED\n"
        "tests/test_pipeline.py::test_summarise_average FAILED\n"
        "FAILED tests/test_pipeline.py::test_summarise_average - AssertionError\n"
        "======================== 1 failed, 2 passed in 0.09s =========================\n"
        "tests/test_pipeline.py:12: AssertionError\n";
    const log_triage::StructuredTriage pt = log_triage::analyze(pytest);
    CHECK(pt.runner == log_triage::Runner::Pytest);
    CHECK_EQ(pt.failed, 1);
    CHECK_EQ(pt.passed, 2);
    CHECK(!pt.failing_tests.empty());
    CHECK(pt.failing_tests[0].find("test_summarise_average") != std::string::npos);
    CHECK(!pt.referenced_paths.empty());
    const std::string ann = log_triage::format_annotation(pt);
    CHECK(ann.find("runner=pytest") != std::string::npos);
    CHECK(ann.find("failing_tests") != std::string::npos);

    const std::string ctest =
        "13/13 Test #13: budget_enforced ..................Subprocess aborted***Exception\n"
        "src/fail.cpp:3: Assertion failed\n"
        "92% tests passed, 1 tests failed out of 13\n"
        "The following tests FAILED:\n"
        "\t 13 - budget_enforced (Subprocess aborted)\n"
        "Errors while running CTest\n";
    const log_triage::StructuredTriage ct = log_triage::analyze(ctest);
    CHECK(ct.runner == log_triage::Runner::CTest);
    CHECK_EQ(ct.failed, 1);
    CHECK(!ct.failing_tests.empty());
    CHECK(ct.failing_tests[0].find("budget_enforced") != std::string::npos);
}

TEST(primary_fingerprint_is_stable_for_identical_diagnostics) {
    const std::string log =
        "src/a.cpp:10:3: error: use of undeclared identifier 'x'\n"
        "src/a.cpp:10:3: error: use of undeclared identifier 'x'\n";
    const std::string a = log_triage::primary_fingerprint(log_triage::analyze(log));
    const std::string b = log_triage::primary_fingerprint(log_triage::analyze(log));
    CHECK(!a.empty());
    CHECK_EQ(a, b);
}
