// Phase 0's exit criterion: "Gate runs and is VERIFIED to run the expected test count."
//
// v1 selected its gate with `ctest -E realmodel` for months. `-E` excludes by NAME, no
// test was named "realmodel", so the pattern matched nothing -- and every "48/48" it
// printed was a number about a set nobody had checked. The lesson is not "use -L": it is
// that a selector must be proven to select, and a count must be pinned by something other
// than itself.
//
// So this test asks ctest what `-L gate` actually resolves to and compares it against a
// manifest checked into the source tree, which records the count and the names
// SEPARATELY. Adding a test without touching the manifest fails on the count. Renaming
// one without touching the manifest fails on the names. Neither can be satisfied by
// editing the other half.
//
// It also proves the selector discriminates, by asserting that a label nobody uses
// selects zero while `gate` selects many. That is the assertion v1 never made, and
// making it is the entire difference.

#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

#include "tests/check.hpp"

namespace {

struct Selection {
    std::set<std::string> names;
    int total = -1; // from ctest's own "Total Tests:" line, parsed independently
    bool ran = false;
};

std::string run_capture(const std::string& cmd) {
    std::string out;
    FILE* p = ::popen(cmd.c_str(), "r");
    if (p == nullptr) {
        return out;
    }
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), p) != nullptr) {
        out.append(buf);
    }
    ::pclose(p);
    return out;
}

// Parses `ctest -N` output: "  Test  #3: test_arena" lines plus "Total Tests: N".
Selection parse_listing(const std::string& text) {
    Selection s;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t hash = line.find('#');
        const std::size_t colon = line.find(": ");
        if (line.find("Test ") != std::string::npos && hash != std::string::npos &&
            colon != std::string::npos && colon > hash) {
            s.names.insert(line.substr(colon + 2));
            continue;
        }
        const std::size_t total = line.find("Total Tests: ");
        if (total != std::string::npos) {
            s.total = std::atoi(line.c_str() + total + 13);
        }
    }
    s.ran = !text.empty();
    return s;
}

Selection list_label(const std::string& label) {
    const std::string cmd = std::string(LMP_CTEST_COMMAND) + " --test-dir " +
                            LMP_BINARY_DIR + " -N" +
                            (label.empty() ? "" : " -L " + label) + " 2>&1";
    return parse_listing(run_capture(cmd));
}

struct Manifest {
    int count = -1;
    std::set<std::string> names;
    int realmodel_count = -1;
};

Manifest read_manifest(const char* path) {
    Manifest m;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (line.rfind("count ", 0) == 0) {
            m.count = std::atoi(line.c_str() + 6);
        } else if (line.rfind("realmodel_count ", 0) == 0) {
            m.realmodel_count = std::atoi(line.c_str() + 16);
        } else {
            m.names.insert(line);
        }
    }
    return m;
}

} // namespace

TEST(the_gate_label_selects_a_nonempty_set) {
    const Selection gate = list_label("gate");
    REQUIRE(gate.ran);
    CHECK(gate.total > 0);
    CHECK(!gate.names.empty());
    // ctest's own total and the names we parsed must agree. If they diverge, the parser
    // is lying and every other assertion in this file is worthless.
    CHECK_EQ(static_cast<int>(gate.names.size()), gate.total);
}

TEST(a_label_nobody_uses_selects_nothing) {
    // The falsification. If this also came back non-empty, label selection would not be
    // discriminating and "-L gate" would be as meaningless as v1's "-E realmodel".
    const Selection bogus = list_label("no_such_label_kjhgf");
    REQUIRE(bogus.ran);
    CHECK_EQ(bogus.total, 0);
    CHECK(bogus.names.empty());

    const Selection gate = list_label("gate");
    CHECK(gate.total > bogus.total);
}

TEST(the_gate_matches_the_pinned_manifest_exactly) {
    const Manifest m = read_manifest(LMP_GATE_MANIFEST);
    REQUIRE(m.count >= 0);
    const Selection gate = list_label("gate");
    REQUIRE(gate.ran);

    // Count and names are pinned independently. Neither half can be repaired by editing
    // the other.
    CHECK_EQ(gate.total, m.count);
    CHECK_EQ(static_cast<int>(m.names.size()), m.count);

    for (const std::string& want : m.names) {
        if (gate.names.count(want) == 0) {
            ::lmp::test::record_failure(__FILE__, __LINE__,
                                        "manifest lists '" + want +
                                            "' but -L gate does not select it");
        }
        ++lmp::test::reg().checks;
    }
    for (const std::string& got : gate.names) {
        if (m.names.count(got) == 0) {
            ::lmp::test::record_failure(__FILE__, __LINE__,
                                        "-L gate selects '" + got +
                                            "' which is not in the manifest");
        }
        ++lmp::test::reg().checks;
    }
}

TEST(realmodel_tests_are_excluded_from_the_gate) {
    // S11.6: real-model tests are labelled and excluded, and never run in parallel.
    // The pin is 0 today because phase 3 has not landed. The day it does, this fails and
    // someone has to state the new number -- which is the point.
    const Manifest m = read_manifest(LMP_GATE_MANIFEST);
    REQUIRE(m.realmodel_count >= 0);
    const Selection real = list_label("realmodel");
    CHECK_EQ(real.total, m.realmodel_count);

    const Selection gate = list_label("gate");
    for (const std::string& r : real.names) {
        CHECK(gate.names.count(r) == 0);
    }
}

TEST(every_declared_test_carries_a_label) {
    // An unlabelled test is invisible to every selector and silently never runs.
    // lmp_add_test() refuses to declare one; this proves the refusal held, by checking
    // that the union of the known labels accounts for every test in the project.
    const Selection all = list_label("");
    const Selection gate = list_label("gate");
    const Selection real = list_label("realmodel");
    REQUIRE(all.ran);
    CHECK_EQ(all.total, gate.total + real.total);
}
