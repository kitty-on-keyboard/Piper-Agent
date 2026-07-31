// The corpora arrive in phase 0, before the components they grade (S18). This test is
// what makes "arrived" mean something.
//
// Two failure modes it exists to catch, both of which actually happened:
//
//  1. `.gitignore`'s `*.log` silently swallowing the committed log corpus. The manifest
//     would still be there, the 25 files it points at would not, and the benchmark would
//     look committed and be unrunnable. So every referenced log is opened and its byte
//     count checked against the manifest's own number.
//  2. Quoting a score without checking what it counts. The pins below are the shape of
//     the key -- case counts, byte totals, the corrections applied to it -- so a key that
//     changed cannot be mistaken for a key that did not.

#include <cstddef>
#include <set>
#include <string>
#include <vector>

// Both corpora expose a header named corpus.hpp, so both are reached through the
// bakeoff root and neither can shadow the other.
#include <blast_radius/corpus.hpp>
#include <log_triage/corpus.hpp>

#include "tests/check.hpp"

using BrCase = blast_radius_corpus::Case;
using LtCase = log_triage_corpus::Case;

namespace {

std::vector<BrCase> load_br(const std::string& file) {
    std::vector<BrCase> cases;
    std::string err;
    const bool ok =
        blast_radius_corpus::load_corpus(std::string(LMP_BAKEOFF_BR_DIR) + "/" + file,
                                         cases, err);
    if (!ok) {
        lmp::test::record_failure(__FILE__, __LINE__, file + ": " + err);
    }
    ++lmp::test::reg().checks;
    return cases;
}

// Removes CSI SGR sequences. One case in the corpus is the colourised build on purpose.
std::string strip_ansi(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    std::size_t i = 0;
    while (i < in.size()) {
        if (in[i] == '\x1b' && i + 1 < in.size() && in[i + 1] == '[') {
            const std::size_t end = in.find('m', i);
            i = (end == std::string::npos) ? in.size() : end + 1;
            continue;
        }
        out.push_back(in[i]);
        ++i;
    }
    return out;
}

const BrCase* find(const std::vector<BrCase>& cases, const std::string& id) {
    for (const BrCase& c : cases) {
        if (c.id == id) {
            return &c;
        }
    }
    return nullptr;
}

} // namespace

// --- blast radius ----------------------------------------------------------

TEST(blast_radius_corpus_loads_at_its_pinned_size) {
    const std::vector<BrCase> cases = load_br("corpus.jsonl");
    // 187 cases: 179 headline + 8 contested. Re-measured from the ported file, not
    // taken from the v1 README.
    CHECK_EQ(cases.size(), std::size_t{187});

    std::size_t contested = 0;
    for (const BrCase& c : cases) {
        contested += c.contested() ? 1U : 0U;
    }
    CHECK_EQ(contested, std::size_t{8});
    CHECK_EQ(cases.size() - contested, std::size_t{179});
}

TEST(blast_radius_holdout_loads_and_is_disjoint) {
    const std::vector<BrCase> corpus = load_br("corpus.jsonl");
    const std::vector<BrCase> holdout = load_br("holdout.jsonl");
    CHECK_EQ(holdout.size(), std::size_t{42});

    // A holdout that overlaps the tuned-against set is not held out.
    std::set<std::string> corpus_ids;
    for (const BrCase& c : corpus) {
        corpus_ids.insert(c.id);
    }
    std::size_t overlap = 0;
    for (const BrCase& h : holdout) {
        overlap += corpus_ids.count(h.id);
    }
    CHECK_EQ(overlap, std::size_t{0});
}

TEST(every_case_id_is_unique) {
    // A duplicate id means one case silently shadows another in any map-keyed report,
    // and the scoreboard's denominator stops matching its numerator.
    for (const char* file : {"corpus.jsonl", "holdout.jsonl"}) {
        const std::vector<BrCase> cases = load_br(file);
        std::set<std::string> ids;
        for (const BrCase& c : cases) {
            ids.insert(c.id);
        }
        CHECK_EQ(ids.size(), cases.size());
    }
}

TEST(every_case_carries_its_reasoning) {
    // "Every case also carries a one-line `why`" -- README.md. A key entry with no
    // argument behind it cannot be disputed, and an undisputable key is the thing S11.3
    // warns about.
    const std::vector<BrCase> cases = load_br("corpus.jsonl");
    std::size_t missing_why = 0;
    std::size_t missing_root = 0;
    for (const BrCase& c : cases) {
        missing_why += c.why.empty() ? 1U : 0U;
        missing_root += c.root.empty() ? 1U : 0U;
    }
    CHECK_EQ(missing_why, std::size_t{0});
    CHECK_EQ(missing_root, std::size_t{0});
}

TEST(the_three_key_corrections_are_applied_and_only_those) {
    // KEY_CORRECTIONS.md records exactly three relabelled cases. Applying a fourth
    // without writing it down fails here; writing one down without applying it fails
    // here too.
    const std::vector<BrCase> cases = load_br("corpus.jsonl");

    for (const char* id : {"benign_ctest", "look_cargo_offline", "chain_mkdir_cmake"}) {
        const BrCase* c = find(cases, id);
        REQUIRE(c != nullptr);
        // Rule 7: these run project code whose bytes are not in the string.
        CHECK(c->status == blast_radius::ParseStatus::PartiallyParsed);
        // The flags are unchanged -- rule 7's second sentence, "the visible flags still
        // stand". Nothing dangerous is visible in any of the three.
        CHECK(!c->truth.writes_outside_workspace);
        CHECK(!c->truth.destroys_data);
        CHECK(!c->truth.network_access);
        CHECK(c->why.find("KEY_CORRECTIONS.md") != std::string::npos);
    }

    // The consistency the corrections were argued from: same rule, same answer.
    for (const char* id : {"chain_make_clean", "indir_make_install", "indir_npm_build",
                           "look_npm_format", "net_npm_install"}) {
        const BrCase* c = find(cases, id);
        REQUIRE(c != nullptr);
        CHECK(c->status == blast_radius::ParseStatus::PartiallyParsed);
    }

    // Nothing else was quietly moved: exactly three cases mention the corrections doc.
    std::size_t annotated = 0;
    for (const BrCase& c : cases) {
        annotated += (c.why.find("KEY_CORRECTIONS.md") != std::string::npos) ? 1U : 0U;
    }
    CHECK_EQ(annotated, std::size_t{3});
}

// --- log triage ------------------------------------------------------------

TEST(log_triage_corpus_loads_with_every_log_present) {
    const std::vector<LtCase> cases = log_triage_corpus::load(LMP_BAKEOFF_LT_DIR, "corpus.jsonl");
    CHECK_EQ(cases.size(), std::size_t{25});

    std::size_t total = 0;
    std::size_t empty_logs = 0;
    for (const LtCase& c : cases) {
        // load() reads each log off disk. An empty one means the file was not there --
        // which is exactly what `*.log` in .gitignore does if the negation is dropped.
        empty_logs += c.log.empty() ? 1U : 0U;
        total += c.log.size();
    }
    CHECK_EQ(empty_logs, std::size_t{0});
    // 971,544 bytes, re-measured from the ported tree.
    CHECK_EQ(total, std::size_t{971544});
}

TEST(log_triage_holdout_loads_and_is_disjoint) {
    const std::vector<LtCase> corpus = log_triage_corpus::load(LMP_BAKEOFF_LT_DIR, "corpus.jsonl");
    const std::vector<LtCase> holdout = log_triage_corpus::load(LMP_BAKEOFF_LT_DIR, "holdout.jsonl");
    CHECK_EQ(holdout.size(), std::size_t{7});

    std::size_t total = 0;
    for (const LtCase& c : holdout) {
        CHECK(!c.log.empty());
        total += c.log.size();
    }
    CHECK_EQ(total, std::size_t{274221});

    std::set<std::string> ids;
    for (const LtCase& c : corpus) {
        ids.insert(c.id);
    }
    std::size_t overlap = 0;
    for (const LtCase& h : holdout) {
        overlap += ids.count(h.id);
    }
    CHECK_EQ(overlap, std::size_t{0});
}

TEST(the_compiler_wrote_the_key_and_the_key_points_into_the_log) {
    // "No diagnostic in this corpus was labelled by hand" -- log_triage/corpus.hpp. The
    // check that keeps that true: the key must still point at its own evidence.
    //
    // The first version of this test asserted that every `rendered` line appears verbatim
    // in its log. Four did not, and the corpus was right and the assertion was wrong --
    // it claimed a property the corpus never offered. Measured rather than assumed
    // (S19.3, attribute by intervention):
    //
    //   * one case, build_color_diagnostics, is the ANSI-colorised log on purpose, so
    //     `rendered` is the de-colourised form. It matches once the escapes are stripped.
    //   * three, all in pytest_failures, are pytest short-summary lines the key composes
    //     from a traceback spread over several lines. There is no single line to match.
    //
    // So the invariant pinned here is the LOCATOR, which is exact for all 61 diagnostics
    // and is also the field the scorer weights at 3 -- "without file:line the agent does
    // not know where to type". `rendered` gets the weaker pin plus a named exception
    // count, so a fourth exception fails rather than blending in.
    const std::vector<LtCase> cases = log_triage_corpus::load(LMP_BAKEOFF_LT_DIR, "corpus.jsonl");

    std::size_t diagnostics = 0;
    std::size_t locator_orphans = 0;
    std::size_t rendered_orphans = 0;
    std::size_t rendered_orphans_outside_pytest = 0;

    for (const LtCase& c : cases) {
        const std::string clean = strip_ansi(c.log);
        for (const log_triage_corpus::Diagnostic& d : c.local) {
            ++diagnostics;
            if (!d.locator.empty() && clean.find(d.locator) == std::string::npos) {
                ++locator_orphans;
            }
            if (clean.find(d.rendered) == std::string::npos) {
                ++rendered_orphans;
                rendered_orphans_outside_pytest += (c.tool == "pytest") ? 0U : 1U;
            }
        }
    }

    CHECK_EQ(diagnostics, std::size_t{61});
    // Exact. Every locator the key names is in the log the key was extracted from.
    CHECK_EQ(locator_orphans, std::size_t{0});
    // The three pytest syntheses, and nothing else.
    CHECK_EQ(rendered_orphans, std::size_t{3});
    CHECK_EQ(rendered_orphans_outside_pytest, std::size_t{0});
}

TEST(the_holdout_is_harder_than_the_tuned_set) {
    // S11.3: assert the holdout stays harder per point. The engine does not exist yet, so
    // the assertion available today is structural -- the holdout's logs are bigger per
    // case, which is what makes them harder to compact within a budget.
    const std::vector<LtCase> corpus = log_triage_corpus::load(LMP_BAKEOFF_LT_DIR, "corpus.jsonl");
    const std::vector<LtCase> holdout = log_triage_corpus::load(LMP_BAKEOFF_LT_DIR, "holdout.jsonl");
    REQUIRE(!corpus.empty());
    REQUIRE(!holdout.empty());

    const std::size_t corpus_avg = 971544 / corpus.size();
    const std::size_t holdout_avg = 274221 / holdout.size();
    CHECK(holdout_avg > corpus_avg);
}
