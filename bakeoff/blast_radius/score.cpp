// Per-entrant scoreboard for the blast-radius-engine cookoff.
//
// One binary per entrant, because every entrant defines the same symbol
// (blast_radius::classify) and ten of them must never meet in one translation
// unit. Loading and scoring live in corpus.hpp so this file and the corpus test
// cannot read the answer key two different ways.
//
//   cmake --build build --target blast_radius_score_incumbent
//   ./build/blast_radius_score_incumbent          # scorecard
//   ./build/blast_radius_score_incumbent "" -v    # plus every imperfect case
//
// Exit code is 0 whenever the corpus loaded and every case was scored. The score
// itself is reported, never asserted here; the ratchet on the incumbent lives in
// src/testing/test_blast_radius_corpus.cpp.

#ifndef ENTRANT_NAME
#define ENTRANT_NAME "incumbent"
#endif

#include "corpus.hpp"
#include "entrant_forward.hpp"

#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace blast_radius_corpus;

std::string describe(const Case& c, const blast_radius::Verdict& v) {
    const auto flags = [](const Capabilities& caps) {
        std::string s;
        for (const FlagSpec& f : kFlags) {
            if (caps.*(f.member)) {
                s.append(f.key).push_back(' ');
            }
        }
        return s.empty() ? std::string("-") : s;
    };
    std::ostringstream os;
    os << "  " << c.id << "  [" << c.category << "]\n"
       << "    cmd    : " << c.command << "\n"
       << "    expect : " << status_name(c.status) << "  " << flags(effective_truth(c)) << "\n"
       << "    got    : " << status_name(v.status) << "  " << flags(effective(v)) << "\n"
       << "    why    : " << c.why << "\n";
    return os.str();
}

// One JSON line per case, expected beside actual. The scorecard answers "how good
// is this entrant"; this answers "which entrant should own this component", which
// is the question a merge-by-component consolidation actually needs. It also makes
// "every entrant fails this case" visible -- which is evidence about the CASE.
std::string as_json(const Case& c, const blast_radius::Verdict& v) {
    const auto bits = [](const Capabilities& caps) {
        std::string s;
        for (const FlagSpec& f : kFlags) {
            s.push_back(caps.*(f.member) ? '1' : '0');
        }
        return s;
    };
    std::ostringstream os;
    os << "CASE {\"entrant\":\"" << ENTRANT_NAME << "\",\"id\":\"" << c.id
       << "\",\"cat\":\"" << c.category << "\",\"want\":\"" << bits(effective_truth(c))
       << "\",\"got\":\"" << bits(effective(v)) << "\",\"want_status\":\""
       << status_name(c.status) << "\",\"got_status\":\"" << status_name(v.status)
       << "\"}";
    return os.str();
}

} // namespace

int main(int argc, char** argv) {
    std::string corpus_path =
#ifdef CORPUS_PATH
        CORPUS_PATH;
#else
        "corpus.jsonl";
#endif
    if (argc > 1 && argv[1][0] != '\0') {
        corpus_path = argv[1];
    }
    bool verbose = false;
    bool per_case_json = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "-v") {
            verbose = true;
        } else if (std::string_view(argv[i]) == "-j") {
            per_case_json = true;
        }
    }

    std::vector<Case> cases;
    std::string error;
    if (!load_corpus(corpus_path, cases, error)) {
        std::fprintf(stderr, "[corpus] %s\n", error.c_str());
        return 2;
    }

    Tally headline;
    Tally contested;
    std::map<std::string, Tally> by_category;
    std::vector<std::string> imperfect;

    for (const Case& c : cases) {
        const blast_radius::Verdict v = run(c);
        const bool exact = score_case(c, v, c.contested() ? contested : headline);
        score_case(c, v, by_category[c.category]);
        if (!exact && verbose) {
            imperfect.push_back(describe(c, v));
        }
        if (per_case_json) {
            std::printf("%s\n", as_json(c, v).c_str());
        }
    }

    std::printf("=== blast-radius scorecard: %s ===\n", ENTRANT_NAME);
    std::printf("corpus: %s\n%zu cases: %d headline + %d contested\n\n",
                corpus_path.c_str(), cases.size(), headline.cases, contested.cases);

    std::printf("HEADLINE (contested excluded)\n");
    std::printf("  weighted misses     %6d   <- PRIMARY metric, lower is better\n",
                headline.weighted_misses);
    std::printf("  raw misses          %6d   of %d flag-slots\n", headline.misses,
                headline.cases * static_cast<int>(kFlagCount));
    std::printf("  false alarms        %6d   <- secondary metric\n", headline.false_alarms);
    std::printf("  status misses       %6d   (needed a sandbox signal, said 'parsed')\n",
                headline.status_misses);
    std::printf("  status false alarms %6d\n", headline.status_false_alarms);
    std::printf("  exact cases         %6d / %d  (%.1f%%)\n\n", headline.exact,
                headline.cases,
                headline.cases ? 100.0 * headline.exact / headline.cases : 0.0);

    std::printf("PER FLAG (headline only)      miss  alarm  weight\n");
    for (std::size_t i = 0; i < kFlagCount; ++i) {
        std::printf("  %-24s %5d  %5d  %5d\n", kFlags[i].key, headline.per_flag_miss[i],
                    headline.per_flag_alarm[i], kFlags[i].miss_weight);
    }

    std::printf("\nPER CATEGORY               cases  wmiss  alarm  exact\n");
    for (const auto& [name, t] : by_category) {
        std::printf("  %-24s %4d  %5d  %5d  %4d\n", name.c_str(), t.cases,
                    t.weighted_misses, t.false_alarms, t.exact);
    }

    std::printf("\nCONTESTED (reported, never ranked: %d cases)\n", contested.cases);
    std::printf("  weighted misses %d, false alarms %d, exact %d\n",
                contested.weighted_misses, contested.false_alarms, contested.exact);

    if (verbose && !imperfect.empty()) {
        std::printf("\n=== %zu imperfect cases ===\n", imperfect.size());
        for (const std::string& s : imperfect) {
            std::fputs(s.c_str(), stdout);
        }
    }

    // One machine-readable line, so a driver can rank entrants without parsing
    // the table above.
    std::printf("\nJSON {\"entrant\":\"%s\",\"weighted_misses\":%d,\"misses\":%d,"
                "\"false_alarms\":%d,\"status_misses\":%d,\"status_false_alarms\":%d,"
                "\"exact\":%d,\"cases\":%d}\n",
                ENTRANT_NAME, headline.weighted_misses, headline.misses,
                headline.false_alarms, headline.status_misses,
                headline.status_false_alarms, headline.exact, headline.cases);
    return 0;
}
