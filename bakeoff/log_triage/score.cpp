// Per-entrant scoreboard for the log-triage cookoff.
//
// One binary per entrant: every entrant defines log_triage::compact and they must never
// meet in one translation unit. Run with -v for a per-case breakdown, -q for one line.
//
//   cmake --build build --target log_triage_score_incumbent
//   ./build/log_triage_score_incumbent -v
//
#include "corpus.hpp"
#include "entrant_bridge.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef CORPUS_DIR
#error "CORPUS_DIR must be defined"
#endif
#ifndef ENTRANT_NAME
#error "ENTRANT_NAME must be defined"
#endif

namespace {

using log_triage_corpus::Case;
using log_triage_corpus::CaseScore;
using log_triage_corpus::Tally;

const char* adapter_label(int kind) {
    switch (kind) {
    case 0:
        return "direct";
    case 1:
        return "unit-searched";
    default:
        return "NO BUDGET";
    }
}

} // namespace

int main(int argc, char** argv) {
    bool verbose = false;
    bool quiet = false;
    bool dump = false;
    // -H scores the HELD-OUT set instead. The engine was written with corpus.jsonl open, so
    // its number there is a memory test; the holdout is the blind one.
    const char* manifest = "corpus.jsonl";
    for (int i = 1; i < argc; ++i) {
        verbose = verbose || std::strcmp(argv[i], "-v") == 0;
        quiet = quiet || std::strcmp(argv[i], "-q") == 0;
        dump = dump || std::strcmp(argv[i], "-j") == 0;
        if (std::strcmp(argv[i], "-H") == 0) {
            manifest = "holdout.jsonl";
        }
    }

    // -j: one line per scoring point, for the oracle-ceiling script. Computing the
    // ceiling BEFORE planning a merge is the step that decides whether the job is
    // merging or writing -- in the blast-radius round the ceiling was only ~18% better
    // than the best single entrant, which meant most of the work had to be written.
    if (dump) {
        for (const Case& c : log_triage_corpus::load(CORPUS_DIR, manifest)) {
            for (std::size_t bi = 0; bi < log_triage_corpus::kBudgetCount; ++bi) {
                const std::size_t budget = log_triage_corpus::kBudgets[bi];
                const std::string out = log_triage_bridge::compact_entrant(c.log, budget);
                const CaseScore s = log_triage_corpus::score_case(c, budget, out);
                std::printf("%s\t%s\t%zu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
                            ENTRANT_NAME, c.id.c_str(), budget, s.weighted(),
                            s.locator_miss, s.locator_total, s.message_miss,
                            s.message_total, s.context_miss, s.context_total,
                            s.over_budget ? 1 : 0, s.passthrough_violation ? 1 : 0);
            }
        }
        return 0;
    }

    const std::vector<Case> cases = log_triage_corpus::load(CORPUS_DIR, manifest);
    if (cases.empty()) {
        std::fprintf(stderr, "no cases loaded from %s\n", CORPUS_DIR);
        return 2;
    }

    Tally total;
    std::vector<std::pair<std::string, Tally>> by_family;
    auto family_tally = [&by_family](const std::string& f) -> Tally& {
        for (auto& e : by_family) {
            if (e.first == f) {
                return e.second;
            }
        }
        by_family.emplace_back(f, Tally{});
        return by_family.back().second;
    };

    if (!quiet) {
        std::printf("== %s ==  adapter: %s (%s)\n\n", ENTRANT_NAME,
                    adapter_label(log_triage_bridge::kAdapterKind),
                    log_triage_bridge::kAdapterNote);
    }

    for (const Case& c : cases) {
        for (std::size_t bi = 0; bi < log_triage_corpus::kBudgetCount; ++bi) {
            const std::size_t budget = log_triage_corpus::kBudgets[bi];
            const std::string out = log_triage_bridge::compact_entrant(c.log, budget);
            const CaseScore s = log_triage_corpus::score_case(c, budget, out);
            log_triage_corpus::accumulate(total, s);
            log_triage_corpus::accumulate(family_tally(c.family), s);

            if (verbose && !s.exact()) {
                std::printf("  %-28s b=%-6zu out=%-7zu w=%-4d", c.id.c_str(), budget,
                            s.out_bytes, s.weighted());
                if (s.over_budget) {
                    std::printf(" OVER-BUDGET");
                }
                if (s.passthrough_violation) {
                    std::printf(" PASSTHRU-VIOLATION");
                }
                if (s.locator_miss) {
                    std::printf(" loc %d/%d", s.locator_miss, s.locator_total);
                }
                if (s.message_miss) {
                    std::printf(" msg %d/%d", s.message_miss, s.message_total);
                }
                if (s.context_miss) {
                    std::printf(" ctx %d/%d", s.context_miss, s.context_total);
                }
                std::printf("\n");
            }
        }
    }

    if (quiet) {
        std::printf("%-14s w=%-6d loc=%-5d msg=%-5d ctx=%-5d over=%-3d pass=%-3d "
                    "exact=%d/%d\n",
                    ENTRANT_NAME, total.weighted, total.locator_miss, total.message_miss,
                    total.context_miss, total.over_budget, total.passthrough_violation,
                    total.exact, total.points);
        return 0;
    }

    auto row = [](const char* label, const Tally& t) {
        std::printf("  %-10s %8d  %6d/%-5d %6d/%-5d %6d/%-5d %5d %5d  %4d/%-3d\n", label,
                    t.weighted, t.locator_miss, t.locator_total, t.message_miss,
                    t.message_total, t.context_miss, t.context_total, t.over_budget,
                    t.passthrough_violation, t.exact, t.points);
    };
    std::printf("\n  %-10s %8s  %12s %12s %12s %5s %5s  %8s\n", "family", "weighted",
                "loc miss/n", "msg miss/n", "ctx miss/n", "over", "pass", "exact");
    for (const auto& e : by_family) {
        row(e.first.c_str(), e.second);
    }
    row("TOTAL", total);
    std::printf("\n  noise: %zu bytes retained that carry no required text\n",
                total.noise_bytes);
    return 0;
}
