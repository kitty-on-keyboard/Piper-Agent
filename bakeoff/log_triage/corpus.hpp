#pragma once
//
// The answer key: loading and scoring for the log-triage cookoff.
//
// corpus.jsonl and logs/ beside this file are the ground truth, and THE COMPILER WROTE
// THEM. Every case is a real source tree with a real defect, built by the real
// cmake/clang/swiftc/python/ctest on the machine that generated it, and compiled twice:
// once with full rendering (that output is the corpus input, byte for byte) and once
// through the compiler's own noise-suppression flags (that output is the key). See
// generate_corpus.py. No diagnostic in this corpus was labelled by hand.
//
// That matters because of what happened in the two previous cookoffs. In the graft round,
// ten implementations each shipped a corpus AND a scorer, all ten reported a 0% false-apply
// rate, and re-scored on one neutral corpus nine of them had false applies -- a benchmark
// whose key comes from the thing under test measures self-consistency. In the blast-radius
// round the corpus was neutral and the key was still ours, and the entrants found four
// defects in it. Here the key is the toolchain's opinion, which is the strongest position
// we have been in: to argue with it you have to argue with clang.
//
// Loading and scoring live here rather than in the scorer binary so the corpus-validation
// test and the per-entrant scoreboards cannot drift into reading the key two different ways.
//
#include <simdjson.h>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace log_triage_corpus {

// The budgets every case is scored at. 8192 is SubprocessVerifier::execute's default and
// 16384 is what the shell tool passes (MAX_OUTPUT_BYTES); 2048 is the pressure test, and
// it is not hypothetical -- it is what a budget looks like once the context builder has
// spent the window on the mission, the plan and three earlier tool results.
inline constexpr std::size_t kBudgets[] = {2048, 8192, 16384};
inline constexpr std::size_t kBudgetCount = sizeof(kBudgets) / sizeof(kBudgets[0]);

// Miss weights. Published here rather than tuned later.
//
//   locator  3  Without file:line the agent does not know where to type. It cannot
//               recover this by reading a file, because it does not know which file.
//   message  2  Without the message it does not know what is wrong -- but it does have
//               the locator, so it can read the source and often see it. Strictly less
//               bad than losing the locator, hence 2 and not 3.
//   context  1  The source line and caret. Losing it is what made the agent edit line 35
//               instead of line 50 in the incident recorded at subprocess_verifier.hpp:407.
//   passthru 3  Rewriting a log that already fits destroys information for no gain.
inline constexpr int kLocatorWeight = 3;
inline constexpr int kMessageWeight = 2;
inline constexpr int kContextWeight = 1;
inline constexpr int kPassthroughWeight = 3;

struct Diagnostic {
    std::string level;    // error | fatal error | note
    std::string path;     // may be empty (linker, python exception, ctest)
    std::string locator;  // "path:line:col", empty when path is
    std::string message;
    std::string rendered; // the whole line as the FULL log printed it
    std::vector<std::string> context; // source + caret lines beneath it
};

struct Case {
    std::string id;
    std::string family; // build | link | bare | swift | other
    std::string tool;   // cmake | clang | swift | python | ctest
    std::string why;
    std::string log; // the complete captured output; the input to compact()
    bool has_primary = false;
    Diagnostic primary;
    std::vector<Diagnostic> local;
    int error_count = 0;
};

struct CaseScore {
    bool over_budget = false;
    bool passthrough_violation = false;
    bool passthrough_expected = false;
    int locator_miss = 0;
    int message_miss = 0;
    int context_miss = 0;
    int locator_total = 0;
    int message_total = 0;
    int context_total = 0;
    std::size_t out_bytes = 0;
    std::size_t noise_bytes = 0;
    std::size_t budget = 0;

    [[nodiscard]] int weighted() const noexcept {
        return kLocatorWeight * locator_miss + kMessageWeight * message_miss +
               kContextWeight * context_miss +
               kPassthroughWeight * (passthrough_violation ? 1 : 0);
    }
    // Every requirement met: within budget, nothing dropped, and a log that already
    // fitted came back untouched.
    [[nodiscard]] bool exact() const noexcept {
        return !over_budget && !passthrough_violation && locator_miss == 0 &&
               message_miss == 0 && context_miss == 0;
    }
};

struct Tally {
    int weighted = 0;
    int locator_miss = 0;
    int message_miss = 0;
    int context_miss = 0;
    // Denominators. A miss count without them says nothing: "34 locator misses" is
    // excellent out of 300 and catastrophic out of 40.
    int locator_total = 0;
    int message_total = 0;
    int context_total = 0;
    int over_budget = 0;
    int passthrough_violation = 0;
    int exact = 0;
    int points = 0; // cases x budgets
    std::size_t noise_bytes = 0;
};

// ---------------------------------------------------------------------------
// Text handling
// ---------------------------------------------------------------------------

// Both sides of every comparison are ANSI-stripped. build_color_diagnostics is the same
// defect as build_undeclared_late with -fcolor-diagnostics forced, which is what happens
// whenever a build runs under a pty; an entrant that strips escapes and one that passes
// them through should be judged on what they KEPT, not on that choice. The escape bytes
// still count against the budget, which is the honest cost of not stripping them.
[[nodiscard]] inline std::string strip_ansi(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) {
        if (s[i] == '\x1b' && i + 1 < s.size() && s[i + 1] == '[') {
            std::size_t j = i + 2;
            while (j < s.size() && ((s[j] >= '0' && s[j] <= '9') || s[j] == ';')) {
                ++j;
            }
            if (j < s.size()) {
                ++j; // the final byte
            }
            i = j;
            continue;
        }
        out.push_back(s[i]);
        ++i;
    }
    return out;
}

[[nodiscard]] inline bool contains(std::string_view haystack, std::string_view needle) {
    return !needle.empty() && haystack.find(needle) != std::string_view::npos;
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

[[nodiscard]] inline std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

[[nodiscard]] inline Diagnostic parse_diagnostic(simdjson::dom::element e) {
    Diagnostic d;
    d.level = std::string(std::string_view(e["level"]));
    d.path = std::string(std::string_view(e["path"]));
    d.locator = std::string(std::string_view(e["locator"]));
    d.message = std::string(std::string_view(e["message"]));
    d.rendered = std::string(std::string_view(e["rendered"]));
    for (simdjson::dom::element c : simdjson::dom::array(e["context"])) {
        d.context.emplace_back(std::string_view(c));
    }
    return d;
}

// `dir` holds the manifest and its log directory. `manifest` selects which set:
// corpus.jsonl (the tuned-against benchmark) or holdout.jsonl (written after the engine,
// scored once). The log path inside each record already names its own directory.
[[nodiscard]] inline std::vector<Case> load(const std::string& dir,
                                            const std::string& manifest = "corpus.jsonl") {
    const std::string jsonl = read_file(dir + "/" + manifest);
    std::vector<Case> cases;
    simdjson::dom::parser parser;
    std::size_t start = 0;
    while (start < jsonl.size()) {
        const std::size_t nl = jsonl.find('\n', start);
        const std::size_t end = (nl == std::string::npos) ? jsonl.size() : nl;
        const std::string_view line(jsonl.data() + start, end - start);
        start = (nl == std::string::npos) ? jsonl.size() : nl + 1;
        if (line.empty()) {
            continue;
        }
        // simdjson requires padded input; a std::string copy gives it the slack.
        simdjson::padded_string padded(line);
        simdjson::dom::element doc = parser.parse(padded);

        Case c;
        c.id = std::string(std::string_view(doc["id"]));
        c.family = std::string(std::string_view(doc["family"]));
        c.tool = std::string(std::string_view(doc["tool"]));
        c.why = std::string(std::string_view(doc["why"]));
        c.error_count = static_cast<int>(int64_t(doc["error_count"]));
        c.log = read_file(dir + "/" + std::string(std::string_view(doc["log"])));

        simdjson::dom::element primary;
        if (!doc["primary"].get(primary) && !primary.is_null()) {
            c.has_primary = true;
            c.primary = parse_diagnostic(primary);
        }
        for (simdjson::dom::element l : simdjson::dom::array(doc["local"])) {
            c.local.push_back(parse_diagnostic(l));
        }
        cases.push_back(std::move(c));
    }
    return cases;
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

// The diagnostics a compacted output is required to carry: the primary, plus every
// diagnostic located in a file the agent can edit.
//
// Deduplicated on the rendered line AND THE LOCATOR TOGETHER, because the primary is
// usually also a local one and SwiftPM prints every diagnostic twice -- but two failures
// at different places are two requirements even when they read identically.
//
// Deduplicating on the rendered line alone silently merged them: rustc renders every type
// error as `error[E0308]: mismatched types`, so ho_rustc_no_cargo's two distinct errors,
// five source lines apart, were scored as one and the second one's address was not
// required at all. A corpus that does not ask for something cannot notice it missing.
[[nodiscard]] inline std::vector<const Diagnostic*> required(const Case& c) {
    std::vector<const Diagnostic*> out;
    auto push = [&out](const Diagnostic* d) {
        for (const Diagnostic* seen : out) {
            if (seen->rendered == d->rendered && seen->locator == d->locator) {
                return;
            }
        }
        out.push_back(d);
    };
    if (c.has_primary) {
        push(&c.primary);
    }
    for (const Diagnostic& d : c.local) {
        push(&d);
    }
    return out;
}

// Score one compacted output.
//
// An OVER-BUDGET result is scored as if it retained nothing. That is not a punishment
// tariff, it is the arithmetic: the caller hard-truncates anything longer than the
// budget, so the bytes past it were never delivered. Without this rule the highest
// score would go to `return std::string(full)`.
[[nodiscard]] inline CaseScore score_case(const Case& c, std::size_t budget,
                                          std::string_view out) {
    CaseScore s;
    s.budget = budget;
    s.out_bytes = out.size();
    s.over_budget = out.size() > budget;
    s.passthrough_expected = c.log.size() <= budget;
    if (s.passthrough_expected) {
        s.passthrough_violation = (out != c.log);
    }

    const std::vector<const Diagnostic*> req = required(c);
    const std::string flat = strip_ansi(out);

    for (const Diagnostic* d : req) {
        if (!d->locator.empty()) {
            ++s.locator_total;
            if (s.over_budget || !contains(flat, d->locator)) {
                ++s.locator_miss;
            }
        }
        if (!d->message.empty()) {
            ++s.message_total;
            if (s.over_budget || !contains(flat, d->message)) {
                ++s.message_miss;
            }
        }
        for (const std::string& ctx : d->context) {
            const std::string trimmed = strip_ansi(ctx);
            if (trimmed.find_first_not_of(" \t") == std::string::npos) {
                continue;
            }
            ++s.context_total;
            if (s.over_budget || !contains(flat, trimmed)) {
                ++s.context_miss;
            }
        }
    }

    // Noise: output bytes on lines that carry none of the required text. Reported and
    // NOT weighted -- some of it (the command echo, the build's final status) is
    // legitimately useful and this corpus does not try to adjudicate which.
    std::size_t essential = 0;
    std::size_t pos = 0;
    while (pos < flat.size()) {
        const std::size_t nl = flat.find('\n', pos);
        const std::size_t end = (nl == std::string::npos) ? flat.size() : nl;
        const std::string_view line(flat.data() + pos, end - pos);
        bool is_essential = false;
        for (const Diagnostic* d : req) {
            if ((!d->locator.empty() && contains(line, d->locator)) ||
                (!d->message.empty() && contains(line, d->message))) {
                is_essential = true;
                break;
            }
            for (const std::string& ctx : d->context) {
                const std::string t = strip_ansi(ctx);
                if (!t.empty() && contains(line, t)) {
                    is_essential = true;
                    break;
                }
            }
            if (is_essential) {
                break;
            }
        }
        if (is_essential) {
            essential += line.size() + 1;
        }
        pos = (nl == std::string::npos) ? flat.size() : nl + 1;
    }
    s.noise_bytes = flat.size() > essential ? flat.size() - essential : 0;
    return s;
}

inline void accumulate(Tally& t, const CaseScore& s) {
    t.weighted += s.weighted();
    t.locator_miss += s.locator_miss;
    t.message_miss += s.message_miss;
    t.context_miss += s.context_miss;
    t.locator_total += s.locator_total;
    t.message_total += s.message_total;
    t.context_total += s.context_total;
    t.over_budget += s.over_budget ? 1 : 0;
    t.passthrough_violation += s.passthrough_violation ? 1 : 0;
    t.exact += s.exact() ? 1 : 0;
    t.noise_bytes += s.noise_bytes;
    ++t.points;
}

} // namespace log_triage_corpus
