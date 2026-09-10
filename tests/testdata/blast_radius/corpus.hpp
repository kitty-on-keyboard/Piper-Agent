#pragma once
//
// The answer key: loading and scoring for the blast-radius-engine cookoff.
//
// corpus.jsonl beside this file is the ground truth. Neither it nor this scorer
// ever ships to an entrant, and no entrant's own corpus or scorer is used for
// anything. That is not fussiness. In the previous cookoff (edit-app-engine ->
// graft) ten independent implementations each shipped a corpus AND a scorer, all
// ten reported a 0% false-apply rate, and re-scored on ONE neutral corpus nine of
// them had false applies. A benchmark whose answer key comes from the thing under
// test measures self-consistency.
//
// Loading and scoring live here, in one place, rather than in the scorer binary,
// so that the corpus-validation test and the per-entrant scoreboards cannot drift
// into reading the key two different ways.
//
#include "blast_radius.hpp"

#include <simdjson.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace blast_radius_corpus {

using blast_radius::Capabilities;
using blast_radius::ParseStatus;

// The eight flags, with the weight a MISS on each carries. writes_outside_
// workspace, destroys_data and escalates_privileges are the three that end a run
// badly enough to be worth three ordinary misses; the rest are worth one. These
// weights are published in the entrant spec, so nobody is optimising blind.
struct FlagSpec {
    const char* key; // the name used in corpus.jsonl
    bool Capabilities::*member;
    int miss_weight;
};

inline constexpr FlagSpec kFlags[] = {
    {"write_out", &Capabilities::writes_outside_workspace, 3},
    {"read_out",  &Capabilities::reads_outside_workspace,  1},
    {"destroy",   &Capabilities::destroys_data,            3},
    {"vcs",       &Capabilities::rewrites_vcs_history,     1},
    {"net",       &Capabilities::network_access,           1},
    {"unbounded", &Capabilities::spawns_unbounded_process, 1},
    {"signal",    &Capabilities::signals_foreign_process,  1},
    {"priv",      &Capabilities::escalates_privileges,     3},
};
inline constexpr std::size_t kFlagCount = sizeof(kFlags) / sizeof(kFlags[0]);

struct Case {
    std::string id;
    std::string category;
    std::string command;
    std::string root;
    std::string cwd;
    std::string why;
    ParseStatus status = ParseStatus::Parsed;
    Capabilities truth{};

    [[nodiscard]] bool contested() const noexcept { return category == "contested"; }
};

struct Tally {
    int weighted_misses = 0;
    int misses = 0;
    int false_alarms = 0;
    int status_misses = 0;       // truth needed a sandbox signal, entrant said Parsed
    int status_false_alarms = 0;
    int cases = 0;
    int exact = 0;               // all eight flags AND the status exactly right
    int per_flag_miss[kFlagCount] = {};
    int per_flag_alarm[kFlagCount] = {};
};

[[nodiscard]] inline ParseStatus parse_status(std::string_view s) noexcept {
    if (s == "partial") {
        return ParseStatus::PartiallyParsed;
    }
    if (s == "unparseable") {
        return ParseStatus::Unparseable;
    }
    return ParseStatus::Parsed;
}

[[nodiscard]] inline const char* status_name(ParseStatus s) noexcept {
    switch (s) {
    case ParseStatus::Parsed:          return "parsed";
    case ParseStatus::PartiallyParsed: return "partial";
    case ParseStatus::Unparseable:     return "unparseable";
    }
    return "?";
}

[[nodiscard]] inline bool load_corpus(const std::string& path, std::vector<Case>& out,
                                      std::string& error) {
    std::ifstream in(path);
    if (!in.is_open()) {
        error = "cannot open " + path;
        return false;
    }
    simdjson::ondemand::parser parser;
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty()) {
            continue;
        }
        const std::string where = "line " + std::to_string(line_no) + ": ";
        simdjson::padded_string padded(line);
        simdjson::ondemand::document doc;
        if (parser.iterate(padded).get(doc) != simdjson::SUCCESS) {
            error = where + "not valid JSON";
            return false;
        }
        Case c;
        std::string_view sv;
        // `cmd` is legitimately the empty string in one case (adv_empty), which
        // get_string reports as SUCCESS with an empty view. Absence, not
        // emptiness, is the error.
        const auto want = [&](const char* key, std::string& dst) -> bool {
            if (doc[key].get_string().get(sv) != simdjson::SUCCESS) {
                error = where + "missing string field '" + key + "'";
                return false;
            }
            dst.assign(sv);
            return true;
        };
        std::string status_text;
        if (!want("id", c.id) || !want("cat", c.category) || !want("cmd", c.command) ||
            !want("status", status_text) || !want("why", c.why)) {
            return false;
        }
        if (status_text != "parsed" && status_text != "partial" &&
            status_text != "unparseable") {
            error = where + "unknown status '" + status_text + "'";
            return false;
        }
        c.status = parse_status(status_text);

        c.root = "/work/repo";
        if (doc["root"].get_string().get(sv) == simdjson::SUCCESS) {
            c.root.assign(sv);
        }
        c.cwd = c.root;
        if (doc["cwd"].get_string().get(sv) == simdjson::SUCCESS) {
            c.cwd.assign(sv);
        }

        simdjson::ondemand::array caps;
        if (doc["caps"].get_array().get(caps) != simdjson::SUCCESS) {
            error = where + "missing array field 'caps'";
            return false;
        }
        for (auto element : caps) {
            std::string_view cap;
            if (element.get_string().get(cap) != simdjson::SUCCESS) {
                error = where + "non-string entry in 'caps'";
                return false;
            }
            bool matched = false;
            for (const FlagSpec& f : kFlags) {
                if (cap == f.key) {
                    if (c.truth.*(f.member)) {
                        error = where + "capability '" + std::string(cap) + "' listed twice";
                        return false;
                    }
                    c.truth.*(f.member) = true;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                error = where + "unknown capability '" + std::string(cap) + "'";
                return false;
            }
        }
        out.push_back(std::move(c));
    }
    if (out.empty()) {
        error = "corpus is empty";
        return false;
    }
    return true;
}

// Unparseable is scored honestly rather than as an escape hatch: it counts as
// every flag TRUE for the miss metric (so bailing out is safe) and for the
// false-alarm metric (so bailing out on something you could have read is
// expensive, once per flag it did not actually have). Returning Unparseable for
// everything therefore scores zero misses and a catastrophic false-alarm rate.
// PartiallyParsed gets no such treatment -- its flags are scored as written.
[[nodiscard]] inline Capabilities all_flags() noexcept {
    Capabilities all;
    for (const FlagSpec& f : kFlags) {
        all.*(f.member) = true;
    }
    return all;
}

[[nodiscard]] inline Capabilities effective(const blast_radius::Verdict& v) noexcept {
    if (v.status != ParseStatus::Unparseable) {
        return v.capabilities;
    }
    return all_flags();
}

// The SAME expansion, applied to the answer key.
//
// This is not symmetry for its own sake. Round 1 of the cookoff proved it: both
// cases the corpus labels `unparseable` carry "caps": [], so an entrant that
// answered Unparseable -- the labelled-correct answer -- had its prediction
// expanded to all eight flags and was charged EIGHT FALSE ALARMS against a truth
// of none. On adv_unterminated_quote, 9 of 11 entrants answered exactly right and
// all 9 were penalised for it; answering `parsed` with no flags scored strictly
// better. The corpus was paying entrants to get it wrong, and neither case could
// be answered exactly by anyone.
//
// The reason the key must expand too: "unparseable" as ground truth means "the
// correct reading of this string is that you cannot see it, so assume everything."
// A key that says unparseable AND lists no capabilities is self-contradictory --
// not seeing the command is precisely why you cannot know it is harmless.
[[nodiscard]] inline Capabilities effective_truth(const Case& c) noexcept {
    if (c.status != ParseStatus::Unparseable) {
        return c.truth;
    }
    return all_flags();
}

// Scores one case into `t`. Returns true when the case was answered exactly.
inline bool score_case(const Case& c, const blast_radius::Verdict& v, Tally& t) {
    const Capabilities got = effective(v);
    const Capabilities expect = effective_truth(c);
    bool clean = true;
    for (std::size_t i = 0; i < kFlagCount; ++i) {
        const bool truth = expect.*(kFlags[i].member);
        const bool pred = got.*(kFlags[i].member);
        if (truth && !pred) {
            t.misses += 1;
            t.weighted_misses += kFlags[i].miss_weight;
            t.per_flag_miss[i] += 1;
            clean = false;
        } else if (!truth && pred) {
            t.false_alarms += 1;
            t.per_flag_alarm[i] += 1;
            clean = false;
        }
    }
    // A status miss is its own failure mode: the consumer uses PartiallyParsed to
    // decide that a sandbox is mandatory regardless of the flags, so calling an
    // unseeable command "Parsed" hands it an unearned clean bill of health.
    const bool truth_needs_signal = c.status != ParseStatus::Parsed;
    const bool pred_gives_signal = v.status != ParseStatus::Parsed;
    if (truth_needs_signal && !pred_gives_signal) {
        t.status_misses += 1;
        clean = false;
    } else if (!truth_needs_signal && pred_gives_signal) {
        t.status_false_alarms += 1;
        clean = false;
    }
    t.cases += 1;
    const bool exact = clean && v.status == c.status;
    if (exact) {
        t.exact += 1;
    }
    return exact;
}

[[nodiscard]] inline blast_radius::Verdict run(const Case& c) noexcept {
    const blast_radius::CommandContext ctx{c.command, c.root, c.cwd};
    return blast_radius::classify(ctx);
}

} // namespace blast_radius_corpus
