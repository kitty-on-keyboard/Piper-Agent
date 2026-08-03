#include "src/loop/verification.hpp"

#include <algorithm>
#include <cctype>

namespace lmp::loop {
namespace {

std::string trim(std::string s) {
    const auto not_space = [](unsigned char c) { return std::isspace(c) == 0; };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

// A trailing pipe into a pager or a truncator: `| tail -20`, `| head -n 5`, `| cat`.
//
// These are not neutral, and that is the point of removing them rather than tolerating
// them. A shell pipeline exits with the status of its LAST element, so
// `pytest tests/ | tail -20` exits 0 when pytest fails, when pytest errors, and when
// there is no pytest at all -- `tail` succeeded either way.
//
// MEASURED: a run declared `python -m pytest tests/ -v --tb=short 2>&1 | tail -20` as its
// contract in an EMPTY workspace. `python` does not exist on this host, so the baseline
// was `command not found` -- and it was recorded as PASSING, because tail exited 0. The
// agent was told its tests already passed before it had written a line, and no failing
// test it wrote afterwards could ever be seen to fail.
//
// Only formatters are stripped. `| grep -q FAIL` and `| wc -l` are predicates whose
// status is load-bearing, and removing one would change what is being checked.
std::size_t formatter_pipe_at(const std::string& s) {
    static constexpr std::string_view kFormatters[] = {"cat", "tail", "head",
                                                       "tee", "less", "more"};
    const std::size_t bar = s.find_last_of('|');
    // `||` is an or-list, not a pipe, and its right side is a real command.
    if (bar == std::string::npos || bar == 0 || s[bar - 1] == '|' ||
        (bar + 1 < s.size() && s[bar + 1] == '|')) {
        return std::string::npos;
    }
    const std::size_t word = s.find_first_not_of(" \t", bar + 1);
    if (word == std::string::npos) {
        return std::string::npos;
    }
    const std::size_t end = s.find_first_of(" \t", word);
    const std::string name =
        s.substr(word, end == std::string::npos ? std::string::npos : end - word);
    for (std::string_view f : kFormatters) {
        if (name == f) {
            return bar;
        }
    }
    return std::string::npos;
}

} // namespace

std::string executable_form(std::string_view command) {
    std::string s(command);

    // Strip reporting wrappers that change nothing about what is being verified. This
    // is why: a proof of falsifiability is expensive (it breaks and restores the
    // workspace), and if `cmake --build build` and `cmake --build build; echo $?` had
    // separate identities the agent would pay for the proof twice and, worse, could
    // present the unproven variant as evidence.
    static constexpr std::string_view kTrailers[] = {"; echo $?", "&& echo $?",
                                                     "; echo done", "2>&1", "| cat"};
    bool changed = true;
    while (changed) {
        changed = false;
        s = trim(s);
        for (std::string_view t : kTrailers) {
            if (s.size() > t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0) {
                s.resize(s.size() - t.size());
                changed = true;
            }
        }
        // After the fixed trailers, so `... | tail -20` and then the `2>&1` it was hiding
        // both come off in one pass.
        if (const std::size_t bar = formatter_pipe_at(s); bar != std::string::npos) {
            s.resize(bar);
            changed = true;
        }
    }
    return s;
}

namespace {

std::string collapse_spaces_for_identity(const std::string& s) {
    std::string out;
    bool in_space = false;
    for (char c : s) {
        const bool space = std::isspace(static_cast<unsigned char>(c)) != 0;
        if (space) {
            in_space = true;
            continue;
        }
        if (in_space && !out.empty()) {
            out.push_back(' ');
        }
        in_space = false;
        out.push_back(c);
    }
    return out;
}

} // namespace

// IDENTITY, not something to run. The whitespace collapse is exactly why this and
// executable_form() are two functions: `pytest -k "a  or  b"` is the same CHECK as its
// single-spaced spelling and must share one ledger entry, but it is NOT the same command,
// and running this form would silently rewrite the model's own quoted arguments.
std::string canonicalize_check(std::string_view command) {
    return collapse_spaces_for_identity(executable_form(command));
}

bool Verifier::is_proven(const std::string& command) const {
    const std::string canon = canonicalize_check(command);
    if (std::find(proven_.begin(), proven_.end(), canon) != proven_.end()) {
        return true;
    }
    // A red OBSERVED earlier in this run is the proof, and it is free.
    //
    // This is the FAIL_TO_PASS discipline: run the check before the fix, see it red, fix,
    // see it green. That sequence is exactly "this check has been shown capable of
    // failing", and it is what the surrounding industry actually does -- reverting a
    // patch to manufacture a red is mutation testing, a QA activity, not an inline agent
    // step. prove_falsifiable() remains for checks a run wants to prove deliberately.
    //
    // A refusal is skipped: the command never ran, so it is not evidence (S6.2).
    for (const context::VerificationRecord& v : ctx_.verifications()) {
        if (v.ran && !v.passed && v.contract == canon) {
            return true;
        }
    }
    return false;
}

bool Verifier::run_and_record(const std::string& command, int approved_tier) {
    return run_and_record_as(command, approved_tier, canonicalize_check(command));
}

bool Verifier::run_and_record_as(const std::string& command, int approved_tier,
                                 const std::string& contract_id) {
    // What RUNS has the status-swallowing wrappers removed and nothing else changed.
    // A contract came to be recorded green on the strength of `tail`'s exit status; this
    // is the fix. It cannot run something weaker than what was asked for -- it removes
    // only the part that was hiding the answer. Deliberately NOT canonicalize_check(),
    // which also collapses whitespace: that is right for a ledger key and wrong for a
    // command line.
    const std::string to_run = executable_form(command);
    const tools::ToolResult r =
        registry_.execute("shell", {{"command", to_run}}, approved_tier);

    context::VerificationRecord rec;
    rec.contract = contract_id;
    // Refused is NOT failed (S6.2): the command never ran, so it is not evidence in
    // either direction, and recording it as a failure would send the agent off fixing
    // a build that was never attempted.
    //
    // A command the shell could not execute is the same case wearing a different exit
    // code -- see ToolResult::never_executed(). `python: command not found` is not the
    // test suite failing, and counting it as a red would hand the run a falsifiability
    // proof for a check that has never once been executed.
    rec.passed = r.status == tools::Status::Ok;
    rec.ran = r.status != tools::Status::Refused && !r.never_executed();
    // Asked BEFORE this record joins the ledger, so a check cannot prove itself.
    rec.falsifiable = is_proven(contract_id);
    rec.detail = r.status == tools::Status::Refused ? "REFUSED (never ran): " + r.summary
                 : r.never_executed()
                     ? "NEVER RAN (the command could not be executed, so this is not "
                       "evidence either way): " + r.summary
                     : r.summary;
    ctx_.record_verification(rec);
    return rec.passed;
}

bool Verifier::prove_falsifiable(const std::string& command, int approved_tier,
                                 const std::function<bool()>& breaker,
                                 const std::function<bool()>& restore) {
    const std::string canon = canonicalize_check(command);
    if (is_proven(command)) {
        return true; // paid for once
    }

    const auto run = [&]() {
        // Same reason as run_and_record_as: a green/red/green sequence read off a
        // formatter's exit status would prove nothing at all.
        return registry_.execute("shell", {{"command", executable_form(command)}},
                                 approved_tier)
                   .status == tools::Status::Ok;
    };

    // Green first: a check that is already red proves nothing by being broken.
    if (!run()) {
        return false;
    }
    if (!breaker()) {
        return false;
    }
    const bool went_red = !run();
    if (!restore()) {
        return false;
    }
    const bool back_green = run();

    // All three must hold: green -> red under intervention -> green again. Anything
    // less and the check is not measuring what it claims.
    if (!went_red || !back_green) {
        return false;
    }
    proven_.push_back(canon);
    return true;
}

} // namespace lmp::loop
