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

} // namespace

std::string canonicalize_check(std::string_view command) {
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
    }

    // Collapse internal whitespace so spacing is not an identity.
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
    const tools::ToolResult r =
        registry_.execute("shell", {{"command", command}}, approved_tier);

    context::VerificationRecord rec;
    rec.contract = contract_id;
    // Refused is NOT failed (S6.2): the command never ran, so it is not evidence in
    // either direction, and recording it as a failure would send the agent off fixing
    // a build that was never attempted.
    rec.passed = r.status == tools::Status::Ok;
    rec.ran = r.status != tools::Status::Refused;
    // Asked BEFORE this record joins the ledger, so a check cannot prove itself.
    rec.falsifiable = is_proven(contract_id);
    rec.detail = r.status == tools::Status::Refused
                     ? "REFUSED (never ran): " + r.summary
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
        return registry_.execute("shell", {{"command", command}}, approved_tier).status ==
               tools::Status::Ok;
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
