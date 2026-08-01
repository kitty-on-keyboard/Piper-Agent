// The approval policy's pure functions (S7.2), declared in agent.hpp.
//
// Split out of agent.cpp because they are the one part of the gate that is decidable
// without an Agent: given a RiskHint and the thresholds, the answer is a value. That makes
// them directly testable, which test_loop does, and it keeps the loop file about the loop.

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "src/loop/agent.hpp"

namespace lmp::loop {

double risk_score(const tools::RiskHint& hint) {
    // The published weights from bakeoff/blast_radius: write_out, destroy and priv are
    // worth three ordinary capabilities. Partial parse is itself risk -- it is the
    // signal that says "sandbox this regardless of the flags".
    const auto& c = hint.caps;
    double score = 0.0;
    score += c.writes_outside_workspace ? 0.30 : 0.0;
    score += c.destroys_data ? 0.30 : 0.0;
    score += c.escalates_privileges ? 0.30 : 0.0;
    score += c.rewrites_vcs_history ? 0.10 : 0.0;
    score += c.reads_outside_workspace ? 0.10 : 0.0;
    score += c.network_access ? 0.10 : 0.0;
    score += c.spawns_unbounded_process ? 0.10 : 0.0;
    score += c.signals_foreign_process ? 0.15 : 0.0;
    if (hint.status == blast_radius::ParseStatus::PartiallyParsed) {
        score += 0.20;
    } else if (hint.status == blast_radius::ParseStatus::Unparseable) {
        score += 0.40;
    }
    return score > 1.0 ? 1.0 : score;
}

bool is_irreversible(const tools::RiskHint& hint) noexcept {
    const auto& c = hint.caps;
    return c.destroys_data || c.writes_outside_workspace || c.escalates_privileges ||
           c.rewrites_vcs_history;
}

bool is_allowlisted(const std::string& command, const std::vector<std::string>& allowed) {
    const auto trim = [](std::string s) {
        const auto space = [](unsigned char ch) { return std::isspace(ch) == 0; };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), space));
        s.erase(std::find_if(s.rbegin(), s.rend(), space).base(), s.end());
        return s;
    };
    const std::string cmd = trim(command);
    if (cmd.empty()) {
        return false;
    }
    // Anything that can chain, substitute or redirect is out of scope for prefix
    // matching. `pytest` on the list must never authorise `pytest; rm -rf ~`.
    if (cmd.find_first_of(";|&`<>") != std::string::npos ||
        cmd.find("$(") != std::string::npos) {
        return false;
    }
    for (const std::string& raw : allowed) {
        const std::string entry = trim(raw);
        if (entry.empty()) {
            continue;
        }
        if (cmd == entry) {
            return true;
        }
        // Followed by a space, so `git st` never matches an entry of `git s`.
        if (cmd.size() > entry.size() && cmd.compare(0, entry.size(), entry) == 0 &&
            cmd[entry.size()] == ' ') {
            return true;
        }
    }
    return false;
}

Approval route_approval(const tools::RiskHint& hint, const HitlThresholds& t) {
    const double score = risk_score(hint);
    if (score >= t.reject_above_risk) {
        return Approval::Reject;
    }
    if (score <= t.auto_approve_below_risk) {
        return Approval::AutoApprove;
    }
    return Approval::Escalate;
}

} // namespace lmp::loop
