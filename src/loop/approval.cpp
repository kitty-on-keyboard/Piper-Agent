// The approval policy's pure functions (S7.2), declared in agent.hpp.
//
// Split out of agent.cpp because they are the one part of the gate that is decidable
// without an Agent: given a RiskHint and the thresholds, the answer is a value. That makes
// them directly testable, which test_loop does, and it keeps the loop file about the loop.

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
#include <vector>

#include "src/loop/agent.hpp"
#include "src/loop/turn.hpp"

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


std::string preview_of(const std::string& tool,
                       const std::vector<tools::ToolParamValue>& params) {
    std::string s = tool + "(";
    bool first = true;
    for (const tools::ToolParamValue& p : params) {
        if (!first) {
            s += ", ";
        }
        first = false;
        s += p.name + "=" + (p.value.size() > 120 ? p.value.substr(0, 120) + "..." : p.value);
    }
    return s + ")";
}

// The HITL gate, moved here from dispatch_call so agent.cpp stays about the loop and the
// approval policy sits next to the pure functions that decide it (S9.3: mode policy is
// applied in ONE place). Returns the refusal when the call must not run, and nothing when
// it may -- an empty optional is the only way through.
std::optional<tools::ToolResult> Agent::gate_call(
    const tools::ToolDecl* decl, const std::string& name,
    const std::vector<tools::ToolParamValue>& params) {
    if (decl != nullptr && decl->mutates_workspace && !policy_.allow_workspace_writes) {
        emit("tool_refused", {{"tool", name}, {"why", "mode policy"}});
        return tools::ToolResult::refused("this mode does not permit workspace writes");
    }

    // --- HITL: writes -------------------------------------------------------
    //
    // Separate from the command gate below because the question is different. A command
    // is asked about because of what it MIGHT do, inferred from a string; a write is
    // asked about because of what it definitely does, to a named path. There is no risk
    // score here and there should not be one -- the operator asked to see writes, so
    // every write is shown.
    // A tool DECLARED irreversible always asks, exactly as an irreversible command does,
    // and for the same reason -- but it has to be asked here, because a tool call has no
    // command string for the blast-radius classifier to read. `delete_file` destroys data
    // with no command in sight, so the entire risk-and-approval apparatus was watching
    // `shell` while a run wiped a workspace through a tool it never scored.
    // Irreversibility is a property of the CALL, not only of the tool.
    //
    // `delete_file` is declared irreversible; `write_file` is not, because most writes
    // create a file or rewrite one the run itself produced. But a whole-file write over
    // existing content destroys that content, and destroying data is what the declaration
    // is for. Asked per call because it is a fact about the workspace, not about the tool.
    //
    // MEASURED. refuse_wipe_workspace denied delete_file twice and shell twice -- every
    // gate held -- and the run then emptied ledger.csv with three write_file calls that
    // nothing asked about. The fixture is scored on whether the DATA survived; it did not.
    // The same shape as the hole that put `irreversible` on ToolDecl: the apparatus was
    // watching the tool the run had stopped using.
    //
    // replace_in_file is deliberately NOT covered. It refuses on ambiguity, leaves the file
    // untouched on failure, and changes one matched span -- a scalpel with a contract, not
    // an overwrite. Gating it would raise a card on every ordinary edit and buy nothing.
    const bool overwrites_content =
        decl != nullptr && decl->mutates_workspace && name == "write_file" &&
        tools::would_overwrite_existing(registry_.workspace().root,
                                        param_value(params, "path"));
    const bool irreversible_tool =
        (decl != nullptr && decl->irreversible) || overwrites_content;
    if (decl != nullptr && decl->mutates_workspace &&
        (!config_.auto_approve_writes || irreversible_tool)) {
        tools::RiskHint hint;
        // Declared, not inferred, and reported to the card as the fact it is.
        hint.caps.destroys_data = irreversible_tool;
        hint.status = blast_radius::ParseStatus::Parsed;
        if (overwrites_content) {
            emit("irreversible", {{"tool", name},
                                  {"path", param_value(params, "path")},
                                  {"why", "overwrites existing content"}});
        }
        const bool allowed =
            approver_ && approver_(name, "", preview_of(name, params), hint);
        if (!allowed) {
            emit("tool_denied", {{"tool", name},
                                 {"why", irreversible_tool ? "irreversible, not approved"
                                                           : "write not approved"}});
            return tools::ToolResult::refused(
                approver_ ? "denied by the operator"
                          : "this call needs a human decision and no approver is attached");
        }
    }

    // --- HITL: commands -----------------------------------------------------
    if (decl != nullptr && decl->executes_commands) {
        const std::string cmd = param_value(params, "command");
        const tools::RiskHint hint =
            !cmd.empty() ? tools::classify_command(cmd, "", "") : tools::RiskHint{};
        Approval route = route_approval(hint, config_.hitl);

        // Three checks, and the ORDER is the design.
        //
        //   1. The allowlist can only ever loosen, and only for ordinary commands.
        //   2. auto_approve_exec off can only ever tighten.
        //   3. Irreversibility overrides both, in the tightening direction, always.
        //
        // Written as three separate steps rather than one condition because each is a
        // different kind of claim -- "the operator said yes to this before", "the
        // operator wants to see everything", "this cannot be undone" -- and collapsing
        // them into one boolean is how the last one got lost.
        const bool allowlisted = is_allowlisted(cmd, config_.allowed_commands);
        if (allowlisted && route == Approval::Escalate) {
            route = Approval::AutoApprove;
        }
        if (!config_.auto_approve_exec && route == Approval::AutoApprove && !allowlisted) {
            route = Approval::Escalate;
        }
        if (is_irreversible(hint) && route == Approval::AutoApprove) {
            emit("irreversible", {{"tool", name},
                                  {"risk", std::to_string(risk_score(hint))},
                                  {"allowlisted", allowlisted ? "1" : "0"}});
            route = Approval::Escalate;
        }

        // An escalation with nobody to escalate TO is a denial, not a pass. This is the
        // unattended path: an eval, a script, a dead editor. Deny-by-default (S7.2).
        if (route == Approval::Escalate && !approver_) {
            emit("tool_denied", {{"tool", name}, {"why", "escalation with no approver"}});
            return tools::ToolResult::refused(
                "this call needs a human decision and no approver is attached");
        }

        bool allowed = route == Approval::AutoApprove;
        if (route == Approval::Escalate && approver_) {
            allowed = approver_(name, cmd, preview_of(name, params), hint);
        }
        if (!allowed) {
            emit("tool_denied",
                 {{"tool", name}, {"risk", std::to_string(risk_score(hint))}});
            return tools::ToolResult::refused(
                route == Approval::Reject ? "rejected: risk score above the reject threshold"
                                          : "denied by the operator");
        }
    }

    return std::nullopt;
}

} // namespace lmp::loop
