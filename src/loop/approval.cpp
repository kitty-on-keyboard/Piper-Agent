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
#include "src/platform/fs.hpp"

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

// Stated as the properties that make a call impossible, never as a list of tool names, so
// a tool added later is filtered by what it declares rather than by whether someone
// remembered to update a list here.
bool Agent::tool_allowed(const tools::ToolDecl& decl) const {
    if (decl.mutates_workspace && !policy_.allow_workspace_writes) {
        return false;
    }
    if (decl.remote && !policy_.allow_workspace_writes) {
        return false;
    }
    if (decl.irreversible && !policy_.allow_destructive) {
        return false;
    }
    // A mode that does not execute can only ever answer these with "T0: this mode does not
    // execute commands". Advertising them is how plan mode came to spend its turns on
    // builds that never ran -- with a persona telling it, in the same prompt, to run the
    // test rather than assert that it would pass.
    //
    // Keyed on the MODE, not on the effective tier. An operator who sets tier 0 on an
    // agent run has not changed what the mode is, and reading the tier here would also
    // have silently disarmed the tests that drive `rm -rf` into the approval gate at tier
    // 0 on purpose.
    if (decl.needs_execution && !policy_.allow_execution) {
        return false;
    }
    if (decl.conversational_only && !policy_.conversational) {
        return false;
    }
    return true;
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
    // A tool that runs in another process is not covered by anything above it. A trusted
    // MCP server's tools are declared with mutates_workspace false -- trust is a statement
    // about the SANDBOX, "this may run outside Seatbelt without a card for every call",
    // and it was being read here as a statement about mode policy. So a trusted server
    // carrying a write-equivalent tool was fully live in plan mode, through a gate whose
    // entire job is that writes do not happen. We cannot see what a remote tool touches,
    // and a mode that permits no writes cannot permit a call whose effects it cannot know.
    if (decl != nullptr && decl->remote && !policy_.allow_workspace_writes) {
        emit("tool_refused", {{"tool", name}, {"why", "mode policy: remote"}});
        return tools::ToolResult::refused(
            "this mode does not permit tools that run outside it");
    }
    // DESTRUCTION IS ITS OWN POWER. `irreversible` already meant "this destroys data that
    // the workspace cannot give back" and was used only to decide whether to raise a card;
    // a mode that is trusted to edit is not thereby trusted to delete, and until this
    // existed there was no way to say so. Debug mode is the whole reason: it needs to
    // write and has no business deleting.
    if (decl != nullptr && decl->irreversible && !policy_.allow_destructive) {
        emit("tool_refused", {{"tool", name}, {"why", "mode policy: destructive"}});
        return tools::ToolResult::refused(
            "this mode does not permit tools that destroy data");
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
    //
    // ITS OWN OUTPUT IS NOT THE OPERATOR'S DATA. A run that writes HostStatsService.swift
    // and then rewrites it eight times as the build teaches it more is iterating, not
    // destroying: the only bytes it overwrites are bytes it wrote a few turns earlier.
    // Gating that raised a card on every one of those turns with auto-approve ON, which
    // is how the switch came to look broken -- while ledger.csv, the case this gate was
    // actually built for, is a file the run never wrote and still asks about.
    // Normalised the way the repeat detector normalises: `ResMon` and `ResMon/` name one
    // directory, and a set keyed on raw bytes would forget its own writes over a slash.
    const std::string write_path =
        platform::lexically_normal(param_value(params, "path"));
    const bool overwrites_content =
        decl != nullptr && decl->mutates_workspace && name == "write_file" &&
        run_wrote_.find(write_path) == run_wrote_.end() &&
        tools::would_overwrite_existing(registry_.workspace().root, write_path);
    const bool irreversible_tool =
        (decl != nullptr && decl->irreversible) || overwrites_content;
    const bool write_gate = decl != nullptr && decl->mutates_workspace;
    // `irreversible_tool` already overrides the blanket switch, which is what covers the
    // one destructive act that can still reach a mode without allow_destructive: a
    // whole-file write over content the run did not itself produce. A tool DECLARED
    // irreversible was refused outright above and never arrives here.
    const bool ask_for_write =
        write_gate && (!config_.auto_approve_writes || irreversible_tool);

    // EVERY DECISION, INCLUDING THE ONES THAT ASK NOTHING.
    //
    // Until this event the gate was only audible when it fired: `irreversible` on a card,
    // `tool_denied` on a refusal, and NOTHING on an auto-approval. So a run where the gate
    // silently passed everything and a run with no gate at all produced identical logs,
    // and "why am I approving every write when the switch is on" had no answer in the
    // record -- it had to be re-derived by reading this function.
    //
    // MEASURED, which is why it is here: a run with auto_approve_writes=1 raised eleven
    // write_file cards. The reason is one line below -- irreversibility overrides the
    // blanket switch -- and nothing in the log said so. `why` now carries it.
    if (write_gate) {
        emit("approval",
             {{"gate", "write"},
              {"tool", name},
              {"path", param_value(params, "path")},
              // ESCALATED, not "asked". The gate decides that a call needs a human
              // decision; whether a CARD is actually shown is the approver's business,
              // and it may answer on its own from consent the operator already gave for
              // this run. Both were briefly called `asked`, which put two events with
              // opposite values against one call.
              {"escalated", ask_for_write ? "1" : "0"},
              {"auto_writes", config_.auto_approve_writes ? "1" : "0"},
              {"destroys_data", irreversible_tool ? "1" : "0"},
              // Whether this run wrote this path earlier. The one input that decides
              // between "iterating on its own output" and "overwriting the operator's".
              {"own_output", run_wrote_.find(write_path) != run_wrote_.end() ? "1" : "0"},
              {"why", !ask_for_write
                          ? "auto_approve_writes is on and nothing is destroyed"
                          : (!config_.auto_approve_writes
                                 ? "auto_approve_writes is off"
                                 : "destroys data, which overrides auto_approve_writes")}});
    }
    if (ask_for_write) {
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
        // THE REAL ROOT, and the real cwd, which is the root -- the same pair the shell
        // tool passes to run_sandboxed. This used to pass ("", ""), and blast_radius
        // defines writes_outside_workspace RELATIVE TO THE ROOT -- so with an empty root
        // every write anywhere was "outside the workspace", every mkdir and compile and
        // redirect scored 0.30, is_irreversible() fired on all of them, and the gate
        // asked about everything while both auto-approve switches were on.
        //
        // MEASURED, in the run that prompted this: the event log shows
        // auto_approve_exec=1 on the policy line and then an "irreversible" card for
        // nearly every shell call the run made, at risks from 0.30 to 0.80. The registry's
        // own advisory call site had the arguments right; this one starved the classifier
        // of the fact that makes "outside" mean anything.
        const std::string& root = registry_.workspace().root;
        const tools::RiskHint hint =
            !cmd.empty() ? tools::classify_command(cmd, root, root) : tools::RiskHint{};
        Approval route = route_approval(hint, config_.hitl);

        // Four checks, and the ORDER is the design.
        //
        //   1. The allowlist loosens an escalation -- and it also survives the
        //      irreversibility override below, because it is the operator's EXPLICIT
        //      decision about this exact command.
        //   2. auto_approve_exec ON loosens an escalation; OFF tightens an auto-approval.
        //   3. Irreversibility overrides the BLANKET policies, in the tightening
        //      direction, always -- but not the per-command rule.
        //
        // The distinction in 3 is what makes the "Always allow" button true. The card it
        // sits on appears precisely for escalated commands, and the commands that
        // escalate under auto-approve are precisely the irreversible ones -- so an
        // override that beat the allowlist made the button a no-op for every command it
        // was ever shown on: the operator clicked "Always allow `rm -rf .build`", the rule
        // was stored, and the same card came back next turn. A blanket switch yielding to
        // the hint is caution; a named, remembered rule yielding to it is the machine
        // overruling the person it just asked.
        //
        // Written as separate steps rather than one condition because each is a
        // different kind of claim -- "the operator said yes to this before", "the
        // operator wants to see everything", "this cannot be undone" -- and collapsing
        // them into one boolean is how the last one got lost.
        const bool allowlisted = is_allowlisted(cmd, config_.allowed_commands);
        if (allowlisted && route == Approval::Escalate) {
            route = Approval::AutoApprove;
        }
        // ON LOOSENS. This switch used to be tightening-only: it could turn an
        // auto-approval into an escalation and never the reverse, so the route came
        // entirely from the risk thresholds and turning "auto-approve command execution"
        // ON changed nothing an operator could observe. It was the same shape as the
        // sandbox_tier and require_approval fields before it -- generated, wired,
        // acknowledged in the editor, consumed by nobody.
        //
        // Reject is deliberately NOT loosened: that is the risk score being above the
        // reject threshold, which is a policy ceiling and not a question to ask. And the
        // irreversible check below still runs afterwards, so this cannot approve something
        // that destroys data -- it can only skip the card for an ORDINARY command, which
        // is exactly what the switch says it does.
        if (config_.auto_approve_exec && route == Approval::Escalate) {
            route = Approval::AutoApprove;
        }
        if (!config_.auto_approve_exec && route == Approval::AutoApprove && !allowlisted) {
            route = Approval::Escalate;
        }
        if (is_irreversible(hint) && route == Approval::AutoApprove && !allowlisted) {
            emit("irreversible", {{"tool", name},
                                  {"risk", std::to_string(risk_score(hint))},
                                  {"allowlisted", "0"}});
            route = Approval::Escalate;
        }

        // The command gate's counterpart to the write event above, emitted once the route
        // is settled and before anyone is asked. Carries every input the four steps read,
        // so a surprising card can be explained from the log instead of from this file:
        // the risk the classifier produced, both switches, and whether the operator's own
        // allowlist rule was what let it through.
        emit("approval", {{"gate", "command"},
                          {"tool", name},
                          {"command", cmd},
                          {"escalated", route == Approval::Escalate ? "1" : "0"},
                          {"route", route == Approval::AutoApprove  ? "auto"
                                    : route == Approval::Escalate   ? "ask"
                                                                    : "reject"},
                          {"risk", std::to_string(risk_score(hint))},
                          {"destroys_data", is_irreversible(hint) ? "1" : "0"},
                          {"allowlisted", allowlisted ? "1" : "0"},
                          {"auto_exec", config_.auto_approve_exec ? "1" : "0"}});

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
