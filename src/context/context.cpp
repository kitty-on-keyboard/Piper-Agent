#include "src/context/context.hpp"

#include <algorithm>

namespace lmp::context {
namespace {

std::string first_line(const std::string& s, std::size_t cap) {
    const std::size_t nl = s.find('\n');
    std::string line = s.substr(0, nl == std::string::npos ? s.size() : nl);
    if (line.size() > cap) {
        line.resize(cap);
        line += "...";
    }
    return line;
}

} // namespace

std::size_t ContextStore::compact_oldest(std::size_t keep_recent) {
    if (recent_.size() <= keep_recent) {
        return 0;
    }
    const std::size_t drop = recent_.size() - keep_recent;

    // Summarize rather than announce (S8.3). Each line keeps the anchor: which tool
    // ran, whether it failed, and the first line of what came back. A run that trims
    // twice must still answer a question whose evidence appeared before the first trim,
    // and the anchor is what carries that.
    std::string span;
    span += "Earlier in this run (turns 1-" + std::to_string(drop) + ", events " +
            std::to_string(recent_.front().first_event_seq) + "-" +
            std::to_string(recent_[drop - 1].last_event_seq) + "):\n";
    for (std::size_t i = 0; i < drop; ++i) {
        const TurnRecord& t = recent_[i];
        if (t.tool_name.empty()) {
            span += "- said: " + first_line(t.assistant_text, 160) + "\n";
            continue;
        }
        span += "- " + t.tool_name;
        if (!t.tool_args_summary.empty()) {
            span += "(" + first_line(t.tool_args_summary, 80) + ")";
        }
        span += t.observation_is_error ? " FAILED: " : " -> ";
        span += first_line(t.observation, 200) + "\n";
    }
    spans_.push_back(std::move(span));
    recent_.erase(recent_.begin(), recent_.begin() + static_cast<std::ptrdiff_t>(drop));
    ++compactions_;
    return drop;
}

std::vector<Message> ContextStore::render(std::string_view tool_guidance) const {
    std::vector<Message> out;

    // T0: the mission, and the only place the deliverable is named.
    std::string system;
    system += std::string(tool_guidance);
    system += "\n\n# Mission\n\n" + mission_;

    // T1: pinned state. Present every turn, never summarized away.
    if (!checklist_.empty()) {
        system += "\n\n# Checklist\n\n";
        for (const ChecklistItem& c : checklist_) {
            system += (c.done ? "- [x] " : "- [ ] ") + c.text + "\n";
        }
    }
    if (!deliverables_.empty()) {
        system += "\n# Deliverables produced so far\n\n";
        for (const std::string& d : deliverables_) {
            system += "- " + d + "\n";
        }
    }
    if (!verifications_.empty()) {
        system += "\n# Verification ledger\n\n";
        for (const VerificationRecord& v : verifications_) {
            system += std::string(v.passed ? "- PASS " : "- FAIL ") + v.contract;
            // A green that has not been proven capable of red is reported as unproven,
            // to the model as well as the UI (S10.2).
            system += v.passed && !v.falsifiable ? "  (UNPROVEN: this check has not been"
                                                   " shown capable of failing)\n"
                                                 : "\n";
        }
    }
    out.push_back({Role::System, std::move(system)});

    // T3: compacted spans, oldest first, as observed history.
    for (const std::string& span : spans_) {
        out.push_back({Role::User, span});
    }

    // T2: recent turns, verbatim. The assistant's answer body and the observation it
    // got back -- the observation goes in as a tool_response, which is Qwen's shape.
    for (const TurnRecord& t : recent_) {
        if (!t.assistant_text.empty()) {
            out.push_back({Role::Assistant, t.assistant_text});
        }
        if (!t.observation.empty()) {
            out.push_back({Role::ToolResponse, t.observation});
        }
    }
    return out;
}

} // namespace lmp::context
