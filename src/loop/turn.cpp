#include "src/loop/turn.hpp"

#include <algorithm>

namespace lmp::loop {

std::string_view to_string(Outcome o) noexcept {
    switch (o) {
        case Outcome::ToolCallExecuted:
            return "ToolCallExecuted";
        case Outcome::ToolCallRefused:
            return "ToolCallRefused";
        case Outcome::TextOnly:
            return "TextOnly";
        case Outcome::LengthCapped:
            return "LengthCapped";
        case Outcome::Cancelled:
            return "Cancelled";
        case Outcome::BackendError:
            return "BackendError";
    }
    return "BackendError";
}

ModePolicy ModePolicy::for_mode(Mode m) noexcept {
    switch (m) {
        case Mode::Plan:
            return {0, false}; // T0: no execution, no writes
        case Mode::Debug:
            return {1, false}; // T1 sandbox, read and run, but no workspace mutation
        case Mode::Agent:
            return {1, true};
    }
    return {0, false}; // an unknown mode is the most restrictive, never the least
}

Outcome classify_turn(const model::GenResult& gen, const model::TurnGrammar& grammar,
                      bool executed, bool refused) {
    // "Did this call actually EXECUTE?" is asked FIRST (S9.1). Everything else is a
    // property of a turn that did not run a tool, so nothing downstream can overwrite
    // an execution that happened.
    if (executed) {
        return Outcome::ToolCallExecuted;
    }
    if (refused) {
        return Outcome::ToolCallRefused;
    }
    switch (gen.status) {
        case model::GenStatus::Cancelled:
            return Outcome::Cancelled;
        case model::GenStatus::BackendError:
            return Outcome::BackendError;
        case model::GenStatus::LengthCapped:
            // NOT completion. v1 blurred these and reported a truncated turn as a
            // finished one.
            return Outcome::LengthCapped;
        case model::GenStatus::Complete:
            break;
    }
    // Accepted by the grammar with no tool call: a text answer.
    return grammar.has_tool_call() ? Outcome::ToolCallRefused : Outcome::TextOnly;
}

std::string RepeatDetector::key(const std::string& tool,
                                const std::vector<tools::ToolParamValue>& params) {
    std::string k = tool;
    for (const tools::ToolParamValue& p : params) {
        k += '\x1f';
        k += p.name;
        k += '\x1e';
        k += p.value;
    }
    return k;
}

std::size_t RepeatDetector::seen_count(
    const std::string& tool, const std::vector<tools::ToolParamValue>& params) const {
    const std::string k = key(tool, params);
    for (const auto& [seen_key, count] : seen_) {
        if (seen_key == k) {
            return count;
        }
    }
    return 0;
}

void RepeatDetector::record(const std::string& tool,
                            const std::vector<tools::ToolParamValue>& params) {
    const std::string k = key(tool, params);
    for (auto& [seen_key, count] : seen_) {
        if (seen_key == k) {
            ++count;
            return;
        }
    }
    seen_.emplace_back(k, 1);
}

Corrective choose_corrective(const TurnResult& turn, const RepeatDetector& repeats,
                             int iterations_used, const Budget& budget,
                             bool wall_clock_exhausted) {
    // Ranked; the highest applicable one wins, and only one is returned (S9.2).
    if (wall_clock_exhausted || iterations_used >= budget.max_iterations) {
        return Corrective::HaltOnBudget;
    }
    if (turn.outcome == Outcome::ToolCallExecuted && turn.tool_result.ok() &&
        repeats.seen_count(turn.tool_name, turn.tool_params) > 1) {
        return Corrective::BreakRepeat;
    }
    // The model described a verification but did not make one. Synthesizing the call is
    // a mechanism; asking it to please run the build would be prose.
    if (turn.outcome == Outcome::TextOnly && !turn.assistant_text.empty()) {
        const std::string& t = turn.assistant_text;
        const bool claims_verification =
            t.find("should pass") != std::string::npos ||
            t.find("should now build") != std::string::npos ||
            t.find("should work") != std::string::npos;
        if (claims_verification) {
            return Corrective::SynthesizeVerification;
        }
    }
    return Corrective::None;
}

CompletionVerdict evaluate_completion(const context::ContextStore& ctx) {
    // Driven by the checklist and the ledgers. Nothing here reads the model's prose for
    // a sense that it sounded finished (S10.4).
    const auto& checklist = ctx.checklist();
    if (checklist.empty()) {
        return {false, "no checklist: the run has not stated what it must produce"};
    }
    const std::size_t open = static_cast<std::size_t>(
        std::count_if(checklist.begin(), checklist.end(),
                      [](const context::ChecklistItem& c) { return !c.done; }));
    if (open > 0) {
        return {false, std::to_string(open) + " checklist item(s) still open"};
    }
    if (ctx.deliverables().empty()) {
        return {false, "checklist is complete but no deliverable was recorded"};
    }
    const auto& vs = ctx.verifications();
    if (vs.empty()) {
        return {false, "no verification has been run"};
    }
    for (const context::VerificationRecord& v : vs) {
        if (!v.passed) {
            return {false, "verification still failing: " + v.contract};
        }
        // A green counts only if that exact check has been proven capable of red
        // (S10.2). An unproven green does not complete a run.
        if (!v.falsifiable) {
            return {false, "verification '" + v.contract +
                               "' passed but has never been shown capable of failing; "
                               "it does not count as evidence yet"};
        }
    }
    return {true, "checklist complete, deliverables recorded, verifications proven"};
}

} // namespace lmp::loop
