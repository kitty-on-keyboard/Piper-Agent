#include "src/loop/turn.hpp"

#include <algorithm>

// For lexically_normal: a repeat is the same call, not the same bytes. See
// RepeatDetector::key.
#include "src/platform/fs.hpp"

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
        // T0: no execution, no writes, and it TALKS. The first three fields were the
        // whole of plan mode for a long time, and they are the half that never mattered:
        // a mode the model is not told it is in, whose write tools are still advertised
        // to it, spends its turns discovering the refusals one at a time.
        case Mode::Plan:
            return {0, false, false, false, true};
        // Debug WRITES. It could not, which made it useless for the one thing it is named
        // after -- you cannot add a log line, cannot save a reproduction, cannot apply the
        // fix you just proved. What it still cannot do is destroy: instrumenting a bug
        // never requires deleting a file, so the power is not granted.
        case Mode::Debug:
            return {1, true, false, true, false};
        case Mode::Agent:
            return {1, true, true, true, false};
    }
    // An unknown mode is the most restrictive, never the least -- and note that the most
    // restrictive is NOT conversational: a mode nobody declared has no operator waiting on
    // it, and yielding to a human who is not there is a hang, not a safety property.
    return {0, false, false, false, false};
}

// Written as what the mode IS and what it is FOR, not as a list of prohibitions. The
// prohibitions are already enforced twice -- the tool is not advertised and the gate would
// refuse it -- so spending prompt on them would be telling the model not to do something
// it has no way to do. What it cannot get anywhere else is the purpose.
std::string_view mode_brief(Mode m) noexcept {
    switch (m) {
        case Mode::Plan:
            return "# Plan mode\n"
                   "\n"
                   "You are in Plan mode to inspect the codebase, ask design choices, and present a plan. "
                   "Nothing you do here changes a file or runs a command -- execution tools are not loaded.\n"
                   "\n"
                   "- Direct Tool Execution: Execute read tool calls (`read_file`, `read_many`, `list_dir`, `find_files`, `search`) directly. "
                   "Do NOT output standalone conversational updates or text commentary explaining what you intend to read.\n"
                   "- Ask Design Options (`ask_question`) EARLY: when the request leaves a design or visual choice open "
                   "(e.g. animation style, colour direction, dashboard structure), invoke `ask_question` with 2 to 4 interactive options "
                   "(passed via `question` and `options`, one option per line) so the human can click an option card in the UI. "
                   "Ask as soon as you know enough to name the alternatives -- that is usually after reading a handful of files, "
                   "NOT after reading the whole codebase. Their answer changes what is worth reading next. "
                   "Asking the question as plain text does not present the card and does not reach them.\n"
                   "- Final Plan Submission (`exit_plan_mode`): When your investigation is finished, call `exit_plan_mode` with your completed plan markdown.\n";
        case Mode::Debug:
            return "# Debug mode\n"
                   "\n"
                   "You are finding out why something is wrong, and the answer has to be "
                   "observed rather than argued. You can read, run and edit; you cannot "
                   "delete.\n"
                   "\n"
                   "- Reproduce it first. A failure you have watched happen is worth more "
                   "than any amount of reading, and until you have one you are guessing "
                   "about which of several stories is true.\n"
                   "- Instrument rather than theorise. Add the log line, print the value, "
                   "run the command -- and then READ what came back. A hypothesis you did "
                   "not test is not evidence, however well it fits.\n"
                   "- Narrow before you fix. Get to the smallest thing that still fails; "
                   "a fix applied to the whole area is a fix you cannot prove.\n"
                   "- Once you have a reproduction, continue from that evidence and your "
                   "last working note. Do not restart by re-explaining the whole project.\n"
                   "- Then fix it, and run the same reproduction again. Answering in text "
                   "without a tool call ends the run as your final answer -- so do not "
                   "conclude until you have watched the reproduction pass.\n";
        case Mode::Agent:
            return "";
    }
    return "";
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

// A repeat is the same CALL, not the same bytes. `ResMon` and `ResMon/` name one
// directory, and keying on the raw value made them two different calls -- so a run could
// alternate the trailing slash and repeat itself forever without the detector ever
// counting past one.
//
// MEASURED: a real run in the editor spent its entire 80-turn budget alternating
// `list_dir ResMon` and `list_dir ResMon/`, learning nothing. Normalising the path
// arguments is what makes the two the same key -- and under the cache it is also what
// makes the second spelling a free answer instead of a second execution.
//
// Only path-shaped parameters are normalised. A `command` or a `content` argument is raw
// text where a trailing slash is a real difference.
//
// And one parameter is dropped from the key entirely: `replace_in_file`'s `new_text`.
//
// What decides whether that call can do anything is the path and `old_text` -- the text
// being searched for. If old_text is not in the file the call fails for EVERY new_text, and
// if it is, the first call consumed it. So (path, old_text) is the whole identity of the
// call, and including new_text meant a model could vary the replacement by one character
// and mint a fresh key for a call that cannot behave any differently.
//
// This is a claim about the tool's contract, not a similarity threshold. `write_file`'s
// `content` stays in the key, because two different contents genuinely are two different
// calls -- the identical-content case is caught upstream now, by the write door refusing
// to write bytes the file already holds (tools::CommitOutcome::unchanged), which both
// costs less and tells the model something a repeat count cannot.
std::string RepeatDetector::key(const std::string& tool,
                                const std::vector<tools::ToolParamValue>& params) {
    std::string k = tool;
    for (const tools::ToolParamValue& p : params) {
        if (tool == "replace_in_file" && p.name == "new_text") {
            continue;
        }
        k += '\x1f';
        k += p.name;
        k += '\x1e';
        k += p.name == "path" ? platform::lexically_normal(p.value) : p.value;
    }
    return k;
}

std::size_t RepeatDetector::seen_count(
    const std::string& tool, const std::vector<tools::ToolParamValue>& params) const {
    const std::string k = key(tool, params);
    for (const auto& [seen_key, call] : seen_) {
        if (seen_key == k) {
            return call.count;
        }
    }
    return 0;
}

void RepeatDetector::record(const std::string& tool,
                            const std::vector<tools::ToolParamValue>& params, bool ok,
                            const std::string& summary, std::size_t writes_now) {
    const std::string k = key(tool, params);
    for (auto& [seen_key, call] : seen_) {
        if (seen_key == k) {
            ++call.count;
            call.last_ok = ok;
            call.last_summary = summary;
            call.writes_at = writes_now;
            return;
        }
    }
    seen_.emplace_back(k, SeenCall{1, ok, summary, writes_now});
}

const RepeatDetector::SeenCall* RepeatDetector::cached(
    const std::string& tool, const std::vector<tools::ToolParamValue>& params,
    std::size_t writes_now) const {
    const SeenCall* prev = previous(tool, params);
    if (prev == nullptr) {
        return nullptr;
    }
    // Valid only while the workspace freshness epoch is unchanged: one write, successful
    // shell, remote call, or external invalidation bumps it. A failed call is never
    // treated as a fresh prior -- retry after an error is legitimate.
    if (prev->last_ok && prev->writes_at == writes_now) {
        return prev;
    }
    return nullptr;
}

const RepeatDetector::SeenCall* RepeatDetector::previous(
    const std::string& tool, const std::vector<tools::ToolParamValue>& params) const {
    const std::string k = key(tool, params);
    for (const auto& [seen_key, call] : seen_) {
        if (seen_key == k) {
            return &call;
        }
    }
    return nullptr;
}

std::string param_value(const std::vector<tools::ToolParamValue>& params,
                        std::string_view name) {
    for (const tools::ToolParamValue& p : params) {
        if (p.name == name) {
            return p.value;
        }
    }
    return {};
}

} // namespace lmp::loop
