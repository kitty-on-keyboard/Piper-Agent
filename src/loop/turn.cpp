#include "src/loop/turn.hpp"

#include <algorithm>
#include <regex>
#include <string>

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

int enumerated_choice_lines(std::string_view text) {
    // Keep in lockstep with Q_ENUM_LINE in webview.ts. The `i` flag is icase here.
    static const std::regex kEnumLine(
        R"(^(?:option\s+[0-9a-z]+\s*[:.)]|[0-9]{1,2}\s*[.)]\s|[a-z]\s*[.)]\s))",
        std::regex::icase);
    int n = 0;
    std::string line;
    const auto count_if_marker = [&](std::string s) {
        const auto a = s.find_first_not_of(" \t\r");
        if (a == std::string::npos) {
            return;
        }
        const auto b = s.find_last_not_of(" \t\r");
        s = s.substr(a, b - a + 1);
        if (std::regex_search(s, kEnumLine)) {
            ++n;
        }
    };
    for (const char c : text) {
        if (c == '\n') {
            count_if_marker(std::move(line));
            line.clear();
        } else {
            line += c;
        }
    }
    count_if_marker(std::move(line));
    return n;
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

// The half of a working brief that is NOT about which mode you are in: how to show the
// operator what you are doing, how to verify, how to edit, and what to do when you are
// going in circles. Every mode that can write needs all of it, so it is shared rather than
// copied -- a lesson learned in one of them is a lesson both of them get.
//
// THE CHECKLIST SECTION IS WHERE "CALL `plan`" NOW LIVES, and this is the only place in
// the prompt that says to. It is a mode brief rather than a mechanism on purpose: the
// eighth-pass rewrite deleted the grammar mask that made `plan` the only callable tool
// until the run had a checklist, and the deletion stands. But that mask's own comment
// recorded the measurement that bounds this text -- with `plan` merely available and a
// description telling the model to call it first, it did not -- so if runs still come back
// with no checklist, the answer is a mechanism, not more words here. The mode brief is the
// lever that did not exist when forcing was judged necessary; it gets one honest try.
//
// DEBUG MODE HAD NONE OF IT. It got five lines about reproducing and instrumenting and
// nothing about the mechanics of changing code, and the difference showed on a measured
// pair of runs against the same bug: agent mode fixed a SwiftUI layout in minutes, debug
// mode spent seventeen turns on it, landed one edit, and stopped. Its failures were the
// ones this text names and its brief did not -- it rewrote a whole file from a copy read
// several turns earlier and broke the build, re-read one unchanged file four times, and
// emitted the same five-item diagnosis on four separate turns without acting on it.
// Diagnosis was never the part debug mode was failing at.
constexpr std::string_view kWorkingDiscipline =
    "## The checklist\n"
    "\n"
    "- OPEN A MULTI-STEP TASK BY CALLING `plan`, before you start the work. The "
    "checklist it draws is the operator's only view of what you are doing -- without "
    "one they are watching an opaque run and cannot tell a long job from a stuck one. "
    "List the items you expect to do, then go and do them.\n"
    "- Tick each item off as it lands, by calling `plan` again with the whole list. An "
    "item is done when you have SEEN it work, not when you have made the edit you "
    "believe finishes it.\n"
    "- Items left open mean the run is unfinished, whatever your closing message says.\n"
    "\n"
    "## Verifying\n"
    "\n"
    "- BUILD OR TEST AFTER YOU EDIT, every time, and READ the output. "
    "Editing is the cheap half. Finding out whether it compiled is the half "
    "that decides whether you did anything.\n"
    "- COMPILER ERRORS GO STALE THE MOMENT YOU EDIT. The list in front of "
    "you describes the files as they were. If you are about to reason about "
    "an error for the second time, you do not need more thought, you need a "
    "fresh build -- run it.\n"
    "- Fix one cause at a time and rebuild. A batch of speculative fixes "
    "applied to one error list tells you nothing about which of them worked, "
    "and the next build's errors will not line up with your model of the "
    "file.\n"
    "- Do not finish on an unverified edit. Either the build is green, or "
    "you can say plainly what is still broken and what you tried.\n"
    "\n"
    "## Editing\n"
    "\n"
    "- Prefer targeted edits (`replace_in_file`) over rewriting a whole file. "
    "A whole-file write of something you last read several turns ago silently "
    "discards anything that changed underneath you, and is how a file ends up "
    "back at a state you already fixed.\n"
    "- If an edit comes back saying the file already contained those bytes, "
    "NOTHING CHANGED. Re-sending it will change nothing again. Read the file "
    "and work from what is actually on disk, not from what you believe you "
    "wrote.\n"
    "- Match the surrounding code: its naming, its idiom, its comment density. "
    "A correct change in a foreign style is still a change someone has to "
    "undo.\n"
    "- Read before you edit anything you have not read this run. The file may "
    "not be what you remember, and an edit built on a guess usually costs more "
    "turns than the read would have.\n"
    "\n"
    "## Getting unstuck\n"
    "\n"
    "- REPEATING YOURSELF IS THE SIGNAL. If you have made the same edit, or "
    "written the same diagnosis, twice and the situation has not moved, the "
    "approach is wrong -- not under-applied. Stop, get a fresh reading of the "
    "actual state, and form a different explanation.\n"
    "- When an error names a symbol, go and look at where that symbol is "
    "defined rather than inferring what it must be. One read settles what "
    "several turns of reasoning cannot.\n"
    "- Answering in text without a tool call ends the run as your final "
    "answer. Say it only when you mean it.\n";

// Written as what the mode IS and what it is FOR, not as a list of prohibitions. The
// prohibitions are already enforced twice -- the tool is not advertised and the gate would
// refuse it -- so spending prompt on them would be telling the model not to do something
// it has no way to do. What it cannot get anywhere else is the purpose.
//
// Returns by value because two of the three modes are now a mode-specific opening plus
// kWorkingDiscipline. The brief is built once per run, into the STABLE system prefix, so
// the allocation is not on any path that repeats.
std::string mode_brief(Mode m) {
    switch (m) {
        case Mode::Plan:
            return "# Plan mode\n"
                   "\n"
                   "You are in Plan mode to inspect the codebase, ask design choices, and present a plan. "
                   "Nothing you do here changes a file or runs a command -- write and execution tools are not loaded. "
                   "Read tools are, including a connected MCP server's tools that declare themselves read-only.\n"
                   "\n"
                   "- Direct Tool Execution: Execute read tool calls (`read_file`, `read_many`, `list_dir`, `find_files`, `search`) directly. "
                   "Do NOT output standalone conversational updates or text commentary explaining what you intend to read.\n"
                   "- Ask Design Options (`ask_user`) EARLY: when the request leaves a design or visual choice open "
                   "(e.g. animation style, colour direction, dashboard structure), invoke `ask_user` with 2 to 4 interactive options "
                   "(passed via `question` and `options`, one option per line) so the human can click an option card in the UI. "
                   "Ask as soon as you know enough to name the alternatives -- that is usually after reading a handful of files, "
                   "NOT after reading the whole codebase. Their answer changes what is worth reading next. "
                   "Asking the question as plain text does not present the card and does not reach them.\n"
                   "- Final Plan Submission (`exit_plan_mode`): When your investigation is finished, call `exit_plan_mode` with your completed plan markdown.\n";
        // DEBUG MODE FIXES THINGS. It is an implementation run that leads with evidence,
        // not a separate, weaker kind of run that hands its findings to someone else --
        // which is what the old brief left it sounding like, and what it then did: the
        // measured run produced a tidy numbered list of five defects and stopped, twice,
        // with the work undone and a green build it had never used to check anything.
        case Mode::Debug:
            return std::string(
                       "# Debug mode\n"
                       "\n"
                       "You are finding out WHY something is wrong and then FIXING it. You "
                       "have the same powers as an implementation run -- read, run, edit -- "
                       "weighted toward evidence: the cause is something you establish, not "
                       "something you infer from code that looks suspicious. The only thing "
                       "you cannot do is delete. Finding the cause is half the job; the run "
                       "is not done until the fix is in and checked.\n"
                       "\n"
                       "## Diagnosing\n"
                       "\n"
                       "- Get the failure in front of you before you change anything. Run "
                       "it, and read what came back. One watched failure settles in a turn "
                       "what reading only makes plausible.\n"
                       "- WHEN YOU CANNOT RUN IT -- a layout that is wrong on screen, a "
                       "build that is already green, any symptom you have no way to "
                       "reproduce from here -- do not fall back on scanning for code that "
                       "looks wrong. Trace the path from the symptom back to the code that "
                       "produces it and name the mechanism: this value, computed here, is "
                       "what puts that element off the edge. That gives you something the "
                       "fix can be checked against.\n"
                       "- Instrument when the answer is not visible. Adding a log line, "
                       "printing the value, writing a scratch harness -- that is what the "
                       "write power is for here, and it is the work, not a detour. Add it, "
                       "RUN it, read the output, and take temporary instrumentation back "
                       "out once it has answered you.\n"
                       "- Narrow before you fix. Get to the smallest thing that still "
                       "fails; a change applied across a whole area is a change you cannot "
                       "attribute.\n"
                       "- A LIST OF SUSPECTS IS NOT A DIAGNOSIS, and writing one out again "
                       "is not progress. Naming five things that might be involved and "
                       "changing all of them leaves you not knowing which one it was, and "
                       "usually leaves four unnecessary changes behind. Take the one you "
                       "can show is responsible, fix it, check it, then take the next.\n"
                       "- Then make the same observation again -- rerun the reproduction, "
                       "rebuild, look at the value you printed. A fix you have not "
                       "re-checked against the original evidence is a guess in better "
                       "formatting.\n"
                       "\n") +
                   std::string(kWorkingDiscipline);
        // AGENT MODE HAD NO BRIEF AT ALL until 2026-08-08, which is why the mode that
        // actually writes code was the only one never told to check its own work. Plan
        // mode is told how to ask; Debug mode is told to run the reproduction again and
        // not to conclude until it passes. Agent mode -- the one that ships changes --
        // got an empty string, and behaved exactly like something with no instructions:
        // a dashboard rewrite wrote nine files, ran `swift build` ONCE, and then spent 44
        // turns editing against that one stale error list without ever building again.
        case Mode::Agent:
            return std::string(
                       "# Agent mode\n"
                       "\n"
                       "You are changing this codebase, and a change you have not seen work "
                       "is not a change, it is a claim. You can read, run and edit. The "
                       "loop that matters is small and you should be in it constantly: read "
                       "enough to be specific, make one coherent edit, build it, read what "
                       "came back.\n"
                       "\n") +
                   std::string(kWorkingDiscipline);
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

// A repeat is the same CALL, not the same bytes. `proj` and `proj/` name one
// directory, and keying on the raw value made them two different calls -- so a run could
// alternate the trailing slash and repeat itself forever without the detector ever
// counting past one.
//
// MEASURED: a real run in the editor spent its entire 80-turn budget alternating
// `list_dir proj` and `list_dir proj/`, learning nothing. Normalising the path
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

namespace {

[[nodiscard]] std::string_view trim_ws(std::string_view s) noexcept {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r')) {
        s.remove_prefix(1);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) {
        s.remove_suffix(1);
    }
    return s;
}

[[nodiscard]] bool is_leaked_tool_xml_line(std::string_view line) noexcept {
    const std::string_view t = trim_ws(line);
    if (t == "</function>" || t == "</parameter>" || t == "</tool_call>" ||
        t == "<tool_call>") {
        return true;
    }
    return t.size() >= 10 && t.substr(0, 10) == "<function=";
}

} // namespace

std::string strip_leaked_tool_xml(std::string value) {
    while (!value.empty()) {
        const std::size_t nl = value.find_last_of('\n');
        const std::string_view last =
            nl == std::string::npos ? std::string_view(value)
                                    : std::string_view(value).substr(nl + 1);
        if (trim_ws(last).empty() || is_leaked_tool_xml_line(last)) {
            value = nl == std::string::npos ? std::string{} : value.substr(0, nl);
            continue;
        }
        break;
    }
    constexpr std::string_view kClosers[] = {"</parameter>", "</function>", "</tool_call>"};
    bool again = true;
    while (again) {
        again = false;
        for (const std::string_view c : kClosers) {
            if (value.size() >= c.size() &&
                value.compare(value.size() - c.size(), c.size(), c.data(), c.size()) == 0) {
                value.resize(value.size() - c.size());
                while (!value.empty() &&
                       (value.back() == ' ' || value.back() == '\t')) {
                    value.pop_back();
                }
                again = true;
            }
        }
    }
    return value;
}

} // namespace lmp::loop
