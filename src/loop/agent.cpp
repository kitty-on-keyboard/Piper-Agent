#include "src/loop/agent.hpp"

#include <algorithm>
#include <cassert> // the emit() field-name guard
#include <cctype> // ordinal_width's digit test
#include <chrono>
#include <cstdlib> // getenv/atoi, for the LMP_TRACE_TEXT gate
#include <memory>
#include <string_view>
#include <unordered_map>

#include "src/loop/parallel_calls.hpp"
#include "src/loop/token_stream.hpp"
#include "src/model/mlx_backend.hpp"
#include "src/platform/fs.hpp"
#include "src/tools/apply_patch.hpp"
#include "src/tools/log_triage.hpp"

namespace lmp::loop {
namespace {

// The log records what the harness DID -- every prompt, every result -- and nothing the
// model SAID. That asymmetry is why a run that burned 40 turns without writing a file
// could not be diagnosed from its own trace: `generation tokens=224` followed by a turn
// with no tool_result says a turn produced nothing, and cannot say why.
//
// Off by default because a turn's reasoning is the largest thing in the run and the log
// is also the UI feed. `LMP_TRACE_TEXT=1` turns it on for a diagnostic run; the events
// go to the same writer as everything else, so the ordering against `prompt` and
// `tool_result` is the real one rather than two files to correlate by timestamp.
bool trace_text_enabled() {
    static const bool on = [] {
        const char* s = std::getenv("LMP_TRACE_TEXT");
        return s != nullptr && std::atoi(s) != 0;
    }();
    return on;
}

// Long enough to see a whole argument -- a truncated write_file is exactly the case
// where the interesting part is the end -- and bounded so one traced turn cannot be the
// whole log.
constexpr std::size_t kTraceFieldCap = 8192;

std::string capped(std::string s) {
    if (s.size() <= kTraceFieldCap) {
        return s;
    }
    s.resize(kTraceFieldCap);
    s += "\n[...truncated]";
    return s;
}

// Trailing slice of think for the next prompt when the answer channel was empty.
// Full CoT stays on the thinking stream (S5.7); this is continuity, not a dump.
constexpr std::size_t kWorkingNoteCap = 512;

std::string working_note_from_reasoning(std::string_view reasoning) {
    std::size_t begin = 0;
    while (begin < reasoning.size() &&
           (reasoning[begin] == ' ' || reasoning[begin] == '\t' || reasoning[begin] == '\n' ||
            reasoning[begin] == '\r')) {
        ++begin;
    }
    std::size_t end = reasoning.size();
    while (end > begin && (reasoning[end - 1] == ' ' || reasoning[end - 1] == '\t' ||
                           reasoning[end - 1] == '\n' || reasoning[end - 1] == '\r')) {
        --end;
    }
    if (begin >= end) {
        return {};
    }
    std::string_view body = reasoning.substr(begin, end - begin);
    if (body.size() <= kWorkingNoteCap) {
        return std::string(body);
    }
    std::size_t start = body.size() - kWorkingNoteCap;
    // Prefer a clean cut at a newline inside the window; otherwise take the tail.
    const std::size_t nl = body.find('\n', start);
    if (nl != std::string_view::npos && nl + 1 < body.size()) {
        start = nl + 1;
    }
    return std::string(body.substr(start));
}

bool is_blank(std::string_view s) {
    return s.find_first_not_of(" \t\r\n") == std::string_view::npos;
}

// How repetitive a generation is, measured on the text rather than guessed from its
// length. Reported ALWAYS, unlike the text itself, because these three numbers are small
// and a degenerate turn is invisible without them.
//
// MEASURED: one turn of a real run emitted "I'll fix all compilation errors
// systematically. Let me read all source files first." roughly two hundred times and then
// hit the token cap. In the event log that turn is `generation tokens=4096 status=1` and
// nothing else -- indistinguishable from a legitimately long write_file. The run had 66
// turns and several like it; the trace could not tell them apart, so nothing in the
// harness could either.
struct TextShape {
    std::size_t lines = 0;
    std::size_t distinct = 0;
    std::size_t worst_line_repeats = 0; // how often the most-repeated non-blank line occurs
};

TextShape shape_of(const std::string& text) {
    TextShape s;
    std::unordered_map<std::string_view, std::size_t> counts;
    std::size_t at = 0;
    while (at <= text.size()) {
        std::size_t nl = text.find('\n', at);
        if (nl == std::string::npos) {
            nl = text.size();
        }
        const std::string_view line(text.data() + at, nl - at);
        at = nl + 1;
        // Blank and near-blank lines repeat in every healthy generation (indentation,
        // paragraph breaks) and would dominate the count without saying anything.
        if (line.find_first_not_of(" \t\r") == std::string_view::npos) {
            continue;
        }
        ++s.lines;
        const std::size_t n = ++counts[line];
        s.worst_line_repeats = std::max(s.worst_line_repeats, n);
    }
    s.distinct = counts.size();
    return s;
}

// When a generation is worth flagging as degenerate rather than merely long.
//
// The discriminator is HOW MUCH DISTINCT CONTENT there is, not how often the commonest
// line recurs. That distinction was found by testing rather than reasoning: a repeat-count
// threshold flags a perfectly good `write_file` of a source file, because real code
// repeats `    }` a hundred times in two hundred lines. Measured on four inputs --
//
//   the real failure  lines=200 distinct=  1 worst=200   <- 0.5% distinct
//   a source file     lines=244 distinct=125 worst=120   <- 51% distinct
//   a brace-heavy file lines=200 distinct=101 worst=100  <- 50% distinct
//   a short answer    lines=  2 distinct=  2 worst=  1
//
// -- the repeat counts of the last three are indistinguishable from the first's, and the
// distinct ratio separates them by an order of magnitude. Both halves still have to hold:
// the floor keeps short answers out, since two identical lines out of two is 100% repeat
// and no evidence of anything.
constexpr std::size_t kRepeatFloor = 8;
constexpr std::size_t kDistinctCeilingPercent = 25;

bool looks_degenerate(const TextShape& s) {
    return s.lines >= kRepeatFloor && s.worst_line_repeats >= kRepeatFloor &&
           s.distinct * 100 <= s.lines * kDistinctCeilingPercent;
}

// A PER-TURN SEED DERIVED FROM THE RUN'S. SplitMix64's finalizer, which is cheap and
// decorrelates adjacent indices -- turn 4 and turn 5 must not get neighbouring draw
// sequences, since consecutive turns are exactly the ones that were coming out identical.
std::uint64_t seed_for_turn(std::uint64_t run_seed, std::uint64_t turn) {
    std::uint64_t z = run_seed + 0x9E3779B97F4A7C15ULL * (turn + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

const char* phase_name(model::TurnPhase phase) {
    switch (phase) {
        case model::TurnPhase::Think:
            return "think";
        case model::TurnPhase::Text:
            return "text";
        case model::TurnPhase::ToolCall:
            return "tool";
        case model::TurnPhase::Done:
            return "done";
    }
    return "unknown";
}

// The escapes a checklist item can carry, decoded. Anything else keeps its literal
// character, which is what the line parser would have seen anyway.
//
// ONE TABLE, read by both JSON shapes below, because they differ only in their framing and
// a second copy of this switch is a second copy to drift.
std::string decode_json_escapes(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 >= s.size()) {
            out += s[i];
            continue;
        }
        switch (s[++i]) {
            case 'n': out += '\n'; break;
            case 't': out += '\t'; break;
            case 'r': break;
            default: out += s[i]; break;
        }
    }
    return out;
}

// A JSON array of strings, rewritten as one element per line. Empty when the text is not
// one, which is the signal to parse it as the newline-separated list it claims to be.
//
// THE TEST IS DELIBERATELY NARROW: first non-space character `[`, next non-space character
// `"`, and a `]` at the end. That is a JSON array and cannot be anything else -- in
// particular it cannot be the checklist item `[ ] ship the parser`, which is the one
// string a looser test would eat. A malformed array (an unterminated quote) flattens to
// nothing and falls through to the line parser, which is the behaviour that was there
// before this function existed.
std::string flatten_json_array(const std::string& raw) {
    const std::size_t open = raw.find_first_not_of(" \t\r\n");
    if (open == std::string::npos || raw[open] != '[') {
        return {};
    }
    const std::size_t first = raw.find_first_not_of(" \t\r\n", open + 1);
    if (first == std::string::npos || raw[first] != '"') {
        return {};
    }
    const std::size_t close = raw.find_last_not_of(" \t\r\n");
    if (close == std::string::npos || raw[close] != ']') {
        return {};
    }

    std::string out;
    std::size_t i = first;
    while (i < close) {
        if (raw[i] != '"') {
            ++i;
            continue;
        }
        // Scan to the closing quote, stepping over an escaped one rather than ending on it.
        const std::size_t begin = i + 1;
        std::size_t end = begin;
        while (end < close && raw[end] != '"') {
            end += raw[end] == '\\' ? 2 : 1;
        }
        if (end >= close) {
            // An unterminated final string means the text was not the array it looked like.
            return {};
        }
        out += decode_json_escapes(std::string_view(raw).substr(begin, end - begin));
        out += '\n';
        i = end + 1;
    }
    return out;
}

// A bare JSON string, unquoted and decoded. Empty when the text is not one.
//
// THE SHAPE THAT COLLAPSED A 17-ITEM CHECKLIST INTO ONE ITEM. `items` is a list, so a
// model that reaches for JSON sends one of two things: an array, which flatten_json_array
// handles, or -- when it has joined the list up itself -- a single string, `"[x] a\n[x] b"`.
// Nothing decoded the second shape, so the quotes and the two-character `\n` sequences
// arrived as text, the line parser found no newline to split on, and the whole plan became
// one item whose label began with a quote mark.
//
// MEASURED, run 3 of 2026-08-09: a 17-item checklist became `items=1 open=1` mid-run, the
// panel showed the entire plan as one unfinished line with `\n[x]` through it, and the
// model spent its last four turns re-sending the same call -- which is the `stalled`
// ending the operator was shown, 28 turns into a 200-turn budget.
std::string unquote_json_string(const std::string& raw) {
    const std::size_t open = raw.find_first_not_of(" \t\r\n");
    if (open == std::string::npos || raw[open] != '"') {
        return {};
    }
    const std::size_t close = raw.find_last_not_of(" \t\r\n");
    if (close <= open || raw[close] != '"') {
        return {};
    }
    return decode_json_escapes(std::string_view(raw).substr(open + 1, close - open - 1));
}

// `12. ` / `12) ` at `i`, or 0 when there is no ordinal there. Digits only: a line that
// opens with the word "Step" is text, and guessing at prose is how a parser starts eating
// item labels.
std::size_t ordinal_width(const std::string& line, std::size_t i) {
    std::size_t j = i;
    while (j < line.size() && std::isdigit(static_cast<unsigned char>(line[j])) != 0) {
        ++j;
    }
    if (j == i || j >= line.size() || (line[j] != '.' && line[j] != ')')) {
        return 0;
    }
    ++j;
    // The separator has to be followed by space or the item text; `1.5x faster` is not an
    // ordinal, and neither is a bare `1.` with nothing after it.
    if (j >= line.size() || (line[j] != ' ' && line[j] != '\t')) {
        return 0;
    }
    while (j < line.size() && (line[j] == ' ' || line[j] == '\t')) {
        ++j;
    }
    return j - i;
}

// One item per line: an optional `- `/`* ` bullet, an optional `1. ` ordinal, an optional
// `[ ]`/`[x]` marker, then the text -- with the ordinal and the marker accepted in either
// order. Tolerant of prose-ish markdown, because refusing a checklist over a dash would be
// theatre.
//
// THE NUMBERED LIST NOBODY COULD TICK. The marker was only ever looked for at the head of
// the line, so `1. [x] Explore the workspace` -- an ordered list with a checkbox, which is
// what a model writes when it is told to number its plan AND to mark items '[x]' -- parsed
// as an UNCHECKED item whose text began "[x]". Every tick the model sent landed in the
// label instead of the state.
//
// MEASURED, run 4 of 2026-08-12: six `plan` calls, ticking one more item each time, and
// all six logged `open=8` out of 8. The operator watched a checklist that never moved
// while the run worked through it, and the panel only filled in at the end because the
// final restate happened to lead with the marker. Nothing was wrong with the model's
// bookkeeping; we were parsing its ticks into the text.
//
// Two passes, because `[x] 1. Item` and `1. [x] Item` are both in the wild and a fixed
// order only ever fixes one of them. The ordinal STAYS in the text -- the model numbered
// its own plan and the panel does not number for it.
std::vector<context::ChecklistItem> parse_checklist_lines(const std::string& source) {
    std::vector<context::ChecklistItem> items;
    std::size_t at = 0;
    while (at < source.size()) {
        std::size_t nl = source.find('\n', at);
        if (nl == std::string::npos) {
            nl = source.size();
        }
        std::string line = source.substr(at, nl - at);
        at = nl + 1;
        std::size_t i = line.find_first_not_of(" \t-*");
        if (i == std::string::npos) {
            continue;
        }
        bool done = false;
        // Kept so the item reads the way the model wrote it; only the marker is consumed.
        std::string ordinal;
        for (int pass = 0; pass < 2; ++pass) {
            if (line.compare(i, 3, "[x]") == 0 || line.compare(i, 3, "[X]") == 0) {
                done = true;
                i += 3;
                break;
            }
            if (line.compare(i, 3, "[ ]") == 0) {
                i += 3;
                break;
            }
            if (pass != 0) {
                break; // one ordinal, then the marker; `1. 2. x` is text, not two numbers
            }
            const std::size_t ord = ordinal_width(line, i);
            if (ord == 0) {
                break;
            }
            ordinal = line.substr(i, ord);
            i += ord;
        }
        const std::size_t text_at = line.find_first_not_of(" \t", i);
        if (text_at == std::string::npos) {
            continue; // a marker or a number with nothing after it is not an item
        }
        items.push_back({ordinal + line.substr(text_at), done});
    }
    return items;
}

// Does this single item carry a SECOND checkbox? Then the list arrived unsplit: a real
// one-item plan does not mention another item's marker.
bool holds_a_whole_list(const std::string& text) {
    return text.find("[ ]") != std::string::npos || text.find("[x]") != std::string::npos ||
           text.find("[X]") != std::string::npos;
}

// Is there an ESCAPED newline here with the next item's checkbox behind it? That is an
// unsplit list and nothing else.
//
// The bare `\n` test this replaces was not good enough: `[ ] make write_file emit \n rather
// than a newline` is ONE item that talks about the escape, and decoding it tore the item in
// half at the word "rather". What separates the two cases is what FOLLOWS the escape -- a
// marker (through any bullet or indent) means another item starts there.
bool escaped_newline_starts_an_item(const std::string& s) {
    for (std::size_t at = s.find("\\n"); at != std::string::npos; at = s.find("\\n", at + 2)) {
        const std::size_t next = s.find_first_not_of(" \t-*", at + 2);
        if (next != std::string::npos && s[next] == '[') {
            return true;
        }
    }
    return false;
}

// The note a cache-served repeat carries. A named constant because record_call() has to
// recognise and strip it: the observation the model reads includes the note, and the
// cache must store the bytes WITHOUT it -- otherwise the third repeat serves a result
// with two notes stacked on it, and the fourth three.
constexpr const char kCacheNote[] =
    "\n(unchanged: this exact call already ran and returned the result above; "
    "nothing has been written since, so it was not re-executed.)";

// Named `exit_plan_mode` until that tool stopped existing outside plan mode, which made the
// advice unfollowable in the two modes that see this note most. `ask_question` is available
// everywhere, so it is the one worth naming.
constexpr const char kRepeatNote[] =
    "\n[Note: This tool call produced identical output to a previous call. Do not repeat the same query. Explore subdirectories, inspect code files, or put the choice to the human with 'ask_question'.]";

std::string without_cache_note(std::string summary) {
    const std::string_view rnote(kRepeatNote);
    if (summary.size() >= rnote.size() &&
        std::string_view(summary).substr(summary.size() - rnote.size()) == rnote) {
        summary.resize(summary.size() - rnote.size());
    }
    const std::string_view note(kCacheNote);
    if (summary.size() >= note.size() &&
        std::string_view(summary).substr(summary.size() - note.size()) == note) {
        summary.resize(summary.size() - note.size());
    }
    return summary;
}

// Workspace-derived observations that must revalidate current state (never serve a
// cached summary as authority after shell/MCP/editor/external changes).
bool observes_workspace(const std::string& tool) {
    return tool == "read_file" || tool == "read_many" || tool == "read_slice" ||
           tool == "list_dir" || tool == "search" || tool == "find_files" ||
           tool == "locate_symbol" || tool == "git_status" || tool == "git_diff" ||
           tool == "git_log";
}

// A PROGRESS DISPLAY IS NOT PROGRESS. `plan` writes no bytes and observes nothing -- it
// restates the model's own checklist for the operator's panel -- so a turn that only
// called it has neither written nor learned, whatever the panel now says.
//
// It counted as progress because observation_is_new() returns true for anything that does
// not read the workspace, which reset the inert-turn counter on every call. Measured: 56
// of one run's 98 tool calls were `plan`, and the ending could not see any of them.
bool is_display_only(const std::string& tool) {
    return tool == "plan";
}

// WHICH COPY OF THIS FILE IS THE FILE, said on the copy that is.
//
// Appended, never rewritten, so it costs nothing but the tokens: the stale snapshots stay
// where they are and this says plainly that they are stale. That is the whole of what the
// model was missing -- not room, not a better receipt, just an answer to "which of these
// four copies is current".
//
// Only fires when the run is actually holding a superseded copy, so a first read and a
// re-read of an unchanged file are both silent (the unchanged case has kRepeatNote).
// MUST RUN LAST, after the repeat note and after the collapse. Both of those compare the
// fresh result against what is already in the context, and both normalise with
// without_cache_note() -- so a note appended ahead of them makes the comparison miss and
// silently disables them. Measured the moment it was written the other way round: an
// unchanged re-read got called stale, and the identity collapse stopped collapsing.
void note_stale_copies(const context::ContextStore& ctx, const std::string& path,
                       const std::string& current, tools::ToolResult& result) {
    const std::size_t stale = ctx.stale_copies_of_path(path, current);
    if (stale == 0) {
        return;
    }
    result.summary += "\n[Note: this is the CURRENT content of " + path + ". " +
                      std::to_string(stale) + " earlier cop" +
                      (stale == 1 ? "y" : "ies") + " of this same file appear" +
                      (stale == 1 ? "s" : "") +
                      " higher up in this conversation and " +
                      (stale == 1 ? "is" : "are") +
                      " OUT OF DATE -- the file has been edited since. Work from this copy "
                      "and ignore the earlier ones; do not read the file again to resolve "
                      "the difference.]";
}

// read_many is excluded: its observation is a multi-file bundle, not one path key.
bool is_content_read(const std::string& tool) {
    return tool == "read_file" || tool == "read_slice";
}

} // namespace

// The build command this workspace obviously has, or empty when it is not obvious.
//
// WHY THE HARNESS GUESSES AT ALL. The post-write check is the only verification this
// harness performs, and it defaulted to empty -- so out of the box, a writing run had no
// feedback loop whatsoever. Measured 2026-08-08: nine files rewritten, `swift build` run
// ONCE at turn 22, then 44 turns of editing against that one stale error list. Asking the
// operator to configure a build command before the agent can check its own work is not a
// setting, it is the agent not working.
//
// ONLY UNAMBIGUOUS MARKERS. Each of these names exactly one build command for the whole
// workspace with no configuration step in between. Deliberately absent: CMakeLists.txt
// (the command depends on a build directory that may not be configured), package.json
// (`build` may not exist as a script, and may mean bundling rather than checking), and
// Makefile (the default target is anyone's guess). A wrong guess is worse than none: it
// spends a shell call per write and teaches the model to distrust the check.
//
// The operator's own setting always wins -- this is consulted only when theirs is empty.
std::string detected_verify_command(const std::string& workspace_root) {
    platform::WorkspaceFs fs(workspace_root);
    if (!fs.valid()) {
        return {};
    }
    const platform::DirectoryContents root = fs.list_directory(".");
    if (!root.ok()) {
        return {};
    }
    bool package_swift = false;
    bool cargo_toml = false;
    bool go_mod = false;
    for (const platform::DirectoryEntry& e : root.entries) {
        if (e.kind != platform::DirectoryEntryKind::File) {
            continue;
        }
        package_swift = package_swift || e.name == "Package.swift";
        cargo_toml = cargo_toml || e.name == "Cargo.toml";
        go_mod = go_mod || e.name == "go.mod";
    }
    // One marker or none. Two build systems in one root is exactly the ambiguity this
    // refuses to guess through.
    const int markers = static_cast<int>(package_swift) + static_cast<int>(cargo_toml) +
                        static_cast<int>(go_mod);
    if (markers != 1) {
        return {};
    }
    if (package_swift) {
        return "swift build";
    }
    if (cargo_toml) {
        return "cargo build";
    }
    return "go build ./...";
}

Agent::Agent(const model::QwenTokenizer& tok, model::InferenceBackend& backend,
             tools::Registry& registry, context::ContextStore& ctx,
             platform::EventLogWriter& log, const platform::Clock& clock,
             AgentConfig config)
    : tok_(tok), backend_(backend), registry_(registry), ctx_(ctx), log_(log),
      clock_(clock), config_(config), policy_(ModePolicy::for_mode(config.mode)) {
    // The operator's tier, when they named one. Plan mode is exempt in the one direction
    // that matters: it pins T0, so "no execution" cannot be undone by a settings field.
    if (config_.sandbox_tier_override >= 0 && config_.mode != Mode::Plan) {
        policy_.sandbox_tier = config_.sandbox_tier_override;
    }
    tools_guidance_ =
        registry_.tools_json([this](const tools::ToolDecl& d) { return tool_allowed(d); });
    std::string withheld_by_mode;
    for (const parsephony::ToolSpec& s : registry_.guard_specs()) {
        const tools::ToolDecl* d = registry_.find(s.name);
        if (d != nullptr && !tool_allowed(*d)) {
            withheld_by_mode += withheld_by_mode.empty() ? "" : ",";
            withheld_by_mode += s.name;
            continue;
        }
        mode_specs_.push_back(s);
    }
    // In the Agent rather than in the sidecar, so every client gets it -- the eval harness
    // and scripts/drive.py send a mode too, and a brief only the editor's runs received
    // would make the two disagree about what plan mode even is.
    ctx_.set_mode_brief(mode_brief(config_.mode));

    // WHAT THIS MODE TOOK AWAY, once, at the top of the run.
    if (!withheld_by_mode.empty()) {
        emit("mode_tools", {{"withheld", withheld_by_mode},
                            {"samplable", std::to_string(mode_specs_.size())},
                            {"of", std::to_string(registry_.guard_specs().size())}});
    }
    if (config_.operator_verify_contract.empty() && policy_.allow_workspace_writes) {
        // A WRITING RUN WITH NO CHECK HAS NO FEEDBACK LOOP AT ALL. Rather than leave that
        // as a silent default, adopt the workspace's obvious build command when it has
        // one. See detected_verify_command() for why the detection is deliberately narrow.
        std::string detected = detected_verify_command(registry_.workspace().root);
        if (!detected.empty()) {
            config_.operator_verify_contract = detected;
            emit("verify_contract_detected",
                 {{"contract", detected},
                  {"why", "no verify_contract configured; adopted from the workspace"}});
        }
    }
    if (!config_.operator_verify_contract.empty()) {
        emit("operator_contract", {{"contract", config_.operator_verify_contract}});
    } else if (policy_.allow_workspace_writes) {
        // Nothing configured and nothing detected: the run genuinely has no verification,
        // and that is worth one loud line in the trace rather than an absence. It stays a
        // legitimate state -- a read-only question, a project with no build command -- so
        // this is emitted, not enforced.
        emit("no_operator_contract",
             {{"why", "writing mode, no verify_contract configured and none detected"},
              {"consequence", "no check runs after any write; `completed` rests on the "
                              "model's own answer and its own checklist"}});
    }
    if (config_.auto_syntax_check) {
        syntax_ = std::make_unique<tools::SyntaxChecker>(registry_.workspace().root,
                                                         2048);
    }
    // SEEDED FROM THE SESSION, not started empty.
    //
    // `run_wrote_` answers "is this file the run's own output or the operator's data", and
    // a follow-up builds a fresh Agent over the SAME ContextStore -- so an empty set told
    // the second run that everything the first one produced belonged to someone else. A
    // run resuming its own work raised an overwrite card on its own files, which is the
    // gate firing on exactly the case it was built to let through.
    //
    // The deliverable ledger is the durable copy of the same fact and needs no second
    // store: it is written from the one place run_wrote_ is (a successful mutating call
    // with a path), and it outlives the run because the context does.
    for (const std::string& path : ctx_.deliverables()) {
        run_wrote_.insert(platform::lexically_normal(path));
    }

    emit("policy", {{"mode", std::to_string(static_cast<int>(config_.mode))},
                    {"sandbox_tier", std::to_string(policy_.sandbox_tier)},
                    {"auto_approve_exec", config_.auto_approve_exec ? "1" : "0"},
                    {"auto_approve_writes", config_.auto_approve_writes ? "1" : "0"}});
}

// `plan` is declared by the registry but executed HERE: the checklist lives in the
// context store, which the registry has no business reaching into.
//
// Restating replaces the whole list, so ticking an item off is the same call as writing
// it -- one idempotent operation instead of a second tool and a synchronisation problem.
//
// The checklist is DISPLAY, not a gate. It feeds the sidebar panel and the run report's
// unfinished_items, and nothing in the loop reads it to decide anything. The model that
// keeps it current is being a good colleague; the model that does not is not held hostage
// over bookkeeping.
TurnResult::PlanOutcome Agent::apply_plan(const std::vector<tools::ToolParamValue>& params) {
    std::vector<context::ChecklistItem> items;
    const std::string* raw = nullptr;
    for (const auto& p : params) {
        if (p.name == "items") {
            raw = &p.value;
        }
    }
    if (raw == nullptr) {
        return {false, "plan requires 'items'"};
    }
    // ONE ITEM PER LINE -- unless the model sent JSON, which it does, because `items` is a
    // list-shaped parameter and that is what a list looks like to a model that has spent
    // its training on JSON tool calls. Both JSON shapes are handled here, in the order that
    // makes each test unambiguous, and each one is rewritten into the newline-separated
    // list the line parser reads.
    //
    // The line parser split on '\n' and nothing else, so JSON arrived as ONE line, the
    // `[ ]` test failed against the leading `["` or `"[`, and the entire list became a
    // single checklist item whose text was its own JSON source.
    std::string source = flatten_json_array(*raw);
    if (source.empty()) {
        source = unquote_json_string(*raw);
    }
    if (source.empty()) {
        source = *raw;
    }
    items = parse_checklist_lines(source);
    // ESCAPED NEWLINES WITH NO QUOTES AROUND THEM: the same list, sent without the JSON
    // framing that would have identified it. Retried only when the first parse found a
    // single item AND an escape in it has another item's marker behind it, so an item that
    // merely talks about a backslash-n keeps it.
    if (items.size() == 1 && escaped_newline_starts_an_item(source)) {
        std::vector<context::ChecklistItem> retry =
            parse_checklist_lines(decode_json_escapes(source));
        if (retry.size() > items.size()) {
            items = std::move(retry);
        }
    }
    if (items.empty()) {
        return {false, "plan produced no items; give one item per line"};
    }
    // ONE ITEM CARRYING THE WHOLE LIST IS A PARSE FAILURE, NOT A ONE-ITEM PLAN, and the
    // model has to be told so: every shape above can be defeated by a fourth one, and the
    // failure this replaces was silent. A run set a 1-item checklist, read back "checklist
    // set: 0/1 done" -- which is what a healthy one-item plan says -- and re-sent the same
    // call until the run ended. An error is information; a plausible number is not.
    if (items.size() == 1 && holds_a_whole_list(items[0].text)) {
        return {false,
                "plan got ONE item containing other checkbox markers, so the list arrived "
                "unsplit and was not applied. Put each item on its own line with a real "
                "newline between them, or send a JSON array of strings -- not one string "
                "with \\n sequences inside it."};
    }
    const std::size_t open = static_cast<std::size_t>(std::count_if(
        items.begin(), items.end(), [](const context::ChecklistItem& c) { return !c.done; }));
    const std::size_t total = items.size();
    ctx_.set_checklist(std::move(items));
    emit("plan", {{"items", std::to_string(total)}, {"open", std::to_string(open)}});
    // THE PARSED ITEMS, not just how many. `plan` reports a count, and a count is exactly
    // the wrong thing to trust here: this parser tolerates prose-ish markdown, so a
    // checklist that arrives on one line, or with the marker but no text, or with nesting
    // it cannot see, still produces a plausible number. A run was observed showing five
    // items with no text at all in the surface, and the log said `items=5 open=5` -- which
    // is what a healthy plan looks like. The text is the only way to tell them apart.
    {
        std::string joined;
        for (const context::ChecklistItem& item : ctx_.checklist()) {
            joined += joined.empty() ? "" : " | ";
            joined += (item.done ? "[x] " : "[ ] ") +
                      (item.text.empty() ? std::string("<EMPTY>") : item.text);
        }
        emit("checklist", {{"count", std::to_string(total)},
                           {"open", std::to_string(open)},
                           {"items", capped(joined)}});
    }
    if (observer_.on_checklist) {
        observer_.on_checklist(ctx_.checklist());
    }
    return {true, "checklist set: " + std::to_string(total - open) + "/" +
                      std::to_string(total) + " done"};
}

// Non-model feedback on an edit, on the same observation the edit produced. A syntax
// check is a fact about the file the edit just made, delivered where the edit's result
// already is -- it never gates anything and never helps a run end.
void Agent::annotate_with_syntax_check(const std::string& path, tools::ToolResult& result) {
    if (!config_.auto_syntax_check || !syntax_) {
        return;
    }
    const tools::SyntaxVerdict v = syntax_->check(path, policy_.sandbox_tier);
    if (!v.ran) {
        return; // no contract, or it could not be run: say nothing at all
    }
    const auto before = pre_edit_clean_.find(path);
    const bool was_clean = before == pre_edit_clean_.end() || before->second;
    emit("syntax_check",
         {{"path", path}, {"language", v.language}, {"clean", v.clean ? "1" : "0"}});
    if (v.clean) {
        return;
    }
    result.summary += "\n[syntax] " + v.language;
    // A red that was already red is a different fact and a different next move. Without
    // this the model gets told its edit broke a file that arrived broken.
    result.summary += was_clean ? ": FAILED\n" : ": still failing (it was already failing "
                                                 "before this edit)\n";
    result.summary += v.diagnostics;
    result.error_class = tools::ErrorClass::Malformed;
}

// NO FIELD MAY BE CALLED `kind` OR `seq`. The writer stamps both onto every event, so a
// payload field with either name emits the key TWICE in one JSON object -- and every JSON
// parser in existence keeps the last one silently.
//
// MEASURED on the shipped log: 291 events whose own kind field overwrote the event kind,
// and 63 whose ledger position overwrote the writer's sequence number. The log exists to
// make "what did the harness do" answerable; a field that deletes the answer to "what
// KIND of thing was this" is worse than no field.
void Agent::emit(const std::string& kind, std::vector<platform::EventField> fields) {
    for (const platform::EventField& f : fields) {
        assert(f.key != "kind" && f.key != "seq" &&
               "event payload may not shadow the writer's own kind/seq fields");
    }
    platform::Event ev;
    ev.kind = kind;
    ev.fields = std::move(fields);
    log_.append(ev, clock_);
}

TurnResult Agent::step(const model::CancelToken& cancel) {
    TurnResult turn;

    // PHASE MARKERS EXIST BECAUSE A CRASH LEAVES NO OTHER TRACE.
    //
    // The sidecar has twice died with the event log ending on a `turn` and no `run_end`,
    // no crash report, and nothing in the unified log. append() is an unbuffered write(2)
    // per event, so that really is the last thing that happened -- the process dies
    // somewhere between recording one turn and emitting the next `prompt`, and until now
    // that whole span was one dark gap.
    //
    // These split the gap into named steps at the cost of three writes per turn. They are
    // cheap enough to leave on: a syscall against a turn that spends seconds on the GPU
    // does not register, and the alternative is another crash that says nothing.
    emit("phase", {{"at", "render_begin"}});

    // --- prompt assembly ---------------------------------------------------
    const model::ChatTemplate tmpl(tok_);
    const std::vector<model::Message> messages = ctx_.render("");
    model::InferenceTask task;
    // render_with_offsets, NOT a second render of a message sub-list: render() appends the
    // generation prompt, so the first k messages rendered alone are not a token prefix of
    // the whole. Asking for offsets is the only correct way to locate the boundary, and
    // getting it wrong reuses a cache against the wrong prefix without crashing (S5.10).
    std::vector<std::size_t> offsets;
    task.prompt = tmpl.render_with_offsets(messages, tools_guidance_, offsets);
    // The real size of the prompt this turn, free of charge -- it has just been tokenized.
    // Read by collapse_duplicate_read, which must not call prompt_tokens() per read: that
    // re-renders and re-tokenizes the whole context, and a batched turn would do it four
    // times to answer a question this number already answers.
    last_prompt_tokens_ = task.prompt.size();
    // Everything except the live-state block, which changes every turn. The backend
    // snapshots here so the next turn rolls back instead of re-prefilling the context.
    const std::size_t stable = ctx_.stable_message_count("");
    task.checkpoint_at = stable < offsets.size() ? offsets[stable] : 0;
    task.max_new_tokens = config_.max_new_tokens;
    task.sampling = config_.sampling;
    // config_.seed stays authoritative over the sampling block's own field: it is the
    // one the run is reproducible from -- but it is the seed of the RUN, not of the turn,
    // and handing the same value to every generation is what made stuck runs unbreakable.
    //
    // THE BACKEND BUILDS A FRESH Sampler PER GENERATION (mlx_backend.cpp) and seeds it
    // from this field. With one constant for the whole run, every turn replayed the SAME
    // sequence of draws; a model that had settled into a confident repetition then
    // re-emitted it token for token, and nothing the harness appended to the prompt could
    // shift it. That is why every repeat note and every nudge in every stuck trace has
    // looked ignored -- they were being written into a prompt whose continuation had
    // already been decided by the RNG.
    //
    // MEASURED, plan mode, ResMon, temperature 0.6: prompts of 2901 and 3042 tokens
    // produced byte-identical 147-token generations; prompts of 3342 and 3586 produced
    // byte-identical 238-token ones. Two independent draws agreeing for 238 tokens at
    // that temperature does not happen -- the draws were not independent.
    //
    // Mixing the turn index in keeps a run REPRODUCIBLE, which is the property config_.seed
    // exists for: turn n of a given config always gets the same seed. What it stops being
    // is reproducible ACROSS turns of the same run, which was never a feature.
    task.sampling.seed = seed_for_turn(config_.seed, turns_generated_++);

    // Hard model ceiling: prompt + reserved generation must fit. Compaction is supposed
    // to keep the prompt under context_budget_tokens, but a mis-set editor budget or a
    // checkpoint smaller than the default must still refuse rather than OOB the KV.
    if (config_.model_max_sequence_tokens > 0) {
        const auto prompt_n = static_cast<std::int64_t>(task.prompt.size());
        const auto need = prompt_n + static_cast<std::int64_t>(task.max_new_tokens);
        if (need > static_cast<std::int64_t>(config_.model_max_sequence_tokens)) {
            turn.generation.status = model::GenStatus::BackendError;
            turn.generation.error =
                "prompt (" + std::to_string(task.prompt.size()) + ") + max_new_tokens (" +
                std::to_string(task.max_new_tokens) + ") exceeds model maximum sequence "
                "length (" + std::to_string(config_.model_max_sequence_tokens) + ")";
            turn.outcome = Outcome::BackendError;
            emit("prompt", {{"tokens", std::to_string(task.prompt.size())},
                            {"messages", std::to_string(messages.size())},
                            {"refused", "model_max_sequence"},
                            {"error", turn.generation.error}});
            return turn;
        }
    }

    // Every harness->model append is an event. This invariant is what makes "did the
    // model receive this?" answerable (S8.1, S14).
    emit("prompt", {{"tokens", std::to_string(task.prompt.size())},
                    {"messages", std::to_string(messages.size())},
                    {"compactions", std::to_string(ctx_.compaction_count())}});

    // What MLX is holding as this turn's generation begins -- the number that decides
    // whether the next allocation is the one that does not come back. `unload_model`
    // reports the same three at the end of a session, which is exactly when a run that
    // died never gets to.
    {
        const model::MemoryReport mem = model::mlx_memory_report();
        emit("memory", {{"at", "pre_generate"},
                        {"active", std::to_string(mem.active)},
                        {"cache", std::to_string(mem.cache)},
                        {"peak", std::to_string(mem.peak)},
                        {"prompt_tokens", std::to_string(task.prompt.size())}});
    }
    emit("phase", {{"at", "generate_begin"}});

    // --- constrained generation --------------------------------------------
    //
    // The grammar offers the mode's whole tool set, every turn. What used to live here --
    // a plan-only gate, per-turn suppressions, a refusal blocklist, a write floor
    // restoring what the other three took away -- was four narrowings composing blind,
    // and the run that motivated the floor lost its editor to a mechanism whose author
    // did not think it could. The mode's set is a run constant, which is also what keeps
    // the KV prefix stable (S6.4).
    model::TurnGrammar grammar(tok_, mode_specs());
    task.mask = &grammar;

    // Reasoning is surfaced on its own channel, never inlined into the answer (S5.7).
    // The split happens by TOKEN ID upstream; the streamer only routes it, one token at a
    // time, on its own thread so a slow reader cannot throttle the decode loop.
    std::unique_ptr<TokenStreamer> streamer;
    if (observer_.on_token) {
        streamer = std::make_unique<TokenStreamer>(tok_, observer_.on_token);
    }
    // Leave reserved_tool_tokens of the turn budget for tool XML after think ends.
    std::size_t think_cap = 0;
    if (config_.max_think_tokens > 0 && config_.max_new_tokens > 0) {
        const auto reserved = std::max(0, config_.reserved_tool_tokens);
        const auto room =
            std::max(0, config_.max_new_tokens - reserved);
        think_cap = static_cast<std::size_t>(
            std::max(0, std::min(config_.max_think_tokens, room)));
    }
    GrammarSink sink(grammar, streamer.get(), think_cap);
    turn.generation = backend_.generate(task, sink, cancel);

    // Drained and joined BEFORE the text below is read, so what the surface showed and
    // what the transcript records cannot disagree about a turn that is already over.
    if (streamer) {
        streamer->finish();
    }

    // Still decoded in one piece for the transcript and the context store. The streamed
    // concatenation is byte-identical to these (test_token_stream asserts it), so this is
    // the same text, not a second opinion about it.
    turn.reasoning = tok_.decode(grammar.think_ids());
    turn.assistant_text = tok_.decode(grammar.text_ids());
    turn.think_tokens = grammar.think_ids().size();
    turn.text_tokens = grammar.text_ids().size();
    const std::size_t generated =
        static_cast<std::size_t>(std::max(0, turn.generation.tokens_generated));
    turn.tool_tokens =
        generated - std::min(generated, turn.think_tokens + turn.text_tokens);
    if (turn.generation.status == model::GenStatus::LengthCapped) {
        turn.cap_phase = phase_name(grammar.phase());
    } else if (sink.think_capped) {
        // Not a length cap of the turn -- generation continued after think closed -- but
        // still worth naming so a trace can see why reasoning stopped early.
        turn.cap_phase = "think_budget";
    }
    if (observer_.on_perf) {
        // AGAINST THE BUDGET THE RUN IS ACTUALLY MANAGED BY, which is the only denominator
        // that means anything to the person watching the meter.
        observer_.on_perf(turn.generation, task.prompt.size(),
                          static_cast<std::size_t>(std::max(1, config_.context_budget_tokens)),
                          ctx_.compaction_count());
    }

    emit("generation",
         {{"status", std::to_string(static_cast<int>(turn.generation.status))},
          {"tokens", std::to_string(turn.generation.tokens_generated)},
          {"think_tokens", std::to_string(turn.think_tokens)},
          {"text_tokens", std::to_string(turn.text_tokens)},
          {"tool_tokens", std::to_string(turn.tool_tokens)},
          {"cap_phase", turn.cap_phase},
          {"prefill_reused_tokens",
           std::to_string(turn.generation.prefill_reused_tokens)},
          {"ttft_ms", std::to_string(turn.generation.ttft_ms)},
          {"decode_tok_per_s", std::to_string(turn.generation.decode_tok_per_s)}});

    // THE SHAPE OF WHAT WAS SAID, always, even when the text itself is not traced. Three
    // integers per turn, and they separate the two failures that `tokens=4096 status=1`
    // cannot: a long legitimate write, and a model stuck emitting one sentence until the
    // cap. A `degenerate` line in the log is the run saying the model went into a loop --
    // which is not a harness bug, and until now looked exactly like one.
    {
        const TextShape said = shape_of(turn.reasoning + "\n" + turn.assistant_text);
        const bool degenerate = looks_degenerate(said);
        if (degenerate || sink.looped ||
            turn.generation.status == model::GenStatus::LengthCapped) {
            emit("degenerate_text",
                 {{"lines", std::to_string(said.lines)},
                  {"distinct", std::to_string(said.distinct)},
                  {"worst_line_repeats", std::to_string(said.worst_line_repeats)},
                  {"length_capped",
                   turn.generation.status == model::GenStatus::LengthCapped ? "1" : "0"},
                  {"degenerate", degenerate ? "1" : "0"},
                  // Whether the harness CUT it, as against merely noticing afterwards.
                  {"cut_for_looping", sink.looped ? "1" : "0"},
                  {"loop_repeats", std::to_string(sink.loop_repeats)},
                  {"tokens", std::to_string(turn.generation.tokens_generated)}});
        }
    }

    // A turn the breaker cut classifies TextOnly -- no tool ran -- but the text is a
    // cut-off cycle, not an answer, and run() must not end the run on it as though the
    // model had concluded. `cut_for_looping` carries that fact.
    //
    // What must NOT survive is the text. Carrying fifty copies of one paragraph into the
    // next prompt is how a loop seeds its own successor: the next turn renders a context
    // whose most recent content is the cycle, at a fixed seed, and draws it again. So the
    // reasoning and the answer are dropped and replaced by the FACT of the loop.
    if (sink.looped) {
        turn.cut_for_looping = true;
        turn.reasoning.clear();
        turn.assistant_text =
            "(this turn was cut: the same " + std::to_string(loop::LoopBreaker::kWindow) +
            " tokens were emitted " + std::to_string(sink.loop_repeats) +
            " times over. Nothing was produced and nothing ran.)";
    }

    if (trace_text_enabled()) {
        emit("turn_text", {{"reasoning", capped(turn.reasoning)},
                           {"text", capped(turn.assistant_text)},
                           {"calls", std::to_string(grammar.tool_calls().size())}});
    }

    if (!grammar.has_tool_call()) {
        turn.outcome = classify_turn(turn.generation, grammar, false, false);
        return turn;
    }

    // --- the call(s) --------------------------------------------------------
    //
    // A turn may carry several calls (S9.1 amended: one turn, one OUTCOME, but the model
    // may batch independent work into it). The first call is the turn's outcome; the rest
    // execute in order and each gets its own history record. Reading four files used to
    // cost four full prefill+decode round-trips.
    const auto& calls = grammar.tool_calls();

    // Params up front for every call, because the concurrent pass below needs them all
    // before it starts and the serial pass wants the same values.
    std::vector<std::vector<tools::ToolParamValue>> params(calls.size());
    turn.batch_count = calls.size();
    for (std::size_t i = 0; i < calls.size(); ++i) {
        for (const auto& p : calls[i].params) {
            params[i].push_back({p.name, p.value});
        }
    }

    // The ARGUMENTS, which `tool_result` never carried -- it records what came back, and
    // the summary of a shell call does not contain the command that produced it. Reading
    // "Ok, empty output" three turns running tells you nothing; reading the three
    // commands tells you immediately whether the model is repeating itself.
    for (std::size_t i = 0; i < calls.size(); ++i) {
        std::vector<platform::EventField> fields{{"tool", calls[i].name},
                                                 {"index", std::to_string(i)},
                                                 {"batch_index", std::to_string(i)},
                                                 {"batch_count",
                                                  std::to_string(calls.size())}};
        if (trace_text_enabled()) {
            for (const tools::ToolParamValue& p : params[i]) {
                fields.push_back({"arg." + p.name, capped(p.value)});
            }
        }
        emit("tool_call", std::move(fields));
    }

    // The read-only calls of this batch run at once; everything else stays exactly where
    // it was. See parallel_calls.hpp for which calls qualify and why the others cannot.
    // Repeats always revalidate: the detector annotates, it does not skip execution.
    std::vector<std::size_t> parallel;
    for (std::size_t i = 0; i < calls.size(); ++i) {
        if (can_run_in_parallel(calls[i].name)) {
            parallel.push_back(i);
        }
    }
    std::vector<tools::ToolResult> precomputed;
    if (parallel.size() > 1) {
        precomputed = run_calls_concurrently(parallel, [this, &calls, &params](std::size_t i) {
            // ONLY the registry. Every gate and every ledger write stays on this thread,
            // below, in call order.
            return registry_.execute(calls[i].name, params[i], policy_.sandbox_tier);
        });
    }
    const auto precomputed_for = [&](std::size_t i) -> const tools::ToolResult* {
        for (std::size_t k = 0; k < precomputed.size(); ++k) {
            if (parallel[k] == i) {
                return &precomputed[k];
            }
        }
        return nullptr;
    };

    // Serial from here, in index order, so the emits, the history records and the UI rows
    // are what the fully serial path produced. Parallelism must not be observable.
    for (std::size_t i = 0; i < calls.size(); ++i) {
        bool ran = false;
        tools::ToolResult result;
        if (const tools::ToolResult* done = precomputed_for(i); done != nullptr) {
            result = *done;
            ran = adopt_readonly_result(calls[i].name, params[i], result);
        } else {
            result = dispatch_call(calls[i].name, params[i], ran);
        }

        // ASKED HERE, BEFORE record_call FOLDS THIS RESULT IN. Both dispatch paths have
        // finished with the call and neither has touched the detector yet, so this is the
        // one point where "did this differ from last time" is still answerable -- and the
        // one point both paths pass through, so a batched call is measured exactly like a
        // solitary one.
        const bool fresh = ran && observation_is_new(calls[i].name, params[i], result);

        if (i == 0) {
            turn.tool_name = calls[0].name;
            turn.tool_params = params[0];
            turn.tool_result = std::move(result);
            turn.produced_new_information = fresh;
            turn.outcome = classify_turn(turn.generation, grammar, ran, !ran);
        } else {
            TurnResult::ExtraCall extra;
            extra.tool_name = calls[i].name;
            extra.params = params[i];
            extra.result = std::move(result);
            extra.produced_new_information = fresh;
            turn.extra_calls.push_back(std::move(extra));
        }
    }
    return turn;
}

// May this call be run off the agent thread? Only if dispatch_call would have reached
// `Registry::execute` and touched nothing else on the way.
//
// Stated as the properties that make the other branches unreachable, not as a list of tool
// names, so a tool added later is excluded until it is declared harmless: `plan` mutates
// the checklist, `mutates_workspace` opens the write gate and the deliverable ledger,
// `executes_commands` opens the risk classifier and the approver. An unregistered name is
// not eligible either -- dispatch_call has to be the one to produce the typed NotFound.
bool Agent::can_run_in_parallel(const std::string& name) const {
    if (name == "plan") {
        return false;
    }
    const tools::ToolDecl* decl = registry_.find(name);
    if (decl == nullptr) {
        return false;
    }
    // Editor-backed locate_symbol blocks on the sidecar inbox (lmp/code_intel); that
    // wait is single-threaded with approval/edit waits and must not run on a worker.
    if (name == "locate_symbol" && registry_.has_code_intel_sink()) {
        return false;
    }
    // Remote tools run in another process: even a "trusted" one may mutate the
    // workspace outside the write ledger, so its result is neither parallel-safe
    // nor cacheable as a fresh observation.
    return !decl->mutates_workspace && !decl->executes_commands && !decl->irreversible &&
           !decl->remote;
}

// DID THIS TURN MOVE THE RUN FORWARD?
//
// Two ways, and a turn needs only one of them:
//
//   WROTE  -- bytes_changed > 0. A no-op edit (`old_text` matched but `new_text` was
//             identical to it, or a patch that produced no byte change) reports ok with
//             zero bytes and is NOT progress. A real run spent three consecutive turns on
//             exactly those.
//   LEARNED -- an observation whose bytes differ from the last time this call was made.
//
// A turn that did neither is INERT. That includes every text-only turn, which is how the
// old text-only ending is subsumed rather than sitting alongside this one: narrating is
// simply the case of being inert without calling anything.
//
// BATCHED CALLS ARE OR-ED, not just the first one: a turn that re-read one file and wrote
// another has done work, and charging it as inert because the front call was a repeat
// would be the same batching hole that once hid three reads a turn from the detector.
//
// Refused and length-capped turns are neither -- see the caller, which leaves the count
// alone for them. A tool the human declined is not the model failing to progress, and a
// generation cut at the token cap never got to choose.
bool Agent::turn_made_progress(const TurnResult& turn) noexcept {
    if (turn.outcome != Outcome::ToolCallExecuted) {
        return false;
    }
    if (turn.tool_result.bytes_changed > 0 || turn.produced_new_information) {
        return true;
    }
    for (const TurnResult::ExtraCall& extra : turn.extra_calls) {
        if (extra.result.bytes_changed > 0 || extra.produced_new_information) {
            return true;
        }
    }
    return false;
}

// IS THIS OBSERVATION NEW? One definition, used by both dispatch paths and by the loop.
//
// "New" means the bytes differ from what this exact call returned last time. It is
// deliberately byte identity and nothing cleverer: a file that changed produces different
// bytes, and no invalidation rule is needed to say so. Callers must ask BEFORE
// record_call() folds this result into the detector, which is why every use sits on the
// execution path rather than after it.
//
// A call with no prior is new by definition. A FAILED call is treated as new -- an error
// is information, and two identical failures in a row are caught by the inert-turn count
// rather than by pretending the second one said nothing.
bool Agent::observation_is_new(const std::string& name,
                               const std::vector<tools::ToolParamValue>& params,
                               const tools::ToolResult& result) const {
    if (!result.ok()) {
        return true;
    }
    // A malformed `plan` is caught above -- an error is information. A SUCCESSFUL one
    // learns nothing by construction, so it must not reset the inert-turn counter.
    if (is_display_only(name)) {
        return false;
    }
    if (!observes_workspace(name)) {
        // A shell or MCP call that ran to completion is not "new information" just
        // because it is not a read. A repeated `swift build` that prints the same
        // success line teaches the run nothing; treating it as fresh let a verify
        // step reset the inert counter on every turn and hide a thrash loop.
        // Measured: 39 tool calls, 11 writes, 9 plan restatements, ended stalled
        // because each build looked like progress.
        if (name == "shell" || name == "run_command") {
            const RepeatDetector::SeenCall* prior = repeats_.previous(name, params);
            return prior == nullptr || !prior->last_ok ||
                   prior->last_summary != without_cache_note(result.summary);
        }
        return true;
    }
    const RepeatDetector::SeenCall* prior = repeats_.previous(name, params);
    return prior == nullptr || !prior->last_ok ||
           prior->last_summary != without_cache_note(result.summary);
}

// The tail dispatch_call would have run for such a call, minus everything the eligibility
// test already proved unreachable: no deliverable to record (nothing was written), no
// approval. What remains is the executed flag and the event, and BOTH must happen here on
// the agent thread, in call order.
//
// `result` is non-const because a repeat has to be ANNOTATED here exactly as the serial
// path annotates it. It used to be a const reference, so this path emitted `repeat_reread`
// and then could not append kRepeatNote -- a re-read batched behind another call got no
// note at all, while the same re-read sent alone got one.
bool Agent::adopt_readonly_result(const std::string& name,
                                  const std::vector<tools::ToolParamValue>& params,
                                  tools::ToolResult& result) {
    // Annotate before collapse so the prior observation is still byte-identical to the
    // fresh result when measuring redundant re-reads.
    const std::size_t prior_seen = repeats_.seen_count(name, params);
    if (prior_seen > 0 && result.ok() && observes_workspace(name)) {
        const bool unchanged = !observation_is_new(name, params, result);
        emit("repeat_reread",
             {{"tool", name},
              {"prior_count", std::to_string(prior_seen)},
              {"unchanged", unchanged ? "1" : "0"},
              {"read_bytes", std::to_string(result.bytes_read)}});
        if (unchanged) {
            if (result.bytes_read > 0) {
                emit("redundant_read_bytes",
                     {{"tool", name},
                      {"bytes", std::to_string(result.bytes_read)},
                      {"path", param_value(params, "path")}});
            }
            result.summary += kRepeatNote;
        }
    }
    // Same duplicate collapse as the serial path. Safe here for the same reason this path
    // exists at all: a call is only eligible for it if it mutates nothing and executes
    // nothing, and the collapse touches only records already in the context.
    collapse_duplicate_read(name, params, result);
    // LAST: the two mechanisms above both compare against the un-annotated bytes.
    if (is_content_read(name) && result.ok()) {
        note_stale_copies(ctx_, param_value(params, "path"),
                          without_cache_note(result.summary), result);
    }
    emit("tool_result", {{"tool", name},
                         {"status", std::string(tools::to_string(result.status))},
                         {"read_bytes", std::to_string(result.bytes_read)},
                         {"edit_bytes", std::to_string(result.bytes_changed)},
                         {"summary", result.summary}});
    // Refused means the tool NEVER RAN, so it is not an execution (S9.1).
    return result.status != tools::Status::Refused;
}

namespace {

// The range a read call covers, or empty for a whole file. Empty is not "no range" -- it
// is the WIDEST range, and the collapse pointer text relies on that ordering.
std::string read_range(const std::string& tool,
                       const std::vector<tools::ToolParamValue>& params) {
    if (tool != "read_slice") {
        return {};
    }
    return param_value(params, "start_line") + "-" + param_value(params, "end_line");
}

} // namespace

bool Agent::mutates_workspace(const std::string& tool) const {
    const tools::ToolDecl* d = registry_.find(tool);
    return d != nullptr && d->mutates_workspace;
}

// A RE-READ IS ANSWERED, ALWAYS. What it costs is charged to the context, not to the model.
//
// What this replaced: a ledger of (path, range) notes that REFUSED a read whose bytes it
// believed were still in the prompt, and told the model to "scroll up". "Scroll up" is
// not an instruction a model can follow -- it has a context window, not a viewport --
// and withholding a tool result trades a bounded cost (a few thousand tokens) for an
// unbounded one (a turn, and then another turn, because the model asks again).
//
// So the read runs, and the DUPLICATE is collapsed instead: the newest copy stays
// verbatim, the copy the model was already holding becomes one line, and the prompt ends
// the turn the size it started. Keyed on byte identity, which needs no invalidation rule:
// a file that changed produces different bytes and nothing collapses.
void Agent::collapse_duplicate_read(const std::string& name,
                                    const std::vector<tools::ToolParamValue>& params,
                                    const tools::ToolResult& result) {
    if (!is_content_read(name) || !result.ok()) {
        return;
    }
    // ONLY UNDER CONTEXT PRESSURE -- because this rewrites HISTORY, and rewritten history
    // is a KV cache thrown away.
    //
    // The collapse edits a turn record that sits INSIDE the stable prefix the backend
    // checkpoints. plan_turn_reuse() finds the divergence at the rewritten message and
    // correctly returns a full re-prefill from token zero. Nothing is stale and nothing
    // is wrong; the saving is simply bought with the entire prefill.
    //
    // MEASURED, and the separation is total:
    //
    //     turns where a collapse fired (n=15)   median TTFT  21,011 ms
    //     turns where none fired      (n=22)    median TTFT     940 ms
    //
    // -- a 22x penalty, with no overlap between the two groups. The run peaked at 34,096
    // tokens against a 96,000-token budget, so all 33 collapses were reclaiming a few KB
    // of a context that was two thirds empty. Below the mark a compaction would trim TO,
    // the bytes are free and the cache is worth more; above it, a collapse spares the run
    // a compaction that costs the same prefill AND destroys information the collapse
    // keeps -- so at that point it is strictly the better of the two.
    const auto budget = static_cast<std::size_t>(std::max(1, config_.context_budget_tokens));
    if (last_prompt_tokens_ <= budget * kCollapseAtPercent / 100) {
        return;
    }

    // Below this, the pointer costs more than the bytes it replaces and the collapse is
    // pure loss -- and short results are things like "(empty file)", where two identical
    // observations are not duplication worth touching.
    static constexpr std::size_t kMinDuplicateBytes = 512;
    if (result.summary.size() < kMinDuplicateBytes) {
        return;
    }
    const std::string path = param_value(params, "path");
    const std::string range = read_range(name, params);
    const std::string current = without_cache_note(result.summary);
    std::size_t collapsed = ctx_.supersede_duplicate_observation(
        current, "(" + path + (range.empty() ? "" : " lines " + range) +
                     " was read again below and is unchanged; this earlier identical copy "
                     "is collapsed to keep one copy in context)");
    // AND THE COPIES THAT ARE NOT IDENTICAL, which are the ones that were doing the harm.
    //
    // Identity-only collapse removes duplicates and keeps every snapshot the run has since
    // edited past -- so at the moment a file has been edited, which is exactly when the
    // context holds contradictory copies of it, this mechanism reclaimed nothing at all.
    // We are already paying the re-prefill by being here; superseding a copy known to be
    // FALSE is worth more than superseding one merely known to be redundant.
    collapsed += ctx_.supersede_stale_copies_of_path(
        path, current,
        "(" + path +
            " was edited after this snapshot was taken and read again below; this earlier "
            "copy is out of date and has been dropped. The current content is further down.)");
    if (collapsed == 0) {
        return;
    }
    // The saving, named. This is the number that justifies the mechanism, and if it stops
    // being large the mechanism should go rather than be tuned.
    emit("duplicate_read_collapsed",
         {{"tool", name},
          {"path", path},
          {"range", range.empty() ? "<whole-file>" : range},
          {"copies_collapsed", std::to_string(collapsed)},
          {"bytes_reclaimed", std::to_string(collapsed * result.summary.size())}});
}

tools::ToolResult Agent::dispatch_call(const std::string& name,
                                       const std::vector<tools::ToolParamValue>& params,
                                       bool& executed) {
    executed = false;

    // `plan` never reaches the registry: the loop owns the checklist.
    if (name == "plan") {
        const TurnResult::PlanOutcome r = apply_plan(params);
        executed = r.ok;
        return r.ok ? tools::ToolResult::okay(r.detail)
                    : tools::ToolResult::error(tools::ErrorClass::Malformed, true, r.detail);
    }

    // The three that END THE RUN. Same reason `plan` is here -- the registry cannot stop
    // the loop -- and they set the halt directly rather than returning a status the loop
    // would have to interpret, because every other way of ending a run is already a named
    // termination_reason and these are more of them.
    //
    // Only exit_plan_mode is mode-bound, and only because leaving plan mode is meaningless
    // in a mode that is not plan mode. The two ASKING tools are available everywhere: an
    // agent run that needs a decision it cannot read out of the code has to be able to ask
    // for it, and while these were conversational-only its only way to raise the question
    // was a text turn -- which, in a working mode, is the ending. The question ended the
    // run instead of asking anything.
    //
    // Refused rather than silently ignored: filtered out of the grammar, so unreachable by
    // sampling and reachable by a synthesized call, and "the loop quietly stopped" is not
    // an outcome worth leaving a path to.
    if (name == "ask_user" || name == "ask_question" || name == "exit_plan_mode") {
        if (name == "exit_plan_mode" && !policy_.conversational) {
            return tools::ToolResult::refused(
                "'exit_plan_mode' is only available in a mode that plans");
        }
        const std::string body =
            param_value(params, (name == "ask_user" || name == "ask_question") ? "question" : "plan");
        if (body.empty()) {
            return tools::ToolResult::error(
                tools::ErrorClass::Malformed, true,
                "'" + name + "' needs a non-empty " +
                    ((name == "ask_user" || name == "ask_question") ? "'question'" : "'plan'"));
        }
        if (name == "ask_question") {
            const std::string options = param_value(params, "options");
            if (options.empty()) {
                return tools::ToolResult::error(
                    tools::ErrorClass::Malformed, true,
                    "'ask_question' requires a non-empty 'options' string with 2-4 selectable options");
            }
        }
        executed = true;
        halted_ = true;
        if (name == "ask_user" || name == "ask_question") {
            const std::string options = param_value(params, "options");
            halt_reason_ = "awaiting_user";
            emit(name, {{"question", body}, {"options", options}});
            if (observer_.on_token) {
                observer_.on_token("answer", body);
            }
            return tools::ToolResult::okay("asked the operator; the run stops here");
        }
        halt_reason_ = "plan_ready";
        emit("plan_ready", {{"chars", std::to_string(body.size())}});
        if (observer_.on_plan_ready) {
            observer_.on_plan_ready(body);
        }
        return tools::ToolResult::okay("presented the plan; the run stops here");
    }

    const tools::ToolDecl* decl = registry_.find(name);
    // Mode policy, then the write gate, then the command gate -- in approval.cpp, with
    // the pure routing functions they drive (S9.3: policy is applied in ONE place).
    if (std::optional<tools::ToolResult> refusal = gate_call(decl, name, params)) {
        return std::move(*refusal);
    }

    // Prior count for annotation after revalidation. Workspace observations always
    // re-execute; RepeatDetector is not an authority for mutable state.
    const std::size_t prior_seen = repeats_.seen_count(name, params);

    // First touch of a path: record whether its syntax check was ALREADY failing, using
    // what is on disk right now -- which is the pre-image, so nothing has to be
    // snapshotted. Costs one extra sandboxed run per file per run, and it is the
    // difference between "your edit broke this" and "this arrived broken".
    if (syntax_ && config_.auto_syntax_check && decl != nullptr &&
        decl->mutates_workspace) {
        const std::string path = param_value(params, "path");
        if (!path.empty() && pre_edit_clean_.find(path) == pre_edit_clean_.end()) {
            const tools::SyntaxVerdict pre = syntax_->check(path, policy_.sandbox_tier);
            if (pre.ran) {
                pre_edit_clean_.emplace(path, pre.clean);
            }
        }
    }

    tools::ToolResult result = registry_.execute(name, params, policy_.sandbox_tier);

    // Refused means the tool NEVER RAN, so it is not an execution (S9.1).
    executed = result.status != tools::Status::Refused;

    // A remote call may have rewritten the workspace in a process we do not see.
    // Invalidate observation freshness whether the call succeeded or failed -- a
    // failed tool can still have side effects, and a stale read after either is a lie.
    if (decl != nullptr && decl->remote && executed) {
        ctx_.invalidate_workspace_freshness();
        emit("workspace_freshness",
             {{"why", "remote_tool"}, {"tool", name},
              {"writes", std::to_string(ctx_.workspace_writes())}});
    }

    // Every successful shell can mutate outside the write ledger (heredocs, redirects,
    // build outputs). Advance the freshness epoch so the next observation revalidates.
    if (decl != nullptr && decl->executes_commands && result.ok() && executed) {
        ctx_.invalidate_workspace_freshness();
        emit("workspace_freshness",
             {{"why", "shell"}, {"tool", name},
              {"writes", std::to_string(ctx_.workspace_writes())}});
    }

    // Annotate a revalidated repeat; never withhold the fresh bytes.
    if (prior_seen > 0 && result.ok() && observes_workspace(name)) {
        const bool unchanged = !observation_is_new(name, params, result);
        emit("repeat_reread",
             {{"tool", name},
              {"prior_count", std::to_string(prior_seen)},
              {"unchanged", unchanged ? "1" : "0"},
              {"read_bytes", std::to_string(result.bytes_read)}});
        if (unchanged) {
            if (result.bytes_read > 0) {
                emit("redundant_read_bytes",
                     {{"tool", name},
                      {"bytes", std::to_string(result.bytes_read)},
                      {"path", param_value(params, "path")}});
            }
            // THE CONSTANT, not a copy of its text. without_cache_note() strips this exact
            // string back off before the summary is cached, so a literal here that drifts
            // from kRepeatNote silently stops being stripped and the notes stack up: two on
            // the third repeat, three on the fourth.
            result.summary += kRepeatNote;
        }
    }

    // A successful write IS the deliverable.
    if (decl != nullptr && decl->mutates_workspace && result.ok()) {
        std::vector<std::string> write_paths;
        const std::string single = param_value(params, "path");
        if (!single.empty()) {
            write_paths.push_back(single);
        } else if (name == "apply_patch") {
            // apply_patch has no `path` param; paths live in the patch body / structured_json.
            if (!result.structured_json.empty() && result.structured_json.front() == '[') {
                std::string cur;
                bool in = false;
                for (char c : result.structured_json) {
                    if (c == '"') {
                        if (in) {
                            write_paths.push_back(cur);
                            cur.clear();
                            in = false;
                        } else {
                            in = true;
                        }
                    } else if (in) {
                        if (c == '\\') {
                            continue; // next char is escaped; keep it plain for paths
                        }
                        cur.push_back(c);
                    }
                }
            }
            if (write_paths.empty()) {
                write_paths = apply_patch::paths_in_patch(param_value(params, "patch"));
            }
        }
        for (const std::string& path : write_paths) {
            if (path.empty()) {
                continue;
            }
            // EVERY WRITE, with the path as the ledger will key it. Writes are the only
            // events that change the workspace, and the deliverable list only reports
            // DISTINCT paths -- so a run editing one file eight times and a run scattering
            // eight files look the same in every other event.
            //
            // `first_touch` is the one that catches the accident this was built for: a run
            // whose workspace root was already `.../ResMon` wrote to `ResMon/Sources/...`,
            // creating a second copy of the tree one level down and then reading the
            // ORIGINAL back and wondering why its edit was missing. Both paths are legal,
            // both writes succeeded, and nothing in the trace marked the moment a second
            // tree appeared.
            const std::string norm = platform::lexically_normal(path);
            // `changed` separates the two things this event used to conflate: a call that
            // moved the workspace, and a call that asked to and found the move already
            // made. Both were logged identically, so the trace of a run rewriting one file
            // twenty times was indistinguishable from a run building twenty files.
            emit("write", {{"path", path},
                           {"normalised", norm},
                           {"tool", name},
                           {"changed", result.mutation_was_noop ? "0" : "1"},
                           {"edit_bytes", std::to_string(result.bytes_changed)},
                           {"first_touch", run_wrote_.count(norm) == 0 ? "1" : "0"},
                           {"distinct_files", std::to_string(ctx_.deliverables().size())}});
            if (!result.mutation_was_noop) {
                ctx_.record_deliverable(path);
                // From here on, a whole-file rewrite of this path is the run editing its
                // own output rather than destroying the operator's data -- which is the
                // difference the write gate needs and did not have. Recorded only on
                // SUCCESS: a write that was refused or failed left the file as the operator
                // had it. Normalised to match the gate's lookup key.
                //
                // A no-op does not claim the path either, and that direction is the safe
                // one: nothing was written, so nothing of the operator's was replaced, and
                // the next rewrite of this file should still stop for a human.
                run_wrote_.insert(norm);
            }
            // The post-edit check goes on the SAME observation rather than becoming a
            // turn of its own: it is a consequence of this edit, not a separate action,
            // and a turn would violate one-turn-one-outcome (S9.1) and burn an iteration.
            annotate_with_syntax_check(path, result);
        }
    }

    // The duplicate collapse, on the SAME success condition as everything above it: a call
    // that was refused or failed read nothing, so there is nothing to have duplicated.
    collapse_duplicate_read(name, params, result);
    // LAST: the two mechanisms above both compare against the un-annotated bytes.
    if (is_content_read(name) && result.ok()) {
        note_stale_copies(ctx_, param_value(params, "path"),
                          without_cache_note(result.summary), result);
    }

    emit("tool_result", {{"tool", name},
                         {"status", std::string(tools::to_string(result.status))},
                         {"read_bytes", std::to_string(result.bytes_read)},
                         {"edit_bytes", std::to_string(result.bytes_changed)},
                         {"summary", result.summary}});
    return result;
}

// The prompt this run would send right now, in REAL tokens rather than an estimate: the
// prompt is rendered and tokenized anyway, so asking the tokenizer costs nothing extra and
// a character heuristic would be wrong exactly where it matters (code and diffs tokenize
// badly).
//
// Rendered the way step() renders it, and that is the whole point of having one function
// for it. The budget check used to build a DIFFERENT prompt: it passed tools_guidance_ to
// ctx_.render() AND to the template, so the entire <tools> block -- every schema for every
// registered tool -- was counted twice, while step() sends it once. The number the run was
// managed against was never the number it was sending.
std::size_t Agent::prompt_tokens() const {
    const model::ChatTemplate tmpl(tok_);
    return tmpl.render(ctx_.render(""), tools_guidance_).size();
}

void Agent::compact_to_budget() {
    // WITH HEADROOM, AND WITH HYSTERESIS. This used to trim only once the prompt had
    // already passed the whole budget, and then dropped exactly one turn -- so a run that
    // crossed the line sat on it, paying a compaction every single turn thereafter and
    // never getting back any slack.
    //
    // Now it starts at kCompactAtPercent and trims down to kCompactToPercent, so a
    // compaction buys enough room for several turns before the next one is due. The gap
    // between the two is what stops the thrash; the low mark is not aggressive because
    // context that is thrown away is re-read later at the cost of a turn (S8.3).
    const auto budget =
        static_cast<std::size_t>(std::max(1, config_.context_budget_tokens));
    const std::size_t high_water = budget * kCompactAtPercent / 100;
    const std::size_t low_water = budget * kCompactToPercent / 100;

    std::size_t tokens = prompt_tokens();
    if (tokens <= high_water) {
        return;
    }
    const std::size_t before = tokens;
    const std::size_t turns_before = ctx_.recent().size();
    while (ctx_.recent().size() > kMinRecentTurns && tokens > low_water) {
        if (ctx_.compact_oldest(ctx_.recent().size() - 1) == 0) {
            break;
        }
        tokens = prompt_tokens();
    }
    // One event per compaction EVENT, carrying both ends of it. Emitting per dropped turn
    // said how often the trim looped and never said whether it achieved anything.
    emit("compaction", {{"tokens_before", std::to_string(before)},
                        {"tokens_after", std::to_string(tokens)},
                        {"budget_tokens", std::to_string(budget)},
                        {"turns_dropped", std::to_string(turns_before - ctx_.recent().size())},
                        {"recent_turns", std::to_string(ctx_.recent().size())}});
}

// Takes whatever the user has said since the last turn boundary into the context.
//
// Everything downstream falls out of ContextStore::add_user_message: the text enters the
// ordered stream at the point it actually arrived, and the latest one is pinned in live
// state where compaction cannot reach it.
std::size_t Agent::take_steering() {
    if (!steer_) {
        return 0;
    }
    const std::vector<std::string> messages = steer_();
    for (const std::string& text : messages) {
        if (text.empty()) {
            continue;
        }
        ctx_.add_user_message(text);
        emit("steer", {{"chars", std::to_string(text.size())},
                       {"at_turn", std::to_string(ctx_.recent().size())}});
    }
    return messages.size();
}

// THE ONLY VERIFICATION IN THE HARNESS, and it is the operator's, verbatim.
//
// The command runs through the same shell path as the model's own calls, tier-gated but
// never carded -- it is the operator's own standing instruction, and raising a card to
// ask the operator whether the operator may check the work would be theatre. Its output
// and exit status become an observation the model reads on the next turn; PASS and FAIL
// are reports of the exit status, not judgements about the work.
//
// What this deliberately does not do: canonicalize the command, strip its pipes, decide
// its exit status "cannot mean what a criterion needs" or demand it be seen to fail
// first. The previous harness did all four, and a measured run deadlocked inside them
// while its contract recorded "ran: 0" eight times without a single execution. If the
// operator's command swallows its own status, the harness reports the status it got --
// the command is the operator's to fix, and the output is in front of both of them.
void Agent::run_operator_check(const char* why) {
    const std::string& command = config_.operator_verify_contract;
    if (command.empty()) {
        return;
    }
    const tools::ToolResult r = registry_.execute(
        "shell", {{"command", command}}, policy_.sandbox_tier);

    context::CheckResult check;
    check.command = command;
    // Refused (the sandbox said no) and never-executed (exit 126/127: the command does
    // not exist here) both mean the check never ran, and that is a distinct fact from
    // failing: a red that never executed says nothing about the workspace.
    check.ran = r.status != tools::Status::Refused && !r.never_executed();
    check.passed = r.ok();
    check.detail = r.summary;
    ctx_.set_last_check(check);

    emit("verification", {{"contract", command},
                          {"why", why},
                          {"ran", check.ran ? "1" : "0"},
                          {"passed", check.passed ? "1" : "0"}});

    // The OUTPUT, not just the verdict. A run handed "- FAIL swift build" has been told
    // what it already assumed; what changes the next edit is the compiler's actual
    // complaint, and this record is its only route into the prompt.
    context::TurnRecord marker;
    marker.tool_name = "operator_check";
    marker.tool_args_summary = command;
    marker.observation =
        "(operator check `" + command + "`: " +
        (!check.ran ? "COULD NOT RUN" : check.passed ? "PASS" : "FAIL") +
        (check.detail.empty() ? std::string(")")
                              : ". Output:\n" + check.detail + ")");
    marker.observation_is_error = !check.passed;

    // Safe stuck-run signals. Observations only — never a tool lock, never "the contract
    // is wrong". The operator contract stays authoritative.
    if (!check.ran) {
        ++could_not_run_streak_;
        same_diag_streak_ = 0;
        last_primary_diag_fp_.clear();
        if (could_not_run_streak_ >= 2) {
            marker.observation +=
                "\n(System Observation: the operator check could not run "
                "repeatedly — treat this as an environment/contract problem: the "
                "command may be missing, the sandbox may refuse it, or the contract "
                "may not be executable here. Do not keep editing as if the check "
                "failed.)";
        }
    } else if (!check.passed) {
        could_not_run_streak_ = 0;
        const log_triage::StructuredTriage triage = log_triage::analyze(check.detail);
        const std::string fp = log_triage::primary_fingerprint(triage);
        if (!fp.empty() && fp == last_primary_diag_fp_) {
            ++same_diag_streak_;
        } else {
            last_primary_diag_fp_ = fp;
            same_diag_streak_ = fp.empty() ? 0 : 1;
        }
        if (same_diag_streak_ >= 2 && !fp.empty()) {
            // Overlap between deliverable paths and paths cited in diagnostics.
            bool overlap = false;
            for (const std::string& del : ctx_.deliverables()) {
                for (const std::string& cited : triage.referenced_paths) {
                    if (del == cited || del.find(cited) != std::string::npos ||
                        cited.find(del) != std::string::npos) {
                        overlap = true;
                        break;
                    }
                }
                if (overlap) {
                    break;
                }
                // Also match against failing test node ids / primary diagnostic paths.
                for (const log_triage::Diagnostic& d : triage.primary_diagnostics) {
                    if (!d.path.empty() &&
                        (del == d.path || del.find(d.path) != std::string::npos ||
                         d.path.find(del) != std::string::npos)) {
                        overlap = true;
                        break;
                    }
                }
                if (overlap) {
                    break;
                }
            }
            if (!overlap) {
                marker.observation +=
                    "\n(System Observation: the same primary diagnostics keep "
                    "appearing and none of the paths they cite overlap the files this "
                    "run has changed. Reassess the target or ask the user — do not "
                    "blame the verification command, and do not lock tools.)";
            }
        }
    } else {
        could_not_run_streak_ = 0;
        same_diag_streak_ = 0;
        last_primary_diag_fp_.clear();
    }

    ctx_.add_turn(std::move(marker));

    if (observer_.on_verification) {
        observer_.on_verification(check);
    }
}

// One executed call into the repeat cache, with what it returned and where the write
// counter stood. One helper so the primary and the batched calls cannot be recorded
// differently -- the batching hole in the old detector was measured: three reads per
// turn invisible to it, forever.
void Agent::record_call(const std::string& tool,
                        const std::vector<tools::ToolParamValue>& params,
                        const tools::ToolResult& result) {
    // A refusal is not an execution (S9.1), so it is not recorded.
    if (result.status == tools::Status::Refused) {
        return;
    }
    // A cache-served result carries the note; the cache stores the bytes without it,
    // so the next serve appends exactly one.
    repeats_.record(tool, params, result.ok(), without_cache_note(result.summary),
                    ctx_.workspace_writes());
}

RunReport Agent::run(const model::CancelToken& cancel) {
    RunReport report;
    inert_turns_ = 0;
    inert_streak_had_tool_call_ = false;
    executed_tool_calls_in_run_ = 0;
    const auto started = clock_.mono();
    // Run-scoped so shell / git / MCP observe the same token from every execute(),
    // including the parallel read path. Cleared on every exit from this function.
    registry_.set_cancel_token(&cancel);
    struct ClearCancel {
        tools::Registry& reg;
        ~ClearCancel() { reg.set_cancel_token(nullptr); }
    } clear_cancel{registry_};

    while (!halted_) {
        if (cancel.cancelled()) {
            report.termination_reason = "cancelled";
            break;
        }
        // The turn boundary, and the only place the user's words enter a live run.
        report.steers_received += take_steering();

        // --- budgets, checked before a turn is spent -----------------------
        //
        // Ceilings on a runaway, not judgements. Which of the two fired is named in the
        // reason, because "budget_exhausted" once meant two different limits and the
        // wrong dial got raised.
        if (report.iterations >= config_.budget.max_iterations) {
            report.termination_reason = "max_turns";
            break;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                 clock_.mono() - started)
                                 .count();
        if (elapsed >= config_.budget.wall_clock_seconds) {
            report.termination_reason = "wall_clock";
            break;
        }
        // One factual note, once, while there is still room to land a multi-file change.
        // The budget is the harness's; the model cannot observe it any other way.
        if (!budget_note_sent_ &&
            report.iterations >=
                config_.budget.max_iterations - kBudgetWarningTurns) {
            budget_note_sent_ = true;
            const auto remaining_s = config_.budget.wall_clock_seconds - elapsed;
            context::TurnRecord note;
            note.tool_name = "budget";
            note.observation = "(" +
                std::to_string(config_.budget.max_iterations - report.iterations) +
                " turns and ~" + std::to_string(remaining_s) +
                "s of wall clock remain in this run's budget.)";
            ctx_.add_turn(std::move(note));
            emit("budget_note",
                 {{"turns_left", std::to_string(config_.budget.max_iterations -
                                                report.iterations)},
                  {"seconds_left", std::to_string(remaining_s)}});
        }

        const std::size_t writes_before_turn = ctx_.workspace_writes();
        TurnResult turn = step(cancel);
        ++report.iterations;

        // ONE LINE PER TURN, always. The log had every ingredient of a turn and no record
        // of the turn itself, so reconstructing "what did iteration 41 actually do" meant
        // correlating four event kinds by sequence number and guessing at the boundaries.
        emit("turn", {{"n", std::to_string(report.iterations)},
                      {"outcome", std::string(to_string(turn.outcome))},
                      {"tool", turn.tool_name.empty() ? "-" : turn.tool_name},
                      {"batched", std::to_string(turn.extra_calls.size())},
                      {"batch_count", std::to_string(turn.batch_count)},
                      {"think_tokens", std::to_string(turn.think_tokens)},
                      {"text_tokens", std::to_string(turn.text_tokens)},
                      {"tool_tokens", std::to_string(turn.tool_tokens)},
                      {"cap_phase", turn.cap_phase},
                      {"read_bytes", std::to_string(turn.tool_result.bytes_read)},
                      {"edit_bytes", std::to_string(turn.tool_result.bytes_changed)},
                      {"ok", turn.tool_result.ok() ? "1" : "0"},
                      {"deliverables", std::to_string(ctx_.deliverables().size())},
                      {"workspace_writes", std::to_string(ctx_.workspace_writes())},
                      {"open_items", std::to_string(ctx_.open_checklist_items())},
                      {"compactions", std::to_string(ctx_.compaction_count())}});

        if (turn.outcome == Outcome::BackendError) {
            report.termination_reason = "backend_error";
            break;
        }
        if (turn.outcome == Outcome::Cancelled) {
            report.termination_reason = "cancelled";
            break;
        }

        if (observer_.on_turn) {
            observer_.on_turn(turn, turn.generation.ttft_ms);
        }

        // Record the turn -- observations only, nothing inferred about the workspace
        // (S8.4). A capped working note from think may fill an empty answer channel on a
        // tool turn so the next prompt has continuity; that is model speech, not a
        // workspace fact.
        context::TurnRecord rec;
        rec.assistant_text = turn.assistant_text;
        rec.tool_name = turn.tool_name;
        rec.tool_args_summary = preview_of(turn.tool_name, turn.tool_params);
        rec.tool_call_text = call_surface_form(turn.tool_name, turn.tool_params);
        // Whole-file reads are SNAPSHOTS, and the context has to be able to find the ones
        // a later edit has invalidated. See ContextStore::stale_copies_of_path.
        if (is_content_read(turn.tool_name) && turn.tool_result.ok()) {
            rec.observed_path = param_value(turn.tool_params, "path");
        }
        rec.observation = turn.tool_result.summary;
        rec.observation_is_error = !turn.tool_result.ok();
        rec.last_event_seq = log_.events_written();

        // Tool turns often leave the answer channel empty (reasoning lived in <think>).
        // Carry a trailing slice so the next turn continues instead of re-deriving.
        // Never on LengthCapped or cut_for_looping -- that re-seeds a stuck think.
        if (turn.outcome == Outcome::ToolCallExecuted && !turn.cut_for_looping &&
            is_blank(rec.assistant_text) && !turn.reasoning.empty()) {
            rec.assistant_text = working_note_from_reasoning(turn.reasoning);
        }

        // The same floor Registry::execute puts under its tools, applied to the paths
        // that do not go through it. render() drops an empty observation, so without
        // this the turn leaves no trace in the next prompt and the model repeats it.
        if (rec.observation.empty() && turn.outcome == Outcome::ToolCallExecuted) {
            rec.observation = "(" + turn.tool_name +
                              (turn.tool_result.ok()
                                   ? " succeeded and produced no output)"
                                   : " failed, with no detail)");
        }

        // A turn that hit the token cap mid-thought leaves no answer and no call. Full
        // reasoning is not backfilled here (that would re-seed a looping think). The
        // truncation itself becomes the observation so the next prompt is not
        // byte-identical -- otherwise a fixed seed redraws the same continuation forever.
        //
        // Observed: twelve consecutive turns, prompt `tokens=2044 messages=11` every
        // time, generation `tokens=4096` every time, until the wall clock killed it.
        //
        // AND IT MUST SAY WHICH PHASE RAN OUT, because the remedy differs and the model
        // cannot see the counter. Capping inside the TOOL phase does not mean the model
        // never got started -- it means the call it was emitting was too long to finish.
        // The old text said "before any tool call was made; nothing ran" in every case,
        // which reads as "you did not begin" and hid the only fact that would have
        // changed the next move.
        //
        // Measured on a synthwave-theme run: two consecutive turns capped at 4030 and
        // 4018 TOOL tokens, both whole-file rewrites of a file too big to emit in one
        // generation. Told only that nothing ran, the model never learned the size was
        // the problem -- it spent its last four turns re-reading that same file,
        // byte-identically, and the run ended `stalled` with one of fourteen items done.
        if (turn.outcome == Outcome::LengthCapped) {
            if (turn.cap_phase == "tool") {
                rec.observation =
                    "(your tool call was CUT OFF after " + std::to_string(turn.tool_tokens) +
                    " tokens -- too long to finish in one turn, so nothing ran and nothing "
                    "changed on disk. Sending it again unchanged will hit the same limit. "
                    "Make a SMALLER call: a targeted `replace_in_file` for the one section "
                    "you are changing rather than a whole-file write, or split the change "
                    "into several edits and apply them one at a time.)";
            } else {
                rec.observation = "(generation hit the token cap during " +
                                  (turn.cap_phase.empty() ? std::string("this turn")
                                                          : turn.cap_phase) +
                                  "; no tool call was made and nothing ran.)";
            }
            rec.observation_is_error = true;
        }
        ctx_.add_turn(std::move(rec));

        // EVERY CALL THE TURN MADE, not just the one at the front of it. Batching made
        // the primary-only version a hole big enough to drive a whole failure through:
        // three reads per turn invisible to the detector no matter how often they came
        // back.
        if (turn.outcome == Outcome::ToolCallExecuted) {
            ++executed_tool_calls_in_run_;
            record_call(turn.tool_name, turn.tool_params, turn.tool_result);
        }
        // PROGRESS RESETS THE STREAK -- not merely having called something. Measured
        // before the ending check below, and after record_call only because the answer was
        // computed on the execution path, where the detector had not yet been updated.
        const bool progressed = turn_made_progress(turn);
        if (progressed) {
            inert_turns_ = 0;
            inert_streak_had_tool_call_ = false;
        }

        // A turn that called only `plan` is not progress, and a run that keeps doing it
        // is not working -- it is performing progress for the operator's panel. The
        // inert counter alone cannot see it because any real edit between two `plan`
        // calls resets the streak; this counter does not reset on an edit, only on a
        // turn that did something else entirely.
        const bool plan_only_turn =
            turn.outcome == Outcome::ToolCallExecuted && turn.extra_calls.empty() &&
            is_display_only(turn.tool_name);
        if (plan_only_turn) {
            ++consecutive_plan_only_turns_;
        } else if (turn.outcome != Outcome::ToolCallExecuted ||
                   !is_display_only(turn.tool_name)) {
            consecutive_plan_only_turns_ = 0;
        }

        // A turn whose only write was a small edit is not the same as a turn that moved
        // the run toward done. A run that makes many tiny edits without ever closing a
        // checklist item or running a build is polishing, not finishing; this counter
        // exists to say so before the turn budget does.
        const bool micro_edit_turn =
            turn.outcome == Outcome::ToolCallExecuted && turn.extra_calls.empty() &&
            turn.tool_result.bytes_changed > 0 &&
            turn.tool_result.bytes_changed <= 64 &&
            (turn.tool_name == "replace_in_file" || turn.tool_name == "write_file");
        if (micro_edit_turn) {
            ++consecutive_micro_edit_turns_;
        } else if (turn.outcome == Outcome::ToolCallExecuted &&
                   turn.tool_result.bytes_changed > 64) {
            consecutive_micro_edit_turns_ = 0;
        }
        for (const TurnResult::ExtraCall& extra : turn.extra_calls) {
            record_call(extra.tool_name, extra.params, extra.result);
        }

        // Calls batched behind the first each get their own record and their own UI row.
        for (std::size_t i = 0; i < turn.extra_calls.size(); ++i) {
            const TurnResult::ExtraCall& extra = turn.extra_calls[i];
            context::TurnRecord er;
            er.tool_name = extra.tool_name;
            er.tool_args_summary = param_value(extra.params, "path");
            if (er.tool_args_summary.empty()) {
                er.tool_args_summary = param_value(extra.params, "command");
            }
            // Batched calls carry their surface form too. A turn that read four files
            // showed one call and three tool_responses with nothing that produced them,
            // which is the same missing-call defect at four times the rate.
            er.tool_call_text = call_surface_form(extra.tool_name, extra.params);
            if (is_content_read(extra.tool_name) && extra.result.ok()) {
                er.observed_path = param_value(extra.params, "path");
            }
            er.observation = extra.result.summary;
            er.observation_is_error = !extra.result.ok();
            er.first_event_seq = log_.events_written();
            er.last_event_seq = er.first_event_seq;
            ctx_.add_turn(std::move(er));

            if (observer_.on_turn) {
                TurnResult as_turn;
                as_turn.outcome = extra.result.status == tools::Status::Refused
                                      ? Outcome::ToolCallRefused
                                      : Outcome::ToolCallExecuted;
                as_turn.tool_name = extra.tool_name;
                as_turn.tool_params = extra.params;
                as_turn.tool_result = extra.result;
                as_turn.batch_index = i + 1;
                as_turn.batch_count = turn.batch_count;
                observer_.on_turn(as_turn, 0.0);
            }
        }

        // A RUN THAT HAS ENDED ITSELF IS NOT A RUN TO KEEP DRIVING. ask_user and
        // exit_plan_mode set the halt during dispatch; the turn and its batched calls
        // are recorded above, so the operator's reply continues from a complete context.
        if (halted_) {
            report.termination_reason = halt_reason_;
            break;
        }

        // A RUN THAT KEEPS RESTATING ITS CHECKLIST IS NOT WORKING. The inert counter
        // cannot see it because any real edit between two `plan` calls resets the
        // streak; this counter does not reset on an edit, only on a turn that did
        // something else entirely. Two in a row is a warning; three ends the run.
        if (consecutive_plan_only_turns_ >= 2 && !policy_.conversational) {
            context::TurnRecord note;
            note.observation =
                "[Note: You have restated the plan " +
                std::to_string(consecutive_plan_only_turns_) +
                " turns in a row without doing any of it. The panel already shows the "
                "list; calling `plan` again changes nothing. Pick the next open item "
                "and do it NOW -- make the edit, run the build, or call `ask_question` "
                "if you are blocked. The run will end if the next turn is also only a "
                "plan update.]";
            ctx_.add_turn(std::move(note));
            emit("plan_spin_warning",
                 {{"consecutive", std::to_string(consecutive_plan_only_turns_)}});
        }
        if (consecutive_plan_only_turns_ >= 3 && !policy_.conversational) {
            report.termination_reason = "stalled";
            emit("stalled",
                 {{"why", "plan_spin"},
                  {"consecutive_plan_only",
                   std::to_string(consecutive_plan_only_turns_)},
                  {"wrote_bytes_in_run", std::to_string(ctx_.workspace_writes())}});
            break;
        }

        // A RUN THAT ONLY POLISHES IS NOT FINISHING. Many tiny edits in a row without
        // a build, a closed checklist item, or a larger change is the shape of a run
        // that has lost the plot. The threshold is high enough that a real fix --
        // which often lands as a few small edits -- does not trip it.
        if (consecutive_micro_edit_turns_ >= 5 && !policy_.conversational) {
            context::TurnRecord note;
            note.observation =
                "[Note: You have made " +
                std::to_string(consecutive_micro_edit_turns_) +
                " small edits in a row without running a build or closing a checklist "
                "item. Stop polishing and verify the whole: run the build, then either "
                "close the item you are on or call `ask_question` if you are blocked.]";
            ctx_.add_turn(std::move(note));
            emit("micro_edit_warning",
                 {{"consecutive", std::to_string(consecutive_micro_edit_turns_)}});
        }

        // Compaction, not eviction (S8.3) -- and only when the BUDGET says so.
        compact_to_budget();

        // THE OPERATOR'S CHECK, after any turn that wrote. The output lands as an
        // observation before the model's next move, so every edit is followed by the one
        // reading that can tell the model what its edits actually did.
        if (!config_.operator_verify_contract.empty() &&
            ctx_.workspace_writes() > writes_before_turn) {
            run_operator_check("post_write");
        }

        // AN INERT TURN IS ONE THAT NEITHER WROTE NOR LEARNED, and a run that produces
        // several in a row is done -- either because the model is answering (it stopped
        // calling tools) or because it is spinning (it kept calling them to no effect).
        // Everything else -- whether the answer is right, whether the work is finished --
        // stays the operator's judgement, informed by the operator's check.
        //
        // This replaces an ending that watched TEXT-ONLY turns and reset on any executed
        // call. That signal was wrong in both directions at once: it could not see a run
        // repeating itself (each repeat reset the count), and it killed runs that were
        // plainly working (two narration turns in a row, 23 turns of budget left, edits
        // still landing). See kRunNudgesBeforeEnding for both measurements.
        //
        // It is still not adjudication. The harness forms no view on whether the work is
        // right; it observes that nothing has changed and nothing has been learned for
        // several turns running, which is the same kind of fact as the turn and wall-clock
        // budgets -- and unlike those, it is a fact about this run rather than a ceiling.
        //
        // A turn the breaker CUT is exempt: the model did not conclude, the harness
        // interrupted it. The fact of the cut is already in the context; the run
        // continues, and the budgets bound a model that can produce nothing else.
        //
        // REFUSED and LENGTH-CAPPED turns leave the count alone rather than incrementing
        // it. A tool the human declined is not the model failing to progress, and a
        // generation cut at the token cap never got to choose.
        const bool countable = turn.outcome == Outcome::TextOnly ||
                               turn.outcome == Outcome::ToolCallExecuted;
        if (!progressed && countable && !turn.cut_for_looping) {
            // Last look at the inbox before anything else. A human watching a run drift
            // toward an ending is exactly the human who types "keep going" -- and ending
            // the run a moment after they said it, having already read it off the pipe,
            // would be the worst possible time to stop listening.
            const std::size_t rescued = take_steering();
            if (rescued > 0) {
                report.steers_received += rescued;
                continue;
            }

            // NUDGE, and end only after the model has ignored the nudges.
            //
            // The count is of CONSECUTIVE INERT turns -- progress resets it -- so this is
            // not a budget for narration across a run, it is how many times in a row the
            // model may neither write nor learn anything.
            const std::size_t allowed =
                policy_.conversational ? kPlanNudgesBeforeEnding : kRunNudgesBeforeEnding;
            ++inert_turns_;
            const bool spun = turn.outcome == Outcome::ToolCallExecuted;
            inert_streak_had_tool_call_ = inert_streak_had_tool_call_ || spun;
            const char* const why = spun ? "no_progress" : "text_only_turn";

            if (inert_turns_ <= allowed) {
                // THE NOTE SAYS WHICH FAILURE THIS IS. "Call a tool now" is the wrong
                // advice for a model that just called one and got back bytes it already
                // had -- and it is exactly what the old single note told it, which is how
                // a nudge came to reinforce the re-read loop it was meant to break.
                //
                // A `plan` TURN IS ITS OWN FAILURE AND NEEDS ITS OWN NOTE. The spun note
                // below explains a byte-identical read or a no-op edit, and a model that
                // restated its checklist did neither -- it read advice about files it never
                // touched. Measured on the run that produced this: four plan-only turns,
                // each one nudged with the file note, none of them broken out of.
                const bool display_only_turn =
                    spun && turn.extra_calls.empty() && is_display_only(turn.tool_name);
                context::TurnRecord note;
                note.observation =
                    display_only_turn
                    ? "[Note: That turn called only `plan`. Restating the checklist writes "
                      "nothing and reads nothing, so the run is exactly where it was before "
                      "it -- and the panel already shows the list you just sent. Do the next "
                      "open item NOW: make the edit, or run the build to see where you stand. "
                      "Turns that neither write nor learn anything end the run.]"
                    : spun ? "[Note: That call changed nothing and returned nothing you did not "
                           "already have -- either the file was byte-identical to a copy already "
                           "in this conversation, or the edit matched but wrote no bytes. Calling "
                           "it again will produce the same result. Scroll up and use what is "
                           "already here: make the NEXT edit, run the build to see where you "
                           "stand, or call 'ask_question' if you need a decision from the human. "
                           "Do not re-read a file this run has already read unless something has "
                           "written to it since.]"
                    : policy_.conversational
                        ? "[Note: You are in Plan mode and your last turn called no tool. Do not output "
                          "standalone text updates or commentary -- they do not reach the human. "
                          "If you were about to read something, call the tool NOW "
                          "(`read_file`, `read_many`, `list_dir`, `find_files`, `search`). "
                          "If you have a design choice to put to the human, call 'ask_question' with 2-4 options. "
                          "If your plan is ready, call 'exit_plan_mode'. "
                          "You do not need to read every file before asking or planning.]"
                        : "[Note: Your last turn called no tool. If you were about to act -- reading, "
                          "editing, or running something -- call the tool NOW; saying you will act does "
                          "not perform the action, and this run has not finished the work yet. "
                          "If you are genuinely done and this text WAS your final answer, say so "
                          "plainly and the run will end shortly.]";
                ctx_.add_turn(std::move(note));
                emit("nudged", {{"why", why},
                                {"consecutive", std::to_string(inert_turns_)}});
                continue;
            }
            if (policy_.conversational) {
                report.termination_reason = "awaiting_user";
                emit("yielded", {{"why", why},
                                 {"consecutive", std::to_string(inert_turns_)}});
                break;
            }
            // TWO ENDINGS, ONE COUNT. A run that only ever narrated is handing back, and
            // `ended` keeps its existing meaning -- including its eligibility for
            // completed=true, which is decided below against the checklist and the check.
            // A run that kept calling tools to no effect is STALLED, and stalled is never
            // completion: it is not a reason the completed=true block below recognises, so
            // it falls through to false without needing to say so twice.
            report.termination_reason = inert_streak_had_tool_call_ ? "stalled" : "ended";
            emit(report.termination_reason,
                 {{"why", why},
                  {"consecutive", std::to_string(inert_turns_)},
                  {"wrote_bytes_in_run", std::to_string(ctx_.workspace_writes())}});
            break;
        }
    }
    if (report.termination_reason.empty()) {
        report.termination_reason = halted_ ? halt_reason_ : "loop_exit";
    }

    report.compactions = ctx_.compaction_count();
    report.unfinished_items = ctx_.open_checklist_items();

    // `completed` is THREE observed facts, not a verdict: the model answered, its own
    // checklist has nothing left open, and -- when the operator configured a check --
    // that check's latest reading passed. A run that answered with no check configured
    // completes on the first two alone, and the report says which claim is being made by
    // carrying the check (or its absence) in the trace.
    //
    // The checklist clause is the 2026-08-08 addition, and it is arithmetic rather than
    // judgement: the list is the MODEL'S OWN, written by its own `plan` call. A run that
    // ended with 10 of its own 10 items open and reported completed=true was not making a
    // debatable claim, it was contradicting itself -- and `unfinished_items` was computed
    // AFTER this block, so the number existed and simply was not consulted.
    //
    // This still says nothing about whether the work is right. It says the model did not
    // finish the list it wrote.
    if (report.termination_reason == "ended") {
        const bool list_clear = report.unfinished_items == 0;
        if (config_.operator_verify_contract.empty()) {
            report.completed = list_clear;
        } else {
            // A run that wrote nothing never triggered the post-write reading; take one
            // now so the ending is judged against the workspace as the run left it.
            if (!ctx_.last_check().has_value()) {
                run_operator_check("final");
            }
            report.completed = list_clear && ctx_.last_check().has_value() &&
                               ctx_.last_check()->passed;
        }
    }
    emit("run_end", {{"termination_reason", report.termination_reason},
                     {"iterations", std::to_string(report.iterations)},
                     {"completed", report.completed ? "true" : "false"},
                     {"unfinished_items", std::to_string(report.unfinished_items)},
                     {"steers_received", std::to_string(report.steers_received)}});
    return report;
}

} // namespace lmp::loop
