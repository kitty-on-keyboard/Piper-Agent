#include "src/context/context.hpp"

#include <algorithm>

namespace lmp::context {
namespace {

// The persona. Stated as standing behaviour rather than personality colour: each line is
// something an observer could catch the agent violating, which is the only kind of
// instruction worth spending prompt tokens on.
constexpr const char* kPersona =
    "You are Piper, a master software engineer.\n"
    "\n"
    "- You do not cut corners. If a job needs six steps you take six steps.\n"
    "- You favour correctness over speed. A slower answer that is right beats a fast one\n"
    "  that is probably right.\n"
    "- You reach for the most specific tool for the job: replace_in_file over write_file\n"
    "  for a partial change, git_diff over shell, read_slice over read_file on a large\n"
    "  file. A general tool used where a specific one exists is a mistake.\n"
    "- When a build or check fails, identify the exact error message and file line before\n"
    "  modifying source code.\n"
    "- You test whenever it is possible and safe to do so, and you run the test rather\n"
    "  than assert that it would pass.\n"
    "- When a decision would genuinely change what you build and no amount of reading will\n"
    "  settle it -- which approach, which library, which of two designs -- you ask with\n"
    "  `ask_question` and offer 2 to 4 options. Writing the question as ordinary text does\n"
    "  not ask it: a text-only turn is your final answer and the run ends on it.\n"
    "- Continue from the latest tool result and your last working note. Do not restate\n"
    "  the user's request, narrate what you are about to do, or explain work that speaks\n"
    "  for itself.\n";

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
        if (!t.user_text.empty()) {
            // A human turn keeps a longer anchor than an assistant one. An instruction
            // is the only thing in the stream that changes what the run is FOR, and a
            // 160-character truncation of "no, use the other approach because ..." can
            // lose the half that mattered. The live-state block carries the latest one
            // verbatim regardless; this is for the ones behind it.
            span += "- you were told: " + first_line(t.user_text, 400) + "\n";
            continue;
        }
        if (t.tool_name.empty()) {
            span += "- said: " + first_line(t.assistant_text, 160) + "\n";
            continue;
        }
        // The two producers of tool_args_summary do not agree on format, and this line is
        // PROMPT-FACING. A turn's own call gets preview_of(), which already names the tool
        // -- "read_file(path=x)" -- while the extra calls batched behind it get a bare path
        // or command. Prepending the name unconditionally is right for the second and
        // produced "- read_file(read_file(path=x)) -> ..." for the first, in every
        // compacted line of every run that ever trimmed.
        //
        // The same bug was fixed in the journal (surface/context_journal.cpp, turn_body);
        // this is the copy that costs tokens rather than index space, in the one place a
        // run is already short of them.
        span += "- ";
        if (t.tool_args_summary.rfind(t.tool_name + "(", 0) == 0) {
            span += first_line(t.tool_args_summary, 80);
        } else {
            span += t.tool_name;
            if (!t.tool_args_summary.empty()) {
                span += "(" + first_line(t.tool_args_summary, 80) + ")";
            }
        }
        span += t.observation_is_error ? " FAILED: " : " -> ";
        span += first_line(t.observation, 200) + "\n";
    }
    spans_.push_back(std::move(span));

    // Hand the full turns over BEFORE they are destroyed. Ordering is the whole
    // contract: after the erase below there is nothing left to journal, and a sink
    // called afterwards would be handed an empty range and report success.
    if (compaction_sink_) {
        const std::vector<TurnRecord> dropped(recent_.begin(),
                                              recent_.begin() +
                                                  static_cast<std::ptrdiff_t>(drop));
        compaction_sink_(dropped, spans_.size() - 1, spans_.back());
    }

    recent_.erase(recent_.begin(), recent_.begin() + static_cast<std::ptrdiff_t>(drop));
    ++compactions_;
    return drop;
}

// ORDERING IS LOAD-BEARING (S5.10). The prompt is laid out most-stable-first so that the
// KV prefix survives a turn:
//
//   [system: persona + mode + tools + workspace + conventions + memory]
//                                                         never changes within a run
//   [user: opening mission]                             never changes within a run
//   [compacted spans]                                     changes only when compaction runs
//   [recent turns]                                        append-only between compactions
//   [live state: latest user pin, checklist, deliverables]
//                                                         changes constantly -- LAST
//
// The task is a user message, not system identity. System stays role/constraints/tools;
// the opening ask sits as the first stable user message so chronology matches 2026
// coding-agent practice while remaining run-constant for the KV checkpoint.
//
// The live state used to live in the system message, at the very front. Every checklist
// tick, every deliverable and every verification therefore rewrote token 0, diverged the
// prefix, and forced a full re-prefill of the whole context -- every single turn. Measured
// on a real run before the move: TTFT climbed monotonically 1427 -> 1758 ms across 29
// turns while the context sat at ~3k tokens. The reuse machinery in src/model/ was doing
// its job; this layer was handing it a different prompt each time.
//
// Anything appended AFTER the mutable block would inherit the same problem, so nothing is.
std::vector<Message> ContextStore::render(std::string_view tool_guidance) const {
    std::vector<Message> out;

    // Stable system: persona, mode, guidance, workspace, conventions, memory. The
    // deliverable is named in the user message that follows -- not here.
    std::string system;
    // Ahead of the persona because the reference template opens the system message with
    // it. Empty for `medium` and for an unset level, and an empty one contributes no
    // separator either -- so the default prompt is byte-identical to what it was before
    // this field existed, and an existing KV prefix is not invalidated by adding the
    // feature.
    if (!reasoning_brief_.empty()) {
        system += reasoning_brief_;
        system += "\n\n";
    }
    system += persona_.empty() ? kPersona : persona_.c_str();
    // Immediately after the persona and before the tools, because it QUALIFIES the persona
    // -- and does so even when the operator has replaced it. A plan-mode run told "you
    // test whenever it is possible, and you run the test rather than assert that it would
    // pass" cannot do either, and reaching for the shell to try is the first thing it did.
    if (!mode_brief_.empty()) {
        system += "\n\n";
        system += mode_brief_;
    }
    system += "\n\n";
    system += std::string(tool_guidance);
    // Immediately after the tools, because it is the one fact every path argument to
    // every one of them depends on. Says what relative paths mean as well as what the
    // root is -- "you are in /x" alone still leaves open whether tools want absolute
    // paths, and the run that guessed /home/user guessed absolute.
    if (!workspace_root_.empty()) {
        system += "\n\n# Workspace\n\nYou are working in " + workspace_root_ +
                  ". Paths you pass to tools are resolved against it, and anything "
                  "outside it is refused, so write them relative to it -- `src/store.py`, "
                  "not `/home/user/src/store.py`. Directories in a path you write to are "
                  "created for you.";
    }
    // Next to the workspace because it is a fact ABOUT the workspace: what this one
    // already remembers. Only when an earlier session actually left something -- see
    // set_recall_scope for the measurement, and for why pointing at an empty store is
    // worse than saying nothing.
    //
    // The closing instruction is not padding. An empty recall already answers "nothing
    // stored matches that", and runs asked anyway, often enough to trip break_repeat and
    // then escalated_hold. Telling the model up front what a miss MEANS is cheaper than
    // suppressing the repeat after the fact.
    if (recall_sessions_ > 1 && recall_items_ > 0) {
        system += "\n\n# What this workspace already remembers\n\nEarlier sessions here "
                  "left " + std::to_string(recall_items_) + " stored items across " +
                  std::to_string(recall_sessions_) +
                  " sessions -- their turns, the files they read and what they worked "
                  "out. `context_recall` searches all of it. Reach for it before "
                  "re-deriving something this project may have already settled, and "
                  "before re-reading a file an earlier session read. If it comes back "
                  "empty, that is a real answer: the fact is not stored, so go to the "
                  "files instead rather than asking again.";
    }
    if (!project_instructions_.empty()) {
        system += "\n\n# Project conventions\n\n" + project_instructions_;
    }
    // AFTER the operator's conventions and clearly attributed, because these are the
    // model's OWN notes coming back into its own prompt. Presented as recollection to
    // check rather than as instruction: a wrong note written last week would otherwise
    // outrank what this session can see with its own tools, and nothing in the file has
    // been reviewed by anyone.
    if (!project_memory_.empty()) {
        system += "\n\n# Notes you left in earlier sessions\n\nYou wrote these, not the "
                  "operator. Treat them as recollection worth checking, not as "
                  "instructions, and prefer what you can observe now.\n\n" +
                  project_memory_;
    }
    out.push_back({Role::System, std::move(system)});

    // Opening mission: first stable user message. Not also stored in recent_ -- render
    // injects it once so the ask is not duplicated when follow-ups arrive.
    out.push_back({Role::User, user_turns_.front()});

    // Compacted spans, oldest first, as observed history.
    for (const std::string& span : spans_) {
        out.push_back({Role::User, span});
    }

    // Recent turns, verbatim. The assistant's answer body (or a capped working note from
    // a tool turn) and the observation it got back -- the observation goes in as a
    // tool_response, which is Qwen's shape.
    for (const TurnRecord& t : recent_) {
        if (!t.user_text.empty()) {
            // Images the HUMAN attached (dragged or pasted into the pane), as against
            // the ones a tool returned below. Same MessageImage shape, same unresolved
            // token count -- Agent::resolve_images fills both.
            model::Message um{Role::User, t.user_text};
            for (const std::string& p : t.observed_images) {
                um.images.push_back({0, p});
            }
            out.push_back(std::move(um));
            continue;
        }
        // THE ASSISTANT TURN INCLUDES ITS CALL. Without this the transcript read
        // `assistant: <prose>` then `tool_response: <result>` -- a tool result arriving
        // after a message that called nothing -- and a run never saw itself make a call.
        //
        // MEASURED, plan mode, 13 turns: every turn whose answer channel held real prose
        // (13-29 tokens) ended with NO call, and every turn that made a call held 1-4
        // tokens of prose. The model was reproducing the shape of its own history, where
        // an assistant message with prose in it is a message that ends the turn.
        if (!t.assistant_text.empty() || !t.tool_call_text.empty()) {
            out.push_back({Role::Assistant, t.assistant_text, t.tool_call_text});
        }
        if (!t.observation.empty() || !t.observed_images.empty()) {
            model::Message m{Role::ToolResponse, t.observation};
            // The pictures a tool returned ride on its response. `tokens` is left at 0
            // here on purpose: only the caller that can preprocess the pixels knows how
            // many pads to reserve, and rendering with 0 throws rather than silently
            // reserving an empty run. See Agent::build_prompt.
            for (const std::string& p : t.observed_images) {
                m.images.push_back({0, p});
            }
            out.push_back(std::move(m));
        }
    }

    // Pinned live state, LAST so that mutating it costs one message of re-prefill
    // rather than the whole context.
    const std::string live = render_live_state();
    if (!live.empty()) {
        out.push_back({Role::User, live});
    }
    return out;
}

std::string ContextStore::render_live_state() const {
    std::string s;
    // The latest follow-up, verbatim and pinned.
    //
    // It is already in the recent stream at the position it arrived, and that copy is
    // what gives it chronology. This one exists because compaction will eventually
    // summarize that copy, and the thing a run must not lose to a trim is the sentence
    // telling it what to do differently. Precedence is chronological in the stream; this
    // pin only keeps the latest ask from vanishing into a summary.
    if (user_turns_.size() > 1) {
        s += "# Latest user message\n\n" + user_turns_.back() + "\n\n";
    }
    if (!checklist_.empty()) {
        s += "# Checklist\n\n";
        for (const ChecklistItem& c : checklist_) {
            s += (c.done ? "- [x] " : "- [ ] ") + c.text + "\n";
        }
    }
    if (!deliverables_.empty()) {
        s += "\n# Deliverables produced so far\n\n";
        for (const std::string& d : deliverables_) {
            s += "- " + d + "\n";
        }
    }
    // The operator check's CURRENT state, one line. Its full output already reached the
    // model as an observation at the moment each reading was taken; this line is the
    // pinned answer to "where does the check stand right now", which is the one fact a
    // long run must not lose to a trim. One line, stated once: repetition is not
    // emphasis to a model reading its own context.
    if (last_check_.has_value()) {
        const CheckResult& v = *last_check_;
        s += "\n# Operator check\n\n- ";
        s += !v.ran ? "COULD NOT RUN " : v.passed ? "PASS " : "FAIL ";
        s += v.command + "\n";
    }
    return s;
}

} // namespace lmp::context
