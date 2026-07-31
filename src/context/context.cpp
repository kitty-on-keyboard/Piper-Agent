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
    "- You test whenever it is possible and safe to do so, and you run the test rather\n"
    "  than assert that it would pass.\n"
    "- You are to the point. You do not restate the task, narrate what you are about to\n"
    "  do, or explain work that speaks for itself.\n";

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

// ORDERING IS LOAD-BEARING (S5.10). The prompt is laid out most-stable-first so that the
// KV prefix survives a turn:
//
//   [system: guidance + project conventions + mission]  never changes within a run
//   [compacted spans]                                   changes only when compaction runs
//   [recent turns]                                      append-only between compactions
//   [live state: checklist, deliverables, ledger]       changes constantly -- so it goes LAST
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

    // T0: persona, guidance, the project's own conventions, and the mission -- the only
    // place the deliverable is named. Fixed for the lifetime of the run, and FIRST,
    // because everything ahead of a change is what stays cached.
    std::string system;
    system += persona_.empty() ? kPersona : persona_.c_str();
    system += "\n\n";
    system += std::string(tool_guidance);
    if (!project_instructions_.empty()) {
        system += "\n\n# Project conventions\n\n" + project_instructions_;
    }
    system += "\n\n# Mission\n\n" + user_turns_.front();
    out.push_back({Role::System, std::move(system)});

    // T3: compacted spans, oldest first, as observed history.
    for (const std::string& span : spans_) {
        out.push_back({Role::User, span});
    }

    // T2: recent turns, verbatim. The assistant's answer body and the observation it
    // got back -- the observation goes in as a tool_response, which is Qwen's shape.
    for (const TurnRecord& t : recent_) {
        if (!t.user_text.empty()) {
            out.push_back({Role::User, t.user_text});
            continue;
        }
        if (!t.assistant_text.empty()) {
            out.push_back({Role::Assistant, t.assistant_text});
        }
        if (!t.observation.empty()) {
            out.push_back({Role::ToolResponse, t.observation});
        }
    }

    // T1: pinned live state, LAST so that mutating it costs one message of re-prefill
    // rather than the whole context.
    const std::string live = render_live_state();
    if (!live.empty()) {
        out.push_back({Role::User, live});
    }
    return out;
}

std::string ContextStore::render_live_state() const {
    std::string s;
    // The standing instruction, verbatim and pinned.
    //
    // It is already in the recent stream at the position it arrived, and that copy is
    // what gives it chronology. This one exists because compaction will eventually
    // summarize that copy, and the thing a run must not lose to a trim is the sentence
    // telling it what to do differently. Duplicated deliberately: T0 names the mission,
    // and until this existed there was nowhere for "and now do it the other way" to live
    // that a long run could not forget.
    if (user_turns_.size() > 1) {
        s += "# Standing instruction (most recent; supersedes the mission where they "
             "conflict)\n\n" +
             user_turns_.back() + "\n";
        if (plan_stale_) {
            s += "\nThe checklist below predates it.\n";
        }
        s += "\n";
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
    if (!verifications_.empty()) {
        s += "\n# Verification ledger\n\n";
        for (const VerificationRecord& v : verifications_) {
            s += std::string(v.passed ? "- PASS " : "- FAIL ") + v.contract;
            // A green that has not been proven capable of red is reported as unproven,
            // to the model as well as the UI (S10.2).
            s += v.passed && !v.falsifiable ? "  (UNPROVEN: this check has not been"
                                              " shown capable of failing)\n"
                                            : "\n";
        }
    }
    return s;
}

} // namespace lmp::context
