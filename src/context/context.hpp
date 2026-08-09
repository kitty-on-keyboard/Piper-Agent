#pragma once
//
// Context store -- event-sourced, tiered, compacting (spec S8).
//
// "~55% of measured agent failure mass is context, and it fails silently -- the run does
// not error, it just gets worse." Built on day one here; v1 deferred it for the whole
// project.
//
// TIERS
//   T0  Opening mission as the first stable user message (not system identity).
//   T1  Pinned state: checklist, deliverable ledger, the operator check's last reading.
//   T2  Recent turns, verbatim (assistant text / working note + observations).
//   T3  Compacted spans: a summary plus a provenance pointer to the full event range.
//
// COMPACTION, NOT EVICTION (S8.3). On trim the dropped span is SUMMARIZED, never
// announced as dropped. Since a trim pays a full re-prefill anyway, it spends that cost
// on a summary rather than a notice.
//
// PROMPT PURITY (S8.4). Only facts observed through a tool result in THIS run reach the
// prompt. No inferred workspace description, no guessed deliverable names. This was v1's
// root bug class, and add_observation() is the only door.
//
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "src/model/chat_template.hpp"

namespace lmp::context {

using model::Message;
using model::Role;
using model::TokenId;

struct ChecklistItem {
    std::string text;
    bool done = false;
};

// One turn's worth of history: what the model said, and what it observed back.
//
// A record carrying only `user_text` is something the HUMAN said -- the opening mission
// or a later instruction. It sits in the same ordered stream as everything else because
// when an instruction arrived relative to what the model had already seen is the whole
// meaning of "no, use the other approach".
struct TurnRecord {
    std::string user_text;        // set only on a human turn; the rest are then empty
    // Answer body, or a capped working note backfilled from think on a tool turn when
    // the model left the answer channel empty. Full think stays on the thinking stream
    // (S5.7); only a short trailing slice may enter the next prompt for continuity.
    std::string assistant_text;
    std::string tool_name;        // empty when the turn was text-only
    std::string tool_args_summary;
    std::string observation;      // the tool result summary -- an OBSERVED fact
    bool observation_is_error = false;
    std::uint64_t first_event_seq = 0; // provenance into the event log
    std::uint64_t last_event_seq = 0;
};

// One reading of the OPERATOR's check command -- the only verification in the harness.
// A report of what a command did, never a verdict about the work: `ran` says whether it
// executed at all (a refusal is not evidence either way), `passed` reports its exit
// status, and `detail` is its output, which is the part the model actually acts on.
struct CheckResult {
    std::string command;
    bool ran = false;
    bool passed = false;
    std::string detail;
};

class ContextStore {
  public:
    // The opening mission. Rendered as the first stable user message (run-constant for
    // the KV prefix), not as system identity. Fixed once set; later follow-ups append.
    // An agent you can only launch is a batch job; the store has to outlive one mission
    // for a follow-up to continue a conversation instead of restarting it.
    explicit ContextStore(std::string mission) { user_turns_.push_back(std::move(mission)); }

    // --- T1 pinned ---------------------------------------------------------
    //
    // The checklist is the model's own progress display: it feeds the sidebar panel and
    // the run report's unfinished_items, and nothing reads it to decide anything. The
    // eighth pass removed its gate along with the verification ledger's -- a tick is a
    // self-report, and a harness that enforces self-reports is adjudicating.
    void set_checklist(std::vector<ChecklistItem> items) {
        checklist_ = std::move(items);
    }
    [[nodiscard]] const std::vector<ChecklistItem>& checklist() const noexcept {
        return checklist_;
    }
    [[nodiscard]] std::size_t open_checklist_items() const noexcept {
        return static_cast<std::size_t>(std::count_if(
            checklist_.begin(), checklist_.end(), [](const ChecklistItem& c) { return !c.done; }));
    }

    // The operator check's latest reading. One slot, not a ledger: the live-state block
    // renders the current state of the check, and the model reads each reading's full
    // output as an observation at the moment it happens -- history lives in the turn
    // stream, where history lives for everything else.
    void set_last_check(CheckResult r) { last_check_ = std::move(r); }
    [[nodiscard]] const std::optional<CheckResult>& last_check() const noexcept {
        return last_check_;
    }
    // Deduplicated: editing one file four times produces one deliverable, not four.
    void record_deliverable(std::string path) {
        // The COUNTER is not deduplicated, and that is the point of it being separate.
        // "How many files exist" and "how much work has happened since that check last
        // ran" are different questions, and only the second one can tell an unmoved
        // failure from an unfinished one. A run iterating on a single file writes it
        // eight times and the deliverable list never moves.
        ++workspace_writes_;
        if (std::find(deliverables_.begin(), deliverables_.end(), path) ==
            deliverables_.end()) {
            deliverables_.push_back(std::move(path));
        }
    }
    [[nodiscard]] const std::vector<std::string>& deliverables() const noexcept {
        return deliverables_;
    }
    [[nodiscard]] std::size_t workspace_writes() const noexcept {
        return workspace_writes_;
    }
    // Something outside the native write door may have changed the workspace: an MCP
    // call, an opaque shell, an editor-side edit the harness did not apply. Bumps the
    // same counter the repeat cache keys on, without claiming a deliverable path.
    void invalidate_workspace_freshness() noexcept { ++workspace_writes_; }

    // The repo's own conventions (AGENTS.md / CLAUDE.md / .cursorrules), loaded once and
    // pinned in the STABLE part of the prompt -- they never change within a run.
    void set_project_instructions(std::string text) {
        project_instructions_ = std::move(text);
    }
    [[nodiscard]] const std::string& project_instructions() const noexcept {
        return project_instructions_;
    }

    // Replaces the built-in persona. The editor keeps one of these per mode, so `plan`
    // can be told to think out loud where `agent` is told to be terse. Empty keeps the
    // built-in -- which is why this can be a plain string rather than an optional.
    //
    // It sits at the very front of the STABLE block, so changing it between runs costs a
    // full re-prefill and changing it within one is impossible by construction.
    // Notes the agent wrote in EARLIER sessions, loaded once at session start. Kept
    // separate from project_instructions_ because their trust levels differ: conventions
    // are the operator's, these are the model's own recollection.
    void set_project_memory(std::string text) { project_memory_ = std::move(text); }
    [[nodiscard]] const std::string& project_memory() const noexcept {
        return project_memory_;
    }

    void set_persona(std::string text) { persona_ = std::move(text); }

    // WHAT THIS MODE IS, in the model's own prompt. A separate channel from the persona
    // for one reason: the persona is the operator's to replace and this is not.
    //
    // `prompts.<mode>` in the editor sets the persona, defaults to empty, and REPLACES the
    // built-in wholesale -- so for as long as mode existed, the only channel that could
    // have told the model it was in plan mode was a setting nobody had filled in, and
    // every mode sent the identical text. The model was never told. It was simply given a
    // tool list it could not use and left to discover the refusals one turn at a time.
    //
    // Rendered after the persona and before the tools, because it qualifies the persona --
    // which tells every run to go and run its tests, advice that is unfollowable at T0.
    void set_mode_brief(std::string text) { mode_brief_ = std::move(text); }

    // The absolute path the run is rooted at. Stated in the prompt because the model
    // otherwise has to GUESS it, and a wrong guess is not cheap: every path tool resolves
    // against this root and refuses anything outside it, so the guess costs a refusal per
    // turn until something happens to reveal the truth.
    //
    // MEASURED: a real run opened with `list_dir /home/user` (refused), then
    // `mkdir -p /home/user/src` (denied by the sandbox), and did not learn where it
    // actually was until turn 25, when it thought to run `pwd`. Stable for the run, so it
    // costs nothing after the first prefill.
    void set_workspace_root(std::string path) { workspace_root_ = std::move(path); }

    // --- T2 recent ----------------------------------------------------------
    // The most an observation may carry into the prompt. The tool layer is supposed to
    // bound every result before it gets here -- tool_result.hpp says so -- but only the
    // shell path actually did, and read_file handed over whole files. This is the door
    // that makes the next tool which forgets fail loudly instead of silently.
    void set_observation_budget(std::size_t n) noexcept { observation_budget_ = n; }

    // How many observations this store had to cut down. Non-zero means a tool handed
    // over more than it was allowed to -- a defect in that tool, visible to a test
    // without the process having to die to report it.
    [[nodiscard]] std::size_t truncated_observations() const noexcept {
        return truncated_observations_;
    }

    // The ONLY door for run facts. Everything it stores was observed through a tool
    // result in this run.
    void add_turn(TurnRecord t) {
        // CLAMP. The line below used to be an assert, and the comment above it said
        // "assert in tests, clamp in a real run" -- but assertions are on in EVERY
        // configuration here, including the shipped sidecar, so the abort always won and
        // the clamp was unreachable. That is not a hypothetical: read_many returned
        // 38,768 bytes against this 32 KB budget, the assert fired, and the sidecar died
        // mid-run and took a 19 GB loaded model with it. Three sessions ended that way
        // before the abort was traced, because SIGABRT left no crash report.
        //
        // The tool layer must still bound its own results and now does
        // (tools::kObservationBudgetBytes, one number shared by both sides). This door
        // stays as the backstop for the next tool that forgets, and a backstop that kills
        // the process is not a backstop. The violation is recorded rather than asserted:
        // truncated_observations() is what a test reads to prove a tool went over.
        if (t.observation.size() > observation_budget_) {
            ++truncated_observations_;
            t.observation.resize(observation_budget_);
            t.observation += "\n[truncated at the observation budget]\n";
        }
        recent_.push_back(std::move(t));
        // Durable BEFORE anything can drop it. See set_turn_sink().
        if (turn_sink_) {
            turn_sink_(recent_.back());
        }
    }
    [[nodiscard]] const std::vector<TurnRecord>& recent() const noexcept { return recent_; }

    // Collapses EARLIER records whose observation is byte-identical to `observation` down
    // to `replacement`, and returns how many were collapsed.
    //
    // This is what a re-read costs now: nothing. The read runs, the model gets the bytes it
    // asked for, and the copy it was already holding turns into one line -- so the prompt
    // ends the turn the same size it started, with the newest copy live.
    //
    // BYTE IDENTITY IS THE WHOLE TEST, and it is why this replaced a ledger of paths and
    // invalidation rules. That ledger had to PREDICT whether a file had changed, from a
    // path string, against tools that rename and shell commands that can do anything -- and
    // when it predicted wrong it fed the model stale bytes, or refused a read of a file that
    // had moved. Identical bytes cannot be stale: if the file changed, the new observation
    // differs and nothing collapses. There is no rule to get wrong.
    // Whether these exact bytes are ALREADY in the prompt.
    //
    // Asked before the new record is added, this answers "did that read tell the model
    // anything it was not already holding?" -- which is the one question the no-progress
    // counter could not ask, and the reason a run could spend thirty turns re-reading four
    // files with `no_progress_streak=0` on every single line of the trace.
    //
    // Byte identity, for the same reason supersede_duplicate_observation uses it: a file
    // that changed produces different bytes, so a re-read after a write is never mistaken
    // for a redundant one and there is no staleness rule to get wrong.
    //
    // Stays correct as copies are collapsed, because the collapse always leaves exactly
    // ONE live copy -- the newest. The third read of an unchanged file still finds the
    // second read's bytes sitting in recent_.
    [[nodiscard]] bool has_observation(const std::string& observation) const {
        if (observation.empty()) {
            return false;
        }
        return std::any_of(recent_.begin(), recent_.end(), [&](const TurnRecord& t) {
            return t.observation == observation;
        });
    }

    std::size_t supersede_duplicate_observation(const std::string& observation,
                                                const std::string& replacement) {
        if (observation.empty()) {
            return 0;
        }
        std::size_t collapsed = 0;
        for (TurnRecord& t : recent_) {
            if (t.observation == observation) {
                t.observation = replacement;
                ++collapsed;
            }
        }
        return collapsed;
    }

    // What the human said, mid-run or between runs. Not prompt IMPURITY: a user
    // instruction is an observed fact about this session, in the same sense a tool
    // result is (S8.4). What stays forbidden is text nobody said -- an inferred
    // workspace description or a guessed deliverable name.
    void add_user_message(std::string text) {
        user_turns_.push_back(text);
        TurnRecord t;
        t.user_text = std::move(text);
        add_turn(std::move(t));
    }
    [[nodiscard]] const std::vector<std::string>& user_turns() const noexcept {
        return user_turns_;
    }

    // --- T3 compacted -------------------------------------------------------
    [[nodiscard]] std::size_t compaction_count() const noexcept { return compactions_; }

    // Called with the turns compact_oldest() is about to erase, BEFORE it erases them.
    //
    // Compaction has always been lossy: the span keeps one anchor line per turn and the
    // rest is gone. The summary even prints the event range it was made from -- a
    // provenance pointer with nothing on the other end. This is the other end. The
    // sidecar wires it to src/pcc, and the trim stops being destructive: the prompt keeps
    // the summary, the full text stays one query away.
    //
    // A SINK rather than a store reference, for two reasons. render() stays pure and
    // diffable, which is the property the whole prompt-purity argument rests on; and this
    // layer keeps knowing nothing about databases, so a ContextStore in a unit test needs
    // no fixture. Unset, behaviour is exactly what it was.
    // `summary` is the span this compaction just produced -- the same text render() will
    // put in the prompt. Passed rather than left for the sink to rebuild from `dropped`,
    // because a sink that reformats the turns itself is a second implementation of the
    // summary that can drift from the one the model actually sees.
    using CompactionSink =
        std::function<void(const std::vector<TurnRecord>& dropped, std::size_t span_index,
                           std::string_view summary)>;
    void set_compaction_sink(CompactionSink sink) { compaction_sink_ = std::move(sink); }

    // Called with every turn AS IT IS RECORDED, before anything can drop it.
    //
    // The compaction sink above was for a long time the only door to the durable store,
    // and it opens only when the prompt crosses the compaction threshold. On real work it
    // frequently never does -- a long run adds little per turn -- so a workspace with
    // several complete 80-turn runs through it ended with a context database holding its
    // schema and ZERO rows. The one component whose whole job is to remember what got
    // trimmed had never been handed anything to remember.
    //
    // MEASURED 2026-08-03 on ~/Desktop/Agent_testing/ResMon: `select kind,count(*) from
    // item group by kind` returned nothing, and the event log had no compaction events to
    // explain it. Nothing was broken; the door was simply never reached.
    //
    // So the durable write now happens at the moment the turn exists, and compaction goes
    // back to being what its name says -- a summarizer -- instead of doubling as the only
    // way anything is ever persisted.
    using TurnSink = std::function<void(const TurnRecord& turn)>;
    void set_turn_sink(TurnSink sink) { turn_sink_ = std::move(sink); }

    // Moves the oldest `keep_recent`-excess turns into a summarized span. Returns the
    // number of turns compacted. The summary keeps every observation's ANCHOR (tool
    // name + whether it errored + the first line of what it observed), because the
    // evidence a later question needs is usually the anchor, not the prose around it.
    std::size_t compact_oldest(std::size_t keep_recent);

    // --- rendering ----------------------------------------------------------
    // Assembles the message list. Deterministic, pure, no clock, no I/O -- so a prompt
    // is diffable and a purity violation is a unit test, not a run-time surprise.
    [[nodiscard]] std::vector<Message> render(std::string_view tool_guidance) const;

    // How many of render()'s messages form the STABLE prefix -- everything except the
    // live-state block, which is the one part that changes every turn. This is the
    // boundary the KV checkpoint is taken at (S5.10): turn N's stable prefix is a pure
    // prefix of turn N+1's, so a cache rolled back to here needs only the new turn record
    // and the new live state prefilled, instead of the whole context.
    [[nodiscard]] std::size_t stable_message_count(std::string_view tool_guidance) const {
        const std::size_t n = render(tool_guidance).size();
        return render_live_state().empty() ? n : n - 1;
    }

    [[nodiscard]] const std::string& mission() const noexcept { return user_turns_.front(); }
    [[nodiscard]] const std::vector<std::string>& compacted_spans() const noexcept {
        return spans_;
    }

    // The mutable block, rendered LAST by render(). Exposed so a caller can size it.
    [[nodiscard]] std::string render_live_state() const;

  private:
    // user_turns_[0] is the opening mission and never changes; the rest are instructions
    // that arrived later. front() is therefore always valid.
    std::vector<std::string> user_turns_;
    std::string persona_;
    std::string mode_brief_;
    std::string workspace_root_;
    std::string project_instructions_;
    std::string project_memory_;
    std::vector<ChecklistItem> checklist_;
    std::optional<CheckResult> last_check_;
    std::vector<std::string> deliverables_;
    std::vector<TurnRecord> recent_;
    std::vector<std::string> spans_;
    CompactionSink compaction_sink_;
    TurnSink turn_sink_;
    std::size_t compactions_ = 0;
    // Every successful write, not every distinct path. See record_deliverable().
    std::size_t workspace_writes_ = 0;
    // Generous by default so a caller that never sets it keeps today's behaviour; the
    // sidecar sets the real one from the workspace's budgets.
    std::size_t observation_budget_ = 1U << 20;
    std::size_t truncated_observations_ = 0;
};

} // namespace lmp::context
