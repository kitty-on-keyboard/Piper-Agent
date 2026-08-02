#pragma once
//
// Context store -- event-sourced, tiered, compacting (spec S8).
//
// "~55% of measured agent failure mass is context, and it fails silently -- the run does
// not error, it just gets worse." Built on day one here; v1 deferred it for the whole
// project.
//
// TIERS
//   T0  Mission. Immutable, always present, THE ONLY THING THAT NAMES THE DELIVERABLE.
//   T1  Pinned state: checklist, deliverable ledger, verification ledger, artifacts.
//   T2  Recent turns, verbatim.
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
#include <string>
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
    std::string assistant_text;   // the answer body; reasoning is NOT carried forward
    std::string tool_name;        // empty when the turn was text-only
    std::string tool_args_summary;
    std::string observation;      // the tool result summary -- an OBSERVED fact
    bool observation_is_error = false;
    std::uint64_t first_event_seq = 0; // provenance into the event log
    std::uint64_t last_event_seq = 0;
};

struct VerificationRecord {
    std::string contract;   // the canonical check, e.g. "cmake --build build"
    bool passed = false;
    bool falsifiable = false; // this exact check has been PROVEN capable of red
    // Whether the command actually EXECUTED. A refusal is not evidence in either
    // direction (S6.2), so it must never be mistaken for the red that proves a check
    // capable of failing.
    bool ran = false;
    std::string detail;
    // Where this reading sits in the conversation. Evidence gathered BEFORE the user's
    // latest instruction cannot discharge that instruction, and without a position on
    // the timeline there is no way to tell the two apart: a follow-up would complete
    // instantly on the previous run's green.
    std::size_t seq = 0;
};

class ContextStore {
  public:
    // The opening mission. Still the only text in the STABLE prompt block that may name
    // the deliverable (S8.2 T0), and still fixed once set -- but it is now the FIRST user
    // turn rather than the only one. An agent you can only launch is a batch job; the
    // store has to outlive one mission for a follow-up to continue a conversation
    // instead of restarting it.
    explicit ContextStore(std::string mission) { user_turns_.push_back(std::move(mission)); }

    // --- T1 pinned ---------------------------------------------------------
    void set_checklist(std::vector<ChecklistItem> items) {
        checklist_ = std::move(items);
        plan_stale_ = false;
    }
    [[nodiscard]] const std::vector<ChecklistItem>& checklist() const noexcept {
        return checklist_;
    }
    [[nodiscard]] std::size_t open_checklist_items() const noexcept {
        return static_cast<std::size_t>(std::count_if(
            checklist_.begin(), checklist_.end(), [](const ChecklistItem& c) { return !c.done; }));
    }

    // WHO CHOSE THE PROOF (S10.4).
    //
    // The checklist stopped gating completion because a tick is a self-report. The verify
    // contract is the same class of thing one level down, and it was never noticed: the
    // model picks the command that counts as proof, and the harness then rigorously
    // verifies whatever the model picked. `rename_across_files` ended
    // completed=yes verified=yes solved=NO by declaring `pytest -q`, making it pass, and
    // stopping -- while the mission also required no residual `calc_total`. Every gate
    // worked. The contract was weaker than the mission and nothing said so.
    //
    // This cannot be fixed by inspecting the contract: whether a command covers a mission
    // is not decidable from either string. What CAN be fixed is the pretence that the two
    // kinds of contract are the same. An operator contract is evidence about the mission;
    // a model contract is evidence about what the model decided to check.
    enum class ContractSource : std::uint8_t { Model, Operator };

    // Operator contracts WIN and are permanent: `plan` may not replace one. Otherwise a
    // run could talk its way out of the criterion it was given by restating the plan.
    void set_verify_contract(std::string c, ContractSource source = ContractSource::Model) {
        if (contract_source_ == ContractSource::Operator && source != ContractSource::Operator) {
            return;
        }
        verify_contract_ = std::move(c);
        contract_source_ = source;
    }
    [[nodiscard]] const std::string& verify_contract() const noexcept {
        return verify_contract_;
    }
    [[nodiscard]] ContractSource verify_contract_source() const noexcept {
        return contract_source_;
    }

    // True when an instruction has landed that the current checklist predates. Cleared
    // only by restating the checklist. The loop turns this into a MECHANISM -- `plan`
    // becomes the sole callable tool -- so a steering message cannot be received and
    // then quietly ignored (S9.2).
    [[nodiscard]] bool plan_is_stale() const noexcept { return plan_stale_; }

    void record_verification(VerificationRecord v) {
        // Takes its OWN position on the timeline rather than borrowing the current
        // turn's. A verification runs mid-turn, before that turn is recorded, so
        // reusing the turn counter would stamp evidence with the position of the last
        // COMPLETED turn -- and a check run immediately after an instruction would tie
        // with it instead of postdating it.
        v.seq = ++seq_;
        verifications_.push_back(std::move(v));
    }
    [[nodiscard]] const std::vector<VerificationRecord>& verifications() const noexcept {
        return verifications_;
    }
    // Deduplicated: editing one file four times produces one deliverable, not four.
    void record_deliverable(std::string path) {
        if (std::find(deliverables_.begin(), deliverables_.end(), path) ==
            deliverables_.end()) {
            deliverables_.push_back(std::move(path));
        }
    }
    [[nodiscard]] const std::vector<std::string>& deliverables() const noexcept {
        return deliverables_;
    }

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

    // --- T2 recent ----------------------------------------------------------
    // The most an observation may carry into the prompt. The tool layer is supposed to
    // bound every result before it gets here -- tool_result.hpp says so -- but only the
    // shell path actually did, and read_file handed over whole files. This is the door
    // that makes the next tool which forgets fail loudly instead of silently.
    void set_observation_budget(std::size_t n) noexcept { observation_budget_ = n; }

    // The ONLY door for run facts. Everything it stores was observed through a tool
    // result in this run.
    void add_turn(TurnRecord t) {
        // Assert in tests (assertions are on in EVERY configuration here), clamp in a
        // real run: a run that has already produced an over-budget observation is better
        // served by a truncated one than by an abort.
        assert(t.observation.size() <= observation_budget_ &&
               "observation exceeds the prompt budget; the tool layer must bound it");
        if (t.observation.size() > observation_budget_) {
            t.observation.resize(observation_budget_);
            t.observation += "\n[truncated at the observation budget]\n";
        }
        ++seq_;
        recent_.push_back(std::move(t));
    }
    [[nodiscard]] const std::vector<TurnRecord>& recent() const noexcept { return recent_; }

    // What the human said, mid-run or between runs. Not prompt IMPURITY: a user
    // instruction is an observed fact about this session, in the same sense a tool
    // result is (S8.4). What stays forbidden is text nobody said -- an inferred
    // workspace description or a guessed deliverable name.
    //
    // Two things follow from an instruction landing, and both are mechanisms:
    // the plan goes stale, and the position is remembered so evidence gathered
    // beforehand cannot be mistaken for evidence that discharges it.
    void add_user_message(std::string text) {
        user_turns_.push_back(text);
        TurnRecord t;
        t.user_text = std::move(text);
        add_turn(std::move(t));
        last_directive_seq_ = seq_;
        plan_stale_ = true;
    }
    [[nodiscard]] const std::vector<std::string>& user_turns() const noexcept {
        return user_turns_;
    }
    [[nodiscard]] std::size_t last_directive_seq() const noexcept {
        return last_directive_seq_;
    }

    // --- T3 compacted -------------------------------------------------------
    [[nodiscard]] std::size_t compaction_count() const noexcept { return compactions_; }

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
    std::string project_instructions_;
    std::string project_memory_;
    std::string verify_contract_;
    ContractSource contract_source_ = ContractSource::Model;
    std::vector<ChecklistItem> checklist_;
    std::vector<VerificationRecord> verifications_;
    std::vector<std::string> deliverables_;
    std::vector<TurnRecord> recent_;
    std::vector<std::string> spans_;
    std::size_t compactions_ = 0;
    // Monotonic position in the conversation. Never reset -- compaction moves turns into
    // spans but must not renumber the timeline, or evidence would appear to move.
    std::size_t seq_ = 0;
    std::size_t last_directive_seq_ = 0;
    // Generous by default so a caller that never sets it keeps today's behaviour; the
    // sidecar sets the real one from the workspace's budgets.
    std::size_t observation_budget_ = 1U << 20;
    bool plan_stale_ = false;
};

} // namespace lmp::context
