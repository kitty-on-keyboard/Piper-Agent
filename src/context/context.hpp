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
struct TurnRecord {
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
};

class ContextStore {
  public:
    // The mission is immutable and set once. It is the only text in the whole prompt
    // that may name the deliverable (S8.2 T0).
    explicit ContextStore(std::string mission) : mission_(std::move(mission)) {}

    // --- T1 pinned ---------------------------------------------------------
    void set_checklist(std::vector<ChecklistItem> items) { checklist_ = std::move(items); }
    [[nodiscard]] const std::vector<ChecklistItem>& checklist() const noexcept {
        return checklist_;
    }
    void record_verification(VerificationRecord v) { verifications_.push_back(std::move(v)); }
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

    // --- T2 recent ----------------------------------------------------------
    // The ONLY door for run facts. Everything it stores was observed through a tool
    // result in this run.
    void add_turn(TurnRecord t) { recent_.push_back(std::move(t)); }
    [[nodiscard]] const std::vector<TurnRecord>& recent() const noexcept { return recent_; }

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

    [[nodiscard]] const std::string& mission() const noexcept { return mission_; }
    [[nodiscard]] const std::vector<std::string>& compacted_spans() const noexcept {
        return spans_;
    }

    // The mutable block, rendered LAST by render(). Exposed so a caller can size it.
    [[nodiscard]] std::string render_live_state() const;

  private:
    const std::string mission_;
    std::string project_instructions_;
    std::vector<ChecklistItem> checklist_;
    std::vector<VerificationRecord> verifications_;
    std::vector<std::string> deliverables_;
    std::vector<TurnRecord> recent_;
    std::vector<std::string> spans_;
    std::size_t compactions_ = 0;
};

} // namespace lmp::context
