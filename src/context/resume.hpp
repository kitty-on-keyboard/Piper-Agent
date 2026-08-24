#pragma once
//
// Crash-safe resume from the event log (P2 §11).
//
// Reconstructs a ContextStore from recorded observations WITHOUT re-executing tools.
// PCC remains cross-session recall; this path is turn-accurate replay of the same run's
// event stream. Auto-resume is refused when an edit was in flight or when the bound
// workspace / model / protocol / tool-schema identity no longer matches.
//
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "src/context/context.hpp"
#include "src/platform/event_log.hpp"

namespace lmp::context {

// Identity bound into a `run_begin` event and checked before auto-resume.
struct ResumeIdentity {
    std::string workspace_root;
    std::string run_id;
    std::string model_identity;       // model_dir or equivalent checkpoint id
    std::string protocol_version;
    std::string tool_schema_hash;     // sha256 of tools_json (or empty if unknown)
    std::string last_acked_edit_id;   // empty when no edit has completed
};

struct ResumeGate {
    bool allowed = false;
    std::string why;
};

// True only when every bound field matches and no edit is outstanding.
[[nodiscard]] ResumeGate can_auto_resume(const ResumeIdentity& checkpoint,
                                         const ResumeIdentity& current,
                                         bool edit_in_flight);

struct ResumeRebuild {
    bool ok = false;
    std::string why;
    ResumeIdentity identity;
    bool edit_in_flight = false;
    // Populated only when ok. Mission text comes from run_begin (required).
    ContextStore store{""};
};

// Walks `events` and rebuilds ContextStore from tool_result / steer / write / checklist
// observations. Never calls a tool. Requires a `run_begin` event carrying mission +
// identity fields; when `run_id_filter` is non-empty, only that run's events apply.
[[nodiscard]] ResumeRebuild reconstruct_context(const std::vector<platform::Event>& events,
                                                std::string_view run_id_filter = {});

// Convenience: read path then reconstruct. File-open failure sets ok=false.
[[nodiscard]] ResumeRebuild reconstruct_context_from_log(const std::string& path,
                                                         std::string_view run_id_filter = {});

// ONE RUN, as a chooser needs to see it -- enough to decide whether to resume, and
// nothing that would require replaying the run to compute.
struct RunSummary {
    std::string run_id;
    std::string mission;
    std::string workspace_root;
    std::string model_identity;
    std::uint64_t started_wall_ns = 0;
    // A `run_end` was seen for this run. Its ABSENCE is the interesting case: it means the
    // process died mid-run, which is exactly what resume exists for.
    bool finished = false;
    std::string termination_reason;
    int iterations = 0;
    bool completed = false;
    // Observations recoverable from the log, i.e. how much conversation a resume gets
    // back. Zero means there is nothing to resume TO, however healthy the run looks.
    int observations = 0;
};

// Every run the log knows about, oldest first. Derived from the SAME event stream
// reconstruct_context replays, so a run that appears here can always be rebuilt -- a
// separate index could drift from the log and offer runs that no longer exist.
[[nodiscard]] std::vector<RunSummary> list_runs(const std::vector<platform::Event>& events);

[[nodiscard]] std::vector<RunSummary> list_runs_from_log(const std::string& path);

// Field helpers shared with the sidecar emitter.
[[nodiscard]] std::string field_or(const platform::Event& ev, std::string_view key);
void set_field(std::vector<platform::EventField>& fields, std::string key, std::string value);

} // namespace lmp::context
