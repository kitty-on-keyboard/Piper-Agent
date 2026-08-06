#pragma once
//
// Crash-safe resume from the event log (P2 §11).
//
// Reconstructs a ContextStore from recorded observations WITHOUT re-executing tools.
// PCC remains cross-session recall; this path is turn-accurate replay of the same run's
// event stream. Auto-resume is refused when an edit was in flight or when the bound
// workspace / model / protocol / tool-schema identity no longer matches.
//
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

// Field helpers shared with the sidecar emitter.
[[nodiscard]] std::string field_or(const platform::Event& ev, std::string_view key);
void set_field(std::vector<platform::EventField>& fields, std::string key, std::string value);

} // namespace lmp::context
