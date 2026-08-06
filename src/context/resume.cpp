#include "src/context/resume.hpp"

#include <algorithm>

namespace lmp::context {
namespace {

[[nodiscard]] bool status_is_error(std::string_view status) {
    return status != "ok" && !status.empty();
}

[[nodiscard]] ResumeIdentity identity_from_run_begin(const platform::Event& ev) {
    ResumeIdentity id;
    id.workspace_root = field_or(ev, "workspace_root");
    id.run_id = field_or(ev, "run_id");
    id.model_identity = field_or(ev, "model_identity");
    id.protocol_version = field_or(ev, "protocol_version");
    id.tool_schema_hash = field_or(ev, "tool_schema_hash");
    id.last_acked_edit_id = field_or(ev, "last_acked_edit");
    return id;
}

[[nodiscard]] bool event_matches_run(const platform::Event& ev, std::string_view run_id) {
    if (run_id.empty()) {
        return true;
    }
    const std::string rid = field_or(ev, "run_id");
    return rid.empty() || rid == run_id;
}

} // namespace

std::string field_or(const platform::Event& ev, std::string_view key) {
    for (const platform::EventField& f : ev.fields) {
        if (f.key == key) {
            return f.value;
        }
    }
    return {};
}

void set_field(std::vector<platform::EventField>& fields, std::string key, std::string value) {
    for (platform::EventField& f : fields) {
        if (f.key == key) {
            f.value = std::move(value);
            return;
        }
    }
    fields.push_back({std::move(key), std::move(value)});
}

ResumeGate can_auto_resume(const ResumeIdentity& checkpoint, const ResumeIdentity& current,
                           bool edit_in_flight) {
    if (edit_in_flight) {
        return {false, "edit was in flight; refuse auto-resume"};
    }
    if (checkpoint.workspace_root.empty() || current.workspace_root.empty() ||
        checkpoint.workspace_root != current.workspace_root) {
        return {false, "workspace identity changed or missing"};
    }
    if (checkpoint.run_id.empty() || checkpoint.run_id != current.run_id) {
        return {false, "run/session id mismatch"};
    }
    if (checkpoint.model_identity.empty() ||
        checkpoint.model_identity != current.model_identity) {
        return {false, "model identity mismatch"};
    }
    if (checkpoint.protocol_version.empty() ||
        checkpoint.protocol_version != current.protocol_version) {
        return {false, "protocol version mismatch"};
    }
    if (checkpoint.tool_schema_hash != current.tool_schema_hash) {
        return {false, "tool-schema hash mismatch"};
    }
    return {true, "ok"};
}

ResumeRebuild reconstruct_context(const std::vector<platform::Event>& events,
                                  std::string_view run_id_filter) {
    ResumeRebuild out;
    const platform::Event* begin = nullptr;
    for (const platform::Event& ev : events) {
        if (ev.kind != "run_begin") {
            continue;
        }
        if (!event_matches_run(ev, run_id_filter)) {
            continue;
        }
        begin = &ev;
        if (!run_id_filter.empty()) {
            break;
        }
        // Without a filter, the latest run_begin wins.
    }
    if (begin == nullptr) {
        out.why = "no run_begin event";
        return out;
    }
    out.identity = identity_from_run_begin(*begin);
    const std::string mission = field_or(*begin, "mission");
    if (mission.empty() || out.identity.run_id.empty()) {
        out.why = "run_begin missing mission or run_id";
        return out;
    }
    const std::string& run_id = out.identity.run_id;

    // Rebuild into a fresh store. ContextStore requires a non-empty mission at construct.
    out.store = ContextStore(mission);
    out.store.set_workspace_root(out.identity.workspace_root);

    std::string open_edit;
    std::string last_assistant;
    for (const platform::Event& ev : events) {
        if (ev.seq < begin->seq) {
            continue;
        }
        if (!event_matches_run(ev, run_id)) {
            continue;
        }
        if (ev.kind == "edit_begin") {
            open_edit = field_or(ev, "request_id");
            continue;
        }
        if (ev.kind == "edit_end") {
            const std::string rid = field_or(ev, "request_id");
            if (!rid.empty()) {
                out.identity.last_acked_edit_id = rid;
            }
            if (rid == open_edit) {
                open_edit.clear();
            }
            continue;
        }
        if (ev.kind == "steer") {
            // Steer payload is length-only in the live emitter; text arrives via
            // ContextStore add_user_message. When a `text` field is present (tests /
            // richer traces), restore it; otherwise skip rather than invent content.
            const std::string text = field_or(ev, "text");
            if (!text.empty()) {
                out.store.add_user_message(text);
            }
            continue;
        }
        if (ev.kind == "turn_text") {
            last_assistant = field_or(ev, "text");
            continue;
        }
        if (ev.kind == "write") {
            const std::string path = field_or(ev, "path");
            if (!path.empty()) {
                out.store.record_deliverable(path);
            }
            continue;
        }
        if (ev.kind == "tool_result") {
            TurnRecord rec;
            rec.assistant_text = last_assistant;
            last_assistant.clear();
            rec.tool_name = field_or(ev, "tool");
            rec.observation = field_or(ev, "summary");
            rec.observation_is_error = status_is_error(field_or(ev, "status"));
            rec.first_event_seq = ev.seq;
            rec.last_event_seq = ev.seq;
            if (rec.observation.empty() && !rec.tool_name.empty()) {
                rec.observation = "(" + rec.tool_name +
                                  (rec.observation_is_error ? " failed, with no detail)"
                                                           : " succeeded and produced no output)");
            }
            out.store.add_turn(std::move(rec));
            continue;
        }
        if (ev.kind == "run_end") {
            // Soft boundary; keep scanning for a trailing edit_end on the same run.
            continue;
        }
    }
    out.edit_in_flight = !open_edit.empty();
    out.ok = true;
    out.why = out.edit_in_flight ? "rebuilt; edit in flight" : "ok";
    return out;
}

ResumeRebuild reconstruct_context_from_log(const std::string& path,
                                           std::string_view run_id_filter) {
    std::vector<platform::Event> events;
    std::size_t skipped = 0;
    std::string error;
    if (!platform::read_event_log(path, events, skipped, error)) {
        ResumeRebuild out;
        out.why = std::move(error);
        return out;
    }
    (void)skipped;
    return reconstruct_context(events, run_id_filter);
}

} // namespace lmp::context
