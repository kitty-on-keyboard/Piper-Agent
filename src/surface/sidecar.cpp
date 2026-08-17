// The sidecar: one process, one model, the loop (spec S4.1, S12.3).
//
// The extension spawns exactly one of these. It owns the model and the loop, speaks
// lmp/* over stdio, and exits on stdin EOF so a dead parent never orphans a process
// holding ~19 GB of weights.

#include <execinfo.h>
#include <unistd.h>

#include <algorithm>
#include <csignal>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/context/resume.hpp"
#include "src/loop/agent.hpp"
#include "src/model/family_traits.hpp"
#include "src/model/mlx_backend.hpp"
#include "src/model/model_limits.hpp"
// The recall tools' token counter is built here, so this file needs the estimator it
// falls back to when no model is loaded.
#include "src/pcc/recall.hpp"
#include "src/platform/event_log.hpp"
#include "src/platform/fs.hpp"
#include "src/surface/protocol_generated.hpp"
#include "src/surface/mcp_settings.hpp"
#include "src/surface/session.hpp"
#include "src/surface/transport.hpp"
#include "src/surface/wire.hpp"

namespace {

using namespace lmp;
using surface::wire::json_escape;
using surface::wire::notify;
using surface::wire::reply_error;
using surface::wire::reply_result;

// Leaves WITHOUT joining the reader thread.
//
// That thread is parked in read() on stdin and only EOF releases it, so a client that
// asks for shutdown and keeps the pipe open would hang forever in reader.join() -- alive,
// idle, and holding ~19 GB of weights. That is the precise outcome the "exit on EOF" rule
// exists to prevent (S12.3), reached through the front door. Observed: the first
// end-to-end run replied {"ok":true} to lmp/shutdown and then sat at 35% of system memory
// until its parent was killed.
//
// The model is already unloaded by the time anything calls this -- the backend dies with
// the run -- so the only state left worth caring about is the log.
[[noreturn]] void exit_now(platform::EventLogWriter& log) {
    log.close();
    std::fflush(nullptr);
    ::_exit(0);
}

// RiskHint -> the wire's capability chips. The parse status travels ALONGSIDE the flags
// rather than being folded into the risk number: "the full effect depends on bytes that
// are not in this string" is the single most useful thing the card can tell a human, and
// a scalar cannot say it.
protocol::CapabilityChips chips_of(const tools::RiskHint& hint) {
    protocol::CapabilityChips c;
    c.writes_outside_workspace = hint.caps.writes_outside_workspace;
    c.reads_outside_workspace = hint.caps.reads_outside_workspace;
    c.destroys_data = hint.caps.destroys_data;
    c.rewrites_vcs_history = hint.caps.rewrites_vcs_history;
    c.network_access = hint.caps.network_access;
    c.spawns_unbounded_process = hint.caps.spawns_unbounded_process;
    c.signals_foreign_process = hint.caps.signals_foreign_process;
    c.escalates_privileges = hint.caps.escalates_privileges;
    switch (hint.status) {
        case blast_radius::ParseStatus::Parsed:
            c.parse_status = "parsed";
            break;
        case blast_radius::ParseStatus::PartiallyParsed:
            c.parse_status = "partially_parsed";
            break;
        case blast_radius::ParseStatus::Unparseable:
            c.parse_status = "unparseable";
            break;
    }
    return c;
}

// The one run_end a run gets when it fails before the loop ever turns.
void end_run(const std::string& id, const std::string& reason) {
    protocol::RunEndNotification end;
    end.run_id = id;
    end.termination_reason = reason;
    end.iterations = 0;
    end.completed = false;
    end.unfinished_items = 0;
    notify(end);
}

// EVERY parameter the call was made with, as a JSON object.
//
// This used to send `tool_params[0].value` -- the first parameter's raw text and nothing
// else. For the tools whose whole point is a second parameter that was a hole with a UI
// on the other side of it: `ask_question` carries `question` AND `options`, the view
// JSON.parse()s this field to find them, and what arrived was bare text that threw. The
// options never reached the card, so the card that renders selectable items could not
// render them for any call, however well formed.
//
// A JSON object rather than a positional list because the view asks for parameters by
// NAME, and the order the model happened to emit them in is not a contract.
[[nodiscard]] std::string tool_args_json(const std::vector<tools::ToolParamValue>& params) {
    if (params.empty()) {
        return {};
    }
    std::string out = "{";
    for (std::size_t i = 0; i < params.size(); ++i) {
        if (i > 0) {
            out += ',';
        }
        // json_escape() RETURNS THE QUOTES. Adding another pair here produces ""name"" --
        // invalid JSON, which the view's JSON.parse rejects, which lands it right back on
        // the "no named arguments" path this function exists to get it off.
        out += json_escape(params[i].name);
        out += ':';
        out += json_escape(params[i].value);
    }
    out += '}';
    return out;
}

// The UI feed for one run. Every notification is serialized by the GENERATED code for
// its type, so a schema change the sidecar has not caught up with is a compile error
// rather than a field the extension silently reads as its default (S4.4).
loop::Observer make_observer(const std::string& id) {
    loop::Observer obs;
    obs.on_token = [id](const std::string& channel, const std::string& text) {
        protocol::TokenNotification n;
        n.run_id = id;
        n.channel = channel;
        n.text = text;
        notify(n);
    };
    obs.on_turn = [id](const loop::TurnResult& t, double duration_ms) {
        protocol::TurnNotification n;
        n.run_id = id;
        n.outcome = std::string(loop::to_string(t.outcome));
        n.tool_name = t.tool_name;
        n.tool_args = tool_args_json(t.tool_params);
        n.tool_status = std::string(tools::to_string(t.tool_result.status));
        n.summary = t.tool_result.summary;
        n.duration_ms = duration_ms;
        n.think_tokens = static_cast<std::int64_t>(t.think_tokens);
        n.text_tokens = static_cast<std::int64_t>(t.text_tokens);
        n.tool_tokens = static_cast<std::int64_t>(t.tool_tokens);
        n.batch_index = static_cast<std::int64_t>(t.batch_index);
        n.batch_count = static_cast<std::int64_t>(t.batch_count);
        n.read_bytes = static_cast<std::int64_t>(t.tool_result.bytes_read);
        n.edit_bytes = static_cast<std::int64_t>(t.tool_result.bytes_changed);
        n.cap_phase = t.cap_phase;
        notify(n);
    };
    obs.on_verification = [id](const context::CheckResult& v) {
        protocol::VerificationNotification n;
        n.run_id = id;
        n.contract = v.command;
        n.ran = v.ran;
        n.passed = v.passed;
        n.detail = v.detail;
        notify(n);
    };
    obs.on_checklist = [id](const std::vector<context::ChecklistItem>& items) {
        protocol::ChecklistNotification n;
        n.run_id = id;
        n.items_json = "[";
        for (std::size_t i = 0; i < items.size(); ++i) {
            n.items_json += i == 0 ? "" : ",";
            n.items_json += R"({"text":)" + json_escape(items[i].text) + R"(,"done":)" +
                            (items[i].done ? "true" : "false") + "}";
        }
        n.items_json += "]";
        notify(n);
    };
    obs.on_plan_ready = [id](const std::string& plan) {
        protocol::PlanReadyNotification n;
        n.run_id = id;
        n.plan = plan;
        notify(n);
    };
    obs.on_perf = [id](const model::GenResult& g, std::size_t used, std::size_t max,
                       std::size_t compactions) {
        protocol::PerfNotification n;
        n.run_id = id;
        n.sample.ttft_ms = g.ttft_ms;
        n.sample.prefill_tok_per_s = g.prefill_tok_per_s;
        n.sample.decode_tok_per_s = g.decode_tok_per_s;
        n.sample.context_used = static_cast<std::int64_t>(used);
        n.sample.context_max = static_cast<std::int64_t>(max);
        n.sample.tokens_generated = g.tokens_generated;
        n.sample.prefill_reused_tokens =
            static_cast<std::int64_t>(g.prefill_reused_tokens);
        n.sample.compactions = static_cast<std::int64_t>(compactions);
        notify(n);
    };
    return obs;
}

// The inbox of a run that is already in flight: approvals (S7.2) and steering (S4.5).
//
// Both are the same problem wearing different clothes -- the human said something while
// the model was working -- so they are one class over one channel. The run thread is
// already the channel's only consumer; handing either job to a second consumer would
// break the SPSC invariant to no purpose.
//
// The two differ only in urgency. An approval BLOCKS: nothing can proceed until the human
// answers. Steering does not block anything; it accumulates and is collected at the next
// turn boundary, which is why `take_messages` never waits.
//
// Deny-by-default survives intact. An `approved: true` carrying the request_id of the
// card still on screen is the ONLY path that returns true from ask(); a denial, a stale
// card, a cancel, a shutdown, a malformed reply and a dead parent all return false.
class RunInbox {
  public:
    RunInbox(platform::SpscChannel<std::string>& inbox, std::string run_id,
             std::string workspace_root, model::CancelToken& cancel)
        : inbox_(inbox), run_id_(std::move(run_id)),
          workspace_root_(std::move(workspace_root)), cancel_(cancel) {}

    // Everything the user has said since the last call, oldest first. Non-blocking: the
    // run is mid-flight and has work to get back to.
    std::vector<std::string> take_messages() {
        std::string msg;
        while (inbox_.try_pop(msg)) {
            (void)handle(msg, nullptr, nullptr);
        }
        std::vector<std::string> out;
        out.swap(steering_);
        return out;
    }

    // Whether the operator has already consented to this run's remaining whole-file
    // writes. See the `allow_writes_for_run` comment in protocol/schema.json for why this
    // is run-scoped rather than an allowlist of paths.
    [[nodiscard]] bool writes_allowed_for_run() const noexcept {
        return allow_writes_for_run_;
    }

    // Whether the operator clicked "Always allow" on this command earlier in this run.
    // Fully-parsed commands match VERBATIM. Opaque (PartiallyParsed / Unparseable)
    // commands match a key of workspace + command + classifier caps + script digests, so
    // approving `bash build.sh` does not survive a rewrite of build.sh.
    [[nodiscard]] bool command_allowed_for_run(const std::string& command,
                                               const tools::RiskHint& hint) const {
        const std::string opaque =
            loop::opaque_run_consent_key(workspace_root_, command, hint);
        if (!opaque.empty()) {
            return std::find(run_opaque_keys_.begin(), run_opaque_keys_.end(), opaque) !=
                   run_opaque_keys_.end();
        }
        return std::find(run_allowed_commands_.begin(), run_allowed_commands_.end(),
                         command) != run_allowed_commands_.end();
    }

    bool ask(const std::string& tool, const std::string& command,
             const std::string& preview, const tools::RiskHint& hint) {
        const std::string request_id = run_id_ + "/" + std::to_string(++seq_);
        protocol::ApprovalRequestNotification req;
        req.request_id = request_id;
        req.run_id = run_id_;
        req.tool_name = tool;
        req.preview = preview;
        req.command = command;
        // Sent rather than re-derived in the view from the chips, so the UI and the gate
        // cannot disagree about which calls are unwaivable.
        req.irreversible = loop::is_irreversible(hint);
        // Whether a persisted rule could ever match this command again. The view used to
        // decide this from `irreversible` alone and offered "Always allow" on chained
        // commands, whose stored rules the matcher can never hit.
        req.can_remember = loop::can_persist_allowlist_rule(command, hint);
        req.risk = loop::risk_score(hint);
        req.capabilities = chips_of(hint);
        // Held so the answer can name what it consented TO. `lmp/approve` carries a
        // request_id and a flag, not the command -- and taking the command from the reply
        // would let a malformed one grant consent for something the card never showed.
        pending_command_ = command;
        pending_hint_ = hint;
        notify(req);

        bool answer = false;
        std::string msg;
        while (true) {
            if (cancel_.cancelled()) {
                // The reader thread already flipped the token; leave without pinning the
                // run on a card nobody will answer.
                return false;
            }
            if (inbox_.drained()) {
                // The parent is gone, so there is nobody left to ask. Deny, rather than
                // block forever on a dead pipe holding 19 GB of weights.
                return false;
            }
            if (!inbox_.try_pop(msg)) {
                std::this_thread::yield();
                continue;
            }
            if (handle(msg, &request_id, &answer)) {
                return answer;
            }
        }
    }

    // The workspace edit round trip (S12.4). Structurally identical to ask(): send a
    // request, block on the run thread for the matching reply, treat a dead parent as a
    // refusal rather than waiting forever.
    //
    // It BLOCKS for the same reason a card does. The tool result has to say whether the
    // bytes landed -- an edit reported as applied while the editor never wrote it would
    // leave the model reasoning about a file that does not exist, which is the silent
    // failure this whole path is meant to remove.
    tools::EditOutcome apply_edit(const tools::EditIntent& intent) {
        const std::string request_id = run_id_ + "/edit/" + std::to_string(++seq_);
        // edit_begin/edit_end bracket the blocking wait so crash-safe resume can refuse
        // auto-resume when an edit was in flight (P2 §11).
        if (log_ != nullptr && clock_ != nullptr) {
            platform::Event begin;
            begin.kind = "edit_begin";
            begin.fields = {{"run_id", run_id_},
                            {"request_id", request_id},
                            {"path", intent.abs_path}};
            log_->append(begin, *clock_);
        }
        protocol::EditNotification req;
        req.request_id = request_id;
        req.run_id = run_id_;
        req.path = intent.abs_path;
        req.new_content = intent.new_content;
        req.expected_version = intent.expected_version;
        req.expected_absent = intent.expected_absent;
        notify(req);

        std::string msg;
        while (true) {
            if (cancel_.cancelled()) {
                emit_edit_end(request_id, false, "cancelled");
                return {false, "cancelled"};
            }
            if (inbox_.drained()) {
                emit_edit_end(request_id, false, "the editor is gone");
                return {false, "the editor is gone"};
            }
            if (!inbox_.try_pop(msg)) {
                std::this_thread::yield();
                continue;
            }
            if (handle_edit_reply(msg, request_id)) {
                emit_edit_end(request_id, edit_answer_.applied, edit_answer_.error);
                return edit_answer_;
            }
        }
    }

    void set_event_log(platform::EventLogWriter* log, const platform::Clock* clock) {
        log_ = log;
        clock_ = clock;
    }

    tools::CodeIntelOutcome apply_code_intel(const tools::CodeIntelQuery& query) {
        const std::string request_id = run_id_ + "/intel/" + std::to_string(++seq_);
        protocol::CodeIntelNotification req;
        req.request_id = request_id;
        req.run_id = run_id_;
        req.op = query.op;
        req.query = query.query;
        req.path = query.path;
        req.line = query.line;
        req.character = query.character;
        notify(req);

        std::string msg;
        while (true) {
            if (cancel_.cancelled()) {
                return {false, {}, "cancelled"};
            }
            if (inbox_.drained()) {
                return {false, {}, "the editor is gone"};
            }
            if (!inbox_.try_pop(msg)) {
                std::this_thread::yield();
                continue;
            }
            if (handle_code_intel_reply(msg, request_id)) {
                return code_intel_answer_;
            }
        }
    }

    // True when a shutdown arrived mid-run: acknowledged there, but only actionable once
    // the run has unwound.
    [[nodiscard]] bool shutdown_requested() const noexcept { return shutdown_; }

  private:
    // Services one message while an edit is outstanding. Returns true only when it
    // ANSWERED that edit; everything else falls through to the ordinary routing, so
    // steering typed while a write was in flight is queued rather than lost.
    bool handle_edit_reply(const std::string& msg, const std::string& awaiting) {
        const std::string method = surface::method_of(msg);
        if (method == "lmp/edit_applied") {
            const std::string id = surface::string_field(msg, "id");
            if (surface::string_field(msg, "request_id") == awaiting) {
                reply_result(id, R"({"accepted":true})");
                edit_answer_.applied = surface::bool_field(msg, "applied");
                edit_answer_.error = surface::string_field(msg, "error");
                return true;
            }
            // An answer to an edit this run has already moved past. Say so rather than
            // letting it decide the CURRENT write.
            reply_result(id, R"({"accepted":false})");
            return false;
        }
        // Cancel / shutdown under an edit must unblock the wait the same way they do
        // under a card. handle() with a null awaiting only ACKs them; without this the
        // edit wait could pin 19 GB until the parent drained.
        if (method == "lmp/cancel" || method == "lmp/shutdown") {
            bool ignored = false;
            (void)handle(msg, &awaiting, &ignored);
            edit_answer_ = {false, method == "lmp/cancel" ? "cancelled" : "shutdown"};
            return true;
        }
        (void)handle(msg, nullptr, nullptr);
        return false;
    }

    bool handle_code_intel_reply(const std::string& msg, const std::string& awaiting) {
        const std::string method = surface::method_of(msg);
        if (method == "lmp/code_intel_result") {
            const std::string id = surface::string_field(msg, "id");
            if (surface::string_field(msg, "request_id") == awaiting) {
                reply_result(id, R"({"accepted":true})");
                code_intel_answer_.ok = surface::bool_field(msg, "ok");
                code_intel_answer_.error = surface::string_field(msg, "error");
                code_intel_answer_.result_text = surface::string_field(msg, "result_text");
                return true;
            }
            reply_result(id, R"({"accepted":false})");
            return false;
        }
        if (method == "lmp/cancel" || method == "lmp/shutdown") {
            bool ignored = false;
            (void)handle(msg, &awaiting, &ignored);
            code_intel_answer_ = {false, {},
                                  method == "lmp/cancel" ? "cancelled" : "shutdown"};
            return true;
        }
        (void)handle(msg, nullptr, nullptr);
        return false;
    }

    // Services one message. Returns true only when it ANSWERED the approval named by
    // `awaiting` -- so the same routing serves the blocking and non-blocking paths, and a
    // message cannot mean one thing at a turn boundary and another under a card.
    bool handle(const std::string& msg, const std::string* awaiting, bool* answer) {
        const std::string method = surface::method_of(msg);
        const std::string id = surface::string_field(msg, "id");

        if (method == "lmp/message") {
            const std::string text = surface::string_field(msg, "text");
            if (text.empty()) {
                reply_error(id, "lmp/message requires a non-empty 'text'");
                return false;
            }
            // Collected, not applied. The model is generating; the instruction lands at
            // the next turn boundary, where the run can actually act on it. This holds
            // even when a card is on screen -- steering typed while the human was
            // deciding is queued, not dropped, and not mistaken for the answer.
            steering_.push_back(text);
            reply_result(id, R"({"accepted":true,"run_id":")" + run_id_ +
                                 R"(","started_run":false})");
            return false;
        }
        if (method == "lmp/approve") {
            if (awaiting != nullptr && surface::string_field(msg, "request_id") == *awaiting) {
                reply_result(id, R"({"accepted":true})");
                *answer = surface::bool_field(msg, "approved");
                // Read even on a denial, and deliberately: "no to this one, yes to the
                // rest" is not a coherent answer, so the flag only ever arrives with an
                // approval from the UI. Latching it here rather than in the caller keeps
                // the run-scoped consent in the one object that owns the card protocol.
                if (surface::bool_field(msg, "allow_writes_for_run")) {
                    allow_writes_for_run_ = true;
                }
                // Same latch, the command gate's half. Only on an APPROVAL and only for a
                // card that actually carried a command: "always allow" attached to a
                // denial, or to a write card, is not a coherent answer and the UI never
                // sends one.
                if (*answer && surface::bool_field(msg, "allow_command_for_run") &&
                    !pending_command_.empty() &&
                    !command_allowed_for_run(pending_command_, pending_hint_)) {
                    const std::string opaque = loop::opaque_run_consent_key(
                        workspace_root_, pending_command_, pending_hint_);
                    if (!opaque.empty()) {
                        run_opaque_keys_.push_back(opaque);
                    } else {
                        run_allowed_commands_.push_back(pending_command_);
                    }
                }
                return true;
            }
            // An answer to a card this run has already moved past, or one nobody asked
            // for. Say so rather than letting it decide the CURRENT call -- an approval
            // is for one specific command, not for a position in a queue.
            reply_result(id, R"({"accepted":false})");
            return false;
        }
        if (method == "lmp/cancel") {
            // The reader thread set the token when it framed the message; this is only
            // the acknowledgement. Under a card, deny and let the loop notice the token.
            reply_result(id, R"({"accepted":true})");
            if (awaiting != nullptr) {
                *answer = false;
                return true;
            }
            return false;
        }
        if (method == "lmp/shutdown") {
            shutdown_ = true;
            reply_result(id, R"({"ok":true})");
            if (awaiting != nullptr) {
                *answer = false;
                return true;
            }
            return false;
        }
        reply_error(id, "a run is in flight; '" + method +
                            "' cannot be serviced until it ends. To talk to the run "
                            "while it works, use lmp/message.");
        return false;
    }

    void emit_edit_end(const std::string& request_id, bool applied, const std::string& error) {
        if (log_ == nullptr || clock_ == nullptr) {
            return;
        }
        platform::Event end;
        end.kind = "edit_end";
        end.fields = {{"run_id", run_id_},
                      {"request_id", request_id},
                      {"applied", applied ? "1" : "0"},
                      {"error", error}};
        log_->append(end, *clock_);
    }

    platform::SpscChannel<std::string>& inbox_;
    std::string run_id_;
    std::string workspace_root_;
    model::CancelToken& cancel_;
    platform::EventLogWriter* log_ = nullptr;
    const platform::Clock* clock_ = nullptr;
    std::vector<std::string> steering_;
    std::uint64_t seq_ = 0;
    tools::EditOutcome edit_answer_;
    tools::CodeIntelOutcome code_intel_answer_;
    bool shutdown_ = false;
    // Latched by one card, spent by the rest of the run, and gone when the run ends. A
    // member of the inbox rather than of the session, so it cannot outlive the mission
    // the operator was actually looking at when they granted it.
    bool allow_writes_for_run_ = false;
    // The command on the card currently in front of the operator, and the ones they have
    // said "always" to during this run. Run-scoped for the same reason the write flag is:
    // the persistent copy is the extension's `allowedCommands`, and consent given to one
    // mission is not consent to the next.
    std::string pending_command_;
    tools::RiskHint pending_hint_;
    std::vector<std::string> run_allowed_commands_;
    // Opaque-script consent keys (see loop::opaque_run_consent_key). Separate from the
    // verbatim list so a digest-bound approval cannot be smuggled in as a prefix match.
    std::vector<std::string> run_opaque_keys_;
};

// How much the operator wants to be asked, and what may run without asking.
//
// sandbox_tier and require_approval were on the wire, generated on both sides, and read
// by NOBODY: the tier came from the mode and the approval routing came from the risk
// thresholds, so both switches in the editor were decoration. The same failure as the
// sampling block before it -- a setting that is plumbed but not consumed looks exactly
// like one that works.
[[nodiscard]] bool apply_autonomy(const std::string& id, const std::string& message,
                                  loop::AgentConfig& config) {
    const double tier = surface::double_field(message, "sandbox_tier", -1.0);
    if (tier >= 0.0) {
        const int t = static_cast<int>(tier);
        if (t > 3) {
            reply_error(id, "sandbox_tier must be 0 (no execution), 1 (Seatbelt), "
                            "2 (container) or 3 (UNSANDBOXED on the host); got " +
                                std::to_string(t));
            return false;
        }
        config.sandbox_tier_override = t;
    }

    // Presence-checked, not just read: absent must keep the AgentConfig default rather
    // than collapsing to false, or every client that predates these fields would silently
    // have both switches flipped.
    if (surface::has_field(message, "auto_approve_exec")) {
        config.auto_approve_exec = surface::bool_field(message, "auto_approve_exec");
    }
    if (surface::has_field(message, "auto_approve_writes")) {
        config.auto_approve_writes = surface::bool_field(message, "auto_approve_writes");
    }
    // require_approval is the operator saying "ask me about everything", so it is a FLOOR
    // over the two specific switches rather than a third one competing with them. It can
    // only ever tighten: `require_approval: true` beats `auto_approve_exec: true`, and
    // never the other way round.
    if (surface::bool_field(message, "require_approval")) {
        config.auto_approve_exec = false;
        config.auto_approve_writes = false;
    }

    // Newline-separated, because the generated protocol has no array type and a newline
    // is the one character a shell command cannot carry unescaped.
    const std::string allowed = surface::string_field(message, "allowed_commands");
    for (std::size_t at = 0; at < allowed.size();) {
        const std::size_t nl = allowed.find('\n', at);
        const std::size_t end = nl == std::string::npos ? allowed.size() : nl;
        if (end > at) {
            config.allowed_commands.push_back(allowed.substr(at, end - at));
        }
        at = end + 1;
    }
    return true;
}

// Reads the run's settings off the start message into the config. Returns false (having
// answered with an error) if a setting is present but unusable.
//
// Before this existed the sidecar built a default AgentConfig and dropped `settings` on
// the floor, so every knob the extension contributes -- sampling, mode, the budget --
// was inert: changing lmPipe.sampling.temperature in the editor did nothing at all. The
// defaults happened to equal Qwen3's recommended operating point, so the effect was
// invisible rather than wrong, which is the worst way for it to have been broken.
//
// Each fallback is the value already in `config`, so an absent field keeps the pinned
// Qwen default rather than collapsing to zero (S5.9).
[[nodiscard]] bool apply_settings(const std::string& id, const std::string& message,
                                  loop::AgentConfig& config) {
    const std::string mode = surface::string_field(message, "mode");
    if (mode == "plan") {
        config.mode = loop::Mode::Plan;
    } else if (mode == "debug") {
        config.mode = loop::Mode::Debug;
    } else if (mode == "agent" || mode.empty()) {
        config.mode = loop::Mode::Agent;
    } else {
        // No silent fall back to the most permissive mode (S13). An unrecognised mode is
        // a settings bug, and guessing Agent for it is how a plan-only run gets to write.
        reply_error(id, "unrecognised mode '" + mode + "'; expected plan, debug or agent");
        return false;
    }

    auto& s = config.sampling;
    const auto real = [&message](std::string_view key, float fallback) {
        return static_cast<float>(
            surface::double_field(message, key, static_cast<double>(fallback)));
    };
    s.temperature = real("temperature", s.temperature);
    s.top_p = real("top_p", s.top_p);
    s.top_k = static_cast<std::int32_t>(surface::double_field(message, "top_k", s.top_k));
    s.min_p = real("min_p", s.min_p);
    s.repetition_penalty = real("repetition_penalty", s.repetition_penalty);
    config.seed = static_cast<std::uint64_t>(
        surface::double_field(message, "seed", static_cast<double>(config.seed)));

    // Positive, finite, and in int32 range. The wire carries JSON numbers as doubles, so
    // a negative, NaN, or 1e20 used to cast into a silent wrap / max(1, x) later and look
    // like a working setting. Refuse at the boundary instead.
    const auto positive_i32 = [&](const char* key, std::int32_t fallback,
                                  std::int32_t* out) -> bool {
        if (!surface::has_field(message, key)) {
            *out = fallback;
            return true;
        }
        const double v = surface::double_field(message, key, static_cast<double>(fallback));
        if (!(v > 0.0) || v > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
            reply_error(id, std::string(key) + " must be a positive integer fitting in "
                                              "int32; got " +
                                std::to_string(v));
            return false;
        }
        *out = static_cast<std::int32_t>(v);
        return true;
    };
    if (!positive_i32("context_budget_tokens", config.context_budget_tokens,
                      &config.context_budget_tokens) ||
        !positive_i32("max_new_tokens", config.max_new_tokens, &config.max_new_tokens)) {
        return false;
    }
    std::int32_t max_iters = config.budget.max_iterations;
    std::int32_t wall = config.budget.wall_clock_seconds;
    if (!positive_i32("max_iterations", max_iters, &max_iters) ||
        !positive_i32("wall_clock_seconds", wall, &wall)) {
        return false;
    }
    config.budget.max_iterations = static_cast<int>(max_iters);
    config.budget.wall_clock_seconds = static_cast<int>(wall);

    // The operator's check -- the only verification the harness runs. Absent keeps the
    // config default (empty: no check), same rule as every field above.
    if (surface::has_field(message, "verify_contract")) {
        config.operator_verify_contract = surface::string_field(message, "verify_contract");
    }

    return apply_autonomy(id, message, config);
}

// Where the surface learns whether this process can answer a prompt at all.
//
// Sent on EVERY transition and once unsolicited after lmp/ready, so a client that
// attaches at any moment is told the truth rather than inferring it. Before this the
// only evidence a model was loaded was the absence of an error, and the absence of an
// error is exactly what a 19 GB load in progress looks like.
void notify_model(const char* state, const std::string& model_dir,
                  const std::string& detail = {}, double elapsed_ms = 0.0) {
    protocol::ModelStatusNotification n;
    n.state = state;
    n.model_dir = model_dir;
    n.detail = detail;
    n.elapsed_ms = elapsed_ms;
    // Answered from the checkpoint on disk, so it is right on `loading` too -- the
    // surface can settle the control before the weights are up, and an empty model_dir
    // (the `unloaded` status) answers false without touching the filesystem.
    n.supports_reasoning_effort =
        !model_dir.empty() && model::supports_reasoning_effort(model_dir);
    notify(n);
}

// Loads the weights, announcing the attempt before it blocks and the outcome after.
//
// The announcement is the whole point of the split. `loading` goes out first because the
// load owns this thread for tens of seconds; a status that could only be sent once the
// work was over would tell the operator nothing they had not already worked out.
[[nodiscard]] surface::ModelLoad ensure_model(surface::Session& session,
                                              const std::string& model_dir,
                                              const std::string& draft_model_dir,
                                              platform::EventLogWriter& log,
                                              const platform::Clock& clock) {
    if (session.holds(model_dir, draft_model_dir)) {
        return {true, {}, 0.0};
    }
    notify_model(protocol::modelstate_values::kLoading, model_dir);
    const surface::ModelLoad loaded =
        surface::load_model(session, model_dir, draft_model_dir, clock);
    // The memory the load left behind, and the cache ceiling that now bounds it. Logged
    // because the only reason we know the allocator cache reached 18 GB on top of a 20 GB
    // model -- and took the process down with it -- is that `unload_model` happened to
    // report both numbers at the END of a session. A run that dies mid-way never gets
    // there, so the same facts have to be on the record at the start.
    if (loaded.ok) {
        const model::MemoryReport m = model::mlx_memory_report();
        platform::Event ev;
        ev.kind = "model_memory";
        ev.fields = {{"active", std::to_string(m.active)},
                     {"cache", std::to_string(m.cache)},
                     {"cache_limit", std::to_string(model::mlx_cache_limit())}};
        log.append(ev, clock);
    }
    // `model_dir` names the checkpoint the status is ABOUT, on the failure too: "which
    // path did that come from" is the first question a load error raises, and a client
    // that had to remember what it asked for could only ever guess at an unsolicited one.
    notify_model(loaded.ok ? protocol::modelstate_values::kReady
                           : protocol::modelstate_values::kFailed,
                 model_dir, loaded.error, loaded.elapsed_ms);
    return loaded;
}

// Turns the loop once over an EXISTING session. Returns true when the run should be the
// process's last. The caller has already replied to the request that triggered it, since
// this blocks for as long as the mission takes.
bool run_loop(const std::string& run_id, surface::Session& session,
              platform::SpscChannel<std::string>& inbox, model::CancelToken& cancel,
              platform::EventLogWriter& log, const platform::Clock& clock) {
    loop::Agent agent(*session.tok, *session.backend, *session.registry, *session.ctx, log,
                      clock, session.config);
    agent.set_observer(make_observer(run_id));

    RunInbox run_inbox(inbox, run_id, session.registry->workspace().root, cancel);
    run_inbox.set_event_log(&log, &clock);
    agent.set_approver([&run_inbox, &log, &clock](const std::string& tool,
                                                  const std::string& command,
                                                  const std::string& preview,
                                                  const tools::RiskHint& hint) {
        // An empty command is how gate_call spells "this is the write gate" -- the
        // command path always passes the command verbatim, because a truncated preview is
        // the wrong thing to build an allowlist rule from.
        const bool write_gate = command.empty();
        if (write_gate && run_inbox.writes_allowed_for_run()) {
            // Answered without a card, so it must still be answered in the LOG. An
            // auto-approval that leaves no trace is the blind spot the `approval` event
            // was just added to close, and re-opening it one layer up would be worse:
            // here the operator really did consent, and the record should say when.
            platform::Event ev;
            ev.kind = "approval";
            ev.fields.push_back({"gate", "write"});
            ev.fields.push_back({"tool", tool});
            // The ANSWER to the escalation the gate just recorded: no card was shown, and
            // the reason is consent the operator gave earlier in this same run.
            ev.fields.push_back({"card", "0"});
            ev.fields.push_back({"answer", "approved"});
            ev.fields.push_back({"why", "operator allowed writes for this run"});
            log.append(ev, clock);
            return true;
        }
        // The command gate's half. "Always allow" writes a persistent rule through the
        // extension, and that rule reaches the sidecar at the next `lmp/start` -- so
        // without this the operator clicks "always" and answers the identical card again
        // on the very next turn, which is the button not working.
        if (!write_gate && run_inbox.command_allowed_for_run(command, hint)) {
            platform::Event ev;
            ev.kind = "approval";
            ev.fields.push_back({"gate", "command"});
            ev.fields.push_back({"tool", tool});
            ev.fields.push_back({"command", command});
            ev.fields.push_back({"card", "0"});
            ev.fields.push_back({"answer", "approved"});
            ev.fields.push_back({"why", "operator allowed this command for this run"});
            log.append(ev, clock);
            return true;
        }
        return run_inbox.ask(tool, command, preview, hint);
    });
    agent.set_steer_source([&run_inbox]() { return run_inbox.take_messages(); });

    // Workspace writes go through the EDITOR (S12.4), so undo, dirty buffers and diff
    // review work. Attached only when an editor is actually there to route through:
    // `lmp/edit_applied` is a capability the client advertises at lmp/start, and a client
    // that does not is left on the direct-write path.
    //
    // Deliberately the opposite polarity to the approver's deny-by-default. An absent
    // approver means "nobody is there to ask", so the safe answer is no. An absent edit
    // sink means "there is no editor to route through" -- an eval run, a script -- and
    // refusing there would break every unattended run for no safety gain at all.
    if (session.client_applies_edits) {
        session.registry->set_edit_sink(
            [&run_inbox](const tools::EditIntent& intent) {
                return run_inbox.apply_edit(intent);
            });
    } else {
        session.registry->set_edit_sink(nullptr);
    }
    if (session.client_provides_code_intel) {
        session.registry->set_code_intel_sink(
            [&run_inbox](const tools::CodeIntelQuery& query) {
                return run_inbox.apply_code_intel(query);
            });
    } else {
        session.registry->set_code_intel_sink(nullptr);
    }

    cancel.reset();
    const loop::RunReport report = agent.run(cancel);

    // Anything said in the last moments of the run, after the loop's final drain.
    //
    // The window is narrow -- a message popped while an approval card was up, on a run
    // that then ended before reaching another turn boundary -- but it was already
    // ACKNOWLEDGED with accepted:true when it was popped. Dropping it on the floor would
    // make the harness a liar about the one thing the user can see. It carries into the
    // session instead, so the next follow-up starts with it already said.
    for (const std::string& late : run_inbox.take_messages()) {
        session.ctx->add_user_message(late);
    }

    protocol::RunEndNotification end;
    end.run_id = run_id;
    end.termination_reason = report.termination_reason;
    end.iterations = report.iterations;
    end.completed = report.completed;
    end.unfinished_items = static_cast<std::int64_t>(report.unfinished_items);
    notify(end);
    return run_inbox.shutdown_requested();
}

// A fresh mission. Reuses the loaded weights when the model has not changed -- the
// tokenizer and the checkpoint are the expensive part and neither depends on the mission
// -- but the context store is rebuilt, because `lmp/start` MEANS start over. Continuing a
// conversation is what lmp/message is for.
bool start_mission(const std::string& id, const std::string& message,
                   surface::Session& session, platform::SpscChannel<std::string>& inbox,
                   model::CancelToken& cancel, platform::EventLogWriter& log,
                   const platform::Clock& clock) {
    const std::string mission = surface::string_field(message, "mission");
    const std::string model_dir = surface::string_field(message, "model_dir");
    // Optional. Empty means no draft head, which is the default and the reference path.
    const std::string draft_model_dir = surface::string_field(message, "draft_model_dir");
    const std::string workspace = surface::string_field(message, "workspace_root");
    if (mission.empty() || model_dir.empty() || workspace.empty()) {
        reply_error(id, "mission, model_dir and workspace_root are all required and "
                        "none may be empty (S7.5: no defaultable security input)");
        return false;
    }
    reply_result(id, R"({"run_id":")" + id + R"("})");

    loop::AgentConfig config;
    if (!apply_settings(id, message, config)) {
        return false;
    }
    session.config = config;

    // Still loads on demand, so lmp/load_model is a door into this and not a new
    // precondition -- a headless client that only knows lmp/start keeps working. The
    // difference is that the wait is now narrated: ensure_model says `loading` before it
    // blocks, and the run_end below carries the loader's own words when it cannot.
    const surface::ModelLoad loaded = ensure_model(session, model_dir, draft_model_dir, log, clock);
    if (!loaded.ok) {
        end_run(id, loaded.error);
        return false;
    }

    // Clamp/refuse against the checkpoint ceiling now that it is known. apply_settings
    // cannot do this: the model may not be loaded yet when the wire values arrive.
    session.config.model_max_sequence_tokens = session.model_max_sequence_tokens;
    if (session.model_max_sequence_tokens > 0) {
        if (session.config.context_budget_tokens > session.model_max_sequence_tokens) {
            session.config.context_budget_tokens = session.model_max_sequence_tokens;
        }
        if (session.config.max_new_tokens > session.model_max_sequence_tokens) {
            end_run(id, "max_new_tokens (" + std::to_string(session.config.max_new_tokens) +
                            ") exceeds model maximum sequence length (" +
                            std::to_string(session.model_max_sequence_tokens) + ")");
            return false;
        }
        const auto room =
            session.model_max_sequence_tokens - session.config.context_budget_tokens;
        if (room < 1) {
            end_run(id, "context_budget_tokens leaves no room for generation under the "
                        "model maximum sequence length (" +
                            std::to_string(session.model_max_sequence_tokens) + ")");
            return false;
        }
        if (session.config.max_new_tokens > room) {
            session.config.max_new_tokens = room;
        }
    }

    // CAN THE MACHINE HOLD THIS BUDGET? The clamp above asks only whether the MODEL can
    // address it, which is a different question with a much larger answer.
    //
    // Clamped, not refused, and for the same reason the ceiling above is: a budget that
    // does not fit is an operator mistake with an obvious correct value, not an ambiguity
    // worth ending a run over. Refusing would also make the failure LOUDER than the one it
    // replaces without making it clearer -- the old failure was the process disappearing.
    //
    // Emitted whatever it decides, including when it changes nothing. A silent clamp is
    // how `context_budget_tokens` came to differ from what the settings file said with
    // nothing anywhere recording it, and the arithmetic is the whole value here: the next
    // person to look at a memory death needs the numbers this used, not its conclusion.
    {
        const std::size_t kv_per_token =
            model::kv_bytes_per_token(model_dir) + model::kv_bytes_per_token(draft_model_dir);
        const model::MemoryReport mem = model::mlx_memory_report();
        const std::size_t working_set = model::mlx_cache_limit();
        const int affordable =
            model::max_affordable_context_tokens(kv_per_token, mem.active, working_set);
        const int requested = session.config.context_budget_tokens;
        const bool clamped = affordable > 0 && requested > affordable;
        if (clamped) {
            session.config.context_budget_tokens = affordable;
            // The generation reserve rides on the budget, so a clamp that left it alone
            // could put max_new_tokens above the whole context.
            if (session.config.max_new_tokens > affordable / 2) {
                session.config.max_new_tokens = std::max(1, affordable / 2);
            }
        }
        platform::Event ev;
        ev.kind = "context_budget";
        ev.fields = {
            {"requested", std::to_string(requested)},
            {"affordable", std::to_string(affordable)},
            {"applied", std::to_string(session.config.context_budget_tokens)},
            {"clamped", clamped ? "1" : "0"},
            {"kv_bytes_per_token", std::to_string(kv_per_token)},
            {"weights_bytes", std::to_string(mem.active)},
            {"device_working_set", std::to_string(working_set)},
            {"why", affordable == 0
                        ? "cannot size the KV cache (no MLX, or an unreadable config); the "
                          "operator's budget is left exactly as configured"
                    : clamped ? "the KV cache for the configured budget does not fit beside "
                                "the weights in the device working set"
                              : "the configured budget fits"}};
        log.append(ev, clock);
    }

    // HOW HARD TO THINK. Resolved here, ahead of any session state, because the invalid
    // case ends the run and every other refusal in this function does so before there is
    // a store or a registry to leave half-built.
    //
    // Two independent questions, and they get opposite answers on failure: is the word a
    // level at all, and does THIS checkpoint understand levels?
    //
    // The first is refused. `high` looks entirely reasonable, is what every summary of
    // this feature claims exists, and is exactly what the reference template raises on --
    // so an unknown word ends the run naming the three that work, rather than silently
    // instructing nothing.
    //
    // The second is dropped, silently and on purpose. One settings file is used against
    // both checkpoints on this machine and only Qwen3.8 has levels, so carrying the
    // setting to a checkpoint without them is the NORMAL case, not an operator error.
    // Logged either way, because "I set it to low and nothing changed" is otherwise an
    // unanswerable question.
    std::string reasoning_brief;
    {
        const std::string requested = surface::string_field(message, "reasoning_effort");
        const std::optional<model::ReasoningEffort> level =
            model::parse_reasoning_effort(requested);
        if (!level.has_value()) {
            end_run(id, "reasoning_effort must be one of low, medium or xhigh (got \"" +
                            requested + "\"); note that this checkpoint family has no "
                                        "'high' -- xhigh is the top level");
            return false;
        }
        const bool supported = model::supports_reasoning_effort(model_dir);
        if (supported) {
            reasoning_brief = std::string(model::reasoning_instructions_for(*level));
        }
        platform::Event ev;
        ev.kind = "reasoning_effort";
        ev.fields = {{"requested", requested.empty() ? "(checkpoint default)" : requested},
                     {"supported", supported ? "1" : "0"},
                     {"instructed", reasoning_brief.empty() ? "0" : "1"},
                     {"why", !supported ? "this checkpoint's chat template has no "
                                          "reasoning_effort; the prompt is unchanged"
                             : reasoning_brief.empty()
                                 ? "medium and the checkpoint default both instruct "
                                   "nothing; the prompt is unchanged"
                                 : "the checkpoint's own sentence for this level opens "
                                   "the system prompt"}};
        log.append(ev, clock);
    }

    surface::ensure_registry(session, workspace, message, log, clock);
    const std::string& canonical_workspace = session.registry->workspace().root;

    session.ctx = std::make_unique<context::ContextStore>(mission);
    // Images attached to the OPENING mission. The store takes the mission through its
    // constructor (it is T0 and run-constant), so they cannot ride on it; they arrive as
    // the first human turn behind it instead, which is where the chronology puts them
    // anyway -- the picture was attached to that instruction, not to the session.
    {
        std::vector<std::string> attached =
            surface::parse_string_array(message, "image_paths");
        if (!attached.empty()) {
            session.ctx->add_user_message("(attached)", std::move(attached));
        }
    }
    // A little above the widest single result the tool layer can produce, so the door
    // catches a tool that forgot to bound itself without firing on a legitimate one.
    session.ctx->set_observation_budget(tools::kObservationBudgetBytes);
    // Loaded once into the STABLE part of the prompt: neither the repo's conventions nor
    // the agent's own notes change mid-run, so they cost one prefill for the whole run.
    session.ctx->set_project_instructions(
        surface::load_project_instructions(session.registry->filesystem()));
    session.ctx->set_project_memory(
        surface::load_project_memory(session.registry->filesystem()));
    session.ctx->set_workspace_root(canonical_workspace);
    // Empty keeps the built-in persona; the editor sends the one it holds for this mode.
    session.ctx->set_persona(surface::string_field(message, "system_prompt"));

    // Resolved before any of this was built; empty whenever the level instructs nothing
    // or the checkpoint has no levels, and an empty brief leaves the prompt untouched.
    session.ctx->set_reasoning_brief(std::move(reasoning_brief));
    // Logged the way a failed MCP server is (see connect_mcp_servers): a degradation the
    // run survives, recorded where a postmortem will find it rather than announced. A
    // null journal means this run's compacted turns are gone for good once they are
    // trimmed, which is invisible at the time and matters later -- exactly the shape of
    // thing the event log exists for.
    surface::ContextJournal::Result journal =
        surface::ContextJournal::open(canonical_workspace, id, *session.ctx);
    if (!journal.error.empty()) {
        platform::Event ev;
        ev.kind = "context_journal";
        ev.fields.push_back({"opened", "0"});
        ev.fields.push_back({"error", journal.error});
        log.append(ev, clock);
    }
    session.journal = std::move(journal.journal);

    // The READ side of the store, which for a long time did not exist: src/pcc journalled
    // faithfully and no tool in this binary could get a byte back out. Declared only when
    // there is a journal to read -- a run whose database would not open keeps every other
    // tool and simply does not advertise these two.
    //
    // BOTH LAMBDAS RESOLVE THROUGH `session` AT CALL TIME, and neither may capture what
    // it resolves. ensure_registry() reuses a Registry across missions in the same
    // workspace, while the two things these tools need are replaced on every mission
    // (the journal, just above) and on every model load (the tokenizer). Capturing either
    // by reference would leave mission two reading mission one's freed memory.
    if (session.journal != nullptr) {
        (void)session.registry->declare_context_tools(
            [&session] {
                tools::Registry::ContextSource src;
                if (session.journal != nullptr) {
                    src.store = &session.journal->store();
                    src.session = session.journal->session_id();
                }
                return src;
            },
            [&session](std::string_view text) {
                // THE REAL TOKENIZER, which is the whole reason these tools are native
                // rather than the pcc_mcp_server's. pcc::recall packs to a TOKEN budget
                // and its default counter is bytes/4 -- an estimate src/pcc names as one
                // at every call site because it deliberately does not link src/model.
                // In-process there is no need to estimate.
                return session.tok != nullptr
                           ? session.tok->encode_content(text).size()
                           : pcc::estimate_tokens(text);
            });
    }
    session.client_applies_edits = surface::bool_field(message, "applies_edits");
    session.client_provides_code_intel =
        surface::bool_field(message, "provides_code_intel");

    // Resume binding (P2 §11). PCC stays cross-session recall; this event is the
    // turn-accurate checkpoint identity for crash-safe rebuild from the event log.
    {
        platform::Event begin;
        begin.kind = "run_begin";
        begin.fields = {
            {"run_id", id},
            {"mission", mission},
            {"workspace_root", canonical_workspace},
            {"model_identity", model_dir},
            {"protocol_version", protocol::kProtocolVersion},
            {"tool_schema_hash",
             platform::content_sha256_hex(session.registry->tools_json())},
        };
        log.append(begin, clock);
    }

    return run_loop(id, session, inbox, cancel, log, clock);
}

// Load the weights, as its own act (S12.2). Answers when the load is over; the surface
// follows the attempt through the model_status notifications ensure_model emits.
//
// The reply carries the loader's verbatim error rather than a JSON-RPC error, because
// "this checkpoint has no safetensors" is an ordinary answer to a reasonable question --
// the request was well-formed and was serviced. A protocol error would say the opposite.
void handle_load_model(const std::string& id, const std::string& message,
                       surface::Session& session, platform::EventLogWriter& log,
                       const platform::Clock& clock) {
    const std::string model_dir = surface::string_field(message, "model_dir");
    if (model_dir.empty()) {
        reply_error(id, "lmp/load_model requires a non-empty 'model_dir'");
        return;
    }
    const std::string draft_model_dir = surface::string_field(message, "draft_model_dir");
    const surface::ModelLoad loaded = ensure_model(session, model_dir, draft_model_dir, log, clock);
    reply_result(id, R"({"loaded":)" + std::string(loaded.ok ? "true" : "false") +
                         R"(,"model_dir":)" + json_escape(loaded.ok ? model_dir : "") +
                         R"(,"error":)" + json_escape(loaded.error) + "}");
}

// Give the memory back. One checkpoint is ~19 GB on a 48 GB host, so this is a routine
// request and the only answer used to be closing the editor.
//
// Only reachable with no run in flight: while one is turning, the main loop is inside
// run_loop and RunInbox refuses every method it does not recognise. Freeing the weights
// out from under a generating model is therefore not a case that has to be defended
// against here -- the structure already excludes it.
// The before/after is logged because "unloaded" was a claim nobody could check from
// outside: the reply said true while ~19 GB stayed resident, and the only way to notice
// was Activity Monitor. Now the run that frees nothing says so in its own trace.
void handle_unload_model(const std::string& id, surface::Session& session,
                         platform::EventLogWriter& log, const platform::Clock& clock) {
    const model::MemoryReport before = model::mlx_memory_report();
    surface::unload_model(session);
    const model::MemoryReport after = model::mlx_memory_report();

    platform::Event ev;
    ev.kind = "unload_model";
    ev.fields = {{"active_before", std::to_string(before.active)},
                 {"cache_before", std::to_string(before.cache)},
                 {"active_after", std::to_string(after.active)},
                 {"cache_after", std::to_string(after.cache)}};
    log.append(ev, clock);

    notify_model(protocol::modelstate_values::kUnloaded, "");
    reply_result(id, R"({"unloaded":true})");
}

// A follow-up: the same context, one more user turn, another pass of the loop.
//
// This is the idle half of lmp/message. The in-flight half lives in RunInbox and never
// reaches here -- while a run is turning, the main loop is inside run_loop and is not
// reading the inbox at all, so a message that arrives HERE is by construction one that
// arrived with nothing running.
bool continue_session(const std::string& id, const std::string& message,
                      surface::Session& session,
                      platform::SpscChannel<std::string>& inbox, model::CancelToken& cancel,
                      platform::EventLogWriter& log, const platform::Clock& clock) {
    const std::string text = surface::string_field(message, "text");
    if (text.empty()) {
        reply_error(id, "lmp/message requires a non-empty 'text'");
        return false;
    }
    if (session.ctx == nullptr) {
        reply_error(id, "there is no session to continue; send lmp/start first");
        return false;
    }
    reply_result(id, R"({"accepted":true,"run_id":")" + id + R"(","started_run":true})");
    session.ctx->add_user_message(text,
                                  surface::parse_string_array(message, "image_paths"));
    return run_loop(id, session, inbox, cancel, log, clock);
}

// Routes ONE framed message. Returns true when the process should not survive it.
//
// Split from main because main had grown two jobs -- bring the process up, then route
// forever -- and the second is the one that changes. The exit decision is returned rather
// than taken here so that "what ends this process" stays readable in one place: a
// shutdown with nothing running, and a shutdown that arrived mid-run and had to wait for
// the weights to be freed, are the same fact and now leave by the same door.
bool dispatch_one(const std::string& message, surface::Session& session,
                  platform::SpscChannel<std::string>& inbox, model::CancelToken& cancel,
                  platform::EventLogWriter& log, const platform::Clock& clock) {
    const std::string method = surface::method_of(message);
    const std::string id = surface::string_field(message, "id");

    if (method == "lmp/shutdown") {
        reply_result(id, R"({"ok":true})");
        return true;
    }
    if (method == "lmp/cancel") {
        // The reader thread already set the token the moment the message was framed --
        // this is only the acknowledgement (S4.3).
        reply_result(id, R"({"accepted":true})");
        return false;
    }
    if (method == "lmp/approve") {
        // An approval with no run waiting on it. Answering a card after the run that
        // raised it has ended must not silently look like it landed.
        reply_result(id, R"({"accepted":false})");
        return false;
    }
    if (method == "lmp/load_model") {
        handle_load_model(id, message, session, log, clock);
        return false;
    }
    if (method == "lmp/unload_model") {
        handle_unload_model(id, session, log, clock);
        return false;
    }

    // --- a run --------------------------------------------------------------
    //
    // Both of these BLOCK for the length of a mission. Anything the user says while one
    // is turning is read by the RunInbox inside it, not here.
    if (method == "lmp/start") {
        return start_mission(id, message, session, inbox, cancel, log, clock);
    }
    if (method == "lmp/message") {
        return continue_session(id, message, session, inbox, cancel, log, clock);
    }
    reply_error(id, "unknown method '" + method +
                        "'. This sidecar speaks the private lmp/* namespace; it is not "
                        "MCP and does not claim to be.");
    return false;
}

} // namespace

// THE ONE QUESTION A FATAL-SIGNAL HANDLER ANSWERS: fault or kill?
//
// The sidecar has died twice with the event log ending mid-turn, no crash report, and
// nothing in the unified log. Those three facts together are the signature of SIGKILL --
// which is what the kernel sends when memory runs out, and which by definition cannot be
// caught, reported, or traced. But they are ALSO what a fault would look like if crash
// reporting were simply not configured for this binary, and the two have opposite fixes.
//
// So: if the process dies of a fault, this fires and the log carries `fatal_signal` with
// a backtrace naming the frame. If the process dies and the log carries NOTHING, it was
// killed from outside and memory is the story. Either way the next crash says which,
// instead of leaving both possible.
//
// async-signal-safety: backtrace_symbols_fd and write are safe; the log's append is not,
// so this writes to the log's fd through the same write(2) the writer uses and does not
// touch its state. Then the default handler is restored and the signal re-raised, so the
// process still dies exactly as it would have -- this observes, it does not rescue.
int g_log_fd = -1;

extern "C" void fatal_signal_handler(int sig) {
    if (g_log_fd >= 0) {
        char line[160];
        const int n = std::snprintf(line, sizeof(line),
                                    "{\"kind\":\"fatal_signal\",\"signal\":%d}\n", sig);
        if (n > 0) {
            const ssize_t w = ::write(g_log_fd, line, static_cast<std::size_t>(n));
            (void)w;
        }
        void* frames[32];
        const int depth = ::backtrace(frames, 32);
        ::backtrace_symbols_fd(frames, depth, g_log_fd);
    }
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

int main() {
    platform::SystemClock clock;
    platform::EventLogWriter log;
    // Chosen, not inherited from the launcher's CWD -- see default_event_log_path.
    const std::string log_path =
        platform::default_event_log_path(std::getenv("LMP_EVENT_LOG"), std::getenv("HOME"));

    // The directory is ours to create; the log rotates within it (4 files, 32 MiB each).
    // Errors are swallowed deliberately -- open() below is the real check, and it reports
    // the actual errno against the actual path rather than a guess made one call earlier.
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(log_path).parent_path(), ec);

    const platform::OpenResult opened = log.open({log_path, 32U * 1024 * 1024, 4});
    if (opened.ok) {
        // Armed as soon as there is somewhere to report to, and before a model is loaded
        // -- the deaths under investigation happen deep in a run, but a fault during load
        // would be just as silent.
        g_log_fd = log.fd_for_signal_handler();
        for (const int sig : {SIGSEGV, SIGBUS, SIGABRT, SIGILL, SIGFPE}) {
            std::signal(sig, fatal_signal_handler);
        }
    }
    if (!opened.ok) {
        // Refused loudly (S13). No silent fallback to "run without a trace".
        std::fprintf(stderr, "lmp: cannot open event log at %s: %s\n", log_path.c_str(),
                     opened.error.c_str());
        return 1;
    }

    model::CancelToken cancel;
    platform::SpscChannel<std::string> inbox(256);
    surface::StdinReader reader(inbox, cancel);
    reader.start(STDIN_FILENO);

    surface::wire::write_line(
        std::string(R"({"jsonrpc":"2.0","method":"lmp/ready","params":)") +
        R"({"protocol_version":")" + protocol::kProtocolVersion + R"("}})");

    // Outlives every run in this process: the conversation, and the weights it is having
    // that conversation with.
    surface::Session session;

    // The starting state, said out loud rather than left to be assumed. A fresh process
    // holds no weights, and a surface that had to infer that from silence would render
    // the same blank thing for "no model" as for "still loading" -- which is precisely
    // how a prompt came to sit on 'Thinking' with nothing behind it.
    notify_model(protocol::modelstate_values::kUnloaded, "");

    std::string message;
    while (!inbox.drained()) {
        if (!inbox.try_pop(message)) {
            std::this_thread::yield();
            continue;
        }
        if (dispatch_one(message, session, inbox, cancel, log, clock)) {
            // A shutdown: either with nothing running, or one that arrived mid-run and
            // was acknowledged there. Acted on here, now that the run has unwound and
            // the model is freed.
            exit_now(log);
        }
    }

    reader.join();
    log.close();
    return 0;
}
