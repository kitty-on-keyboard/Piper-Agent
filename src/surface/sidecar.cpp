// The sidecar: one process, one model, the loop (spec S4.1, S12.3).
//
// The extension spawns exactly one of these. It owns the model and the loop, speaks
// lmp/* over stdio, and exits on stdin EOF so a dead parent never orphans a process
// holding ~19 GB of weights.

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>

#include "src/loop/agent.hpp"
#include "src/model/mlx_backend.hpp"
#include "src/platform/event_log.hpp"
#include "src/platform/fs.hpp"
#include "src/surface/mcp_settings.hpp"
#include "src/surface/protocol_generated.hpp"
#include "src/surface/transport.hpp"
#include "src/tools/mcp_host.hpp"

namespace {

using namespace lmp;

// Two threads reach this now: the run thread for turn/verification/perf notifications, and
// the token streamer for per-token text (S5.7 streaming). stdio locks each FILE* call on
// its own, which is not enough -- the payload and its newline are two calls, so without a
// lock here another thread's line can land between them and both messages are corrupt.
// The lock is held across the flush deliberately: a half-written line must never be
// visible to the extension's line-oriented reader.
std::mutex& stdout_lock() {
    static std::mutex m;
    return m;
}

void write_line(const std::string& s) {
    const std::lock_guard<std::mutex> guard(stdout_lock());
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

// Every outbound notification goes through the GENERATED serializer for its type, so
// a schema change that the sidecar has not caught up with is a compile error rather
// than a field the extension silently reads as its default (S4.4).
template <class N>
void notify(const N& n) {
    std::string out;
    out += R"({"jsonrpc":"2.0","method":")";
    out += N::kMethod;
    out += R"(","params":)";
    out += protocol::to_json(n);
    out += "}";
    write_line(out);
}

std::string json_escape(std::string_view in) {
    std::string out;
    (void)platform::append_json_string(out, in);
    return out;
}

void reply_result(const std::string& id, const std::string& body) {
    write_line(R"({"jsonrpc":"2.0","id":")" + id + R"(","result":)" + body + "}");
}

void reply_error(const std::string& id, const std::string& message) {
    write_line(R"({"jsonrpc":"2.0","id":")" + id + R"(","error":{"code":-32000,)" +
               R"("message":)" + json_escape(message) + "}}");
}

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
        n.tool_args = t.tool_params.empty() ? std::string() : t.tool_params[0].value;
        n.tool_status = std::string(tools::to_string(t.tool_result.status));
        n.summary = t.tool_result.summary;
        n.duration_ms = duration_ms;
        notify(n);
    };
    obs.on_verification = [id](const context::VerificationRecord& v) {
        protocol::VerificationNotification n;
        n.run_id = id;
        n.contract = v.contract;
        n.passed = v.passed;
        n.falsifiable = v.falsifiable;
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
    obs.on_perf = [id](const model::GenResult& g, std::size_t used, std::size_t max) {
        protocol::PerfNotification n;
        n.run_id = id;
        n.sample.ttft_ms = g.ttft_ms;
        n.sample.prefill_tok_per_s = g.prefill_tok_per_s;
        n.sample.decode_tok_per_s = g.decode_tok_per_s;
        n.sample.context_used = static_cast<std::int64_t>(used);
        n.sample.context_max = static_cast<std::int64_t>(max);
        n.sample.tokens_generated = g.tokens_generated;
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
    RunInbox(platform::SpscChannel<std::string>& inbox, std::string run_id)
        : inbox_(inbox), run_id_(std::move(run_id)) {}

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
        req.risk = loop::risk_score(hint);
        req.capabilities = chips_of(hint);
        notify(req);

        bool answer = false;
        std::string msg;
        while (true) {
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
    tools::EditOutcome apply_edit(const std::string& path, const std::string& content) {
        const std::string request_id = run_id_ + "/edit/" + std::to_string(++seq_);
        protocol::EditNotification req;
        req.request_id = request_id;
        req.run_id = run_id_;
        req.path = path;
        req.new_content = content;
        notify(req);

        std::string msg;
        while (true) {
            if (inbox_.drained()) {
                return {false, "the editor is gone"};
            }
            if (!inbox_.try_pop(msg)) {
                std::this_thread::yield();
                continue;
            }
            if (handle_edit_reply(msg, request_id)) {
                return edit_answer_;
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
        if (surface::method_of(msg) == "lmp/edit_applied") {
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

    platform::SpscChannel<std::string>& inbox_;
    std::string run_id_;
    std::vector<std::string> steering_;
    std::uint64_t seq_ = 0;
    tools::EditOutcome edit_answer_;
    bool shutdown_ = false;
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

    config.context_budget_tokens = static_cast<std::int32_t>(surface::double_field(
        message, "context_budget_tokens", config.context_budget_tokens));
    config.budget.max_iterations = static_cast<int>(
        surface::double_field(message, "max_iterations", config.budget.max_iterations));
    config.budget.wall_clock_seconds = static_cast<int>(surface::double_field(
        message, "wall_clock_seconds", config.budget.wall_clock_seconds));

    return apply_autonomy(id, message, config);
}

// The repo's own conventions, in the order the surrounding ecosystem settled on. First
// file found wins; they are alternatives, not layers, and concatenating them would let a
// stale `.cursorrules` contradict a current AGENTS.md with no way to tell which won.
//
// Loaded once into the STABLE part of the prompt: conventions do not change mid-run, so
// they cost one prefill for the whole run.
std::string load_project_instructions(const std::string& workspace) {
    static constexpr const char* kNames[] = {"AGENTS.md", "CLAUDE.md", ".cursorrules"};
    for (const char* name : kNames) {
        const platform::FileContents f =
            platform::read_file_whole(workspace + "/" + name, 64U * 1024);
        if (f.status == platform::FsStatus::Ok && !f.bytes.empty()) {
            return "(from " + std::string(name) + ")\n\n" + f.bytes;
        }
    }
    return {};
}

// What the agent told itself, last time it was here.
//
// The counterpart to load_project_instructions above: same stable-prompt slot, same
// once-per-session cost, opposite provenance. Bounded by the same constant the writer
// enforces, so a hand-edited file cannot grow the prompt past what `remember` allows.
std::string load_project_memory(const std::string& workspace) {
    const platform::FileContents f = platform::read_file_whole(
        workspace + "/" + tools::kMemoryFileName, tools::kMemoryMaxBytes);
    return f.status == platform::FsStatus::Ok ? f.bytes : std::string();
}

// What survives between missions.
//
// v1 of this file built the tokenizer, the weights, the registry and the context store
// inside one function and dropped all four on the way out, so every mission was a fresh
// one-shot: no history to follow up on, and a ~19 GB reload to ask a second question.
// The agent could be launched and could be aborted, and that was the entire vocabulary.
//
// The context store is the part that makes a conversation; keeping the weights loaded
// alongside it is what makes a follow-up cost a prefill instead of a minute.
struct Session {
    std::string model_dir;
    std::string workspace;
    std::unique_ptr<model::QwenTokenizer> tok;
    std::unique_ptr<model::MlxBackend> backend;
    std::unique_ptr<tools::Registry> registry;
    // The handlers this installs into the registry share ownership of their clients
    // (McpHost::Connection), so neither destruction order here is a use-after-free --
    // that WAS true of an earlier raw-pointer version and ASan caught it. Kept adjacent
    // to the registry because they are one unit: the registry holds the tools, this holds
    // the processes behind them.
    std::unique_ptr<tools::McpHost> mcp;
    // What the current registry's remote tools were built from. The Registry has no
    // "unregister", so a changed server list means a new Registry, not a patched one.
    std::string mcp_signature;
    std::unique_ptr<context::ContextStore> ctx;
    loop::AgentConfig config;
    // Whether this client can answer lmp/edit. Advertised at lmp/start rather than
    // assumed: the extension can, agent_eval.py cannot, and guessing wrong either wedges
    // an unattended run on a reply that never comes or writes behind the editor's back.
    bool client_applies_edits = false;
};

// Turns the loop once over an EXISTING session. Returns true when the run should be the
// process's last. The caller has already replied to the request that triggered it, since
// this blocks for as long as the mission takes.
bool run_loop(const std::string& run_id, Session& session,
              platform::SpscChannel<std::string>& inbox, model::CancelToken& cancel,
              platform::EventLogWriter& log, const platform::Clock& clock) {
    loop::Agent agent(*session.tok, *session.backend, *session.registry, *session.ctx, log,
                      clock, session.config);
    agent.set_observer(make_observer(run_id));

    RunInbox run_inbox(inbox, run_id);
    agent.set_approver([&run_inbox](const std::string& tool, const std::string& command,
                                    const std::string& preview,
                                    const tools::RiskHint& hint) {
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
            [&run_inbox](const std::string& path, const std::string& content) {
                return run_inbox.apply_edit(path, content);
            });
    } else {
        session.registry->set_edit_sink(nullptr);
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
    end.self_declared = report.self_declared;
    notify(end);
    return run_inbox.shutdown_requested();
}

// Builds the tool registry for this run, and connects the configured MCP servers into it.
//
// Rebuilt rather than patched when the server list changes: the Registry has no
// unregister, so the alternative is remote tools from a previous run outliving the
// settings that authorised them.
void ensure_registry(Session& session, const std::string& workspace,
                     const std::string& message, platform::EventLogWriter& log,
                     const platform::Clock& clock) {
    std::string mcp_signature;
    const std::vector<tools::McpServerConfig> servers =
        surface::parse_mcp_servers(message, mcp_signature);
    if (session.registry != nullptr && session.workspace == workspace &&
        session.mcp_signature == mcp_signature) {
        return;
    }

    tools::WorkspaceContext wctx;
    wctx.root = workspace;
    wctx.max_read_bytes = 4U << 20;
    wctx.max_model_read_bytes = 24U << 10;
    wctx.max_result_bytes = 8192;
    wctx.spool_dir = workspace + "/.lmp_spool";
    wctx.shell_wall_clock_seconds = 300;
    // Registry first, then host: the old registry's handlers are what keep the old
    // clients alive, so releasing it first lets those connections go at the same time
    // rather than one reconfiguration later.
    session.registry = std::make_unique<tools::Registry>(std::move(wctx));
    session.mcp = std::make_unique<tools::McpHost>();
    session.workspace = workspace;

    surface::connect_mcp_servers(*session.mcp, servers, *session.registry, log, clock);
    session.mcp_signature = mcp_signature;
}

// A fresh mission. Reuses the loaded weights when the model has not changed -- the
// tokenizer and the checkpoint are the expensive part and neither depends on the mission
// -- but the context store is rebuilt, because `lmp/start` MEANS start over. Continuing a
// conversation is what lmp/message is for.
bool start_mission(const std::string& id, const std::string& message, Session& session,
                   platform::SpscChannel<std::string>& inbox, model::CancelToken& cancel,
                   platform::EventLogWriter& log, const platform::Clock& clock) {
    const std::string mission = surface::string_field(message, "mission");
    const std::string model_dir = surface::string_field(message, "model_dir");
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

    if (session.tok == nullptr || session.model_dir != model_dir) {
        auto tok = std::make_unique<model::QwenTokenizer>();
        const model::LoadStatus tok_status =
            tok->load(model_dir + "/tokenizer.json", model::Family::Qwen3);
        if (!tok_status.ok) {
            end_run(id, "tokenizer_refused: " + tok_status.error);
            return false;
        }
        auto backend = std::make_unique<model::MlxBackend>(clock);
        const model::LoadStatus backend_status = backend->load({model_dir, ""});
        if (!backend_status.ok) {
            end_run(id, "model_load_failed: " + backend_status.error);
            return false;
        }
        // Assigned only once BOTH succeeded, so a failed reload cannot leave the session
        // holding a tokenizer for one checkpoint and weights for another.
        session.tok = std::move(tok);
        session.backend = std::move(backend);
        session.model_dir = model_dir;
    }

    ensure_registry(session, workspace, message, log, clock);

    session.ctx = std::make_unique<context::ContextStore>(mission);
    // A little above the widest single result the tool layer can produce, so the door
    // catches a tool that forgot to bound itself without firing on a legitimate one.
    session.ctx->set_observation_budget(32U << 10);
    session.ctx->set_project_instructions(load_project_instructions(workspace));
    session.ctx->set_project_memory(load_project_memory(workspace));
    // Empty keeps the built-in persona; the editor sends the one it holds for this mode.
    session.ctx->set_persona(surface::string_field(message, "system_prompt"));
    session.client_applies_edits = surface::bool_field(message, "applies_edits");

    return run_loop(id, session, inbox, cancel, log, clock);
}

// A follow-up: the same context, one more user turn, another pass of the loop.
//
// This is the idle half of lmp/message. The in-flight half lives in RunInbox and never
// reaches here -- while a run is turning, the main loop is inside run_loop and is not
// reading the inbox at all, so a message that arrives HERE is by construction one that
// arrived with nothing running.
bool continue_session(const std::string& id, const std::string& message, Session& session,
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
    session.ctx->add_user_message(text);
    return run_loop(id, session, inbox, cancel, log, clock);
}

} // namespace

int main() {
    platform::SystemClock clock;
    platform::EventLogWriter log;
    const platform::OpenResult opened =
        log.open({"lmp_events.jsonl", 32U * 1024 * 1024, 4});
    if (!opened.ok) {
        // Refused loudly (S13). No silent fallback to "run without a trace" -- the log
        // IS the UI feed, the debugging trace and the replay input.
        std::fprintf(stderr, "lmp: cannot open event log: %s\n", opened.error.c_str());
        return 1;
    }

    model::CancelToken cancel;
    platform::SpscChannel<std::string> inbox(256);
    surface::StdinReader reader(inbox, cancel);
    reader.start(STDIN_FILENO);

    write_line(std::string(R"({"jsonrpc":"2.0","method":"lmp/ready","params":)") +
               R"({"protocol_version":")" + protocol::kProtocolVersion + R"("}})");

    // Outlives every run in this process: the conversation, and the weights it is having
    // that conversation with.
    Session session;

    std::string message;
    while (!inbox.drained()) {
        if (!inbox.try_pop(message)) {
            std::this_thread::yield();
            continue;
        }
        const std::string method = surface::method_of(message);
        const std::string id = surface::string_field(message, "id");

        if (method == "lmp/shutdown") {
            reply_result(id, R"({"ok":true})");
            exit_now(log);
        }
        if (method == "lmp/cancel") {
            // The reader thread already set the token the moment the message was
            // framed -- this is only the acknowledgement (S4.3).
            reply_result(id, R"({"accepted":true})");
            continue;
        }
        if (method == "lmp/approve") {
            // An approval with no run waiting on it. Answering a card after the run
            // that raised it has ended must not silently look like it landed.
            reply_result(id, R"({"accepted":false})");
            continue;
        }

        // --- a run --------------------------------------------------------
        //
        // Both of these BLOCK for the length of a mission. Anything the user says while
        // one is turning is read by the RunInbox inside it, not here.
        bool last_run = false;
        if (method == "lmp/start") {
            last_run = start_mission(id, message, session, inbox, cancel, log, clock);
        } else if (method == "lmp/message") {
            last_run = continue_session(id, message, session, inbox, cancel, log, clock);
        } else {
            reply_error(id, "unknown method '" + method +
                                "'. This sidecar speaks the private lmp/* namespace; it "
                                "is not MCP and does not claim to be.");
            continue;
        }
        if (last_run) {
            // A shutdown arrived mid-run: acknowledged there, acted on here, now that
            // the run has unwound and the model is freed.
            exit_now(log);
        }
    }

    reader.join();
    log.close();
    return 0;
}
