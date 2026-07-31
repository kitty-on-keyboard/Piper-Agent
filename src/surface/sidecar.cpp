// The sidecar: one process, one model, the loop (spec S4.1, S12.3).
//
// The extension spawns exactly one of these. It owns the model and the loop, speaks
// lmp/* over stdio, and exits on stdin EOF so a dead parent never orphans a process
// holding ~19 GB of weights.

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

#include "src/loop/agent.hpp"
#include "src/model/mlx_backend.hpp"
#include "src/platform/event_log.hpp"
#include "src/platform/fs.hpp"
#include "src/surface/protocol_generated.hpp"
#include "src/surface/transport.hpp"

namespace {

using namespace lmp;

void write_line(const std::string& s) {
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

// The approval round-trip (S7.2).
//
// An escalation BLOCKS the run until the human answers, and the wait drains the inbox on
// the run's own thread -- which is already the channel's only consumer. Handing the wait
// to a second consumer would break the SPSC invariant to no purpose: this thread has
// nothing else to do until the answer arrives.
//
// Deny-by-default survives intact. An `approved: true` carrying the request_id of the
// card still on screen is the ONLY path that returns true; a denial, a stale card, a
// cancel, a shutdown, a malformed reply and a dead parent all return false.
class ApprovalBridge {
  public:
    ApprovalBridge(platform::SpscChannel<std::string>& inbox, std::string run_id)
        : inbox_(inbox), run_id_(std::move(run_id)) {}

    bool ask(const std::string& tool, const std::string& preview,
             const tools::RiskHint& hint) {
        const std::string request_id = run_id_ + "/" + std::to_string(++seq_);
        protocol::ApprovalRequestNotification req;
        req.request_id = request_id;
        req.run_id = run_id_;
        req.tool_name = tool;
        req.preview = preview;
        req.risk = loop::risk_score(hint);
        req.capabilities = chips_of(hint);
        notify(req);
        return wait_for_answer(request_id);
    }

    // True when a shutdown arrived while a card was on screen: acknowledged there, but
    // only actionable once the run has unwound.
    [[nodiscard]] bool shutdown_requested() const noexcept { return shutdown_; }

  private:
    bool wait_for_answer(const std::string& request_id) {
        std::string reply;
        while (true) {
            if (inbox_.drained()) {
                // The parent is gone, so there is nobody left to ask. Deny, rather than
                // block forever on a dead pipe holding 19 GB of weights.
                return false;
            }
            if (!inbox_.try_pop(reply)) {
                std::this_thread::yield();
                continue;
            }
            const std::string method = surface::method_of(reply);
            const std::string id = surface::string_field(reply, "id");

            if (method == "lmp/approve" &&
                surface::string_field(reply, "request_id") == request_id) {
                reply_result(id, R"({"accepted":true})");
                return surface::bool_field(reply, "approved");
            }
            if (method == "lmp/approve") {
                // An answer to a card this run has already moved past. Say so rather
                // than letting it decide the CURRENT call -- an approval is for one
                // specific command, not for a position in a queue.
                reply_result(id, R"({"accepted":false})");
                continue;
            }
            if (method == "lmp/cancel") {
                // The reader thread set the token when it framed the message; this is
                // only the acknowledgement. Deny, and the loop notices the token.
                reply_result(id, R"({"accepted":true})");
                return false;
            }
            if (method == "lmp/shutdown") {
                shutdown_ = true;
                reply_result(id, R"({"ok":true})");
                return false;
            }
            reply_error(id, "a run is in flight and is waiting on approval '" +
                                request_id + "'; '" + method +
                                "' cannot be serviced until that run ends");
        }
    }

    platform::SpscChannel<std::string>& inbox_;
    std::string run_id_;
    std::uint64_t seq_ = 0;
    bool shutdown_ = false;
};

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
    return true;
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

// One run, start to finish. Returns true when the run should be the process's last.
bool run_mission(const std::string& id, const std::string& message,
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

    model::QwenTokenizer tok;
    const model::LoadStatus tok_status =
        tok.load(model_dir + "/tokenizer.json", model::Family::Qwen3);
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

    tools::WorkspaceContext wctx;
    wctx.root = workspace;
    wctx.max_read_bytes = 4U << 20;
    wctx.max_result_bytes = 8192;
    wctx.spool_dir = workspace + "/.lmp_spool";
    wctx.shell_wall_clock_seconds = 300;
    tools::Registry registry(std::move(wctx));

    context::ContextStore ctx(mission);
    ctx.set_project_instructions(load_project_instructions(workspace));
    loop::AgentConfig config;
    if (!apply_settings(id, message, config)) {
        return false;
    }
    loop::Agent agent(tok, *backend, registry, ctx, log, clock, config);

    agent.set_observer(make_observer(id));
    ApprovalBridge approvals(inbox, id);
    agent.set_approver([&approvals](const std::string& tool, const std::string& preview,
                                    const tools::RiskHint& hint) {
        return approvals.ask(tool, preview, hint);
    });

    cancel.reset();
    const loop::RunReport report = agent.run(cancel);
    protocol::RunEndNotification end;
    end.run_id = id;
    end.termination_reason = report.termination_reason;
    end.iterations = report.iterations;
    end.completed = report.completed;
    notify(end);
    return approvals.shutdown_requested();
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
        if (method != "lmp/start") {
            reply_error(id, "unknown method '" + method +
                                "'. This sidecar speaks the private lmp/* namespace; it "
                                "is not MCP and does not claim to be.");
            continue;
        }

        // --- a run --------------------------------------------------------
        if (run_mission(id, message, inbox, cancel, log, clock)) {
            // A shutdown arrived while a card was on screen: acknowledged there, acted
            // on here, now that the run has unwound and the model is freed.
            exit_now(log);
        }
    }

    reader.join();
    log.close();
    return 0;
}
