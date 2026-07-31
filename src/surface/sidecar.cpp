// The sidecar: one process, one model, the loop (spec S4.1, S12.3).
//
// The extension spawns exactly one of these. It owns the model and the loop, speaks
// lmp/* over stdio, and exits on stdin EOF so a dead parent never orphans a process
// holding ~19 GB of weights.

#include <unistd.h>

#include <cstdio>
#include <memory>
#include <string>

#include "src/loop/agent.hpp"
#include "src/model/mlx_backend.hpp"
#include "src/platform/event_log.hpp"
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

void reply_error(const std::string& id, const std::string& message) {
    write_line(R"({"jsonrpc":"2.0","id":")" + id + R"(","error":{"code":-32000,)" +
               R"("message":)" + json_escape(message) + "}}");
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
            write_line(R"({"jsonrpc":"2.0","id":")" + id + R"(","result":{"ok":true}})");
            break;
        }
        if (method == "lmp/cancel") {
            // The reader thread already set the token the moment the message was
            // framed -- this is only the acknowledgement (S4.3).
            write_line(R"({"jsonrpc":"2.0","id":")" + id +
                       R"(","result":{"accepted":true}})");
            continue;
        }
        if (method != "lmp/start") {
            reply_error(id, "unknown method '" + method +
                                "'. This sidecar speaks the private lmp/* namespace; it "
                                "is not MCP and does not claim to be.");
            continue;
        }

        // --- a run --------------------------------------------------------
        const std::string mission = surface::string_field(message, "mission");
        const std::string model_dir = surface::string_field(message, "model_dir");
        const std::string workspace = surface::string_field(message, "workspace_root");
        if (mission.empty() || model_dir.empty() || workspace.empty()) {
            reply_error(id, "mission, model_dir and workspace_root are all required and "
                            "none may be empty (S7.5: no defaultable security input)");
            continue;
        }
        write_line(R"({"jsonrpc":"2.0","id":")" + id + R"(","result":{"run_id":")" + id +
                   R"("}})");

        model::QwenTokenizer tok;
        const model::LoadStatus tok_status =
            tok.load(model_dir + "/tokenizer.json", model::Family::Qwen3);
        if (!tok_status.ok) {
            protocol::RunEndNotification end;
            end.run_id = id;
            end.termination_reason = "tokenizer_refused: " + tok_status.error;
            end.iterations = 0;
            end.completed = false;
            notify(end);
            continue;
        }

        auto backend = std::make_unique<model::MlxBackend>(clock);
        const model::LoadStatus backend_status = backend->load({model_dir, ""});
        if (!backend_status.ok) {
            protocol::RunEndNotification end;
            end.run_id = id;
            end.termination_reason = "model_load_failed: " + backend_status.error;
            end.iterations = 0;
            end.completed = false;
            notify(end);
            continue;
        }

        tools::WorkspaceContext wctx;
        wctx.root = workspace;
        wctx.max_read_bytes = 4U << 20;
        wctx.max_result_bytes = 8192;
        wctx.spool_dir = workspace + "/.lmp_spool";
        wctx.shell_wall_clock_seconds = 300;
        tools::Registry registry(std::move(wctx));

        context::ContextStore ctx(mission);
        loop::AgentConfig config;
        loop::Agent agent(tok, *backend, registry, ctx, log, clock, config);

        // Every UI notification is serialized by the GENERATED code for its type.
        loop::Observer obs;
        obs.on_token = [&id](const std::string& channel, const std::string& text) {
            protocol::TokenNotification n;
            n.run_id = id;
            n.channel = channel;
            n.text = text;
            notify(n);
        };
        obs.on_turn = [&id](const loop::TurnResult& t, double duration_ms) {
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
        obs.on_verification = [&id](const context::VerificationRecord& v) {
            protocol::VerificationNotification n;
            n.run_id = id;
            n.contract = v.contract;
            n.passed = v.passed;
            n.falsifiable = v.falsifiable;
            n.detail = v.detail;
            notify(n);
        };
        obs.on_perf = [&id](const model::GenResult& g, std::size_t used, std::size_t max) {
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
        agent.set_observer(std::move(obs));
        // Without a UI approver wired, escalation denies. Deny-by-default is the only
        // honest behaviour when there is nobody to ask (S7.2).
        agent.set_approver(nullptr);

        cancel.reset();
        const loop::RunReport report = agent.run(cancel);
        protocol::RunEndNotification end;
        end.run_id = id;
        end.termination_reason = report.termination_reason;
        end.iterations = report.iterations;
        end.completed = report.completed;
        notify(end);
    }

    reader.join();
    log.close();
    return 0;
}
