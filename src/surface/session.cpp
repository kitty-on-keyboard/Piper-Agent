#include "src/surface/session.hpp"

#include <chrono>
#include <utility>

#include "src/platform/fs.hpp"
#include "src/surface/mcp_settings.hpp"

namespace lmp::surface {
namespace {

double ms_since(const platform::Clock& clock, platform::MonoTime start) {
    return std::chrono::duration<double, std::milli>(clock.mono() - start).count();
}

} // namespace

ModelLoad load_model(Session& session, const std::string& model_dir,
                     const platform::Clock& clock) {
    const platform::MonoTime started = clock.mono();
    if (model_dir.empty()) {
        return {false, "model_dir is required and empty (S7.5: no defaultable input)", 0.0};
    }

    // Built into locals first. MlxBackend refuses a second load on the same instance
    // (S5.11), so a reload has to be a NEW instance -- and until both halves have
    // succeeded the session must keep the model it already had. Assigning as we went
    // would mean a failed reload leaves the session holding a tokenizer for one
    // checkpoint and weights for another, which generates fluent nonsense rather than
    // failing.
    auto tok = std::make_unique<model::QwenTokenizer>();
    const model::LoadStatus tok_status =
        tok->load(model_dir + "/tokenizer.json", model::Family::Qwen3);
    if (!tok_status.ok) {
        return {false, "tokenizer_refused: " + tok_status.error, ms_since(clock, started)};
    }

    // The old weights are released BEFORE the new ones are read, or the peak is two
    // checkpoints at once: 38 GB on a 48 GB host, which does not degrade, it takes the
    // machine down. This is the same rule as "never run two MLX processes at once",
    // reached from inside one process.
    session.backend.reset();
    session.tok.reset();
    session.model_dir.clear();

    auto backend = std::make_unique<model::MlxBackend>(clock);
    const model::LoadStatus backend_status = backend->load({model_dir, ""});
    if (!backend_status.ok) {
        return {false, "model_load_failed: " + backend_status.error,
                ms_since(clock, started)};
    }

    session.tok = std::move(tok);
    session.backend = std::move(backend);
    session.model_dir = model_dir;
    return {true, {}, ms_since(clock, started)};
}

void unload_model(Session& session) {
    // Journal before ctx (its sink points into ctx), registry before host (the registry's
    // handlers are what keep the MCP connections alive). Both orders are load-bearing and
    // both are the same ones the construction paths use in reverse.
    session.journal.reset();
    session.ctx.reset();
    session.registry.reset();
    session.mcp.reset();
    session.mcp_signature.clear();
    session.backend.reset();
    session.tok.reset();
    session.model_dir.clear();
}

void ensure_registry(Session& session, const std::string& workspace,
                     const std::string& message, platform::EventLogWriter& log,
                     const platform::Clock& clock) {
    std::string mcp_signature;
    const std::vector<tools::McpServerConfig> servers =
        parse_mcp_servers(message, mcp_signature);
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

    connect_mcp_servers(*session.mcp, servers, *session.registry, log, clock);
    session.mcp_signature = mcp_signature;
}

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

std::string load_project_memory(const std::string& workspace) {
    // Bounded by the same constant the writer enforces, so a hand-edited file cannot
    // grow the prompt past what `remember` allows.
    const platform::FileContents f = platform::read_file_whole(
        workspace + "/" + tools::kMemoryFileName, tools::kMemoryMaxBytes);
    return f.status == platform::FsStatus::Ok ? f.bytes : std::string();
}

} // namespace lmp::surface
