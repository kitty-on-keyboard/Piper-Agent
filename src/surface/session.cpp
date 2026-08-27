#include "src/surface/session.hpp"

#include <algorithm>
#include <cctype>
#include <string_view>

#include <chrono>
#include <utility>

#include "src/model/model_limits.hpp"
#include "src/platform/fs.hpp"
#include "src/surface/mcp_settings.hpp"

namespace lmp::surface {
namespace {

double ms_since(const platform::Clock& clock, platform::MonoTime start) {
    return std::chrono::duration<double, std::milli>(clock.mono() - start).count();
}

} // namespace

ModelLoad load_model(Session& session, const std::string& model_dir,
                     const std::string& draft_model_dir, const platform::Clock& clock) {
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
    session.draft_model_dir.clear();

    auto backend = std::make_unique<model::MlxBackend>(clock);
    model::MlxBackendConfig backend_cfg;
    backend_cfg.model_dir = model_dir;
    backend_cfg.draft_model_dir = draft_model_dir;
    // THE PRODUCT'S VISION POLICY, and the only place it is decided: keep the tower
    // whenever the checkpoint has one.
    //
    // This line is the whole reason `view_image` did not work. `with_vision` defaults to
    // off and nothing outside the tests ever set it, so every real run loaded a seeing
    // checkpoint blind -- the tool ran, reported the picture's cost, and the turn after
    // it died with "the prompt carries an image but the model was loaded without its
    // vision tower". A capability nothing switches on is a capability that does not ship.
    //
    // Unconditional `true` is not the fix: the loader REFUSES vision against a text-only
    // export by design, so asking always would turn every text-only checkpoint into a
    // load failure. Ask what is there, then request exactly that.
    //
    // The cost is 0.92 GB resident, about 6% on top of the 15.13 GB text weights, paid on
    // every run whether or not it looks at anything. That is the right trade here: the
    // load is one-shot per process (S5.11), so a tower skipped at load cannot be added
    // when the first image arrives -- the alternative to paying always is not paying
    // later, it is a full 19 GB reload mid-run.
    backend_cfg.with_vision = model::checkpoint_declares_vision(model_dir);
    const model::LoadStatus backend_status = backend->load(backend_cfg);
    if (!backend_status.ok) {
        return {false, "model_load_failed: " + backend_status.error,
                ms_since(clock, started)};
    }

    session.tok = std::move(tok);
    session.backend = std::move(backend);
    session.model_dir = model_dir;
    session.draft_model_dir = draft_model_dir;
    session.model_max_sequence_tokens =
        static_cast<std::int32_t>(model::load_max_position_embeddings(model_dir));
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
    session.model_max_sequence_tokens = 0;
}

void ensure_registry(Session& session, const std::string& workspace,
                     const std::string& message, platform::EventLogWriter& log,
                     const platform::Clock& clock) {
    std::string settings_signature;
    std::vector<tools::McpServerConfig> from_settings =
        parse_mcp_servers(message, settings_signature);
    // AND the project's own `.mcp.json`. Every other agent on the machine reads it, and
    // `godoer connect` writes it -- so a project configured correctly for Claude Code,
    // Cursor and Gemini CLI reached LM_Pipe with no servers, no event, and a model that
    // burned six turns discovering by trial that the tools its AGENTS.md named were not
    // there. See parse_mcp_json_file for why trust is not inherited from it.
    std::string file_signature;
    std::size_t trusted_ignored = 0;
    std::vector<tools::McpServerConfig> from_file =
        parse_mcp_json_file(workspace, file_signature, trusted_ignored);
    const std::size_t from_file_count = from_file.size();
    const bool had_file = from_file_count > 0 || trusted_ignored > 0;
    const std::vector<tools::McpServerConfig> servers =
        merge_mcp_servers(std::move(from_settings), std::move(from_file));
    // BOTH halves, so the registry is rebuilt when either config moves. Keyed on the two
    // sources rather than the merged list because the merge is lossy: a file server
    // shadowed by a settings one of the same name leaves no trace in the result, and a
    // change to the shadowed entry must still not look like no change at all.
    const std::string mcp_signature = settings_signature + "\x1e" + file_signature;
    // What the model loaded RIGHT NOW can do, not what the last one could. A registry
    // built against a seeing checkpoint offers `view_image`, and reusing it after a
    // reload onto a text-only one would offer a tool whose every result is unusable --
    // so the capability joins workspace and MCP config in what makes a registry stale.
    const bool can_see = session.backend != nullptr && session.backend->can_see();
    if (session.registry != nullptr && session.workspace == workspace &&
        session.mcp_signature == mcp_signature &&
        session.registry->workspace().model_can_see == can_see) {
        return;
    }

    tools::WorkspaceContext wctx;
    wctx.root = workspace;
    wctx.max_read_bytes = 4U << 20;
    wctx.max_model_read_bytes = 24U << 10;
    wctx.max_result_bytes = 8192;
    wctx.max_observation_bytes = tools::kObservationBudgetBytes;
    wctx.spool_dir = workspace + "/.lmp_spool";
    wctx.shell_wall_clock_seconds = 300;
    wctx.model_can_see = can_see;
    // Registry first, then host: the old registry's handlers are what keep the old
    // clients alive, so releasing it first lets those connections go at the same time
    // rather than one reconfiguration later.
    session.registry = std::make_unique<tools::Registry>(std::move(wctx));
    session.mcp = std::make_unique<tools::McpHost>();
    session.workspace = workspace;

    // Before connecting, so the file is on the record even when every server in it fails
    // to start -- and so `trusted_ignored` is visible whether or not the entry connected.
    if (had_file) {
        platform::Event ev;
        ev.kind = "mcp_config_file";
        ev.fields.push_back({"path", workspace + "/.mcp.json"});
        ev.fields.push_back({"servers", std::to_string(from_file_count)});
        ev.fields.push_back({"trusted_ignored", std::to_string(trusted_ignored)});
        if (trusted_ignored > 0) {
            ev.fields.push_back(
                {"why", "trusted is never inherited from a file that arrives with a "
                        "checkout; name the server in lmPipe.mcpServers to vouch for it"});
        }
        log.append(ev, clock);
    }
    connect_mcp_servers(*session.mcp, servers, *session.registry, log, clock);
    session.mcp_signature = mcp_signature;
}

std::string load_project_instructions(const platform::WorkspaceFs& workspace) {
    static constexpr const char* kNames[] = {"AGENTS.md", "CLAUDE.md", ".cursorrules"};
    for (const char* name : kNames) {
        const platform::FileContents f =
            workspace.read_file_whole(name, 64U * 1024);
        if (f.status == platform::FsStatus::Ok && !f.bytes.empty()) {
            return "(from " + std::string(name) + ")\n\n" + f.bytes;
        }
    }
    return {};
}

std::string load_project_memory(const platform::WorkspaceFs& workspace) {
    // Bounded by the same constant the writer enforces, so a hand-edited file cannot
    // grow the prompt past what `remember` allows.
    const platform::FileContents f =
        workspace.read_file_whole(tools::kMemoryFileName, tools::kMemoryMaxBytes);
    return f.status == platform::FsStatus::Ok ? f.bytes : std::string();
}

namespace {

// A tool answers to `n` if it is registered under that exact name, or under an MCP
// namespace ending in it. `mcp__godoer__godot_guide` answers to `godot_guide`, which is
// what the conventions call it and what the model will write.
bool registry_answers_to(const tools::Registry& registry, const std::string& n) {
    const std::string suffix = "__" + n;
    for (const parsephony::ToolSpec& s : registry.guard_specs()) {
        if (s.name == n) {
            return true;
        }
        if (s.name.size() > suffix.size() &&
            s.name.compare(s.name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return true;
        }
    }
    return false;
}

// lower_snake_case, at least one underscore, long enough not to be a stray word. The
// underscore is doing most of the work: it separates `godot_build_scene` from `capture`
// and `plan` without needing a list of English words.
bool looks_like_a_tool_name(std::string_view s) {
    if (s.size() < 5 || s.size() > 64 || s.find('_') == std::string_view::npos) {
        return false;
    }
    if (s.front() < 'a' || s.front() > 'z' || s.back() == '_') {
        return false;
    }
    for (const char c : s) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
        if (!ok) {
            return false;
        }
    }
    return true;
}

// Within a short reach behind the name, does the prose say this is something to CALL?
// Without this every snake_case identifier a README happens to quote is a candidate, and
// the note is only worth its tokens if it is about tools.
bool called_like_a_tool(const std::string& text, std::size_t tick) {
    static constexpr std::string_view kVerbs[] = {"call", "run", "use", "tool", "invoke"};
    const std::size_t reach = 64;
    const std::size_t from = tick > reach ? tick - reach : 0;
    std::string window = text.substr(from, tick - from);
    for (char& c : window) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    for (const std::string_view v : kVerbs) {
        if (window.find(v) != std::string::npos) {
            return true;
        }
    }
    return false;
}

} // namespace

std::vector<std::string> unknown_tool_names(const std::string& conventions,
                                            const tools::Registry& registry) {
    // Bounded so a file full of snake_case cannot grow the system prompt without limit.
    // The cap is on what is REPORTED, not on what is scanned: eight names is already more
    // than enough to tell the model a whole toolset is absent.
    static constexpr std::size_t kMaxReported = 8;
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < conventions.size() && out.size() < kMaxReported) {
        const std::size_t open = conventions.find('`', i);
        if (open == std::string::npos) {
            break;
        }
        const std::size_t close = conventions.find('`', open + 1);
        if (close == std::string::npos) {
            break;
        }
        const std::string name = conventions.substr(open + 1, close - open - 1);
        i = close + 1;
        if (!looks_like_a_tool_name(name) || !called_like_a_tool(conventions, open)) {
            continue;
        }
        if (registry_answers_to(registry, name)) {
            continue;
        }
        if (std::find(out.begin(), out.end(), name) == out.end()) {
            out.push_back(name);
        }
    }
    return out;
}

std::string unknown_tools_note(const std::vector<std::string>& names) {
    if (names.empty()) {
        return {};
    }
    std::string list;
    for (const std::string& n : names) {
        list += list.empty() ? "" : ", ";
        list += "`" + n + "`";
    }
    // Says what to DO, not merely what is missing. The failure this exists for was not a
    // model that lacked the fact -- it was a model that spent six turns establishing it by
    // trial and then had no move left. Naming `ask_user` is the whole point: an absent
    // toolchain is the operator's problem to solve, and a run that asks in turn 1 costs
    // one turn where the same run stalling costs fourteen.
    return "\n\n# Tools named above that DO NOT EXIST in this run\n\n" + list +
           " appear in the conventions above, but no tool by those names is available to "
           "you -- they are almost certainly from an MCP server that is not connected. "
           "Nothing you do can call them. Do not plan around them, do not shell out "
           "looking for them, and do not tick a checklist item that needed one. If the "
           "work genuinely requires them, say so plainly and call `ask_user` so the human "
           "can connect the server; otherwise do the work with the tools you do have.";
}

} // namespace lmp::surface
