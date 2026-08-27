// A diagnostic driver, not a test (S19.3): points McpHost at a REAL MCP server and prints
// what the agent would actually get.
//
//   mcp_host_probe [--trusted] [--call TOOL --args JSON] -- <command> [args...]
//
// The gate tests drive our own mcp_demo_server, which we wrote and which therefore cannot
// surprise us. This exists for the servers we did not write -- the official reference
// servers, and whatever the operator wants to configure -- because "the schema converted,
// the names were usable, the tool answered" is a claim about someone else's software.
//
// Run it against a server BEFORE putting it in settings: it prints the registered names,
// the parameters the grammar will enforce, and whether each call raises an approval card.
//
//   mcp_host_probe -- node .../server-filesystem/dist/index.js /tmp
//
#include <cstdio>
#include <string>
#include <vector>

#include "src/tools/mcp_host.hpp"
#include "src/tools/registry.hpp"

namespace {

using namespace lmp;

const char* type_name(parsephony::ParamType t) {
    switch (t) {
        case parsephony::ParamType::Text: return "text";
        case parsephony::ParamType::Json: return "json";
        case parsephony::ParamType::Number: return "number";
        case parsephony::ParamType::Boolean: return "bool";
        case parsephony::ParamType::Object: return "object";
        case parsephony::ParamType::Array: return "array";
    }
    return "?";
}

int usage() {
    std::fprintf(stderr,
                 "usage: mcp_host_probe [--trusted] [--call TOOL --args JSON] "
                 "-- <command> [args...]\n");
    return 2;
}

// Returns false on a usage error.
bool parse_args(int argc, char** argv, tools::McpServerConfig& cfg, std::string& call_tool,
                std::string& call_args) {
    int i = 1;
    for (; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--") {
            ++i;
            break;
        }
        if (a == "--trusted") {
            cfg.trusted = true;
        } else if (a == "--call" && i + 1 < argc) {
            call_tool = argv[++i];
        } else if (a == "--args" && i + 1 < argc) {
            call_args = argv[++i];
        } else {
            return false;
        }
    }
    if (i >= argc) {
        return false;
    }
    cfg.command = argv[i++];
    for (; i < argc; ++i) {
        cfg.args.emplace_back(argv[i]);
    }
    return true;
}

void print_registered(const tools::Registry& registry) {
    for (const tools::ToolDecl& d : registry.decls()) {
        if (!d.remote) {
            continue;
        }
        std::printf("  %s%s%s%s\n", d.name.c_str(), d.irreversible ? "  [card]" : "",
                    d.mutates_workspace ? "  [write]" : "  [read]",
                    d.needs_execution ? "  [exec]" : "");
        for (const parsephony::ParamSpec& p : d.spec.params) {
            std::printf("      %s: %s%s\n", p.name.c_str(), type_name(p.type),
                        p.required ? " (required)" : "");
        }
    }
}

// Deliberately through the REGISTRY, not the client: this is the path the agent takes,
// including the namespacing and the result mapping.
int call_through_registry(tools::Registry& registry, const std::string& server,
                          const std::string& tool, const std::string& args_json) {
    std::vector<tools::ToolParamValue> params;
    const auto args = nlohmann::json::parse(args_json, nullptr, false);
    if (args.is_object()) {
        for (const auto& [k, v] : args.items()) {
            params.push_back({k, v.is_string() ? v.get<std::string>() : v.dump()});
        }
    }
    // Call by the name the server declared. registered_mcp_tool_name is the
    // registration-time chooser; after connect the short name is already taken, so
    // using it here would invent a namespaced alias that was never registered.
    std::string full = tool;
    if (registry.find(full) == nullptr) {
        full = tools::namespaced_tool_name(server, tool);
    }
    const tools::ToolResult r = registry.execute(full, params, 0);
    std::printf("\ncall %s -> %s\n", full.c_str(), r.ok() ? "ok" : "FAILED");
    std::printf("%s\n", r.summary.c_str());
    return r.ok() ? 0 : 1;
}

tools::WorkspaceContext probe_workspace() {
    tools::WorkspaceContext wctx;
    wctx.root = "/tmp";
    wctx.max_read_bytes = 1U << 20;
    wctx.max_model_read_bytes = 24U << 10;
    wctx.max_result_bytes = 8192;
    wctx.shell_wall_clock_seconds = 60;
    return wctx;
}

} // namespace

int main(int argc, char** argv) {
    tools::McpServerConfig cfg;
    cfg.name = "probe";
    std::string call_tool;
    std::string call_args = "{}";
    if (!parse_args(argc, argv, cfg, call_tool, call_args)) {
        return usage();
    }

    tools::Registry registry(probe_workspace());
    tools::McpHost host;
    const auto report = host.connect_and_register({cfg}, registry);
    if (report.empty()) {
        std::fprintf(stderr, "no report\n");
        return 1;
    }
    const tools::McpServerStatus& st = report[0];
    std::printf("connected: %s\n", st.connected ? "yes" : "no");
    if (!st.error.empty()) {
        std::printf("error: %s\n", st.error.c_str());
        return 1;
    }
    std::printf("registered %zu tool(s)%s\n", st.registered,
                cfg.trusted ? " (trusted: no approval card)"
                            : " (untrusted: every call raises a card)");
    std::printf("instructions %zu chars\n", st.instructions.size());
    for (const std::string& why : st.rejected) {
        std::printf("  REJECTED %s\n", why.c_str());
    }
    print_registered(registry);

    if (!call_tool.empty()) {
        return call_through_registry(registry, cfg.name, call_tool, call_args);
    }
    return 0;
}
