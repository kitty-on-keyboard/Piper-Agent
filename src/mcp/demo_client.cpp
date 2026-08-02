// mcp_probe -- connect to any MCP server over stdio and report what it offers.
//
//   mcp_probe [--call TOOL --args JSON] [--read URI] [--timeout MS] -- <command> [args...]
//
// Written to be pointed at servers we did not write. The cook-off's central failure was
// seven clients that only ever spoke to their own mock; this one exists to be run
// against npx @modelcontextprotocol/server-everything and anything else in the wild.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/mcp/client.hpp"

namespace {

using namespace lmp::mcp;

void print_usage() {
    std::fprintf(stderr,
                 "usage: mcp_probe [--call TOOL] [--args JSON] [--read URI] [--prompt NAME]\n"
                 "                 [--timeout MS] [--quiet] -- <server-command> [args...]\n");
}

std::string first_text(const nlohmann::json& content) {
    if (!content.is_array()) {
        return {};
    }
    for (const auto& block : content) {
        if (block.is_object() && block.value("type", "") == "text") {
            return block.value("text", "");
        }
    }
    return {};
}

struct Args {
    std::string call_tool;
    std::string call_args = "{}";
    std::string read_uri;
    std::string prompt_name;
    int timeout_ms = 30000;
    bool quiet = false;
    std::vector<std::string> command;
    bool ok = true;
    bool help = false;
};

Args parse_args(int argc, char** argv) {
    Args a;
    bool after_ddash = false;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (after_ddash) {
            a.command.push_back(s);
        } else if (s == "--") {
            after_ddash = true;
        } else if (s == "--call" && i + 1 < argc) {
            a.call_tool = argv[++i];
        } else if (s == "--args" && i + 1 < argc) {
            a.call_args = argv[++i];
        } else if (s == "--read" && i + 1 < argc) {
            a.read_uri = argv[++i];
        } else if (s == "--prompt" && i + 1 < argc) {
            a.prompt_name = argv[++i];
        } else if (s == "--timeout" && i + 1 < argc) {
            a.timeout_ms = std::atoi(argv[++i]);
        } else if (s == "--quiet") {
            a.quiet = true;
        } else if (s == "-h" || s == "--help") {
            a.help = true;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", s.c_str());
            a.ok = false;
        }
    }
    if (a.command.empty() && !a.help) {
        a.ok = false;
    }
    return a;
}

void describe(Client& client, const ServerInfo& server) {
    std::printf("server:    %s %s\n", server.name.c_str(), server.version.c_str());
    std::printf("protocol:  %s\n", server.protocol_version.c_str());
    std::printf("caps:      %s\n", server.capabilities.dump().c_str());
    if (!server.instructions.empty()) {
        std::printf("hint:      %s\n", server.instructions.c_str());
    }
    if (server.supports_tools()) {
        const std::vector<Tool> tools = client.list_tools();
        std::printf("\ntools (%zu):\n", tools.size());
        for (const auto& t : tools) {
            std::printf("  %-28s %s\n", t.name.c_str(), t.description.c_str());
        }
    }
    if (server.supports_resources()) {
        const std::vector<Resource> res = client.list_resources();
        std::printf("\nresources (%zu):\n", res.size());
        for (const auto& r : res) {
            std::printf("  %-40s %s\n", r.uri.c_str(), r.name.c_str());
        }
    }
    if (server.supports_prompts()) {
        const std::vector<Prompt> prompts = client.list_prompts();
        std::printf("\nprompts (%zu):\n", prompts.size());
        for (const auto& p : prompts) {
            std::printf("  %-28s %s\n", p.name.c_str(), p.description.c_str());
        }
    }
}

bool run_tool_call(Client& client, const Args& a) {
    const nlohmann::json args =
        nlohmann::json::parse(a.call_args, nullptr, /*allow_exceptions=*/false);
    if (args.is_discarded()) {
        std::fprintf(stderr, "--args is not valid JSON\n");
        return false;
    }
    std::printf("\ncalling %s %s\n", a.call_tool.c_str(), args.dump().c_str());
    const ToolResult r = client.call_tool(
        a.call_tool, args,
        [](double progress, std::optional<double> total, std::string_view message) {
            std::printf("  progress %.0f/%s %.*s\n", progress,
                        total ? std::to_string(*total).c_str() : "?",
                        static_cast<int>(message.size()), message.data());
        });
    std::printf("  isError: %s\n", r.is_error ? "true" : "false");
    std::printf("  text:    %s\n", first_text(r.content).c_str());
    if (r.structured.has_value()) {
        std::printf("  struct:  %s\n", r.structured->dump().c_str());
    }
    return true;
}

void run_extras(Client& client, const Args& a) {
    if (!a.read_uri.empty()) {
        const std::vector<ResourceContents> contents = client.read_resource(a.read_uri);
        std::printf("\nresource %s -> %zu part(s)\n", a.read_uri.c_str(), contents.size());
        for (const auto& c : contents) {
            std::printf("  [%s] %s\n", c.mime_type.c_str(), c.text.value_or("<binary>").c_str());
        }
    }
    if (!a.prompt_name.empty()) {
        const std::vector<PromptMessage> msgs =
            client.get_prompt(a.prompt_name, nlohmann::json::object());
        std::printf("\nprompt %s -> %zu message(s)\n", a.prompt_name.c_str(), msgs.size());
        for (const auto& m : msgs) {
            std::printf("  %s: %s\n", m.role.c_str(), m.content.dump().c_str());
        }
    }
}

void install_diagnostics(Client& client, bool quiet) {
    // The server's stderr is where a failing server explains itself, so it is surfaced
    // rather than swallowed -- but prefixed, so it can never be mistaken for protocol.
    client.on_server_stderr([quiet](std::string_view chunk) {
        if (!quiet) {
            std::fprintf(stderr, "[server] %.*s", static_cast<int>(chunk.size()), chunk.data());
        }
    });
    client.on_log([](LogLevel level, std::string_view logger, const nlohmann::json& data) {
        std::fprintf(stderr, "[log/%s] %s %s\n", std::string(to_string(level)).c_str(),
                     std::string(logger).c_str(), data.dump().c_str());
    });
}

} // namespace

int main(int argc, char** argv) {
    const Args a = parse_args(argc, argv);
    if (a.help) {
        print_usage();
        return 0;
    }
    if (!a.ok) {
        print_usage();
        return 2;
    }

    Client::Options options;
    options.default_timeout = std::chrono::milliseconds(a.timeout_ms);
    Client client(Client::Info{"mcp_probe", "0.1.0"}, options);
    install_diagnostics(client, a.quiet);

    try {
        Subprocess::Options spawn;
        spawn.program = a.command.front();
        spawn.args.assign(a.command.begin() + 1, a.command.end());
        spawn.stderr_mode = StderrMode::kCapture;
        client.connect_stdio(std::move(spawn));

        describe(client, client.initialize());
        if (!a.call_tool.empty() && !run_tool_call(client, a)) {
            return 2;
        }
        run_extras(client, a);
        client.close();
        return 0;
    } catch (const McpError& e) {
        std::fprintf(stderr, "\nMCP error %d: %s\n", e.code(), e.what());
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nfailed: %s\n", e.what());
        return 1;
    }
}
