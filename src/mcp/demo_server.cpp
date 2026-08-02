// A complete MCP server over stdio, for driving from a real MCP client.
//
// It exercises the parts that are easy to get wrong and therefore worth having a live
// example of: a tool that reports progress and honours cancellation, a tool that fails
// as a result rather than as a protocol error, resources, prompts, and completion.
//
// The one rule this file obeys absolutely: nothing is ever written to stdout except by
// the transport. Diagnostics go to stderr. A stray printf here would desync the client's
// framer, which is the single most common way a hand-written MCP server breaks.

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "src/mcp/server.hpp"
#include "src/mcp/transport.hpp"

namespace {

using namespace lmp::mcp;

std::string arg_string(const nlohmann::json& args, const char* key, std::string fallback = {}) {
    if (args.is_object() && args.contains(key) && args[key].is_string()) {
        return args[key].get<std::string>();
    }
    return fallback;
}

double arg_number(const nlohmann::json& args, const char* key, double fallback) {
    if (args.is_object() && args.contains(key) && args[key].is_number()) {
        return args[key].get<double>();
    }
    return fallback;
}

void register_tools(Server& server) {
    Tool echo;
    echo.name = "echo";
    echo.title = "Echo";
    echo.description = "Echo the supplied text back to the caller.";
    echo.input_schema = {{"type", "object"},
                         {"properties", {{"text", {{"type", "string"}, {"description", "Text to echo"}}}}},
                         {"required", nlohmann::json::array({"text"})}};
    server.add_tool(std::move(echo), [](const nlohmann::json& args, RequestContext&) {
        return ToolResult::text(arg_string(args, "text"));
    });

    Tool add;
    add.name = "add";
    add.title = "Add";
    add.description = "Add two numbers.";
    add.input_schema = {{"type", "object"},
                        {"properties", {{"a", {{"type", "number"}}}, {"b", {{"type", "number"}}}}},
                        {"required", nlohmann::json::array({"a", "b"})}};
    // A structured result alongside the text one: 2025-06-18 clients read
    // structuredContent, older ones read the text block, and both are satisfied.
    add.output_schema = nlohmann::json{{"type", "object"},
                                       {"properties", {{"sum", {{"type", "number"}}}}}};
    server.add_tool(std::move(add), [](const nlohmann::json& args, RequestContext&) {
        const double sum = arg_number(args, "a", 0) + arg_number(args, "b", 0);
        ToolResult r = ToolResult::text(std::to_string(sum));
        r.structured = nlohmann::json{{"sum", sum}};
        return r;
    });

    Tool slow;
    slow.name = "slow_count";
    slow.title = "Slow count";
    slow.description = "Count to N, one step every 100ms, reporting progress. Cancellable.";
    slow.input_schema = {{"type", "object"},
                         {"properties", {{"n", {{"type", "integer"}, {"description", "Steps"}}}}},
                         {"required", nlohmann::json::array({"n"})}};
    server.add_tool(std::move(slow), [](const nlohmann::json& args, RequestContext& ctx) {
        const int n = static_cast<int>(arg_number(args, "n", 5));
        for (int i = 0; i < n; ++i) {
            if (ctx.cancelled()) {
                return ToolResult::failure("cancelled after " + std::to_string(i) + " steps");
            }
            ctx.report_progress(static_cast<double>(i + 1), static_cast<double>(n),
                                "step " + std::to_string(i + 1));
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return ToolResult::text("counted to " + std::to_string(n));
    });

    Tool boom;
    boom.name = "always_fails";
    boom.title = "Always fails";
    boom.description = "Demonstrates a tool failure, which is a result and not a protocol error.";
    server.add_tool(std::move(boom), [](const nlohmann::json&, RequestContext&) {
        return ToolResult::failure("this tool fails on purpose");
    });
}

void register_resources(Server& server) {
    Resource readme;
    readme.uri = "mem://readme";
    readme.name = "readme";
    readme.title = "Read me";
    readme.description = "A static text resource.";
    readme.mime_type = "text/plain";
    server.add_resource(std::move(readme), [](const std::string& uri, RequestContext&) {
        return std::vector<ResourceContents>{
            ResourceContents::from_text(uri, "This is a static MCP resource served by lmp_mcp.")};
    });

    Resource config;
    config.uri = "mem://config.json";
    config.name = "config";
    config.mime_type = "application/json";
    server.add_resource(std::move(config), [](const std::string& uri, RequestContext&) {
        return std::vector<ResourceContents>{
            ResourceContents::from_text(uri, R"({"demo":true,"units":"metric"})", "application/json")};
    });
}

void register_prompts(Server& server) {
    Prompt review;
    review.name = "code_review";
    review.title = "Code review";
    review.description = "Ask for a review of a snippet.";
    review.arguments.push_back(PromptArgument{"language", "Source language", true});
    review.arguments.push_back(PromptArgument{"code", "The snippet to review", true});
    server.add_prompt(std::move(review), [](const nlohmann::json& args, RequestContext&) {
        const std::string lang = arg_string(args, "language", "text");
        const std::string code = arg_string(args, "code");
        return std::vector<PromptMessage>{PromptMessage::user_text(
            "Review the following " + lang + " code and list concrete defects:\n\n" + code)};
    });

    server.set_completion_handler(
        [](std::string_view argument_name, std::string_view partial) {
            std::vector<std::string> out;
            if (argument_name != "language") {
                return out;
            }
            for (const char* c : {"c++", "python", "typescript", "rust", "go"}) {
                if (std::string(c).rfind(std::string(partial), 0) == 0) {
                    out.emplace_back(c);
                }
            }
            return out;
        });
}

} // namespace

int main() {
    Server::Info info;
    info.name = "lmp-mcp-demo";
    info.version = "0.1.0";
    info.instructions =
        "Demonstration server for the lmp_mcp C++ implementation. `slow_count` reports "
        "progress and honours cancellation; `always_fails` returns a tool failure.";

    Server server(info);
    register_tools(server);
    register_resources(server);
    register_prompts(server);

    std::fprintf(stderr, "[lmp-mcp-demo] ready on stdio\n");

    StdioTransport transport;
    server.run(transport);

    std::fprintf(stderr, "[lmp-mcp-demo] client disconnected, exiting\n");
    return 0;
}
