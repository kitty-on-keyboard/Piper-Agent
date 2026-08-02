#include "mcp_server.hpp"
#include <iostream>
#include <string>
#include <chrono>

int main() {
    mcp::Server server;

    // Register a simple Echo tool
    mcp::Tool echoTool;
    echoTool.name = "echo";
    echoTool.description = "Echoes back the input text";
    echoTool.inputSchema = {
        {"type", "object"},
        {"properties", {
            {"text", {{"type", "string"}, {"description", "The text to echo"}}}
        }},
        {"required", {"text"}}
    };
    echoTool.handler = [](const mcp::JSONValue& args) -> mcp::ToolResult {
        std::string text = args.value("text", "");
        mcp::JSONValue content = mcp::JSONValue::array({
            {{"type", "text"}, {"text", "Echo: " + text}}
        });
        return {content, false};
    };
    server.registerTool(echoTool);

    // Register a system status tool
    mcp::Tool statusTool;
    statusTool.name = "system_status";
    statusTool.description = "Gets the current system time and status";
    statusTool.inputSchema = {
        {"type", "object"},
        {"properties", mcp::JSONValue::object()} // No arguments required
    };
    statusTool.handler = [](const mcp::JSONValue& /*args*/) -> mcp::ToolResult {
        auto now = std::chrono::system_clock::now();
        // Use standard C++20 chrono formatting if we had <format>, but to keep it simple and portable:
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        char buf[100];
        if (std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now_c))) {
            mcp::JSONValue content = mcp::JSONValue::array({
                {{"type", "text"}, {"text", std::string("System is running fine. Time: ") + buf}}
            });
            return {content, false};
        } else {
            mcp::JSONValue content = mcp::JSONValue::array({
                {{"type", "text"}, {"text", "System is running fine."}}
            });
            return {content, false};
        }
    };
    server.registerTool(statusTool);

    // Run the server (this will block and listen on stdin)
    std::cerr << "[Main] Starting MCP server..." << std::endl;
    server.run();

    return 0;
}