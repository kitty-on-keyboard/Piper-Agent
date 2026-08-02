#include "mcp_server.hpp"
#include <iostream>
#include <string>

int main() {
    mcp::McpServer server;

    // Register a simple Echo tool
    mcp::Tool echoTool{
        .name = "echo",
        .description = "Echos the input message back",
        .inputSchema = {
            {"type", "object"},
            {"properties", {
                {"message", {
                    {"type", "string"},
                    {"description", "The message to echo"}
                }}
            }},
            {"required", {"message"}}
        },
        .handler = [](const mcp::JSONValue& args) -> mcp::ToolResult {
            if (!args.contains("message") || !args["message"].is_string()) {
                throw std::invalid_argument("Missing or invalid 'message' argument");
            }
            std::string message = args["message"];

            mcp::JSONValue content = mcp::JSONValue::array({
                {
                    {"type", "text"},
                    {"text", "Echo: " + message}
                }
            });

            return {content, false};
        }
    };
    server.registerTool(std::move(echoTool));

    // Register a System Status tool
    mcp::Tool sysStatusTool{
        .name = "system_status",
        .description = "Returns current system status dummy data",
        .inputSchema = {
            {"type", "object"},
            {"properties", nlohmann::json::object()}
        },
        .handler = [](const mcp::JSONValue&) -> mcp::ToolResult {
            mcp::JSONValue content = mcp::JSONValue::array({
                {
                    {"type", "text"},
                    {"text", "System is running optimally on C++20."}
                }
            });

            return {content, false};
        }
    };
    server.registerTool(std::move(sysStatusTool));

    std::cerr << "Starting MCP C++20 Server. Ready to receive JSON-RPC over stdin." << std::endl;

    server.start();

    // Main thread can wait or do other things.
    // We'll just wait forever (or until stdin is closed if we handled EOF,
    // but right now it runs until stopped or process is killed).
    // A better approach is to join some condition variable, but for now:
    while (std::cin.good()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    server.stop();
    std::cerr << "MCP Server stopping." << std::endl;

    return 0;
}
