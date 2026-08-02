#include "mcp_server.hpp"

#include <iostream>
#include <string>
#include <chrono>
#include <thread>

int main() {
    // 1. Instantiate the server
    mcp::McpServer server("mcp-cpp-server", "1.0.0");

    // 2. Register tools
    auto& registry = server.getRegistry();

    // Tool 1: system_status
    registry.registerTool({
        "system_status",
        "Returns the current system status and uptime.",
        {
            {"type", "object"},
            {"properties", nlohmann::json::object()}, // No properties needed
            {"required", nlohmann::json::array()}
        },
        [](const mcp::JSONValue& args) -> mcp::ToolResult {
            std::string status = "System is running optimally. Architecture: macOS ARM64.";
            return mcp::ToolResult::text(status);
        }
    });

    // Tool 2: echo
    registry.registerTool({
        "echo",
        "Echos back the provided message.",
        {
            {"type", "object"},
            {"properties", {
                {"message", {
                    {"type", "string"},
                    {"description", "The message to echo back."}
                }}
            }},
            {"required", {"message"}}
        },
        [](const mcp::JSONValue& args) -> mcp::ToolResult {
            if (!args.contains("message") || !args["message"].is_string()) {
                return mcp::ToolResult::error("Missing or invalid 'message' argument");
            }
            std::string msg = args["message"];
            return mcp::ToolResult::text("Echo: " + msg);
        }
    });

    // Tool 3: file_utility (mocked)
    registry.registerTool({
        "file_utility",
        "Reads a file or returns info (mock).",
        {
            {"type", "object"},
            {"properties", {
                {"filename", {
                    {"type", "string"},
                    {"description", "The file to inspect."}
                }}
            }},
            {"required", {"filename"}}
        },
        [](const mcp::JSONValue& args) -> mcp::ToolResult {
            if (!args.contains("filename") || !args["filename"].is_string()) {
                return mcp::ToolResult::error("Missing or invalid 'filename' argument");
            }
            std::string filename = args["filename"];
            return mcp::ToolResult::text("File " + filename + " is 1024 bytes (mock).");
        }
    });

    // 3. Start the server
    std::cerr << "Starting MCP C++ Server..." << std::endl;
    server.start();

    // Block main thread until server is stopped (e.g. by EOF on stdin)
    server.wait();

    std::cerr << "Stopping MCP C++ Server..." << std::endl;
    server.stop(); // Safe to call multiple times

    return 0;
}
