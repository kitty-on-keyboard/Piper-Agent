#include "mcp_server.hpp"
#include <iostream>
#include <chrono>

using namespace mcp;

int main() {
    MCPServer server;

    // Register a simple Echo Tool
    server.registerTool({
        .name = "echo_tool",
        .description = "Echoes back the input message",
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
        .handler = [](const json& args) -> ToolResult {
            if (!args.contains("message") || !args["message"].is_string()) {
                return ToolResult::fromText("Missing or invalid 'message' argument.", true);
            }
            std::string msg = args["message"];
            return ToolResult::fromText("Echo: " + msg);
        }
    });

    // Register a System Status Tool
    server.registerTool({
        .name = "system_status",
        .description = "Returns current system time and status",
        .inputSchema = {
            {"type", "object"},
            {"properties", {}}
        },
        .handler = [](const json& args) -> ToolResult {
            auto now = std::chrono::system_clock::now();
            auto time = std::chrono::system_clock::to_time_t(now);
            std::string timeStr = std::ctime(&time);
            // ctime appends a newline, let's remove it
            if (!timeStr.empty() && timeStr.back() == '\n') {
                timeStr.pop_back();
            }
            return ToolResult::fromText("System is nominal. Current time: " + timeStr);
        }
    });

    // Register a File Utility Tool (Simulated)
    server.registerTool({
        .name = "file_utility",
        .description = "Simulates reading a file",
        .inputSchema = {
            {"type", "object"},
            {"properties", {
                {"filename", {
                    {"type", "string"},
                    {"description", "The file to read"}
                }}
            }},
            {"required", {"filename"}}
        },
        .handler = [](const json& args) -> ToolResult {
            if (!args.contains("filename") || !args["filename"].is_string()) {
                return ToolResult::fromText("Missing 'filename'", true);
            }
            std::string filename = args["filename"];
            return ToolResult::fromText("Simulated content of " + filename + " : [EOF]");
        }
    });

    // Start the server
    server.start();

    // Wait for the server to stop (blocks until stdin is closed)
    server.wait();

    return 0;
}
