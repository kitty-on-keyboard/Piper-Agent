#include "mcp_server.hpp"
#include <iostream>
#include <string>
#include <chrono>

int main() {
    mcp::MCPServer server;

    // Tool 1: System Status
    mcp::Tool systemStatusTool{
        .name = "system_status",
        .description = "Returns the current system status.",
        .inputSchema = {
            {"type", "object"},
            {"properties", nlohmann::json::object()},
            {"required", nlohmann::json::array()}
        },
        .handler = [](const mcp::JSONValue& args) -> mcp::ToolResult {
            mcp::ToolResult result;
            result.isError = false;
            result.content = {
                {
                    {"type", "text"},
                    {"text", "System is running optimally. Time: " + std::to_string(std::chrono::system_clock::now().time_since_epoch().count())}
                }
            };
            return result;
        }
    };
    server.registerTool(systemStatusTool);

    // Tool 2: Echo
    mcp::Tool echoTool{
        .name = "echo",
        .description = "Echoes back the provided message.",
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
            mcp::ToolResult result;

            if (!args.contains("message") || !args["message"].is_string()) {
                result.isError = true;
                result.content = {
                    {{"type", "text"}, {"text", "Missing or invalid 'message' argument."}}
                };
                return result;
            }

            std::string msg = args["message"].get<std::string>();
            result.isError = false;
            result.content = {
                {{"type", "text"}, {"text", "Echo: " + msg}}
            };
            return result;
        }
    };
    server.registerTool(echoTool);

    // Tool 3: File Utility
    mcp::Tool fileUtilityTool{
        .name = "file_utility",
        .description = "Simulates reading a file.",
        .inputSchema = {
            {"type", "object"},
            {"properties", {
                {"filename", {
                    {"type", "string"},
                    {"description", "Name of the file to read"}
                }}
            }},
            {"required", {"filename"}}
        },
        .handler = [](const mcp::JSONValue& args) -> mcp::ToolResult {
            mcp::ToolResult result;

            if (!args.contains("filename") || !args["filename"].is_string()) {
                result.isError = true;
                result.content = {
                    {{"type", "text"}, {"text", "Missing or invalid 'filename' argument."}}
                };
                return result;
            }

            std::string filename = args["filename"].get<std::string>();

            result.isError = false;
            result.content = {
                {{"type", "text"}, {"text", "Content of " + filename + " would be here."}}
            };
            return result;
        }
    };
    server.registerTool(fileUtilityTool);

    // Start server loop
    server.start();

    // Wait until stopped (e.g. standard input is closed)
    server.wait();

    return 0;
}
