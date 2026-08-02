#include "mcp_server.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <thread>
#include <csignal>

mcp::MCPServer* g_server = nullptr;

void signal_handler(int signal) {
    std::cerr << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
    std::exit(signal);
}

int main() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    mcp::MCPServer server("mcp-server-cpp-demo", "1.0.0");
    g_server = &server;

    // Register a simple echo tool
    server.registerTool({
        "echo",
        "Echoes the input message back",
        {
            {"type", "object"},
            {"properties", {
                {"message", {
                    {"type", "string"},
                    {"description", "The message to echo"}
                }}
            }},
            {"required", {"message"}}
        },
        [](const json& arguments) -> mcp::ToolResult {
            std::string msg = arguments.value("message", "default message");
            json content = json::array({
                {
                    {"type", "text"},
                    {"text", "Echo: " + msg}
                }
            });
            return {content, false};
        }
    });

    // Register a system status tool
    server.registerTool({
        "system_status",
        "Returns basic system status information",
        {
            {"type", "object"},
            {"properties", json::object()}
        },
        [](const json&) -> mcp::ToolResult {
            auto now = std::chrono::system_clock::now();
            std::time_t now_c = std::chrono::system_clock::to_time_t(now);

            std::string time_str = std::ctime(&now_c);
            // Remove trailing newline from ctime
            if (!time_str.empty() && time_str.back() == '\n') {
                time_str.pop_back();
            }

            json content = json::array({
                {
                    {"type", "text"},
                    {"text", "System Time: " + time_str + "\nStatus: Running"}
                }
            });
            return {content, false};
        }
    });

    // Register a mock file read tool
    server.registerTool({
        "read_file_mock",
        "Mocks reading a file and returns its content",
        {
            {"type", "object"},
            {"properties", {
                {"filepath", {
                    {"type", "string"},
                    {"description", "The path of the file to read"}
                }}
            }},
            {"required", {"filepath"}}
        },
        [](const json& arguments) -> mcp::ToolResult {
            std::string filepath = arguments.value("filepath", "");
            if (filepath.empty()) {
                return {
                    json::array({
                        {
                            {"type", "text"},
                            {"text", "Error: filepath is required"}
                        }
                    }),
                    true
                };
            }

            json content = json::array({
                {
                    {"type", "text"},
                    {"text", "Mock content for file: " + filepath}
                }
            });
            return {content, false};
        }
    });

    server.start();

    // Block main thread while server is running
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}
