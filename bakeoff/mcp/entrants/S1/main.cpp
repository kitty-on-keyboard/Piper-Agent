#include "mcp_server.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

using namespace mcp;

int main() {
    MCPServer server("test-server", "1.0.0");

    std::vector<Tool> myTools = {
        Tool{
            "system_status",
            "Get the system status",
            JSONValue::object(),
            [](const JSONValue& args) -> ToolResult {
                return {
                    {{{"type", "text"}, {"text", "System is fully operational."}}},
                    false
                };
            }
        },
        Tool{
            "echo",
            "Echo a message",
            {
                {"type", "object"},
                {"properties", {
                    {"message", {{"type", "string"}}}
                }},
                {"required", {"message"}}
            },
            [](const JSONValue& args) -> ToolResult {
                if (!args.contains("message") || !args["message"].is_string()) {
                    return {
                        {{{"type", "text"}, {"text", "Missing or invalid 'message' parameter."}}},
                        true
                    };
                }
                std::string message = args["message"];
                return {
                    {{{"type", "text"}, {"text", "Echo: " + message}}},
                    false
                };
            }
        },
        Tool{
            "math_add",
            "Add two numbers together",
            {
                {"type", "object"},
                {"properties", {
                    {"a", {{"type", "number"}}},
                    {"b", {{"type", "number"}}}
                }},
                {"required", {"a", "b"}}
            },
            [](const JSONValue& args) -> ToolResult {
                if (!args.contains("a") || !args["a"].is_number() ||
                    !args.contains("b") || !args["b"].is_number()) {
                    return {
                        {{{"type", "text"}, {"text", "Missing or invalid 'a' or 'b' parameter."}}},
                        true
                    };
                }
                double a = args["a"];
                double b = args["b"];
                double sum = a + b;

                // Simulate a long running task to prove decoupling
                std::this_thread::sleep_for(std::chrono::milliseconds(500));

                return {
                    {{{"type", "text"}, {"text", std::to_string(sum)}}},
                    false
                };
            }
        }
    };

    server.registerTools(myTools);

    // Start server, which spins up a jthread
    server.start();

    // Block the main thread cleanly until EOF is received and the loop ends
    server.wait();

    server.stop();

    return 0;
}
