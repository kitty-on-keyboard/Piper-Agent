#include "mcp_client.hpp"
#include <iostream>
#include <chrono>

int main() {
    try {
        std::cout << "Starting MCP Client Example..." << std::endl;

        // Launch a mock MCP server that we will create for testing
        // Assuming the mock server is compiled into the build directory
        std::vector<std::string> args;
        mcp::McpClient client("./mock_server", args);

        std::cout << "Sending initialize..." << std::endl;
        nlohmann::json init_params = {
            {"protocolVersion", "2024-11-05"},
            {"capabilities", nlohmann::json::object()},
            {"clientInfo", {
                {"name", "mcp-cpp-example"},
                {"version", "1.0.0"}
            }}
        };

        auto init_future = client.initialize(init_params);
        auto init_result = init_future.get();
        std::cout << "Initialize Result: " << init_result.dump(2) << std::endl;

        client.send_initialized();

        std::cout << "Sending tools/list..." << std::endl;
        auto list_future = client.list_tools();
        auto list_result = list_future.get();
        std::cout << "Tools List Result: " << list_result.dump(2) << std::endl;

        std::cout << "Sending tools/call..." << std::endl;
        auto call_future = client.call_tool("echo", {{"message", "Hello MCP!"}});
        auto call_result = call_future.get();
        std::cout << "Tool Call Result: " << call_result.dump(2) << std::endl;

        std::cout << "Example completed successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
