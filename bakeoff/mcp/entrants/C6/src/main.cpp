#include "mcp_client.hpp"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_server_executable>" << std::endl;
        return 1;
    }

    try {
        std::cout << "Starting MCP Client..." << std::endl;

        // Use argv[1] as the server executable
        std::vector<std::string> args;
        mcp::McpClient client(argv[1], args);
        client.start();

        std::cout << "Sending initialize..." << std::endl;
        mcp::json init_params = {
            {"protocolVersion", "2024-11-05"},
            {"clientInfo", {
                {"name", "cpp-test-client"},
                {"version", "1.0.0"}
            }},
            {"capabilities", mcp::json::object()}
        };

        auto init_future = client.initialize(init_params);
        mcp::json init_res = init_future.get();
        std::cout << "Initialized: " << init_res.dump(2) << std::endl;

        std::cout << "Sending notifications/initialized..." << std::endl;
        client.send_notification("notifications/initialized");

        std::cout << "Requesting tool list..." << std::endl;
        auto tools_future = client.list_tools();
        mcp::json tools_res = tools_future.get();
        std::cout << "Tools: " << tools_res.dump(2) << std::endl;

        std::cout << "Calling tool 'echo'..." << std::endl;
        mcp::json echo_args = {
            {"message", "Hello from C++20 MCP Client!"}
        };
        auto call_future = client.call_tool("echo", echo_args);
        mcp::json call_res = call_future.get();
        std::cout << "Tool Call Result: " << call_res.dump(2) << std::endl;

        std::cout << "Shutting down gracefully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
