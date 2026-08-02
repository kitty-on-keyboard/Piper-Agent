#include "mcp_client.hpp"
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_server> [args...]\n";
        return 1;
    }

    std::string server_path = argv[1];
    std::vector<std::string> args;
    for (int i = 2; i < argc; ++i) {
        args.push_back(argv[i]);
    }

    try {
        std::cout << "Starting MCP client..." << std::endl;
        mcp::McpClient client(server_path, args);

        std::cout << "Initializing..." << std::endl;
        auto init_future = client.initialize();
        auto init_result = init_future.get();
        std::cout << "Initialized: " << init_result.dump(2) << std::endl;

        std::cout << "Sending initialized notification..." << std::endl;
        client.send_initialized_notification();

        std::cout << "\nListing tools..." << std::endl;
        auto list_future = client.list_tools();
        auto list_result = list_future.get();
        std::cout << "Tools: " << list_result.dump(2) << std::endl;

        std::cout << "\nCalling a tool (assuming 'echo_tool' exists)..." << std::endl;
        nlohmann::json args_json = {{"message", "Hello from C++20!"}};
        auto call_future = client.call_tool("echo_tool", args_json);

        // Wait for the response with a timeout
        if (call_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready) {
            auto call_result = call_future.get();
            std::cout << "Tool result: " << call_result.dump(2) << std::endl;
        } else {
            std::cerr << "Tool call timed out (mock server might not handle 'echo_tool')" << std::endl;
        }

        std::cout << "\nShutting down gracefully..." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
