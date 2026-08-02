#include "mcp_client.hpp"
#include <iostream>
#include <vector>
#include <string>

int main() {
    try {
        std::cout << "Starting MCP Client..." << std::endl;

        std::vector<std::string> args;
        // In reality, this would point to an actual MCP server executable.
        // Here we point to our mock server for demonstration.
        McpClient client("./mock_server.sh", args);

        std::cout << "Initializing..." << std::endl;
        auto init_fut = client.initialize();
        auto init_res = init_fut.get();
        std::cout << "Init response: " << init_res.dump(2) << std::endl;

        std::cout << "Listing tools..." << std::endl;
        auto tools_fut = client.list_tools();
        auto tools_res = tools_fut.get();
        std::cout << "Tools response: " << tools_res.dump(2) << std::endl;

        std::cout << "Calling tool..." << std::endl;
        auto call_fut = client.call_tool("echo", {{"message", "Hello MCP!"}});
        auto call_res = call_fut.get();
        std::cout << "Call response: " << call_res.dump(2) << std::endl;

        std::cout << "Shutting down..." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
