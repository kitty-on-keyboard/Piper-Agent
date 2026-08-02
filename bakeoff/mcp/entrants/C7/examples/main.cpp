#include "mcp_client.hpp"
#include <iostream>
#include <chrono>

int main() {
    try {
        std::cout << "Starting MCP Client..." << std::endl;

        // As a demonstration, we will spawn a simple mock MCP server
        // In real-world, this would point to the actual MCP server command.
        // For demonstration, we just use 'cat' or similar to prevent crashes, but it won't
        // act as a real MCP server.
        // If there's an actual MCP server, we'd replace these args.

        // This is purely to compile and show usage.
        mcp::McpClient client("/bin/echo", {"{}"});

        // mcp::McpClient client("node", {"path/to/mcp/server.js"}); // Example

        // Initialize
        std::cout << "Initializing..." << std::endl;
        // auto init_res = client.initialize();
        // std::cout << "Init Result: " << init_res.dump(4) << std::endl;

        // List tools
        // std::cout << "Listing tools..." << std::endl;
        // auto tools = client.list_tools();
        // std::cout << "Tools: " << tools.dump(4) << std::endl;

        // Call tool
        // std::cout << "Calling tool..." << std::endl;
        // nlohmann::json args = {{"param1", "value1"}};
        // auto result = client.call_tool("my_tool", args);
        // std::cout << "Tool Result: " << result.dump(4) << std::endl;

        std::cout << "Example completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
