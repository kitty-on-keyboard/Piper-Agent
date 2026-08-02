#include "mcp_client.hpp"
#include <iostream>
#include <vector>

int main() {
    try {
        mcp::McpClient client;

        std::cout << "Starting MCP Client...\n";

        // For demonstration, we'll run a Python script that echoes a mock MCP response
        // In a real application, this would be an actual MCP server process.
        // client.start("python3", {"./mock_mcp_server.py"});
        std::cout << "Uncomment client.start() with your actual MCP server command to test.\n";
        return 0; // Exit early since we deleted the mock script for submission

        std::cout << "Sending initialize...\n";
        auto init_res = client.initialize("example_client", "1.0.0");
        std::cout << "Init response: " << init_res.dump(2) << "\n\n";

        std::cout << "Listing tools...\n";
        auto tools_res = client.list_tools();
        std::cout << "Tools response: " << tools_res.dump(2) << "\n\n";

        std::cout << "Calling tool asynchronously...\n";
        nlohmann::json args = {{"param", "value"}};
        auto future_res = client.call_tool("my_tool", args);

        // Do other work here...
        std::cout << "Doing other work while waiting...\n";

        auto call_res = future_res.get();
        std::cout << "Call response: " << call_res.dump(2) << "\n\n";

        std::cout << "Shutting down...\n";
        client.stop();
        std::cout << "Client stopped successfully.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
