#include "mcp_client.hpp"
#include <iostream>

int main() {
    try {
        mcp::Client client;

        std::cout << "Attempting to connect to MCP server...\n";

        // This is an example of how you would connect to a real MCP server.
        // Replace "your-mcp-server-binary" with an actual MCP server executable.
        std::vector<std::string> args = {"--some-flag"};
        if (!client.connect("your-mcp-server-binary", args)) {
            std::cerr << "Failed to start server\n";
            return 1;
        }

        std::cout << "Sending initialize...\n";
        mcp::json init_res = client.initialize();
        std::cout << "Initialized: " << init_res.dump(4) << "\n";

        client.send_initialized();

        std::cout << "Listing tools...\n";
        mcp::json tools = client.list_tools();
        std::cout << "Tools: " << tools.dump(4) << "\n";

        std::cout << "Calling tool 'echo'...\n";
        mcp::json call_res = client.call_tool("echo", {{"message", "Hello C++20!"}});
        std::cout << "Tool Result: " << call_res.dump(4) << "\n";

        client.stop();
        std::cout << "Client stopped cleanly.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
