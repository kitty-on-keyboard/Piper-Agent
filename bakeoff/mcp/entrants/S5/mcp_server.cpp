#include "mcp_server.hpp"

namespace mcp {

MCPServer::MCPServer() {
}

MCPServer::~MCPServer() {
    stop();
}

void MCPServer::start() {
    if (running_.exchange(true)) {
        return; // Already running
    }

    // Start reader thread using jthread which automatically joins on destruction
    readerThread_ = std::jthread([this]() { readLoop(); });
    std::cerr << "[mcp] Server started. Listening on stdin..." << std::endl;
}

void MCPServer::stop() {
    running_ = false;

    if (readerThread_ && readerThread_->joinable()) {
        readerThread_->request_stop();
    }

    // Wait for all active worker threads to finish to prevent Use-After-Free
    std::unique_lock<std::mutex> lock(workersMutex_);
    workersCv_.wait(lock, [this]() { return activeWorkers_ == 0; });
}

void MCPServer::wait() {
    if (readerThread_ && readerThread_->joinable()) {
        readerThread_->join();
    }
}

void MCPServer::readLoop() {
    std::string line;
    // Keep reading until EOF or running_ becomes false
    while (running_ && std::getline(std::cin, line)) {
        if (line.empty()) continue;

        try {
            json request = json::parse(line);

            // Increment active workers before detaching the thread
            activeWorkers_++;

            // Use 'mutable' lambda so std::move(req) actually moves
            std::thread([this, req = std::move(request)]() mutable {
                dispatchRequest(std::move(req));

                // Decrement active workers and notify stop() if needed
                activeWorkers_--;
                workersCv_.notify_all();
            }).detach();

        } catch (const json::parse_error& e) {
            std::cerr << "[mcp] JSON Parse Error: " << e.what() << "\n";
            sendError(nullptr, -32700, "Parse error");
        }
    }
    running_ = false;
}

void MCPServer::dispatchRequest(json request) {
    json id = nullptr;

    try {
        // Prevent DoS exception: Ensure request is an object before accessing methods
        if (!request.is_object()) {
            sendError(nullptr, -32600, "Invalid Request: Expected JSON object");
            return;
        }

        // Try to get the ID safely
        if (request.contains("id")) {
            id = request["id"];
        }

        // Simple JSON-RPC 2.0 validation
        if (!request.contains("jsonrpc") || request["jsonrpc"] != "2.0") {
            sendError(id, -32600, "Invalid Request: Missing or invalid jsonrpc version");
            return;
        }

        std::string method = request.value("method", "");
        json params = request.value("params", json::object());

        // Routing
        if (method == "initialize") {
            auto result = handleInitialize(params);
            sendResponse({{"jsonrpc", "2.0"}, {"id", id}, {"result", result}});
        }
        else if (method == "notifications/initialized") {
            handleInitialized(params); // Notification, no response
        }
        else if (method == "tools/list") {
            auto result = handleToolsList(params);
            sendResponse({{"jsonrpc", "2.0"}, {"id", id}, {"result", result}});
        }
        else if (method == "tools/call") {
            auto result = handleToolsCall(params);
            sendResponse({{"jsonrpc", "2.0"}, {"id", id}, {"result", result}});
        }
        else if (method == "ping") {
            auto result = handlePing(params);
            sendResponse({{"jsonrpc", "2.0"}, {"id", id}, {"result", result}});
        }
        else {
            sendError(id, -32601, "Method not found: " + method);
        }

    } catch (const std::exception& e) {
        std::cerr << "[mcp] Internal error: " << e.what() << "\n";
        sendError(id, -32603, std::string("Internal error: ") + e.what());
    } catch (...) {
        std::cerr << "[mcp] Unknown internal error\n";
        sendError(id, -32603, "Internal error");
    }
}

json MCPServer::handleInitialize(const json& params) {
    // Protocol capability exchange
    json result = {
        {"protocolVersion", "2024-11-05"}, // Standard MCP version
        {"serverInfo", {
            {"name", "mcp-cpp-server"},
            {"version", "1.0.0"}
        }},
        {"capabilities", {
            {"tools", {}} // Advertising tools capability
        }}
    };
    return result;
}

void MCPServer::handleInitialized(const json& params) {
    std::cerr << "[mcp] Client fully initialized.\n";
}

json MCPServer::handleToolsList(const json& params) {
    json result;
    result["tools"] = registry_.getToolsList();
    return result;
}

json MCPServer::handleToolsCall(const json& params) {
    std::string name = params.value("name", "");
    json arguments = params.value("arguments", json::object());

    auto toolOpt = registry_.getTool(name);
    if (!toolOpt) {
        // Standard MCP tool call error response format usually isn't an RPC error,
        // but rather a tool result with isError = true
        json content = json::array();
        content.push_back({
            {"type", "text"},
            {"text", "Tool not found: " + name}
        });
        return {
            {"content", content},
            {"isError", true}
        };
    }

    try {
        ToolResult res = toolOpt->handler(arguments);

        json contentArray = json::array();
        for (const auto& c : res.content) {
            contentArray.push_back({
                {"type", c.type},
                {"text", c.text}
            });
        }

        return {
            {"content", contentArray},
            {"isError", res.isError}
        };

    } catch (const std::exception& e) {
        std::cerr << "[mcp] Tool handler exception: " << e.what() << "\n";
        json content = json::array();
        content.push_back({
            {"type", "text"},
            {"text", std::string("Tool execution failed: ") + e.what()}
        });
        return {
            {"content", content},
            {"isError", true}
        };
    }
}

json MCPServer::handlePing(const json& params) {
    return json::object(); // Empty object for ping result
}

void MCPServer::sendResponse(const json& response) {
    std::lock_guard<std::mutex> lock(stdoutMutex_);
    std::cout << response.dump() << "\n" << std::flush;
}

void MCPServer::sendError(const json& id, int code, const std::string& message) {
    json error = {
        {"code", code},
        {"message", message}
    };
    json response = {
        {"jsonrpc", "2.0"},
        {"error", error},
        {"id", id}
    };

    std::lock_guard<std::mutex> lock(stdoutMutex_);
    std::cout << response.dump() << "\n" << std::flush;
}

} // namespace mcp
