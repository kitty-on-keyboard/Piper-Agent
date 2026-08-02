#include "mcp_server.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace mcp {

// Standard JSON-RPC error codes
constexpr int PARSE_ERROR = -32700;
constexpr int INVALID_REQUEST = -32600;
constexpr int METHOD_NOT_FOUND = -32601;
constexpr int INVALID_PARAMS = -32602;
constexpr int INTERNAL_ERROR = -32603;

// ToolRegistry Implementation

void ToolRegistry::registerTool(const Tool& tool) {
    std::lock_guard<std::mutex> lock(mutex_);
    tools_[tool.name] = tool;
}

std::vector<Tool> ToolRegistry::listTools() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Tool> result;
    result.reserve(tools_.size());
    for (const auto& [name, tool] : tools_) {
        result.push_back(tool);
    }
    return result;
}

// MCPServer Implementation

MCPServer::MCPServer() {
}

MCPServer::~MCPServer() {
    stop();
}

void MCPServer::registerTool(const Tool& tool) {
    registry_.registerTool(tool);
}

void MCPServer::start() {
    log("Server starting...");
    readerThread_ = std::jthread([this](std::stop_token stoken) {
        readerThreadLoop(stoken);
    });
}

void MCPServer::stop() {
    if (readerThread_.joinable()) {
        readerThread_.request_stop();
        // Since we are reading from stdin which might block, we can't easily cancel a blocking std::getline.
        // We'll rely on the stream closing or process termination.
        // For a robust implementation, we would use select/poll, but std::getline on std::cin is standard for MCP unless non-blocking I/O is configured.
        // Note: For simplicity and portability in C++, blocking std::getline on a thread is used.
    }
}

void MCPServer::wait() {
    // Wait until the reader thread completes (e.g. EOF on stdin)
    if (readerThread_.joinable()) {
        readerThread_.join();
    }

    // Wait for all workers to finish
    std::unique_lock<std::mutex> lock(workersCvMutex_);
    workersCv_.wait(lock, [this]() { return activeWorkers_.load() == 0; });
}

void MCPServer::readerThreadLoop(std::stop_token stoken) {
    std::string line;
    // Read line-delimited JSON from stdin
    while (!stoken.stop_requested() && std::getline(std::cin, line)) {
        if (line.empty()) continue;

        // Dispatch to worker thread for non-blocking standard input processing
        dispatchMessage(line);
    }
    log("Reader thread exiting (EOF or stopped).");
}

void MCPServer::dispatchMessage(std::string rawMessage) {
    activeWorkers_++;
    try {
        std::thread([this, msg = std::move(rawMessage)]() {
            try {
                JSONValue jsonMsg = JSONValue::parse(msg);
                processMessage(jsonMsg);
            } catch (const JSONValue::parse_error& e) {
                log(std::string("JSON Parse Error: ") + e.what());
                // If it's a parse error, we can't extract the id reliably, so id is null
                sendError(nullptr, PARSE_ERROR, "Parse error");
            } catch (const std::exception& e) {
                log(std::string("Worker exception: ") + e.what());
            }

            {
                std::lock_guard<std::mutex> lock(workersCvMutex_);
                activeWorkers_--;
            }
            workersCv_.notify_all();
        }).detach();
    } catch (const std::system_error& e) {
        log(std::string("Failed to spawn worker thread: ") + e.what());
        {
            std::lock_guard<std::mutex> lock(workersCvMutex_);
            activeWorkers_--;
        }
        workersCv_.notify_all();
    }
}

void MCPServer::processMessage(const JSONValue& msg) {
    if (!msg.is_object()) {
        sendError(nullptr, INVALID_REQUEST, "Invalid Request");
        return;
    }

    // Extract ID (could be null for notifications)
    JSONValue id = nullptr;
    if (msg.contains("id")) {
        id = msg["id"];
    }

    // Determine message type
    if (msg.contains("method")) {
        std::string method = msg["method"].get<std::string>();

        try {
            if (method == "initialize") {
                handleInitialize(msg);
            } else if (method == "notifications/initialized") {
                handleInitialized(msg);
            } else if (method == "tools/list") {
                handleToolsList(msg);
            } else if (method == "tools/call") {
                handleToolsCall(msg);
            } else if (method == "ping") {
                handlePing(msg);
            } else {
                if (!id.is_null()) {
                    sendError(id, METHOD_NOT_FOUND, "Method not found");
                }
            }
        } catch (const std::exception& e) {
            log(std::string("Error handling method ") + method + ": " + e.what());
            if (!id.is_null()) {
                sendError(id, INTERNAL_ERROR, e.what());
            }
        }
    } else if (msg.contains("result") || msg.contains("error")) {
        // We are a server, we typically don't receive results unless we send requests.
        // Ignore for now.
    } else {
        if (!id.is_null()) {
            sendError(id, INVALID_REQUEST, "Invalid Request");
        }
    }
}

void MCPServer::handleInitialize(const JSONValue& request) {
    JSONValue id = request["id"];

    JSONValue result = {
        {"protocolVersion", "2024-11-05"},
        {"capabilities", {
            {"tools", nlohmann::json::object()}
        }},
        {"serverInfo", {
            {"name", "mcp-server-cpp"},
            {"version", "1.0.0"}
        }}
    };

    sendResponse(id, result);
}

void MCPServer::handleInitialized(const JSONValue& notification) {
    // Notification, no response needed.
    initialized_ = true;
    log("Server initialized.");
}

void MCPServer::handleToolsList(const JSONValue& request) {
    JSONValue id = request["id"];

    auto tools = registry_.listTools();
    JSONValue jsonTools = JSONValue::array();

    for (const auto& tool : tools) {
        jsonTools.push_back({
            {"name", tool.name},
            {"description", tool.description},
            {"inputSchema", tool.inputSchema}
        });
    }

    JSONValue result = {
        {"tools", jsonTools}
    };

    sendResponse(id, result);
}

void MCPServer::handleToolsCall(const JSONValue& request) {
    JSONValue id = request["id"];

    if (!request.contains("params") || !request["params"].contains("name")) {
        sendError(id, INVALID_PARAMS, "Invalid params: missing tool name");
        return;
    }

    std::string toolName = request["params"]["name"].get<std::string>();
    JSONValue arguments = JSONValue::object();

    if (request["params"].contains("arguments")) {
        arguments = request["params"]["arguments"];
    }

    auto resultOpt = registry_.execute(toolName, arguments);
    if (!resultOpt) {
        // Tool not found
        sendError(id, INVALID_PARAMS, "Tool not found");
        return;
    }

    const auto& toolResult = *resultOpt;
    JSONValue responseResult = {
        {"content", toolResult.content},
        {"isError", toolResult.isError}
    };

    sendResponse(id, responseResult);
}

void MCPServer::handlePing(const JSONValue& request) {
    JSONValue id = request["id"];
    sendResponse(id, JSONValue::object());
}

void MCPServer::sendResponse(const JSONValue& id, const JSONValue& result) {
    JSONValue msg = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"result", result}
    };
    sendRaw(msg);
}

void MCPServer::sendError(const JSONValue& id, int code, std::string_view message) {
    JSONValue msg = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"error", {
            {"code", code},
            {"message", message}
        }}
    };
    sendRaw(msg);
}

void MCPServer::sendRaw(const JSONValue& msg) {
    std::lock_guard<std::mutex> lock(stdoutMutex_);
    std::cout << msg.dump() << "\n";
    std::cout.flush();
}

void MCPServer::log(std::string_view msg) {
    // Strict stdout isolation: all logging goes to stderr
    std::cerr << "[MCP Server] " << msg << "\n";
}

} // namespace mcp
