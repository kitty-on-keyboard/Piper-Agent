#include "mcp_server.hpp"
#include <iostream>
#include <format>
#include <stdexcept>
#include <thread>

namespace mcp {

// ToolRegistry Implementation

void ToolRegistry::registerTool(Tool tool) {
    std::unique_lock lock(mutex_);
    tools_[tool.name] = std::move(tool);
}

std::optional<Tool> ToolRegistry::getTool(std::string_view name) const {
    std::shared_lock lock(mutex_);
    std::string key(name);
    if (auto it = tools_.find(key); it != tools_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::vector<Tool> ToolRegistry::listTools() const {
    std::shared_lock lock(mutex_);
    std::vector<Tool> result;
    result.reserve(tools_.size());
    for (const auto& [name, tool] : tools_) {
        result.push_back(tool);
    }
    return result;
}

ToolResult ToolRegistry::execute(std::string_view name, const JSONValue& arguments) const {
    std::shared_lock lock(mutex_);
    std::string key(name);
    auto it = tools_.find(key);
    if (it == tools_.end()) {
        throw std::runtime_error("Tool not found");
    }
    return it->second.handler(arguments);
}


// McpServer Implementation

McpServer::~McpServer() {
    stop();
}

void McpServer::registerTool(Tool tool) {
    registry_.registerTool(std::move(tool));
}

void McpServer::start() {
    readerThread_ = std::jthread([this](std::stop_token stoken) {
        this->readerLoop(stoken);
    });
}

void McpServer::stop() {
    if (readerThread_.joinable()) {
        readerThread_.request_stop();
        readerThread_.join();
    }
}

void McpServer::readerLoop(std::stop_token stoken) {
    std::string line;
    while (!stoken.stop_requested() && std::getline(std::cin, line)) {
        if (line.empty()) continue;
        dispatchWorker(std::move(line));
    }
}

void McpServer::dispatchWorker(std::string message) {
    std::thread([this, msg = std::move(message)]() {
        processMessage(msg);
    }).detach();
}

void McpServer::processMessage(const std::string& message) {
    try {
        JSONValue parsed = JSONValue::parse(message);

        // JSON-RPC validation
        if (!parsed.contains("jsonrpc") || parsed["jsonrpc"] != "2.0") {
            sendError(nullptr, ErrorCode::InvalidRequest, "Invalid JSON-RPC 2.0 format");
            return;
        }

        if (parsed.contains("id")) {
            handleRequest(parsed);
        } else {
            handleNotification(parsed);
        }
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "Parse Error: " << e.what() << std::endl;
        sendError(nullptr, ErrorCode::ParseError, "Parse error");
    } catch (const std::exception& e) {
        std::cerr << "Internal Error: " << e.what() << std::endl;
        sendError(nullptr, ErrorCode::InternalError, "Internal error");
    }
}

void McpServer::handleRequest(const JSONValue& request) {
    if (!request.contains("method") || !request["method"].is_string()) {
        sendError(request["id"], ErrorCode::InvalidRequest, "Invalid request method");
        return;
    }

    std::string method = request["method"];

    try {
        if (method == "initialize") {
            handleInitialize(request);
        } else if (method == "tools/list") {
            handleToolsList(request);
        } else if (method == "tools/call") {
            handleToolsCall(request);
        } else if (method == "ping") {
            handlePing(request);
        } else {
            sendError(request["id"], ErrorCode::MethodNotFound, "Method not found");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error handling request: " << e.what() << std::endl;
        sendError(request["id"], ErrorCode::InternalError, e.what());
    }
}

void McpServer::handleNotification(const JSONValue& notification) {
    if (!notification.contains("method") || !notification["method"].is_string()) {
        std::cerr << "Invalid notification method" << std::endl;
        return;
    }
    std::string method = notification["method"];
    if (method == "notifications/initialized") {
        initialized_ = true;
        std::cerr << "Server initialized successfully." << std::endl;
    } else {
        std::cerr << "Unhandled notification: " << method << std::endl;
    }
}

void McpServer::handleInitialize(const JSONValue& request) {
    JSONValue result = {
        {"protocolVersion", "2024-11-05"},
        {"capabilities", {
            {"tools", {
                {"listChanged", false}
            }}
        }},
        {"serverInfo", {
            {"name", "mcp-cpp20-server"},
            {"version", "1.0.0"}
        }}
    };
    sendResponse(request["id"], result);
}

void McpServer::handleToolsList(const JSONValue& request) {
    if (!initialized_) {
        sendError(request["id"], ErrorCode::InvalidRequest, "Server not initialized");
        return;
    }

    std::vector<Tool> tools = registry_.listTools();
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
    sendResponse(request["id"], result);
}

void McpServer::handleToolsCall(const JSONValue& request) {
    if (!initialized_) {
        sendError(request["id"], ErrorCode::InvalidRequest, "Server not initialized");
        return;
    }

    if (!request.contains("params") || !request["params"].contains("name")) {
        sendError(request["id"], ErrorCode::InvalidParams, "Missing tool name in params");
        return;
    }

    std::string toolName = request["params"]["name"];
    JSONValue arguments = request["params"].value("arguments", JSONValue::object());

    try {
        ToolResult toolResult = registry_.execute(toolName, arguments);
        JSONValue result = {
            {"content", toolResult.content},
            {"isError", toolResult.isError}
        };
        sendResponse(request["id"], result);
    } catch (const std::exception& e) {
        std::cerr << "Tool execution error: " << e.what() << std::endl;
        sendError(request["id"], ErrorCode::InternalError, e.what());
    }
}

void McpServer::handlePing(const JSONValue& request) {
    sendResponse(request["id"], JSONValue::object());
}

void McpServer::sendResponse(const JSONValue& id, const JSONValue& result) {
    JSONValue response = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"result", result}
    };

    std::string jsonStr = response.dump();

    std::lock_guard lock(stdoutMutex_);
    std::cout << jsonStr << "\n" << std::flush;
}

void McpServer::sendError(const JSONValue& id, ErrorCode code, std::string_view message) {
    JSONValue response = {
        {"jsonrpc", "2.0"},
        {"error", {
            {"code", static_cast<int>(code)},
            {"message", message}
        }}
    };

    if (!id.is_null()) {
        response["id"] = id;
    } else {
        // Technically id should be null if parse error or similar, but adding explicit null
        response["id"] = nullptr;
    }

    std::string jsonStr = response.dump();

    std::lock_guard lock(stdoutMutex_);
    std::cout << jsonStr << "\n" << std::flush;
}

} // namespace mcp
