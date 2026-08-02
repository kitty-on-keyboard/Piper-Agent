#include "mcp_server.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace mcp {

// Error Codes
constexpr int PARSE_ERROR = -32700;
constexpr int INVALID_REQUEST = -32600;
constexpr int METHOD_NOT_FOUND = -32601;
constexpr int INVALID_PARAMS = -32602;
constexpr int INTERNAL_ERROR = -32603;

void ToolRegistry::registerTool(const Tool& tool) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_tools.insert_or_assign(tool.name, tool);
}

std::optional<Tool> ToolRegistry::getTool(std::string_view name) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_tools.find(name);
    if (it != m_tools.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::vector<Tool> ToolRegistry::listTools() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<Tool> tools;
    tools.reserve(m_tools.size());
    for (const auto& [_, tool] : m_tools) {
        tools.push_back(tool);
    }
    return tools;
}

MCPServer::MCPServer(std::string name, std::string version)
    : m_name(std::move(name)), m_version(std::move(version)), m_running(false) {}

MCPServer::~MCPServer() {
    stop();
}

void MCPServer::registerTool(const Tool& tool) {
    m_registry.registerTool(tool);
}

void MCPServer::registerTools(std::span<const Tool> tools) {
    for (const auto& tool : tools) {
        m_registry.registerTool(tool);
    }
}

void MCPServer::start() {
    if (m_running) return;
    m_running = true;
    m_finished = false;
    m_readerThread = std::jthread([this]() {
        loop();

        std::lock_guard<std::mutex> lock(m_waitMutex);
        m_finished = true;
        m_waitCv.notify_all();
    });
}

void MCPServer::wait() {
    std::unique_lock<std::mutex> lock(m_waitMutex);
    m_waitCv.wait(lock, [this]() { return m_finished || !m_running; });
}

void MCPServer::stop() {
    if (!m_running) return;
    m_running = false;

    // Unblock any waiters just in case, though they wait for m_finished or !m_running
    {
        std::lock_guard<std::mutex> lock(m_waitMutex);
        m_waitCv.notify_all();
    }
}

void MCPServer::log(const std::string& message) {
    std::cerr << "[MCPServer] " << message << std::endl;
}

void MCPServer::sendResponse(const JSONValue& id, const JSONValue& result) {
    JSONValue response = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"result", result}
    };
    std::string responseStr = response.dump();

    std::lock_guard<std::mutex> lock(m_stdoutMutex);
    std::cout << responseStr << "\r\n";
    std::cout.flush();
}

void MCPServer::sendError(const JSONValue& id, int code, const std::string& message) {
    JSONValue response = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"error", {
            {"code", code},
            {"message", message}
        }}
    };
    std::string responseStr = response.dump();

    std::lock_guard<std::mutex> lock(m_stdoutMutex);
    std::cout << responseStr << "\r\n";
    std::cout.flush();
}

void MCPServer::loop() {
    std::string line;
    while (m_running && std::getline(std::cin, line)) {
        if (line.empty() || line == "\r") {
            continue;
        }

        try {
            JSONValue message = JSONValue::parse(line);
            handleMessage(message);
        } catch (const nlohmann::json::parse_error& e) {
            log(std::string("Parse error: ") + e.what());
            sendError(nullptr, PARSE_ERROR, "Parse error");
        } catch (const std::exception& e) {
            log(std::string("Internal error: ") + e.what());
            sendError(nullptr, INTERNAL_ERROR, "Internal error");
        }
    }
}

void MCPServer::handleMessage(const JSONValue& message) {
    if (!message.contains("jsonrpc") || message["jsonrpc"] != "2.0") {
        sendError(nullptr, INVALID_REQUEST, "Invalid JSON-RPC version");
        return;
    }

    if (message.contains("id")) {
        handleRequest(message);
    } else {
        handleNotification(message);
    }
}

void MCPServer::handleRequest(const JSONValue& request) {
    auto id = request["id"];
    if (!request.contains("method") || !request["method"].is_string()) {
        sendError(id, INVALID_REQUEST, "Method not provided or invalid");
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
            sendError(id, METHOD_NOT_FOUND, "Method not found: " + method);
        }
    } catch (const std::exception& e) {
        log(std::string("Error handling request: ") + e.what());
        sendError(id, INTERNAL_ERROR, e.what());
    }
}

void MCPServer::handleNotification(const JSONValue& notification) {
    if (!notification.contains("method") || !notification["method"].is_string()) {
        log("Invalid notification");
        return;
    }

    std::string method = notification["method"];
    if (method == "notifications/initialized") {
        handleInitialized(notification);
    } else if (method == "cancelled") {
        // Handle cancel
    } else {
        log("Unhandled notification: " + method);
    }
}

void MCPServer::handleInitialize(const JSONValue& request) {
    JSONValue result = {
        {"protocolVersion", "2024-11-05"},
        {"capabilities", {
            {"tools", {
                {"listChanged", false}
            }}
        }},
        {"serverInfo", {
            {"name", m_name},
            {"version", m_version}
        }}
    };
    sendResponse(request["id"], result);
}

void MCPServer::handleInitialized(const JSONValue& notification) {
    log("Client initialized");
}

void MCPServer::handleToolsList(const JSONValue& request) {
    auto tools = m_registry.listTools();
    JSONValue toolsJson = JSONValue::array();

    for (const auto& tool : tools) {
        toolsJson.push_back({
            {"name", tool.name},
            {"description", tool.description},
            {"inputSchema", tool.inputSchema}
        });
    }

    JSONValue result = {
        {"tools", toolsJson}
    };
    sendResponse(request["id"], result);
}

void MCPServer::handleToolsCall(const JSONValue& request) {
    auto id = request["id"];
    if (!request.contains("params") || !request["params"].contains("name")) {
        sendError(id, INVALID_PARAMS, "Missing tool name in params");
        return;
    }

    std::string name = request["params"]["name"];
    auto toolOpt = m_registry.getTool(name);

    if (!toolOpt) {
        sendError(id, INVALID_PARAMS, "Tool not found: " + name);
        return;
    }

    JSONValue arguments = JSONValue::object();
    if (request["params"].contains("arguments")) {
        arguments = request["params"]["arguments"];
    }

    // Decoupled worker dispatch:
    // Execute tool callback in a separate jthread to avoid stalling protocol parsing.
    // The detached jthread will execute and send the response back securely via thread-safe methods.
    std::jthread worker([this, id, toolOpt, arguments]() {
        try {
            ToolResult toolResult = toolOpt->handler(arguments);
            JSONValue result = {
                {"content", toolResult.content},
                {"isError", toolResult.isError}
            };
            sendResponse(id, result);
        } catch (const std::exception& e) {
            log(std::string("Error executing tool: ") + e.what());
            JSONValue result = {
                {"content", {{
                    {"type", "text"},
                    {"text", std::string("Error executing tool: ") + e.what()}
                }}},
                {"isError", true}
            };
            sendResponse(id, result);
        }
    });
    worker.detach();
}

void MCPServer::handlePing(const JSONValue& request) {
    sendResponse(request["id"], JSONValue::object());
}

} // namespace mcp
