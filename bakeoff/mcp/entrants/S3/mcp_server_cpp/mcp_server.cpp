#include "mcp_server.hpp"

#include <iostream>
#include <sstream>
#include <string>

namespace mcp {

// JSON-RPC Error Codes
constexpr int PARSE_ERROR = -32700;
constexpr int INVALID_REQUEST = -32600;
constexpr int METHOD_NOT_FOUND = -32601;
constexpr int INVALID_PARAMS = -32602;
constexpr int INTERNAL_ERROR = -32603;

McpServer::McpServer(std::string server_name, std::string server_version)
    : m_name(std::move(server_name)), m_version(std::move(server_version)) {
}

McpServer::~McpServer() {
    stop();
}

void McpServer::start() {
    if (m_running.exchange(true)) {
        return; // Already running
    }

    // Start reader thread. Decouples IO reading from potential main thread tasks.
    m_readerThread = std::make_unique<std::jthread>([this](std::stop_token stoken) {
        this->readLoop();
    });
}

void McpServer::stop() {
    if (m_running.exchange(false)) {
        if (m_readerThread && m_readerThread->joinable()) {
            m_readerThread->request_stop();
            if (m_readerThread->get_id() != std::this_thread::get_id()) {
                m_readerThread->join();
            }
        }

        // Clean up workers
        {
            std::lock_guard<std::mutex> lock(m_workersMutex);
            m_workers.clear(); // std::jthread destruction will request_stop and join
        }

        // Notify wait condition
        {
            std::lock_guard<std::mutex> lock(m_waitMutex);
            m_finished = true;
        }
        m_waitCv.notify_all();
    }
}

void McpServer::wait() {
    std::unique_lock<std::mutex> lock(m_waitMutex);
    m_waitCv.wait(lock, [this] { return m_finished; });
}

void McpServer::readLoop() {
    std::string line;
    // Keep reading until EOF or running flag is cleared
    while (m_running && std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }

        // Clean up completed worker threads occasionally to avoid unbounded growth
        {
            std::lock_guard<std::mutex> lock(m_workersMutex);
            std::erase_if(m_workers, [](const std::jthread& t) {
                // If we can get a stop token and the thread is not joinable, it's done.
                // A simpler check is just checking if we can't join it, but in jthread we need to rely on the function finishing.
                // Wait for 0 ms to see if we can acquire the thread (we can't easily poll jthread completion).
                // Actually C++20 doesn't have an easy is_finished() on threads.
                // We'll just leave them in the vector to join on shutdown for simplicity,
                // or we could use std::future, but the scope dictates just basic worker tracking.
                // To keep it simple and safe from OOM in a long-running app, we'll keep them in m_workers.
                return false;
            });
        }

        // Spawn a new jthread to handle the message and store it
        {
            std::lock_guard<std::mutex> lock(m_workersMutex);
            m_workers.emplace_back([this, msg = std::move(line)]() {
                this->handleMessage(msg);
            });
        }
    }
    stop(); // Call stop when EOF is reached
}

void McpServer::handleMessage(const std::string& message) {
    try {
        JSONValue json_msg = JSONValue::parse(message);

        // Validate basic JSON-RPC 2.0 structure
        if (!json_msg.contains("jsonrpc") || json_msg["jsonrpc"] != "2.0") {
            std::cerr << "Invalid jsonrpc version" << std::endl;
            if (json_msg.contains("id")) {
                sendError(json_msg["id"], INVALID_REQUEST, "Invalid JSON-RPC version");
            }
            return;
        }

        if (json_msg.contains("method")) {
            if (json_msg.contains("id")) {
                // Request
                handleRequest(json_msg);
            } else {
                // Notification
                handleNotification(json_msg);
            }
        } else {
            // Response (server typically doesn't handle responses unless it sends requests)
            std::cerr << "Received response payload, ignoring" << std::endl;
        }

    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "JSON Parse error: " << e.what() << std::endl;
        // Parse error means we don't know the ID
        JSONValue null_id = nullptr;
        sendError(null_id, PARSE_ERROR, "Parse error");
    } catch (const std::exception& e) {
        std::cerr << "Exception in handleMessage: " << e.what() << std::endl;
        if (auto json_msg = JSONValue::parse(message, nullptr, false); !json_msg.is_discarded() && json_msg.contains("id")) {
             sendError(json_msg["id"], INTERNAL_ERROR, e.what());
        }
    }
}

void McpServer::handleRequest(const JSONValue& request) {
    const std::string method = request["method"];

    if (method == "initialize") {
        handleInitialize(request);
    } else if (method == "ping") {
        handlePing(request);
    } else if (method == "tools/list") {
        handleToolsList(request);
    } else if (method == "tools/call") {
        handleToolsCall(request);
    } else {
        std::cerr << "Method not found: " << method << std::endl;
        sendError(request["id"], METHOD_NOT_FOUND, "Method not found");
    }
}

void McpServer::handleNotification(const JSONValue& notification) {
    const std::string method = notification["method"];
    // Most notifications we might just log or ignore
    if (method == "notifications/initialized") {
        std::cerr << "Client initialized." << std::endl;
    } else {
        std::cerr << "Unhandled notification: " << method << std::endl;
    }
}

void McpServer::handleInitialize(const JSONValue& request) {
    JSONValue result = {
        {"protocolVersion", "2024-11-05"},
        {"capabilities", {
            {"tools", {}} // Advertising tools capability
        }},
        {"serverInfo", {
            {"name", m_name},
            {"version", m_version}
        }}
    };
    sendResponse(request["id"], result);
}

void McpServer::handlePing(const JSONValue& request) {
    JSONValue result = JSONValue::object();
    sendResponse(request["id"], result);
}

void McpServer::handleToolsList(const JSONValue& request) {
    auto tools = m_registry.getAllTools();
    JSONValue json_tools = JSONValue::array();

    for (const auto& tool : tools) {
        JSONValue t = {
            {"name", tool.name},
            {"description", tool.description},
            {"inputSchema", tool.inputSchema}
        };
        json_tools.push_back(t);
    }

    JSONValue result = {
        {"tools", json_tools}
    };
    sendResponse(request["id"], result);
}

void McpServer::handleToolsCall(const JSONValue& request) {
    try {
        if (!request.contains("params") || !request["params"].contains("name")) {
            sendError(request["id"], INVALID_PARAMS, "Missing tool name in params");
            return;
        }

        std::string name = request["params"]["name"];
        JSONValue args = request["params"].value("arguments", JSONValue::object());

        auto tool = m_registry.getTool(name);
        if (!tool) {
            sendError(request["id"], INVALID_PARAMS, "Tool not found");
            return;
        }

        // Execute tool handler
        ToolResult res = tool->handler(args);

        JSONValue result = {
            {"content", res.content},
            {"isError", res.isError}
        };

        sendResponse(request["id"], result);

    } catch (const std::exception& e) {
        std::cerr << "Error executing tool: " << e.what() << std::endl;
        sendError(request["id"], INTERNAL_ERROR, e.what());
    }
}

void McpServer::sendResponse(const JSONValue& id, const JSONValue& result) {
    JSONValue response = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"result", result}
    };

    std::string out = response.dump();

    std::lock_guard<std::mutex> lock(m_stdoutMutex);
    std::cout << out << "\n" << std::flush;
}

void McpServer::sendError(const JSONValue& id, int code, const std::string& message) {
    JSONValue error = {
        {"code", code},
        {"message", message}
    };

    JSONValue response = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"error", error}
    };

    std::string out = response.dump();

    std::lock_guard<std::mutex> lock(m_stdoutMutex);
    std::cout << out << "\n" << std::flush;
}

} // namespace mcp