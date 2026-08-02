#include "mcp_server.hpp"
#include <iostream>
#include <sstream>
#include <exception>

namespace mcp {

// Standard JSON-RPC Error Codes
constexpr int PARSE_ERROR = -32700;
constexpr int INVALID_REQUEST = -32600;
constexpr int METHOD_NOT_FOUND = -32601;
constexpr int INVALID_PARAMS = -32602;
constexpr int INTERNAL_ERROR = -32603;

MCPServer::MCPServer(const std::string& name, const std::string& version, size_t num_workers)
    : name_(name), version_(version) {
    for (size_t i = 0; i < num_workers; ++i) {
        workers_.emplace_back([this](std::stop_token stoken) { workerLoop(stoken); });
    }
}

MCPServer::~MCPServer() {
    stop();
}

void MCPServer::registerTool(const Tool& tool) {
    toolRegistry_.registerTool(tool);
}

void MCPServer::start() {
    if (running_.exchange(true)) {
        return; // Already running
    }

    std::cerr << "Starting MCP Server..." << std::endl;

    readerThread_ = std::jthread([this](std::stop_token stoken) {
        std::string line;
        while (!stoken.stop_requested() && std::getline(std::cin, line)) {
            if (line.empty()) continue;
            processLine(line);
        }
        running_ = false;
        queueCV_.notify_all();
    });
}

void MCPServer::stop() {
    if (!running_.exchange(false)) {
        return;
    }

    if (readerThread_.joinable()) {
        readerThread_.request_stop();
        readerThread_.join();
    }

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.request_stop();
        }
    }
    queueCV_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void MCPServer::processLine(const std::string& line) {
    try {
        json request = json::parse(line);
        {
            std::lock_guard lock(queueMutex_);
            requestQueue_.push(std::move(request));
        }
        queueCV_.notify_one();
    } catch (const json::parse_error& e) {
        std::cerr << "JSON Parse Error: " << e.what() << std::endl;
        sendError(nullptr, PARSE_ERROR, "Parse error");
    }
}

void MCPServer::workerLoop(std::stop_token stoken) {
    while (!stoken.stop_requested()) {
        json request;
        {
            std::unique_lock lock(queueMutex_);
            queueCV_.wait(lock, stoken, [this]() {
                return !requestQueue_.empty();
            });

            if (stoken.stop_requested() && requestQueue_.empty()) {
                return;
            }

            if (!requestQueue_.empty()) {
                request = std::move(requestQueue_.front());
                requestQueue_.pop();
            }
        }

        if (!request.is_null()) {
            handleRequest(request);
        }
    }
}

void MCPServer::handleRequest(const json& request) {
    if (!request.is_object()) {
        sendError(nullptr, INVALID_REQUEST, "Invalid Request");
        return;
    }

    std::string method;
    json id = nullptr;

    if (request.contains("id")) {
        id = request["id"];
    }

    if (request.contains("method") && request["method"].is_string()) {
        method = request["method"].get<std::string>();
    } else {
        if (!id.is_null()) sendError(id, INVALID_REQUEST, "Invalid Request");
        return;
    }

    try {
        if (method == "initialize") {
            handleInitialize(request);
        } else if (method == "notifications/initialized") {
            handleInitialized(request);
        } else if (method == "tools/list") {
            handleToolsList(request);
        } else if (method == "tools/call") {
            handleToolsCall(request);
        } else if (method == "ping") {
            handlePing(request);
        } else {
            if (!id.is_null()) {
                sendError(id, METHOD_NOT_FOUND, "Method not found");
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Internal Error handling " << method << ": " << e.what() << std::endl;
        if (!id.is_null()) {
            sendError(id, INTERNAL_ERROR, e.what());
        }
    } catch (...) {
        std::cerr << "Unknown Internal Error handling " << method << std::endl;
        if (!id.is_null()) {
            sendError(id, INTERNAL_ERROR, "Internal Error");
        }
    }
}

void MCPServer::sendResponse(const json& response) {
    std::string out = response.dump();
    std::lock_guard lock(stdoutMutex_);
    std::cout << out << "\n";
    std::cout.flush();
}

void MCPServer::sendError(const json& id, int code, const std::string& message) {
    json errorResp = {
        {"jsonrpc", "2.0"},
        {"error", {
            {"code", code},
            {"message", message}
        }}
    };
    if (!id.is_null()) {
        errorResp["id"] = id;
    }
    sendResponse(errorResp);
}

void MCPServer::handleInitialize(const json& request) {
    json id = request.value("id", json(nullptr));
    if (id.is_null()) return;

    json response = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"result", {
            {"protocolVersion", "2024-11-05"},
            {"capabilities", {
                {"tools", json::object()}
            }},
            {"serverInfo", {
                {"name", name_},
                {"version", version_}
            }}
        }}
    };
    sendResponse(response);
}

void MCPServer::handleInitialized(const json& /*request*/) {
    // Notification, no response needed
    std::cerr << "Client initialized." << std::endl;
}

void MCPServer::handleToolsList(const json& request) {
    json id = request.value("id", json(nullptr));
    if (id.is_null()) return;

    auto tools = toolRegistry_.getAllTools();
    json toolsJson = json::array();
    for (const auto& tool : tools) {
        toolsJson.push_back({
            {"name", tool.name},
            {"description", tool.description},
            {"inputSchema", tool.inputSchema}
        });
    }

    json response = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"result", {
            {"tools", toolsJson}
        }}
    };
    sendResponse(response);
}

void MCPServer::handleToolsCall(const json& request) {
    json id = request.value("id", json(nullptr));
    if (id.is_null()) return;

    if (!request.contains("params") || !request["params"].contains("name")) {
        sendError(id, INVALID_PARAMS, "Missing tool name");
        return;
    }

    std::string toolName = request["params"]["name"].get<std::string>();
    json arguments = json::object();
    if (request["params"].contains("arguments")) {
        arguments = request["params"]["arguments"];
    }

    auto toolOpt = toolRegistry_.getTool(toolName);
    if (!toolOpt) {
        sendError(id, INVALID_PARAMS, "Unknown tool: " + toolName);
        return;
    }

    try {
        ToolResult result = toolOpt->handler(arguments);

        // MCP standard for tool result
        json response = {
            {"jsonrpc", "2.0"},
            {"id", id},
            {"result", {
                {"content", result.content},
                {"isError", result.isError}
            }}
        };
        sendResponse(response);
    } catch (const std::exception& e) {
        std::cerr << "Tool execution error: " << e.what() << std::endl;
        json response = {
            {"jsonrpc", "2.0"},
            {"id", id},
            {"result", {
                {"content", {
                    {{"type", "text"}, {"text", e.what()}}
                }},
                {"isError", true}
            }}
        };
        sendResponse(response);
    }
}

void MCPServer::handlePing(const json& request) {
    json id = request.value("id", json(nullptr));
    if (id.is_null()) return;

    json response = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"result", json::object()}
    };
    sendResponse(response);
}

} // namespace mcp
