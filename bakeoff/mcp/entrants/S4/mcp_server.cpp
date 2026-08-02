#include "mcp_server.hpp"
#include <poll.h>
#include <unistd.h>
#include <iostream>
#include <string>

namespace mcp {

Server::Server() {
}

Server::~Server() {
    stop();
}

void Server::run() {
    // Start reader thread
    readerThread_ = std::jthread([this](std::stop_token stoken) {
        readerThreadFunc(stoken);
    });

    // In this basic implementation, we'll block the main thread
    // until the reader thread exits (e.g., stdin closed).
    // Usually, the main thread might do other work or wait on a condition variable.
    // For this design, we can just join or wait.
    if (readerThread_.joinable()) {
        readerThread_.join();
    }
}

void Server::stop() {
    if (readerThread_.joinable()) {
        readerThread_.request_stop();
        readerThread_.join();
    }
}

void Server::log(const std::string& message) {
    // Strict isolation: Logging MUST go to stderr
    std::cerr << "[MCP Log] " << message << std::endl;
}

void Server::sendRaw(const JSONValue& message) {
    std::lock_guard lock(writeMutex_);
    std::cout << message.dump() << "\n";
    std::cout.flush();
}

void Server::sendResponse(const JSONValue& id, const JSONValue& result) {
    JSONValue response = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"result", result}
    };
    sendRaw(response);
}

void Server::sendError(const JSONValue& id, int code, const std::string& message) {
    JSONValue response = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"error", {
            {"code", code},
            {"message", message}
        }}
    };
    sendRaw(response);
}

void Server::readerThreadFunc(std::stop_token stoken) {
    struct pollfd pfd;
    pfd.fd = STDIN_FILENO;
    pfd.events = POLLIN;

    std::string buffer;

    while (!stoken.stop_requested()) {
        int ret = poll(&pfd, 1, 100); // 100ms timeout to allow checking stop_token

        if (ret < 0) {
            log("Poll error on stdin");
            break;
        }

        if (ret == 0) {
            // Timeout
            continue;
        }

        if (pfd.revents & (POLLERR | POLLHUP)) {
            log("Stdin closed or error");
            break;
        }

        if (pfd.revents & POLLIN) {
            // Read available data
            char chunk[4096];
            ssize_t bytesRead = read(STDIN_FILENO, chunk, sizeof(chunk));
            if (bytesRead > 0) {
                buffer.append(chunk, bytesRead);

                // Process lines (line-delimited JSON)
                size_t pos;
                while ((pos = buffer.find('\n')) != std::string::npos) {
                    std::string line = buffer.substr(0, pos);
                    buffer.erase(0, pos + 1);

                    if (!line.empty()) {
                        // Dispatch async to avoid stalling the reader thread.
                        // NOTE: Using detached threads is simple and works well for this lifetime,
                        // but a production framework might use a thread pool to avoid
                        // use-after-free if the Server object is destroyed while threads are running.
                        std::jthread worker([this, line]() {
                            try {
                                JSONValue request = JSONValue::parse(line);
                                dispatchRequest(request);
                            } catch (const nlohmann::json::parse_error& e) {
                                log("Parse error: " + std::string(e.what()));
                                sendError(nullptr, -32600, "Parse Error");
                            }
                        });
                        worker.detach();
                    }
                }
            } else if (bytesRead == 0) {
                log("Stdin reached EOF");
                break;
            } else {
                log("Read error on stdin");
                break;
            }
        }
    }
}

void Server::dispatchRequest(JSONValue request) {
    try {
        if (!request.contains("jsonrpc") || request["jsonrpc"] != "2.0") {
            sendError(request.value("id", JSONValue(nullptr)), -32600, "Invalid Request");
            return;
        }

        std::string method = request.value("method", "");

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
            sendError(request.value("id", JSONValue(nullptr)), -32601, "Method Not Found: " + method);
        }
    } catch (const std::exception& e) {
        log("Exception in dispatch: " + std::string(e.what()));
        sendError(request.value("id", JSONValue(nullptr)), -32603, "Internal Error");
    }
}

void Server::handleInitialize(const JSONValue& request) {
    JSONValue id = request.value("id", JSONValue(nullptr));

    JSONValue result = {
        {"protocolVersion", "2024-11-05"},
        {"capabilities", {
            {"tools", {
                {"listChanged", false}
            }}
        }},
        {"serverInfo", {
            {"name", "mcp-server-cpp"},
            {"version", "1.0.0"}
        }}
    };

    sendResponse(id, result);
    initialized_ = true;
}

void Server::handleInitialized(const JSONValue& request) {
    // Client has received initialize response. Nothing to reply to since it's a notification usually,
    // or just acknowledge. Notifications don't have IDs.
    log("Client initialized");
}

void Server::handleToolsList(const JSONValue& request) {
    JSONValue id = request.value("id", JSONValue(nullptr));

    auto tools = registry_.getTools();
    JSONValue toolsArray = JSONValue::array();

    for (const auto& tool : tools) {
        toolsArray.push_back({
            {"name", tool.name},
            {"description", tool.description},
            {"inputSchema", tool.inputSchema}
        });
    }

    JSONValue result = {
        {"tools", toolsArray}
    };

    sendResponse(id, result);
}

void Server::handleToolsCall(const JSONValue& request) {
    JSONValue id = request.value("id", JSONValue(nullptr));

    if (!request.contains("params") || !request["params"].contains("name")) {
        sendError(id, -32602, "Invalid Params: Missing tool name");
        return;
    }

    std::string name = request["params"]["name"];
    JSONValue arguments = request["params"].value("arguments", JSONValue::object());

    auto toolOpt = registry_.getTool(name);
    if (!toolOpt) {
        sendError(id, -32601, "Tool Not Found: " + name);
        return;
    }

    try {
        ToolResult toolResult = toolOpt->handler(arguments);

        JSONValue result = {
            {"content", toolResult.content},
            {"isError", toolResult.isError}
        };

        sendResponse(id, result);
    } catch (const std::exception& e) {
        log("Tool execution error: " + std::string(e.what()));
        sendError(id, -32603, "Internal Error during tool execution: " + std::string(e.what()));
    }
}

void Server::handlePing(const JSONValue& request) {
    JSONValue id = request.value("id", JSONValue(nullptr));
    sendResponse(id, JSONValue::object()); // Empty object for ping response
}

} // namespace mcp