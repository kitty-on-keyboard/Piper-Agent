#pragma once

#include <string>
#include <vector>
#include <functional>
#include <shared_mutex>
#include <unordered_map>
#include <thread>
#include <stop_token>
#include <mutex>
#include <optional>
#include <iostream>
#include <nlohmann/json.hpp>

namespace mcp {

using JSONValue = nlohmann::json;

// Result of a tool execution
struct ToolResult {
    JSONValue content; // e.g., [{"type": "text", "text": "..."}]
    bool isError = false;
};

// Callback for tool execution
using ToolHandler = std::function<ToolResult(const JSONValue& arguments)>;

// Registration definition of a tool
struct Tool {
    std::string name;
    std::string description;
    JSONValue inputSchema;
    ToolHandler handler;
};

// Thread-safe registry for managing tools
class ToolRegistry {
public:
    void registerTool(Tool tool) {
        std::unique_lock lock(mutex_);
        tools_[tool.name] = std::move(tool);
    }

    std::vector<Tool> getTools() const {
        std::shared_lock lock(mutex_);
        std::vector<Tool> result;
        result.reserve(tools_.size());
        for (const auto& [name, tool] : tools_) {
            result.push_back(tool);
        }
        return result;
    }

    std::optional<Tool> getTool(const std::string& name) const {
        std::shared_lock lock(mutex_);
        auto it = tools_.find(name);
        if (it != tools_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, Tool> tools_;
};

// Main Server class
class Server {
public:
    Server();
    ~Server();

    // Prevent copy and move
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;
    Server(Server&&) = delete;
    Server& operator=(Server&&) = delete;

    // Easy access to register tools
    void registerTool(Tool tool) {
        registry_.registerTool(std::move(tool));
    }

    // Start the reader thread and wait (or detach if designed differently)
    // For simplicity, we can have a run() function that blocks, or start() and wait().
    void run();
    void stop();

private:
    void readerThreadFunc(std::stop_token stoken);
    void dispatchRequest(JSONValue request);

    // Core Handlers
    void handleInitialize(const JSONValue& request);
    void handleInitialized(const JSONValue& request);
    void handleToolsList(const JSONValue& request);
    void handleToolsCall(const JSONValue& request);
    void handlePing(const JSONValue& request);

    // I/O Helpers
    void sendResponse(const JSONValue& id, const JSONValue& result);
    void sendError(const JSONValue& id, int code, const std::string& message);
    void sendRaw(const JSONValue& message);
    void log(const std::string& message);

    ToolRegistry registry_;
    std::jthread readerThread_;
    std::mutex writeMutex_;
    bool initialized_ = false;
};

} // namespace mcp