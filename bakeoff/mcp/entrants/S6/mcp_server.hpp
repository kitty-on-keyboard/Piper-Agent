#pragma once

#include <string>
#include <vector>
#include <functional>
#include <shared_mutex>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <optional>
#include <iostream>
#include <queue>
#include <condition_variable>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace mcp {

struct ToolResult {
    json content;
    bool isError = false;
};

using ToolHandler = std::function<ToolResult(const json& arguments)>;

struct Tool {
    std::string name;
    std::string description;
    json inputSchema;
    ToolHandler handler;
};

class ToolRegistry {
public:
    void registerTool(const Tool& tool) {
        std::unique_lock lock(mutex_);
        tools_[tool.name] = tool;
    }

    std::optional<Tool> getTool(const std::string& name) const {
        std::shared_lock lock(mutex_);
        auto it = tools_.find(name);
        if (it != tools_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    std::vector<Tool> getAllTools() const {
        std::shared_lock lock(mutex_);
        std::vector<Tool> result;
        result.reserve(tools_.size());
        for (const auto& [name, tool] : tools_) {
            result.push_back(tool);
        }
        return result;
    }

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, Tool> tools_;
};

class MCPServer {
public:
    MCPServer(const std::string& name, const std::string& version, size_t num_workers = 4);
    ~MCPServer();

    void registerTool(const Tool& tool);
    void start();
    void stop();

private:
    void processLine(const std::string& line);
    void workerLoop(std::stop_token stoken);
    void handleRequest(const json& request);
    void sendResponse(const json& response);
    void sendError(const json& id, int code, const std::string& message);

    // Handlers
    void handleInitialize(const json& request);
    void handleInitialized(const json& request);
    void handleToolsList(const json& request);
    void handleToolsCall(const json& request);
    void handlePing(const json& request);

    std::string name_;
    std::string version_;

    ToolRegistry toolRegistry_;

    std::mutex stdoutMutex_;

    std::atomic<bool> running_{false};
    std::jthread readerThread_;

    // Thread pool for decoupled worker dispatch
    std::vector<std::jthread> workers_;
    std::queue<json> requestQueue_;
    std::mutex queueMutex_;
    std::condition_variable_any queueCV_;
};

} // namespace mcp
