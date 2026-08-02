#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <functional>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <future>
#include <iostream>
#include <optional>
#include <condition_variable>
#include <span>
#include <concepts>

namespace mcp {

using json = nlohmann::json;

// C++20 Concept: ensure T can be converted to std::string_view
template <typename T>
concept StringLike = std::convertible_to<T, std::string_view>;

// Define Tool Result content structure
struct ToolContent {
    std::string type; // e.g., "text"
    std::string text;
};

// Define the Tool Result output structure
struct ToolResult {
    std::vector<ToolContent> content;
    bool isError = false;

    // Helper constructor to easily create a text-based non-error response using Concepts
    template <StringLike T>
    static ToolResult fromText(T text, bool error = false) {
        return ToolResult{
            .content = {ToolContent{.type = "text", .text = std::string(std::string_view(text))}},
            .isError = error
        };
    }

    // Support std::span for binary/buffer-like text conversions
    static ToolResult fromSpan(std::span<const char> buffer, bool error = false) {
        return ToolResult{
            .content = {ToolContent{.type = "text", .text = std::string(buffer.data(), buffer.size())}},
            .isError = error
        };
    }
};

using ToolHandler = std::function<ToolResult(const json& arguments)>;

// Define a Tool
struct Tool {
    std::string name;
    std::string description;
    json inputSchema;
    ToolHandler handler;
};

// Thread-safe Tool Registry
class ToolRegistry {
public:
    void registerTool(Tool tool) {
        std::unique_lock lock(mutex_);
        tools_[tool.name] = std::move(tool);
    }

    std::vector<json> getToolsList() const {
        std::shared_lock lock(mutex_);
        std::vector<json> result;
        for (const auto& [name, tool] : tools_) {
            result.push_back({
                {"name", tool.name},
                {"description", tool.description},
                {"inputSchema", tool.inputSchema}
            });
        }
        return result;
    }

    std::optional<Tool> getTool(const std::string& name) const {
        std::shared_lock lock(mutex_);
        if (auto it = tools_.find(name); it != tools_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, Tool> tools_;
};

class MCPServer {
public:
    MCPServer();
    ~MCPServer();

    void registerTool(Tool tool) {
        registry_.registerTool(std::move(tool));
    }

    // Start the server (non-blocking if we just start the thread)
    void start();

    // Stop the server gracefully
    void stop();

    // Wait until server shuts down
    void wait();

private:
    void readLoop();
    void dispatchRequest(json request);

    // JSON-RPC Methods
    json handleInitialize(const json& params);
    void handleInitialized(const json& params);
    json handleToolsList(const json& params);
    json handleToolsCall(const json& params);
    json handlePing(const json& params);

    void sendResponse(const json& response);
    void sendError(const json& id, int code, const std::string& message);

    ToolRegistry registry_;
    std::atomic<bool> running_{false};
    std::optional<std::jthread> readerThread_;

    // Use-After-Free protection for detached workers
    std::atomic<int> activeWorkers_{0};
    std::condition_variable workersCv_;
    std::mutex workersMutex_;

    // Used to strictly synchronize stdout output (raw JSON-RPC frames)
    std::mutex stdoutMutex_;
};

} // namespace mcp
