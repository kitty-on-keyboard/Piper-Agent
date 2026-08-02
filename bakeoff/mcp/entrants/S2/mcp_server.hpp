#pragma once

#include <string>
#include <string_view>
#include <functional>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <optional>
#include <condition_variable>
#include <atomic>
#include <nlohmann/json.hpp>
#include <span>

namespace mcp {

using JSONValue = nlohmann::json;

// Example of C++20 Concept usage to constrain tool names or inputs
template <typename T>
concept StringLike = std::convertible_to<T, std::string_view>;

// Standard MCP tool result structure
struct ToolResult {
    bool isError = false;
    JSONValue content; // Usually an array of content objects e.g., [{"type": "text", "text": "..."}]
};

// Handler signature
using ToolHandler = std::function<ToolResult(const JSONValue& arguments)>;

// Abstract Tool structure
struct Tool {
    std::string name;
    std::string description;
    JSONValue inputSchema;
    ToolHandler handler;
};

// Thread-safe registry that indexes registered tools
class ToolRegistry {
public:
    void registerTool(const Tool& tool);

    template<StringLike NameType>
    std::optional<Tool> getTool(const NameType& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tools_.find(std::string(std::string_view(name)));
        if (it != tools_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    std::vector<Tool> listTools() const;

    // Validates incoming call payloads and executes tool handlers.
    // Returns nullopt if the tool is not found.
    template<StringLike NameType>
    std::optional<ToolResult> execute(const NameType& name, const JSONValue& arguments) const {
        auto toolOpt = getTool(name);
        if (!toolOpt) {
            return std::nullopt;
        }
        return toolOpt->handler(arguments);
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Tool> tools_;
};

// Main MCP Server class
class MCPServer {
public:
    MCPServer();
    ~MCPServer();

    // Prevent copy and move
    MCPServer(const MCPServer&) = delete;
    MCPServer& operator=(const MCPServer&) = delete;

    // Provide a declarative C++ API to register custom tools dynamically
    void registerTool(const Tool& tool);

    // Starts the server listening loop
    void start();

    // Graceful shutdown
    void stop();

    // Wait until stopped
    void wait();

private:
    // Asynchronous reader thread
    void readerThreadLoop(std::stop_token stoken);

    // Decoupled worker dispatch
    void dispatchMessage(std::string rawMessage);
    void processMessage(const JSONValue& msg);

    // Handlers for Core MCP Specification Endpoints
    void handleInitialize(const JSONValue& request);
    void handleInitialized(const JSONValue& notification);
    void handleToolsList(const JSONValue& request);
    void handleToolsCall(const JSONValue& request);
    void handlePing(const JSONValue& request);

    // Utility for JSON-RPC
    void sendResponse(const JSONValue& id, const JSONValue& result);
    void sendError(const JSONValue& id, int code, std::string_view message);
    void sendRaw(const JSONValue& msg);

    // Logging strictly to stderr
    void log(std::string_view msg);

    ToolRegistry registry_;

    // Thread safety for stdout
    std::mutex stdoutMutex_;

    bool initialized_ = false;

    std::atomic<int> activeWorkers_{0};
    std::condition_variable workersCv_;
    std::mutex workersCvMutex_;

    // Must be at the bottom so it is destroyed first,
    // preventing access to destroyed mutexes or registries.
    std::jthread readerThread_;
};

} // namespace mcp
