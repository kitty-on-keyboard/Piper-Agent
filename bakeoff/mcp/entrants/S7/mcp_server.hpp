#pragma once

#include <string>
#include <string_view>
#include <functional>
#include <unordered_map>
#include <shared_mutex>
#include <thread>
#include <vector>
#include <optional>
#include <span>
#include <concepts>
#include <mutex>
#include <nlohmann/json.hpp>

namespace mcp {

using JSONValue = nlohmann::json;

// Standardized Tool Result
struct ToolResult {
    JSONValue content;
    bool isError = false;
};

// Concept for a valid tool handler
template <typename T>
concept ToolHandler = requires(T t, const JSONValue& args) {
    { t(args) } -> std::same_as<ToolResult>;
};

// Abstract Tool Structure
struct Tool {
    std::string name;
    std::string description;
    JSONValue inputSchema;
    std::function<ToolResult(const JSONValue&)> handler;
};

// Thread-safe Tool Registry
class ToolRegistry {
public:
    ToolRegistry() = default;

    void registerTool(Tool tool);
    std::optional<Tool> getTool(std::string_view name) const;
    std::vector<Tool> listTools() const;

    // Executes the tool handler
    ToolResult execute(std::string_view name, const JSONValue& arguments) const;

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, Tool> tools_;
};

// Error Codes as per JSON-RPC 2.0
enum class ErrorCode : int {
    ParseError = -32700,
    InvalidRequest = -32600,
    MethodNotFound = -32601,
    InvalidParams = -32602,
    InternalError = -32603
};

class McpServer {
public:
    McpServer() = default;
    ~McpServer();

    // Prevent copy and move
    McpServer(const McpServer&) = delete;
    McpServer& operator=(const McpServer&) = delete;

    // Declarative tool registration
    void registerTool(Tool tool);

    // Starts the listening loop on stdin
    void start();

    // Requests stop and waits for the reader thread
    void stop();

private:
    // Asynchronous reader loop
    void readerLoop(std::stop_token stoken);

    // Decoupled worker dispatch
    void dispatchWorker(std::string message);

    // Internal message processing
    void processMessage(const std::string& message);
    void handleRequest(const JSONValue& request);
    void handleNotification(const JSONValue& notification);

    // Core endpoints
    void handleInitialize(const JSONValue& request);
    void handleToolsList(const JSONValue& request);
    void handleToolsCall(const JSONValue& request);
    void handlePing(const JSONValue& request);

    // Output formatting helpers
    void sendResponse(const JSONValue& id, const JSONValue& result);
    void sendError(const JSONValue& id, ErrorCode code, std::string_view message);

    ToolRegistry registry_;
    std::jthread readerThread_;
    bool initialized_ = false;

    // Mutex for writing to stdout to ensure JSON text frames don't interleave
    std::mutex stdoutMutex_;
};

} // namespace mcp
