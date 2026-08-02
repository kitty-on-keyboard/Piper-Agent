#pragma once

#include <string>
#include <vector>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <optional>
#include <string_view>
#include <iostream>
#include <span>
#include <atomic>
#include <memory>
#include <condition_variable>

#include <nlohmann/json.hpp>

namespace mcp {

using JSONValue = nlohmann::json;

// Standard MCP Tool Result
struct ToolResult {
    std::vector<JSONValue> content;
    bool isError = false;

    // Helper to create simple text result
    static ToolResult text(const std::string& text) {
        JSONValue item = {
            {"type", "text"},
            {"text", text}
        };
        return {{item}, false};
    }

    // Helper to create simple error result
    static ToolResult error(const std::string& error_msg) {
        JSONValue item = {
            {"type", "text"},
            {"text", error_msg}
        };
        return {{item}, true};
    }
};

// Abstract Tool structure
struct Tool {
    std::string name;
    std::string description;
    JSONValue inputSchema;
    std::function<ToolResult(const JSONValue& arguments)> handler;
};

// Thread-safe tool registry
class ToolRegistry {
public:
    void registerTool(Tool tool) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_tools[tool.name] = std::move(tool);
    }

    std::optional<Tool> getTool(const std::string& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_tools.find(name);
        if (it != m_tools.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    std::vector<Tool> getAllTools() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<Tool> result;
        result.reserve(m_tools.size());
        for (const auto& [name, tool] : m_tools) {
            result.push_back(tool);
        }
        return result;
    }

private:
    mutable std::mutex m_mutex;
    std::unordered_map<std::string, Tool> m_tools;
};

// Core MCP Server
class McpServer {
public:
    McpServer(std::string server_name, std::string server_version);
    ~McpServer();

    // Disable copy/move
    McpServer(const McpServer&) = delete;
    McpServer& operator=(const McpServer&) = delete;

    // Provide access to registry to add tools before starting
    ToolRegistry& getRegistry() { return m_registry; }

    // Start the server loop
    void start();

    // Stop the server
    void stop();

    // Block until server shuts down
    void wait();

private:
    std::string m_name;
    std::string m_version;
    ToolRegistry m_registry;
    std::atomic<bool> m_running{false};
    std::unique_ptr<std::jthread> m_readerThread;

    // For waiting in main thread
    std::mutex m_waitMutex;
    std::condition_variable m_waitCv;
    bool m_finished{false};

    // Tracking worker threads
    std::mutex m_workersMutex;
    std::vector<std::jthread> m_workers;

    // Synchronize stdout output
    std::mutex m_stdoutMutex;

    // Core read loop
    void readLoop();

    // Protocol handling
    void handleMessage(const std::string& message);
    void handleRequest(const JSONValue& request);
    void handleNotification(const JSONValue& notification);

    // Endpoints
    void handleInitialize(const JSONValue& request);
    void handlePing(const JSONValue& request);
    void handleToolsList(const JSONValue& request);
    void handleToolsCall(const JSONValue& request);

    // Helpers
    void sendResponse(const JSONValue& id, const JSONValue& result);
    void sendError(const JSONValue& id, int code, const std::string& message);
};

} // namespace mcp
