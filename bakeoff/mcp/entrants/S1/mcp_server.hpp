#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <map>
#include <functional>
#include <mutex>
#include <thread>
#include <optional>
#include <condition_variable>
#include <span>
#include <concepts>
#include <nlohmann/json.hpp>

namespace mcp {

using JSONValue = nlohmann::json;

struct ToolResult {
    std::vector<JSONValue> content;
    bool isError = false;
};

// C++20 Concept for a Tool Handler
template <typename T>
concept ToolHandler = requires(T t, const JSONValue& args) {
    { t(args) } -> std::same_as<ToolResult>;
};

struct Tool {
    std::string name;
    std::string description;
    JSONValue inputSchema;
    std::function<ToolResult(const JSONValue&)> handler;

    // Templated constructor using the concept
    template<ToolHandler H>
    Tool(std::string name_, std::string desc_, JSONValue schema_, H&& handler_)
        : name(std::move(name_)), description(std::move(desc_)),
          inputSchema(std::move(schema_)), handler(std::forward<H>(handler_)) {}
};

class ToolRegistry {
public:
    void registerTool(const Tool& tool);
    std::optional<Tool> getTool(std::string_view name) const;
    std::vector<Tool> listTools() const;

private:
    mutable std::mutex m_mutex;
    std::map<std::string, Tool, std::less<>> m_tools;
};

class MCPServer {
public:
    MCPServer(std::string name, std::string version);
    ~MCPServer();

    // Prevent copying
    MCPServer(const MCPServer&) = delete;
    MCPServer& operator=(const MCPServer&) = delete;

    void registerTool(const Tool& tool);
    // Use std::span (C++20) for registering multiple tools
    void registerTools(std::span<const Tool> tools);

    // Starts the non-blocking reader loop
    void start();

    // Blocks the caller until the reader loop finishes (EOF)
    void wait();

    // Stops the server
    void stop();

private:
    void loop();
    void handleMessage(const JSONValue& message);
    void handleRequest(const JSONValue& request);
    void handleNotification(const JSONValue& notification);

    void handleInitialize(const JSONValue& request);
    void handleInitialized(const JSONValue& notification);
    void handleToolsList(const JSONValue& request);
    void handleToolsCall(const JSONValue& request);
    void handlePing(const JSONValue& request);

    void sendResponse(const JSONValue& id, const JSONValue& result);
    void sendError(const JSONValue& id, int code, const std::string& message);

    void log(const std::string& message);

    std::string m_name;
    std::string m_version;
    bool m_running;

    ToolRegistry m_registry;

    std::jthread m_readerThread;

    std::mutex m_stdoutMutex;

    std::mutex m_waitMutex;
    std::condition_variable m_waitCv;
    bool m_finished = false;
};

} // namespace mcp
