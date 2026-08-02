#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <future>
#include <atomic>
#include <mutex>
#include <thread>
#include <span>
#include <string_view>
#include <nlohmann/json.hpp>

namespace mcp {

class McpError : public std::runtime_error {
public:
    McpError(const std::string& message) : std::runtime_error(message) {}
};

class McpClient {
public:
    McpClient();
    ~McpClient();

    // Delete copy and move
    McpClient(const McpClient&) = delete;
    McpClient& operator=(const McpClient&) = delete;
    McpClient(McpClient&&) = delete;
    McpClient& operator=(McpClient&&) = delete;

    // Lifecycle
    void start(const std::string& command, const std::vector<std::string>& args);
    void stop();

    // Core MCP Operations
    nlohmann::json initialize(const std::string& client_name, const std::string& client_version);
    nlohmann::json list_tools();
    std::future<nlohmann::json> call_tool(const std::string& name, const nlohmann::json& arguments);

private:
    void reader_thread_func(std::stop_token stoken);
    std::future<nlohmann::json> send_request_async(const std::string& method, const nlohmann::json& params);
    nlohmann::json send_request(const std::string& method, const nlohmann::json& params);
    void send_notification(const std::string& method, const nlohmann::json& params);

    // Process state
    pid_t child_pid_ = -1;
    int pipe_stdin_[2] = {-1, -1};
    int pipe_stdout_[2] = {-1, -1};
    int pipe_stderr_[2] = {-1, -1};

    // Threading & Concurrency
    std::jthread reader_thread_;
    std::atomic<std::uint64_t> next_request_id_{1};

    std::mutex promises_mutex_;
    std::unordered_map<std::uint64_t, std::promise<nlohmann::json>> pending_requests_;

    std::mutex write_mutex_;
};

} // namespace mcp
