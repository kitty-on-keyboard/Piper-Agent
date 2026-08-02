#pragma once

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <future>
#include <unordered_map>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <nlohmann/json.hpp>
#include <sys/types.h>

namespace mcp {

class McpClient {
public:
    McpClient(const std::string& command, const std::vector<std::string>& args);
    ~McpClient();

    // Prevent copy and move
    McpClient(const McpClient&) = delete;
    McpClient& operator=(const McpClient&) = delete;
    McpClient(McpClient&&) = delete;
    McpClient& operator=(McpClient&&) = delete;

    // Core MCP Functionality
    nlohmann::json initialize();
    nlohmann::json list_tools();
    nlohmann::json call_tool(std::string_view name, const nlohmann::json& arguments);

    // Asynchronous JSON-RPC
    std::future<nlohmann::json> send_request(std::string_view method, const nlohmann::json& params = nullptr);
    void send_notification(std::string_view method, const nlohmann::json& params = nullptr);

    // Graceful teardown
    void stop();

private:
    void reader_thread_func(std::stop_token stoken);
    void handle_parsed_json(const nlohmann::json& doc);

    void write_to_subprocess(const std::string& data);

    pid_t child_pid_ = -1;
    int pipe_stdin_ = -1;
    int pipe_stdout_ = -1;
    int pipe_stderr_ = -1;

    std::jthread reader_thread_;
    std::atomic<std::uint64_t> next_request_id_{1};

    std::mutex promises_mutex_;
    std::unordered_map<std::uint64_t, std::promise<nlohmann::json>> pending_requests_;
};

} // namespace mcp
