#pragma once

#include <string>
#include <vector>
#include <thread>
#include <future>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <span>
#include <string_view>

#include <sys/types.h> // For pid_t

namespace mcp {

class McpClient {
public:
    // Initialize with a path to the server executable and its arguments
    McpClient(const std::string& server_path, const std::vector<std::string>& args);

    // Disallow copying and moving
    McpClient(const McpClient&) = delete;
    McpClient& operator=(const McpClient&) = delete;
    McpClient(McpClient&&) = delete;
    McpClient& operator=(McpClient&&) = delete;

    // Graceful teardown
    ~McpClient();

    // Standard MCP requests
    std::future<nlohmann::json> initialize();
    void send_initialized_notification();
    std::future<nlohmann::json> list_tools();
    std::future<nlohmann::json> call_tool(const std::string& name, const nlohmann::json& arguments);

private:
    void send_notification(const std::string& method, const nlohmann::json& params = nullptr);
    std::future<nlohmann::json> send_request(const std::string& method, const nlohmann::json& params = nullptr);
    void reader_thread_loop(std::stop_token stoken);
    void process_message(const std::string_view message);

    // Subprocess state
    pid_t server_pid_ = -1;
    int write_pipe_ = -1;
    int read_pipe_ = -1;

    // Concurrency and state
    std::jthread reader_thread_;
    std::atomic<std::uint64_t> next_request_id_{1};

    std::mutex requests_mutex_;
    std::unordered_map<std::uint64_t, std::promise<nlohmann::json>> pending_requests_;
};

} // namespace mcp
