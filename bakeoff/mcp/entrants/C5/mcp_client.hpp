#pragma once

#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <future>
#include <atomic>
#include <unordered_map>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <sys/types.h>

namespace mcp {

class Process {
public:
    Process(const std::string& command, const std::vector<std::string>& args);
    ~Process();

    // Prevent copying and moving
    Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;
    Process(Process&&) = delete;
    Process& operator=(Process&&) = delete;

    ssize_t write(std::string_view data);
    ssize_t read_stdout(std::span<char> buffer);
    ssize_t read_stderr(std::span<char> buffer);

    int stdout_fd() const { return stdout_fd_; }
    int stderr_fd() const { return stderr_fd_; }

    void terminate();

private:
    pid_t pid_ = -1;
    int stdin_fd_ = -1;
    int stdout_fd_ = -1;
    int stderr_fd_ = -1;
};

class McpClient {
public:
    McpClient(const std::string& command, const std::vector<std::string>& args);
    ~McpClient();

    // Lifecycle
    std::future<nlohmann::json> initialize(const nlohmann::json& params);
    void send_initialized();

    // Tools
    std::future<nlohmann::json> list_tools();
    std::future<nlohmann::json> call_tool(const std::string& name, const nlohmann::json& arguments = nlohmann::json::object());

    // Generic RPC
    std::future<nlohmann::json> send_request(const std::string& method, const nlohmann::json& params = nlohmann::json::object());
    void send_notification(const std::string& method, const nlohmann::json& params = nlohmann::json::object());

private:
    void reader_thread_loop(std::stop_token stoken);
    void process_stdout_data(std::string_view data);
    void handle_message(const nlohmann::json& msg);
    std::uint64_t get_next_id() { return next_request_id_++; }

    std::unique_ptr<Process> process_;
    std::jthread reader_thread_;

    std::atomic<std::uint64_t> next_request_id_{1};

    std::mutex requests_mutex_;
    std::unordered_map<std::uint64_t, std::promise<nlohmann::json>> pending_requests_;

    std::string read_buffer_;
};

} // namespace mcp
