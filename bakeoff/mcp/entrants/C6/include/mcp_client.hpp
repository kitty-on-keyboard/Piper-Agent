#pragma once

#include <string>
#include <vector>
#include <string_view>
#include <memory>
#include <thread>
#include <functional>
#include <future>
#include <atomic>
#include <unordered_map>
#include <mutex>
#include <span>
#include <concepts>
#include <nlohmann/json.hpp>

namespace mcp {

// Concept for JSON convertible types
template<typename T>
concept JsonConvertible = requires(T a) {
    { nlohmann::json(a) };
};

// Represents a launched child process with Stdio pipes.
class McpProcess {
public:
    McpProcess(const std::string& command, const std::vector<std::string>& args);
    ~McpProcess();

    // Prevent copying
    McpProcess(const McpProcess&) = delete;
    McpProcess& operator=(const McpProcess&) = delete;

    // Prevent moving for simplicity, or we can implement it
    McpProcess(McpProcess&&) = delete;
    McpProcess& operator=(McpProcess&&) = delete;

    // Read from stdout/stderr of child
    std::string read_stdout();
    std::string read_stderr();
    ssize_t read_stdout_raw(std::span<char> buf);

    // Write to stdin of child
    bool write_stdin(std::string_view data);

    // Get POSIX file descriptors if needed
    int get_stdin_fd() const { return stdin_pipe_[1]; }
    int get_stdout_fd() const { return stdout_pipe_[0]; }
    int get_stderr_fd() const { return stderr_pipe_[0]; }

    void terminate();

private:
    pid_t pid_ = -1;
    int stdin_pipe_[2] = {-1, -1};
    int stdout_pipe_[2] = {-1, -1};
    int stderr_pipe_[2] = {-1, -1};
};

using json = nlohmann::json;

class McpClient {
public:
    McpClient(const std::string& command, const std::vector<std::string>& args);
    ~McpClient();

    // Prevent copying
    McpClient(const McpClient&) = delete;
    McpClient& operator=(const McpClient&) = delete;

    // Start background threads for reading
    void start();

    // Core MCP Operations
    template<JsonConvertible T>
    std::future<json> initialize(const T& params) {
        return send_request("initialize", json(params));
    }

    void send_notification(const std::string& method, const json& params = json::object());

    // Tools
    std::future<json> list_tools();
    std::future<json> call_tool(const std::string& name, const json& arguments);

private:
    void reader_thread_func(std::stop_token stoken);
    void handle_message(const std::string& msg);

    std::future<json> send_request(const std::string& method, const json& params = json::object());
    void write_message(const json& msg);

    std::unique_ptr<McpProcess> process_;
    std::atomic<std::uint64_t> next_id_{1};

    std::mutex requests_mutex_;
    std::mutex write_mutex_;
    std::unordered_map<std::uint64_t, std::promise<json>> pending_requests_;
    std::jthread reader_thread_; // Defined last to be destroyed first
};

} // namespace mcp
