#ifndef MCP_CLIENT_HPP
#define MCP_CLIENT_HPP

#include <string>
#include <vector>
#include <thread>
#include <future>
#include <atomic>
#include <map>
#include <mutex>
#include <string_view>
#include <span>
#include <optional>
#include <sys/types.h>

#include <nlohmann/json.hpp>

class McpClient {
public:
    // Launch the MCP server subprocess and start the reader thread.
    McpClient(const std::string& command, const std::vector<std::string>& args);

    // Graceful teardown (SIGTERM, then SIGKILL timeout), wait for reader thread.
    ~McpClient();

    // Prevent copy and move for simplicity
    McpClient(const McpClient&) = delete;
    McpClient& operator=(const McpClient&) = delete;

    // Core MCP Functionality returning a future to the parsed JSON response

    // Sends "initialize" request, waits for response, then sends "notifications/initialized".
    std::future<nlohmann::json> initialize(const nlohmann::json& client_info = nlohmann::json::object());

    // Sends "tools/list" request.
    std::future<nlohmann::json> list_tools();

    // Sends "tools/call" request.
    std::future<nlohmann::json> call_tool(const std::string& tool_name, const nlohmann::json& arguments = nlohmann::json::object());

private:
    pid_t child_pid_ = -1;
    int pipe_stdin_[2] = {-1, -1};
    int pipe_stdout_[2] = {-1, -1};
    int pipe_stderr_[2] = {-1, -1};

    std::jthread reader_thread_;
    std::atomic<bool> stopping_{false};

    std::atomic<std::uint64_t> next_request_id_{1};

    std::mutex requests_mutex_;
    std::map<std::uint64_t, std::promise<nlohmann::json>> pending_requests_;

    void reader_loop();

    void process_message(const std::string_view message);

    std::future<nlohmann::json> send_request(const std::string& method, const nlohmann::json& params = nlohmann::json::object());
    void send_notification(const std::string& method, const nlohmann::json& params = nlohmann::json::object());

    void write_to_stdin(const std::string& data);
};

#endif // MCP_CLIENT_HPP
