#pragma once

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <future>
#include <optional>
#include <span>
#include <string_view>
#include "json.hpp"

namespace mcp {

using json = nlohmann::json;

class Client {
public:
    Client();
    ~Client();

    // Prevent copy and assignment
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    // Connects to the MCP server by launching a subprocess
    bool connect(const std::string& command, const std::vector<std::string>& args);

    // Stops the MCP server process
    void stop();

    // Core MCP methods
    json initialize();
    void send_initialized();
    json list_tools();
    json call_tool(const std::string& name, const json& args);

private:
    void reader_thread_loop();
    void handle_message(std::string_view message);
    void send_message(const json& message);

    // Process variables
    pid_t pid_ = -1;
    int write_fd_ = -1;
    int read_fd_ = -1;

    // Threading variables
    std::jthread reader_thread_;
    std::atomic<bool> running_{false};

    // Thread-safe writing
    std::mutex write_mutex_;

    // Request tracking
    std::atomic<std::uint64_t> next_id_{1};
    std::mutex requests_mutex_;
    std::unordered_map<std::uint64_t, std::promise<json>> pending_requests_;
};

} // namespace mcp
