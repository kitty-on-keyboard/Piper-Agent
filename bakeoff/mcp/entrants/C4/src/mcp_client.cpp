#include "mcp_client.hpp"

#include <iostream>
#include <system_error>
#include <cstring>
#include <chrono>
#include <unistd.h>
#include <spawn.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>

extern char **environ;

namespace mcp {

McpClient::McpClient() = default;

McpClient::~McpClient() {
    stop();
}

void McpClient::start(const std::string& command, const std::vector<std::string>& args) {
    if (child_pid_ != -1) {
        throw McpError("Process already running");
    }

    if (pipe(pipe_stdin_) == -1 || pipe(pipe_stdout_) == -1 || pipe(pipe_stderr_) == -1) {
        throw std::system_error(errno, std::system_category(), "Failed to create pipes");
    }

    posix_spawn_file_actions_t file_actions;
    if (posix_spawn_file_actions_init(&file_actions) != 0) {
        throw std::system_error(errno, std::system_category(), "Failed to init file actions");
    }

    // Connect pipes to stdin/stdout/stderr of child
    posix_spawn_file_actions_adddup2(&file_actions, pipe_stdin_[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&file_actions, pipe_stdout_[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&file_actions, pipe_stderr_[1], STDERR_FILENO);

    // Close all other pipe ends in child
    posix_spawn_file_actions_addclose(&file_actions, pipe_stdin_[0]);
    posix_spawn_file_actions_addclose(&file_actions, pipe_stdin_[1]);
    posix_spawn_file_actions_addclose(&file_actions, pipe_stdout_[0]);
    posix_spawn_file_actions_addclose(&file_actions, pipe_stdout_[1]);
    posix_spawn_file_actions_addclose(&file_actions, pipe_stderr_[0]);
    posix_spawn_file_actions_addclose(&file_actions, pipe_stderr_[1]);

    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(command.c_str()));
    for (const auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);

    // Make stdout and stderr reader pipes non-blocking in parent
    int flags = fcntl(pipe_stdout_[0], F_GETFL, 0);
    fcntl(pipe_stdout_[0], F_SETFL, flags | O_NONBLOCK);

    flags = fcntl(pipe_stderr_[0], F_GETFL, 0);
    fcntl(pipe_stderr_[0], F_SETFL, flags | O_NONBLOCK);

    int status = posix_spawn(&child_pid_, command.c_str(), &file_actions, nullptr, argv.data(), environ);
    // If command doesn't contain a slash, maybe we should use posix_spawnp. Let's use posix_spawnp instead.
    if (status != 0 && command.find('/') == std::string::npos) {
        status = posix_spawnp(&child_pid_, command.c_str(), &file_actions, nullptr, argv.data(), environ);
    }

    posix_spawn_file_actions_destroy(&file_actions);

    if (status != 0) {
        close(pipe_stdin_[0]); close(pipe_stdin_[1]);
        close(pipe_stdout_[0]); close(pipe_stdout_[1]);
        close(pipe_stderr_[0]); close(pipe_stderr_[1]);
        pipe_stdin_[0] = pipe_stdin_[1] = pipe_stdout_[0] = pipe_stdout_[1] = pipe_stderr_[0] = pipe_stderr_[1] = -1;
        throw std::system_error(status, std::system_category(), "Failed to spawn process");
    }

    // Close child ends in parent
    close(pipe_stdin_[0]); pipe_stdin_[0] = -1;
    close(pipe_stdout_[1]); pipe_stdout_[1] = -1;
    close(pipe_stderr_[1]); pipe_stderr_[1] = -1;

    // Start reader thread
    reader_thread_ = std::jthread([this](std::stop_token st) { reader_thread_func(st); });
}

void McpClient::stop() {
    if (reader_thread_.joinable()) {
        reader_thread_.request_stop();
        reader_thread_.join();
    }

    if (child_pid_ != -1) {
        // Send SIGTERM
        kill(child_pid_, SIGTERM);

        // Wait with timeout for graceful shutdown
        bool exited = false;
        for (int i = 0; i < 50; ++i) { // 5 seconds
            int status;
            pid_t res = waitpid(child_pid_, &status, WNOHANG);
            if (res == child_pid_ || res == -1) {
                exited = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (!exited) {
            // Fallback to SIGKILL
            kill(child_pid_, SIGKILL);
            waitpid(child_pid_, nullptr, 0);
        }
        child_pid_ = -1;
    }

    if (pipe_stdin_[1] != -1) {
        close(pipe_stdin_[1]);
        pipe_stdin_[1] = -1;
    }
    if (pipe_stdout_[0] != -1) {
        close(pipe_stdout_[0]);
        pipe_stdout_[0] = -1;
    }
    if (pipe_stderr_[0] != -1) {
        close(pipe_stderr_[0]);
        pipe_stderr_[0] = -1;
    }
}


void McpClient::reader_thread_func(std::stop_token stoken) {
    std::string buffer;
    char read_buf[4096];

    // For stderr mapping (we just drop or log it)
    char err_buf[4096];

    while (!stoken.stop_requested()) {
        // Drain stderr non-blocking to avoid pipe stall
        while (read(pipe_stderr_[0], err_buf, sizeof(err_buf)) > 0) {}

        ssize_t bytes_read = read(pipe_stdout_[0], read_buf, sizeof(read_buf));
        if (bytes_read > 0) {
            buffer.append(read_buf, bytes_read);

            // Process line-delimited JSON using string_view / spans
            std::string_view sv(buffer);
            size_t start = 0;
            size_t pos;

            while ((pos = sv.find('\n', start)) != std::string_view::npos) {
                std::string_view line = sv.substr(start, pos - start);
                start = pos + 1;

                // Trim
                if (line.empty()) continue;
                size_t first = line.find_first_not_of("\r\n\t ");
                if (first == std::string_view::npos) continue;

                try {
                    auto j = nlohmann::json::parse(line);

                    if (j.contains("id") && j["id"].is_number_integer()) {
                        std::uint64_t id = j["id"].get<std::uint64_t>();

                        std::lock_guard<std::mutex> lock(promises_mutex_);
                        auto it = pending_requests_.find(id);
                        if (it != pending_requests_.end()) {
                            it->second.set_value(j);
                            pending_requests_.erase(it);
                        }
                    } else if (j.contains("method")) {
                        // Handle server-to-client notifications or requests if needed
                    }
                } catch (const nlohmann::json::parse_error& e) {
                    std::cerr << "JSON parse error in reader thread: " << e.what() << "\n";
                }
            }

            // Advance buffer
            if (start > 0) {
                buffer.erase(0, start);
            }
        } else if (bytes_read == 0) {
            // EOF
            break;
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } else {
                // Error reading
                break;
            }
        }
    }

    // Fail all pending promises
    std::lock_guard<std::mutex> lock(promises_mutex_);
    for (auto& [id, promise] : pending_requests_) {
        promise.set_exception(std::make_exception_ptr(McpError("Connection closed")));
    }
    pending_requests_.clear();
}

std::future<nlohmann::json> McpClient::send_request_async(const std::string& method, const nlohmann::json& params) {
    if (pipe_stdin_[1] == -1) {
        throw McpError("Process not running");
    }

    std::uint64_t id = next_request_id_++;
    nlohmann::json req = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"method", method},
    };
    if (!params.is_null()) {
        req["params"] = params;
    }

    std::promise<nlohmann::json> promise;
    auto future = promise.get_future();

    {
        std::lock_guard<std::mutex> lock(promises_mutex_);
        pending_requests_[id] = std::move(promise);
    }

    std::string msg = req.dump() + "\n";

    {
        std::lock_guard<std::mutex> wlock(write_mutex_);
        ssize_t written = write(pipe_stdin_[1], msg.c_str(), msg.size());
        if (written != msg.size()) {
            std::lock_guard<std::mutex> lock(promises_mutex_);
            pending_requests_.erase(id);
            throw McpError("Failed to write request");
        }
    }

    return future;
}

nlohmann::json McpClient::send_request(const std::string& method, const nlohmann::json& params) {
    auto future = send_request_async(method, params);
    auto res = future.get();
    if (res.contains("error")) {
        throw McpError("RPC Error: " + res["error"].dump());
    }
    return res.contains("result") ? res["result"] : nlohmann::json::object();
}

void McpClient::send_notification(const std::string& method, const nlohmann::json& params) {
    if (pipe_stdin_[1] == -1) {
        throw McpError("Process not running");
    }

    nlohmann::json req = {
        {"jsonrpc", "2.0"},
        {"method", method},
    };
    if (!params.is_null()) {
        req["params"] = params;
    }

    std::string msg = req.dump() + "\n";

    std::lock_guard<std::mutex> wlock(write_mutex_);
    ssize_t written = write(pipe_stdin_[1], msg.c_str(), msg.size());
    if (written != msg.size()) {
        throw McpError("Failed to write notification");
    }
}

nlohmann::json McpClient::initialize(const std::string& client_name, const std::string& client_version) {
    nlohmann::json params = {
        {"protocolVersion", "2024-11-05"},
        {"capabilities", nlohmann::json::object()},
        {"clientInfo", {
            {"name", client_name},
            {"version", client_version}
        }}
    };

    auto result = send_request("initialize", params);
    send_notification("notifications/initialized", nlohmann::json::object());
    return result;
}

nlohmann::json McpClient::list_tools() {
    return send_request("tools/list", nlohmann::json::object());
}

std::future<nlohmann::json> McpClient::call_tool(const std::string& name, const nlohmann::json& arguments) {
    nlohmann::json params = {
        {"name", name},
        {"arguments", arguments}
    };
    // Return the future wrapping the unwrapping logic.
    // Wait, future wrapping might be a bit tricky, let's just return what send_request_async gives, and let caller handle unwrapping the result vs error
    // Wait, send_request unwrapped "result" for us. Let's return std::future<nlohmann::json> and let caller handle it, or we can use std::async to unwrap.
    // Let's use std::async to unwrap.
    auto future = send_request_async("tools/call", params);
    return std::async(std::launch::deferred, [f = std::move(future)]() mutable {
        auto res = f.get();
        if (res.contains("error")) {
            throw McpError("RPC Error: " + res["error"].dump());
        }
        return res.contains("result") ? res["result"] : nlohmann::json::object();
    });
}

} // namespace mcp
