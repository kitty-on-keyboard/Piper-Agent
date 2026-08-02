#include "mcp_client.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <signal.h>
#include <poll.h>
#include <cstring>

extern char **environ;

namespace {

void set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1) {
        throw std::system_error(errno, std::generic_category(), "fcntl F_GETFL failed");
    }
    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) {
        throw std::system_error(errno, std::generic_category(), "fcntl F_SETFL failed");
    }
}

} // namespace

McpClient::McpClient(const std::string& command, const std::vector<std::string>& args) {
    if (pipe(pipe_stdin_) == -1 || pipe(pipe_stdout_) == -1 || pipe(pipe_stderr_) == -1) {
        throw std::system_error(errno, std::generic_category(), "pipe creation failed");
    }

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);

    // Redirect stdin
    posix_spawn_file_actions_adddup2(&actions, pipe_stdin_[0], STDIN_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipe_stdin_[1]);

    // Redirect stdout
    posix_spawn_file_actions_adddup2(&actions, pipe_stdout_[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipe_stdout_[0]);

    // Redirect stderr
    posix_spawn_file_actions_adddup2(&actions, pipe_stderr_[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipe_stderr_[0]);

    std::vector<char*> c_args;
    c_args.push_back(const_cast<char*>(command.c_str()));
    for (const auto& arg : args) {
        c_args.push_back(const_cast<char*>(arg.c_str()));
    }
    c_args.push_back(nullptr);

    int status = posix_spawnp(&child_pid_, command.c_str(), &actions, nullptr, c_args.data(), environ);
    posix_spawn_file_actions_destroy(&actions);

    if (status != 0) {
        throw std::system_error(status, std::generic_category(), "posix_spawnp failed");
    }

    // Close unused ends of pipes in parent
    close(pipe_stdin_[0]);
    close(pipe_stdout_[1]);
    close(pipe_stderr_[1]);

    // Set non-blocking for stdout and stderr to avoid hanging on read
    set_nonblocking(pipe_stdout_[0]);
    set_nonblocking(pipe_stderr_[0]);

    // Start reader thread
    reader_thread_ = std::jthread([this](std::stop_token stoken) {
        reader_loop();
    });
}

McpClient::~McpClient() {
    stopping_ = true;

    // Signal the child to terminate
    if (child_pid_ > 0) {
        kill(child_pid_, SIGTERM);

        // Simple fallback to SIGKILL if it doesn't terminate quickly
        int status;
        pid_t res = waitpid(child_pid_, &status, WNOHANG);
        if (res == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            res = waitpid(child_pid_, &status, WNOHANG);
            if (res == 0) {
                kill(child_pid_, SIGKILL);
                waitpid(child_pid_, &status, 0); // Wait for sure
            }
        }
    }

    if (reader_thread_.joinable()) {
        reader_thread_.request_stop();
        // Close pipes to break poll/read
        close(pipe_stdout_[0]);
        close(pipe_stderr_[0]);
        reader_thread_.join();
    }

    close(pipe_stdin_[1]);
}

void McpClient::write_to_stdin(const std::string& data) {
    const char* buf = data.c_str();
    size_t to_write = data.size();
    while (to_write > 0) {
        ssize_t written = write(pipe_stdin_[1], buf, to_write);
        if (written < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "write to stdin failed");
        }
        buf += written;
        to_write -= written;
    }
}

std::future<nlohmann::json> McpClient::send_request(const std::string& method, const nlohmann::json& params) {
    std::uint64_t id = next_request_id_++;
    nlohmann::json req = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"method", method}
    };
    if (!params.empty()) {
        req["params"] = params;
    }

    std::promise<nlohmann::json> promise;
    std::future<nlohmann::json> future = promise.get_future();

    {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        pending_requests_[id] = std::move(promise);
    }

    std::string req_str = req.dump() + "\n";
    write_to_stdin(req_str);

    return future;
}

void McpClient::send_notification(const std::string& method, const nlohmann::json& params) {
    nlohmann::json req = {
        {"jsonrpc", "2.0"},
        {"method", method}
    };
    if (!params.empty()) {
        req["params"] = params;
    }

    std::string req_str = req.dump() + "\n";
    write_to_stdin(req_str);
}

std::future<nlohmann::json> McpClient::initialize(const nlohmann::json& client_info) {
    // Basic MCP initialize payload
    nlohmann::json params = {
        {"protocolVersion", "2024-11-05"},
        {"capabilities", nlohmann::json::object()},
        {"clientInfo", {
            {"name", "mcp-client-cpp"},
            {"version", "0.1.0"}
        }}
    };

    // We send request, and return a future. Once we get response, we send notification in a different thread?
    // It's cleaner if the caller handles it, or we can chain it.
    // For simplicity, we can do it via a wrapper task or let caller do it.
    // Let's implement wrapper task using a generic future wait thread, but we can't easily wait in this function since it's sync.
    // Wait, the signature is std::future<nlohmann::json>. We can return a std::async that waits.

    return std::async(std::launch::async, [this, params]() {
        auto fut = send_request("initialize", params);
        auto result = fut.get();
        send_notification("notifications/initialized");
        return result;
    });
}

std::future<nlohmann::json> McpClient::list_tools() {
    return send_request("tools/list");
}

std::future<nlohmann::json> McpClient::call_tool(const std::string& tool_name, const nlohmann::json& arguments) {
    nlohmann::json params = {
        {"name", tool_name},
        {"arguments", arguments}
    };
    return send_request("tools/call", params);
}

void McpClient::process_message(const std::string_view message) {
    if (message.empty()) return;

    try {
        nlohmann::json response = nlohmann::json::parse(message);

        if (response.contains("id")) {
            std::uint64_t id = response["id"].get<std::uint64_t>();

            std::lock_guard<std::mutex> lock(requests_mutex_);
            auto it = pending_requests_.find(id);
            if (it != pending_requests_.end()) {
                if (response.contains("error")) {
                    // Could throw or just return error json. Returning json is simpler.
                    it->second.set_value(response);
                } else {
                    it->second.set_value(response);
                }
                pending_requests_.erase(it);
            }
        } else {
            // Probably a notification, ignore for now
        }
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "JSON Parse error: " << e.what() << "\nMessage: " << message << std::endl;
    }
}

void McpClient::reader_loop() {
    struct pollfd fds[2];
    fds[0].fd = pipe_stdout_[0];
    fds[0].events = POLLIN;
    fds[1].fd = pipe_stderr_[0];
    fds[1].events = POLLIN;

    std::string stdout_buffer;
    std::string stderr_buffer;

    char read_buf[4096];

    while (!stopping_) {
        int ret = poll(fds, 2, 100); // 100ms timeout to allow checking stopping_

        if (ret < 0) {
            if (errno == EINTR) continue;
            break;
        }

        if (ret > 0) {
            // Read stdout
            if (fds[0].revents & POLLIN) {
                ssize_t bytes_read = read(pipe_stdout_[0], read_buf, sizeof(read_buf));
                if (bytes_read > 0) {
                    stdout_buffer.append(read_buf, bytes_read);

                    // Parse line-delimited JSON
                    size_t pos;
                    while ((pos = stdout_buffer.find('\n')) != std::string::npos) {
                        std::string_view msg(stdout_buffer.data(), pos);
                        process_message(msg);
                        stdout_buffer.erase(0, pos + 1);
                    }
                } else if (bytes_read == 0) {
                    // EOF
                    break;
                }
            }

            // Read stderr (just print or ignore)
            if (fds[1].revents & POLLIN) {
                ssize_t bytes_read = read(pipe_stderr_[0], read_buf, sizeof(read_buf));
                if (bytes_read > 0) {
                    stderr_buffer.append(read_buf, bytes_read);
                    size_t pos;
                    while ((pos = stderr_buffer.find('\n')) != std::string::npos) {
                        std::cerr << "[MCP STDERR] " << stderr_buffer.substr(0, pos) << std::endl;
                        stderr_buffer.erase(0, pos + 1);
                    }
                }
            }
        }
    }
}
