#include "mcp_client.hpp"

#include <iostream>
#include <system_error>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <poll.h>
#include <signal.h>

extern char **environ;

namespace mcp {

McpClient::McpClient(const std::string& server_path, const std::vector<std::string>& args) {
    // Ignore SIGPIPE process-wide to prevent crash on writing to a closed pipe
    signal(SIGPIPE, SIG_IGN);
    int stdin_pipe[2];
    int stdout_pipe[2];

    if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0) {
        throw std::system_error(errno, std::generic_category(), "Failed to create pipes");
    }

    posix_spawn_file_actions_t action;
    posix_spawn_file_actions_init(&action);

    // Close the write end of stdout and read end of stdin for the parent in the child
    posix_spawn_file_actions_addclose(&action, stdout_pipe[0]);
    posix_spawn_file_actions_addclose(&action, stdin_pipe[1]);

    // Dup2 the pipe ends to stdin and stdout in the child
    posix_spawn_file_actions_adddup2(&action, stdin_pipe[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&action, stdout_pipe[1], STDOUT_FILENO);

    // Close the originally opened ends in the child now that they are dup'ed
    posix_spawn_file_actions_addclose(&action, stdin_pipe[0]);
    posix_spawn_file_actions_addclose(&action, stdout_pipe[1]);

    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(server_path.c_str()));
    for (const auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);

    if (posix_spawn(&server_pid_, server_path.c_str(), &action, nullptr, argv.data(), environ) != 0) {
        posix_spawn_file_actions_destroy(&action);
        close(stdin_pipe[0]); close(stdin_pipe[1]);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        throw std::system_error(errno, std::generic_category(), "Failed to spawn server");
    }

    posix_spawn_file_actions_destroy(&action);

    // Close the ends we don't need in the parent
    close(stdin_pipe[0]);
    close(stdout_pipe[1]);

    write_pipe_ = stdin_pipe[1];
    read_pipe_ = stdout_pipe[0];

    // Set non-blocking on the read pipe
    int flags = fcntl(read_pipe_, F_GETFL, 0);
    fcntl(read_pipe_, F_SETFL, flags | O_NONBLOCK);

    reader_thread_ = std::jthread([this](std::stop_token stoken) {
        this->reader_thread_loop(stoken);
    });
}

McpClient::~McpClient() {
    if (reader_thread_.joinable()) {
        reader_thread_.request_stop();
        reader_thread_.join();
    }

    if (server_pid_ > 0) {
        kill(server_pid_, SIGTERM);

        int status;
        pid_t res = waitpid(server_pid_, &status, WNOHANG);
        if (res == 0) {
            // Give it a brief moment
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            res = waitpid(server_pid_, &status, WNOHANG);
            if (res == 0) {
                // Force kill if still running
                kill(server_pid_, SIGKILL);
                waitpid(server_pid_, &status, 0);
            }
        }
    }

    if (write_pipe_ != -1) close(write_pipe_);
    if (read_pipe_ != -1) close(read_pipe_);
}

std::future<nlohmann::json> McpClient::send_request(const std::string& method, const nlohmann::json& params) {
    uint64_t id = next_request_id_++;

    nlohmann::json req = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"method", method}
    };
    if (params != nullptr) {
        req["params"] = params;
    }

    std::string payload = req.dump() + "\n";

    std::promise<nlohmann::json> promise;
    auto future = promise.get_future();

    {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        pending_requests_[id] = std::move(promise);
    }

    std::string framed_payload = "Content-Length: " + std::to_string(payload.size()) + "\r\n\r\n" + payload;

    // Write to pipe (simplistic approach, should handle partial writes in prod)
    ssize_t written = write(write_pipe_, framed_payload.data(), framed_payload.size());
    if (written < 0) {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        pending_requests_.erase(id);
        throw std::system_error(errno, std::generic_category(), "Failed to write to server");
    }

    return future;
}

void McpClient::send_notification(const std::string& method, const nlohmann::json& params) {
    nlohmann::json req = {
        {"jsonrpc", "2.0"},
        {"method", method}
    };
    if (params != nullptr) {
        req["params"] = params;
    }

    std::string payload = req.dump() + "\n";
    std::string framed_payload = "Content-Length: " + std::to_string(payload.size()) + "\r\n\r\n" + payload;

    // Ignore error if we fail to write, as notifications are fire-and-forget
    write(write_pipe_, framed_payload.data(), framed_payload.size());
}

std::future<nlohmann::json> McpClient::initialize() {
    nlohmann::json params = {
        {"protocolVersion", "2024-11-05"},
        {"capabilities", nlohmann::json::object()},
        {"clientInfo", {
            {"name", "mcp-cpp-client"},
            {"version", "1.0.0"}
        }}
    };
    return send_request("initialize", params);
}

void McpClient::send_initialized_notification() {
    send_notification("notifications/initialized");
}

std::future<nlohmann::json> McpClient::list_tools() {
    return send_request("tools/list");
}

std::future<nlohmann::json> McpClient::call_tool(const std::string& name, const nlohmann::json& arguments) {
    nlohmann::json params = {
        {"name", name},
        {"arguments", arguments}
    };
    return send_request("tools/call", params);
}

void McpClient::reader_thread_loop(std::stop_token stoken) {
    std::string buffer;
    char read_buf[4096];

    struct pollfd pfd;
    pfd.fd = read_pipe_;
    pfd.events = POLLIN;

    while (!stoken.stop_requested()) {
        int ret = poll(&pfd, 1, 100); // 100ms timeout

        if (ret > 0 && (pfd.revents & POLLIN)) {
            ssize_t n = read(read_pipe_, read_buf, sizeof(read_buf));
            if (n > 0) {
                buffer.append(read_buf, n);

                // Handle both newline-delimited and Content-Length framed messages
                while (!buffer.empty()) {
                    if (buffer.find("Content-Length: ") == 0) {
                        size_t header_end = buffer.find("\r\n\r\n");
                        if (header_end == std::string::npos) {
                            break; // Wait for full header
                        }

                        size_t content_length = std::stoull(buffer.substr(16, header_end - 16));
                        size_t total_length = header_end + 4 + content_length;

                        if (buffer.size() >= total_length) {
                            std::string message = buffer.substr(header_end + 4, content_length);
                            buffer.erase(0, total_length);
                            if (!message.empty()) {
                                process_message(message);
                            }
                        } else {
                            break; // Wait for full content
                        }
                    } else {
                        // Fallback to newline delimited
                        size_t pos = buffer.find('\n');
                        if (pos != std::string::npos) {
                            std::string message = buffer.substr(0, pos);
                            buffer.erase(0, pos + 1);
                            if (!message.empty()) {
                                process_message(message);
                            }
                        } else {
                            break; // Wait for newline
                        }
                    }
                }
            } else if (n == 0) {
                // EOF
                break;
            } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
                // Error
                break;
            }
        } else if (ret < 0 && errno != EINTR) {
            break; // Error in poll
        }
    }

    // Fulfill remaining promises with exceptions to prevent deadlocks
    std::lock_guard<std::mutex> lock(requests_mutex_);
    for (auto& pair : pending_requests_) {
        try {
            pair.second.set_exception(std::make_exception_ptr(
                std::system_error(EPIPE, std::generic_category(), "Pipe closed prematurely")
            ));
        } catch(...) {} // Ignore if already fulfilled
    }
    pending_requests_.clear();
}

void McpClient::process_message(const std::string_view message) {
    try {
        auto j = nlohmann::json::parse(message);
        if (j.contains("id") && !j["id"].is_null()) {
            uint64_t id = j["id"].get<uint64_t>();

            std::lock_guard<std::mutex> lock(requests_mutex_);
            auto it = pending_requests_.find(id);
            if (it != pending_requests_.end()) {
                if (j.contains("error")) {
                    // For simplicity, we just return the error object as the result here.
                    // In a full implementation, you'd probably want a custom exception.
                    it->second.set_value(j);
                } else if (j.contains("result")) {
                    it->second.set_value(j["result"]);
                } else {
                    it->second.set_value(j);
                }
                pending_requests_.erase(it);
            }
        }
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "JSON parse error: " << e.what() << "\nMessage: " << message << std::endl;
    }
}

} // namespace mcp
