#include "mcp_client.hpp"
#include <iostream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <spawn.h>
#include <poll.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <chrono>

#ifdef __APPLE__
#include <crt_externs.h>
#define environ (*_NSGetEnviron())
#else
extern char** environ;
#endif

namespace mcp {

McpClient::McpClient(const std::string& command, const std::vector<std::string>& args) {
    int stdin_pipe[2];
    int stdout_pipe[2];
    int stderr_pipe[2];

    if (pipe(stdin_pipe) == -1 || pipe(stdout_pipe) == -1 || pipe(stderr_pipe) == -1) {
        throw std::system_error(errno, std::generic_category(), "Failed to create pipes");
    }

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);

    posix_spawn_file_actions_adddup2(&actions, stdin_pipe[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&actions, stdout_pipe[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, stderr_pipe[1], STDERR_FILENO);

    posix_spawn_file_actions_addclose(&actions, stdin_pipe[0]);
    posix_spawn_file_actions_addclose(&actions, stdin_pipe[1]);
    posix_spawn_file_actions_addclose(&actions, stdout_pipe[0]);
    posix_spawn_file_actions_addclose(&actions, stdout_pipe[1]);
    posix_spawn_file_actions_addclose(&actions, stderr_pipe[0]);
    posix_spawn_file_actions_addclose(&actions, stderr_pipe[1]);

    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(command.c_str()));
    for (const auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);

    int status = posix_spawnp(&child_pid_, command.c_str(), &actions, nullptr, argv.data(), environ);

    posix_spawn_file_actions_destroy(&actions);

    if (status != 0) {
        close(stdin_pipe[0]); close(stdin_pipe[1]);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        close(stderr_pipe[0]); close(stderr_pipe[1]);
        throw std::system_error(status, std::generic_category(), "Failed to posix_spawn");
    }

    close(stdin_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    pipe_stdin_ = stdin_pipe[1];
    pipe_stdout_ = stdout_pipe[0];
    pipe_stderr_ = stderr_pipe[0];

    // Set stdout pipe to non-blocking
    int flags = fcntl(pipe_stdout_, F_GETFL, 0);
    fcntl(pipe_stdout_, F_SETFL, flags | O_NONBLOCK);

    reader_thread_ = std::jthread([this](std::stop_token stoken) {
        reader_thread_func(std::move(stoken));
    });
}

McpClient::~McpClient() {
    stop();
}

void McpClient::stop() {
    if (child_pid_ != -1) {
        if (reader_thread_.joinable()) {
            reader_thread_.request_stop();
            reader_thread_.join();
        }

        kill(child_pid_, SIGTERM);

        int status;
        int ret = waitpid(child_pid_, &status, WNOHANG);
        if (ret == 0) {
            // Give it some time
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            ret = waitpid(child_pid_, &status, WNOHANG);
            if (ret == 0) {
                kill(child_pid_, SIGKILL);
                waitpid(child_pid_, &status, 0);
            }
        }
        child_pid_ = -1;
    }

    if (pipe_stdin_ != -1) { close(pipe_stdin_); pipe_stdin_ = -1; }
    if (pipe_stdout_ != -1) { close(pipe_stdout_); pipe_stdout_ = -1; }
    if (pipe_stderr_ != -1) { close(pipe_stderr_); pipe_stderr_ = -1; }

    std::lock_guard<std::mutex> lock(promises_mutex_);
    for (auto& [id, promise] : pending_requests_) {
        try {
            promise.set_exception(std::make_exception_ptr(std::runtime_error("Client stopped before request could complete")));
        } catch (...) {}
    }
    pending_requests_.clear();
}

void McpClient::write_to_subprocess(const std::string& data) {
    if (pipe_stdin_ == -1) return;
    // For standard MCP, write Content-Length header
    std::string formatted = "Content-Length: " + std::to_string(data.size()) + "\r\n\r\n" + data;
    ssize_t written = 0;
    while (written < formatted.size()) {
        ssize_t res = write(pipe_stdin_, formatted.c_str() + written, formatted.size() - written);
        if (res == -1) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "Failed to write to subprocess");
        }
        written += res;
    }
}

std::future<nlohmann::json> McpClient::send_request(std::string_view method, const nlohmann::json& params) {
    std::uint64_t id = next_request_id_++;
    nlohmann::json req = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"method", method}
    };
    if (params != nullptr) {
        req["params"] = params;
    }

    std::promise<nlohmann::json> promise;
    auto future = promise.get_future();

    {
        std::lock_guard<std::mutex> lock(promises_mutex_);
        pending_requests_[id] = std::move(promise);
    }

    write_to_subprocess(req.dump());
    return future;
}

void McpClient::send_notification(std::string_view method, const nlohmann::json& params) {
    nlohmann::json req = {
        {"jsonrpc", "2.0"},
        {"method", method}
    };
    if (params != nullptr) {
        req["params"] = params;
    }

    write_to_subprocess(req.dump());
}

void McpClient::handle_parsed_json(const nlohmann::json& doc) {
    if (doc.contains("id")) {
        // It's a response
        std::uint64_t id = doc["id"].get<std::uint64_t>();
        std::lock_guard<std::mutex> lock(promises_mutex_);
        auto it = pending_requests_.find(id);
        if (it != pending_requests_.end()) {
            it->second.set_value(doc);
            pending_requests_.erase(it);
        }
    } else if (doc.contains("method")) {
        // Server to client notification/request, can be handled here if needed
        // For simplicity, we just ignore it in this basic client
    }
}

void McpClient::reader_thread_func(std::stop_token stoken) {
    pollfd pfd;
    pfd.fd = pipe_stdout_;
    pfd.events = POLLIN;

    std::string buffer;

    while (!stoken.stop_requested()) {
        int ret = poll(&pfd, 1, 100); // 100ms timeout
        if (ret < 0) {
            if (errno == EINTR) continue;
            break;
        }

        if (ret > 0 && (pfd.revents & POLLIN)) {
            char buf[4096];
            ssize_t bytes_read = read(pipe_stdout_, buf, sizeof(buf));
            if (bytes_read > 0) {
                buffer.append(buf, bytes_read);

                // Parse
                while (true) {
                    size_t header_end = buffer.find("\r\n\r\n");
                    if (header_end != std::string::npos) {
                        std::string headers = buffer.substr(0, header_end);
                        size_t content_length_pos = headers.find("Content-Length: ");
                        if (content_length_pos != std::string::npos) {
                            size_t length_start = content_length_pos + 16;
                            size_t length_end = headers.find("\r\n", length_start);
                            if (length_end == std::string::npos) length_end = headers.length();

                            size_t length = 0;
                            try {
                                length = std::stoull(headers.substr(length_start, length_end - length_start));
                            } catch (...) {
                                // Invalid length, drop the message
                                buffer.erase(0, header_end + 4);
                                continue;
                            }

                            if (buffer.size() >= header_end + 4 + length) {
                                std::string payload = buffer.substr(header_end + 4, length);
                                buffer.erase(0, header_end + 4 + length);

                                try {
                                    auto doc = nlohmann::json::parse(payload);
                                    handle_parsed_json(doc);
                                } catch (const std::exception& e) {
                                    std::cerr << "Failed to parse JSON: " << e.what() << std::endl;
                                }
                                continue; // Check for more messages
                            }
                        } else {
                            // No Content-Length header, just skip the headers
                            buffer.erase(0, header_end + 4);
                        }
                    } else {
                        // Might be line-delimited JSON fallback
                        size_t pos = buffer.find('\n');
                        if (pos != std::string::npos && !buffer.starts_with("Content-Length:")) {
                            std::string line = buffer.substr(0, pos);
                            buffer.erase(0, pos + 1);
                            try {
                                if (line.starts_with("{")) {
                                    auto doc = nlohmann::json::parse(line);
                                    handle_parsed_json(doc);
                                }
                            } catch (const std::exception& e) {
                                // Ignored
                            }
                            continue;
                        }
                    }
                    break;
                }
            } else if (bytes_read == 0) {
                // EOF
                break;
            } else {
                if (errno != EINTR && errno != EAGAIN) {
                    break;
                }
            }
        }
    }
}

nlohmann::json McpClient::initialize() {
    nlohmann::json params = {
        {"protocolVersion", "2024-11-05"},
        {"capabilities", nlohmann::json::object()},
        {"clientInfo", {
            {"name", "mcp-cpp-client"},
            {"version", "1.0.0"}
        }}
    };

    auto future = send_request("initialize", params);
    auto response = future.get();

    send_notification("notifications/initialized");
    return response;
}

nlohmann::json McpClient::list_tools() {
    auto future = send_request("tools/list");
    return future.get();
}

nlohmann::json McpClient::call_tool(std::string_view name, const nlohmann::json& arguments) {
    nlohmann::json params = {
        {"name", name},
        {"arguments", arguments}
    };
    auto future = send_request("tools/call", params);
    return future.get();
}

} // namespace mcp
