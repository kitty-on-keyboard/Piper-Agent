#include "mcp_client.hpp"

#include <iostream>
#include <system_error>
#include <stdexcept>
#include <poll.h>
#include <unistd.h>
#include <spawn.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <cstring>
#include <vector>

extern char** environ;

namespace mcp {

// --- Process Implementation ---

Process::Process(const std::string& command, const std::vector<std::string>& args) {
    int stdin_pipe[2];
    int stdout_pipe[2];
    int stderr_pipe[2];

    if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
        throw std::system_error(errno, std::generic_category(), "Failed to create pipes");
    }

    posix_spawn_file_actions_t file_actions;
    posix_spawn_file_actions_init(&file_actions);

    // stdin: child reads from stdin_pipe[0], client writes to stdin_pipe[1]
    posix_spawn_file_actions_adddup2(&file_actions, stdin_pipe[0], STDIN_FILENO);
    posix_spawn_file_actions_addclose(&file_actions, stdin_pipe[0]);
    posix_spawn_file_actions_addclose(&file_actions, stdin_pipe[1]);

    // stdout: child writes to stdout_pipe[1], client reads from stdout_pipe[0]
    posix_spawn_file_actions_adddup2(&file_actions, stdout_pipe[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&file_actions, stdout_pipe[1]);
    posix_spawn_file_actions_addclose(&file_actions, stdout_pipe[0]);

    // stderr: child writes to stderr_pipe[1], client reads from stderr_pipe[0]
    posix_spawn_file_actions_adddup2(&file_actions, stderr_pipe[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&file_actions, stderr_pipe[1]);
    posix_spawn_file_actions_addclose(&file_actions, stderr_pipe[0]);

    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(command.c_str()));
    for (const auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);

    int status = posix_spawnp(&pid_, command.c_str(), &file_actions, nullptr, argv.data(), environ);

    posix_spawn_file_actions_destroy(&file_actions);

    if (status != 0) {
        close(stdin_pipe[0]); close(stdin_pipe[1]);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        close(stderr_pipe[0]); close(stderr_pipe[1]);
        throw std::system_error(status, std::generic_category(), "Failed to spawn process");
    }

    // Close child ends in parent
    close(stdin_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    stdin_fd_ = stdin_pipe[1];
    stdout_fd_ = stdout_pipe[0];
    stderr_fd_ = stderr_pipe[0];

    // Set non-blocking on stdout/stderr
    fcntl(stdout_fd_, F_SETFL, fcntl(stdout_fd_, F_GETFL) | O_NONBLOCK);
    fcntl(stderr_fd_, F_SETFL, fcntl(stderr_fd_, F_GETFL) | O_NONBLOCK);
}

Process::~Process() {
    terminate();
}

void Process::terminate() {
    if (pid_ > 0) {
        kill(pid_, SIGTERM);

        // Wait up to 1 second for graceful termination
        for (int i = 0; i < 10; ++i) {
            int status;
            pid_t wpid = waitpid(pid_, &status, WNOHANG);
            if (wpid > 0) {
                pid_ = -1; // Process exited
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Force kill if still running
        if (pid_ > 0) {
            kill(pid_, SIGKILL);
            waitpid(pid_, nullptr, 0);
            pid_ = -1;
        }
    }

    if (stdin_fd_ != -1) { close(stdin_fd_); stdin_fd_ = -1; }
    if (stdout_fd_ != -1) { close(stdout_fd_); stdout_fd_ = -1; }
    if (stderr_fd_ != -1) { close(stderr_fd_); stderr_fd_ = -1; }
}

ssize_t Process::write(std::string_view data) {
    if (stdin_fd_ == -1) return -1;
    size_t total_written = 0;
    while (total_written < data.size()) {
        ssize_t written = ::write(stdin_fd_, data.data() + total_written, data.size() - total_written);
        if (written < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        total_written += written;
    }
    return total_written;
}

ssize_t Process::read_stdout(std::span<char> buffer) {
    if (stdout_fd_ == -1) return -1;
    return ::read(stdout_fd_, buffer.data(), buffer.size());
}

ssize_t Process::read_stderr(std::span<char> buffer) {
    if (stderr_fd_ == -1) return -1;
    return ::read(stderr_fd_, buffer.data(), buffer.size());
}

// --- McpClient Implementation ---

McpClient::McpClient(const std::string& command, const std::vector<std::string>& args)
    : process_(std::make_unique<Process>(command, args)),
      reader_thread_([this](std::stop_token stoken) { reader_thread_loop(std::move(stoken)); }) {
}

McpClient::~McpClient() {
    reader_thread_.request_stop();
    if (reader_thread_.joinable()) {
        reader_thread_.join();
    }
}

std::future<nlohmann::json> McpClient::send_request(const std::string& method, const nlohmann::json& params) {
    std::uint64_t id = get_next_id();

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
        pending_requests_.emplace(id, std::move(promise));
    }

    std::string msg = req.dump() + "\n";
    process_->write(msg);

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
    std::string msg = req.dump() + "\n";
    process_->write(msg);
}

std::future<nlohmann::json> McpClient::initialize(const nlohmann::json& params) {
    return send_request("initialize", params);
}

void McpClient::send_initialized() {
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
    struct pollfd fds[2];
    fds[0].fd = process_->stdout_fd();
    fds[0].events = POLLIN;
    fds[1].fd = process_->stderr_fd();
    fds[1].events = POLLIN;

    std::vector<char> buffer(8192);

    while (!stoken.stop_requested()) {
        int ret = poll(fds, 2, 100); // 100ms timeout
        if (ret < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (ret == 0) continue; // timeout

        if (fds[0].revents & POLLIN) {
            ssize_t bytes_read = process_->read_stdout(buffer);
            if (bytes_read > 0) {
                process_stdout_data(std::string_view(buffer.data(), bytes_read));
            } else if (bytes_read == 0) {
                // EOF
                break;
            }
        }

        if (fds[1].revents & POLLIN) {
            ssize_t bytes_read = process_->read_stderr(buffer);
            if (bytes_read > 0) {
                // Just consume and ignore stderr, or log it
                // std::cerr << "STDERR: " << std::string_view(buffer.data(), bytes_read);
            }
        }

        if (fds[0].revents & (POLLERR | POLLHUP) || fds[1].revents & (POLLERR | POLLHUP)) {
            break;
        }
    }

    // Fail pending requests on close
    std::lock_guard<std::mutex> lock(requests_mutex_);
    for (auto& [id, promise] : pending_requests_) {
        try {
            throw std::runtime_error("Connection closed");
        } catch(...) {
            promise.set_exception(std::current_exception());
        }
    }
    pending_requests_.clear();
}

void McpClient::process_stdout_data(std::string_view data) {
    read_buffer_.append(data);

    // Simple line-delimited JSON parser (JSON-RPC over stdio typically uses this or Content-Length headers)
    // MCP spec specifies stdio transport as line-delimited JSON messages

    size_t pos;
    while ((pos = read_buffer_.find('\n')) != std::string::npos) {
        std::string line = read_buffer_.substr(0, pos);
        read_buffer_.erase(0, pos + 1);

        // skip empty lines
        if (line.empty() || line.find_first_not_of(" \r") == std::string::npos) {
            continue;
        }

        // Handling Content-Length headers just in case some servers send it over stdio
        if (line.starts_with("Content-Length:")) {
            size_t length = std::stoull(line.substr(15));
            // Read next line (empty)
            size_t next_pos = read_buffer_.find('\n');
            if (next_pos != std::string::npos) {
                read_buffer_.erase(0, next_pos + 1);
            }

            // we need 'length' bytes
            if (read_buffer_.size() >= length) {
                std::string msg = read_buffer_.substr(0, length);
                read_buffer_.erase(0, length);
                try {
                    handle_message(nlohmann::json::parse(msg));
                } catch (const std::exception& e) {
                    std::cerr << "Failed to parse json msg: " << e.what() << std::endl;
                }
            } else {
                // put header back, wait for more data. Not optimal but works for generic streams
                read_buffer_ = line + "\n\r\n" + read_buffer_;
                break;
            }
        } else {
            // Line delimited JSON
            try {
                auto msg = nlohmann::json::parse(line);
                handle_message(msg);
            } catch (const std::exception& e) {
                 // std::cerr << "JSON parse error on line: " << line << " Error: " << e.what() << std::endl;
            }
        }
    }
}

void McpClient::handle_message(const nlohmann::json& msg) {
    if (msg.contains("id")) {
        // It's a response
        std::uint64_t id;
        if (msg["id"].is_number()) {
            id = msg["id"].get<std::uint64_t>();
        } else if (msg["id"].is_string()) {
            id = std::stoull(msg["id"].get<std::string>());
        } else {
            return;
        }

        std::promise<nlohmann::json> promise;
        {
            std::lock_guard<std::mutex> lock(requests_mutex_);
            auto it = pending_requests_.find(id);
            if (it != pending_requests_.end()) {
                promise = std::move(it->second);
                pending_requests_.erase(it);
            } else {
                return; // Not found or already handled
            }
        }

        if (msg.contains("error")) {
            // Can be enhanced to throw specific JSON-RPC exceptions
            try {
                throw std::runtime_error(msg["error"].dump());
            } catch(...) {
                promise.set_exception(std::current_exception());
            }
        } else {
            promise.set_value(msg);
        }
    } else {
        // It's a notification, could add callbacks here
        // std::cout << "Notification received: " << msg.dump() << std::endl;
    }
}

} // namespace mcp
