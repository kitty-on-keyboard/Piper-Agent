#include "mcp_client.hpp"

#include <iostream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <spawn.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>

#ifdef __APPLE__
#include <crt_externs.h>
#else
extern char **environ;
#endif

namespace mcp {

Client::Client() = default;

Client::~Client() {
    stop();
}

bool Client::connect(const std::string& command, const std::vector<std::string>& args) {
    if (pid_ != -1) {
        return false; // Already connected
    }

    int pipe_in[2];
    int pipe_out[2];

    if (pipe(pipe_in) != 0 || pipe(pipe_out) != 0) {
        throw std::system_error(errno, std::generic_category(), "Failed to create pipes");
    }

    posix_spawn_file_actions_t file_actions;
    posix_spawn_file_actions_init(&file_actions);

    // Child reads from pipe_in[0], so dup it to stdin
    posix_spawn_file_actions_adddup2(&file_actions, pipe_in[0], STDIN_FILENO);
    // Child writes to pipe_out[1], so dup it to stdout
    posix_spawn_file_actions_adddup2(&file_actions, pipe_out[1], STDOUT_FILENO);
    // Keep stderr as is, or we could redirect it if needed. For now, leave it.

    // Close all pipe ends in child since they are dup'd or unused
    posix_spawn_file_actions_addclose(&file_actions, pipe_in[0]);
    posix_spawn_file_actions_addclose(&file_actions, pipe_in[1]);
    posix_spawn_file_actions_addclose(&file_actions, pipe_out[0]);
    posix_spawn_file_actions_addclose(&file_actions, pipe_out[1]);

    std::vector<char*> c_args;
    c_args.push_back(const_cast<char*>(command.c_str()));
    for (const auto& arg : args) {
        c_args.push_back(const_cast<char*>(arg.c_str()));
    }
    c_args.push_back(nullptr);

#ifdef __APPLE__
    char **envp = *_NSGetEnviron();
#else
    char **envp = ::environ;
#endif

    int status = posix_spawnp(&pid_, command.c_str(), &file_actions, nullptr, c_args.data(), envp);

    posix_spawn_file_actions_destroy(&file_actions);

    if (status != 0) {
        close(pipe_in[0]);
        close(pipe_in[1]);
        close(pipe_out[0]);
        close(pipe_out[1]);
        throw std::system_error(status, std::generic_category(), "posix_spawnp failed");
    }

    // Parent side
    write_fd_ = pipe_in[1];
    read_fd_ = pipe_out[0];

    // Close unused ends
    close(pipe_in[0]);
    close(pipe_out[1]);

    // Make parent ends non-blocking
    int flags = fcntl(write_fd_, F_GETFL, 0);
    fcntl(write_fd_, F_SETFL, flags | O_NONBLOCK);

    flags = fcntl(read_fd_, F_GETFL, 0);
    fcntl(read_fd_, F_SETFL, flags | O_NONBLOCK);

    running_ = true;
    reader_thread_ = std::jthread(&Client::reader_thread_loop, this);

    return true;
}

void Client::stop() {
    running_ = false;

    if (pid_ != -1) {
        // Graceful termination
        kill(pid_, SIGTERM);

        // Simple timeout for SIGKILL fallback
        int status;
        pid_t res = waitpid(pid_, &status, WNOHANG);
        if (res == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            res = waitpid(pid_, &status, WNOHANG);
            if (res == 0) {
                kill(pid_, SIGKILL);
                waitpid(pid_, &status, 0);
            }
        }
        pid_ = -1;
    }

    // Now that the child is dead (or dying), EOF will be hit on read_fd_
    // Wait for reader thread to finish
    if (reader_thread_.joinable()) {
        reader_thread_.join();
    }

    if (write_fd_ != -1) {
        close(write_fd_);
        write_fd_ = -1;
    }
    if (read_fd_ != -1) {
        close(read_fd_);
        read_fd_ = -1;
    }
}

void Client::send_message(const json& message) {
    std::string serialized = message.dump();
    std::string payload = "Content-Length: " + std::to_string(serialized.size()) + "\r\n\r\n" + serialized;

    std::lock_guard<std::mutex> lock(write_mutex_);

    size_t total_written = 0;
    while (total_written < payload.size()) {
        ssize_t written = write(write_fd_, payload.data() + total_written, payload.size() - total_written);
        if (written < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // If it's non-blocking, we might need to wait or sleep briefly
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            throw std::system_error(errno, std::generic_category(), "Failed to write to MCP server");
        }
        total_written += written;
    }
}

json Client::initialize() {
    uint64_t req_id = next_id_++;

    json req = {
        {"jsonrpc", "2.0"},
        {"id", req_id},
        {"method", "initialize"},
        {"params", {
            {"protocolVersion", "2024-11-05"},
            {"capabilities", json::object()},
            {"clientInfo", {
                {"name", "mcp-client-cpp"},
                {"version", "1.0.0"}
            }}
        }}
    };

    std::future<json> fut;
    {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        fut = pending_requests_[req_id].get_future();
    }

    send_message(req);
    return fut.get();
}

void Client::send_initialized() {
    json req = {
        {"jsonrpc", "2.0"},
        {"method", "notifications/initialized"}
    };
    send_message(req);
}

json Client::list_tools() {
    uint64_t req_id = next_id_++;

    json req = {
        {"jsonrpc", "2.0"},
        {"id", req_id},
        {"method", "tools/list"}
    };

    std::future<json> fut;
    {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        fut = pending_requests_[req_id].get_future();
    }

    send_message(req);
    return fut.get();
}

json Client::call_tool(const std::string& name, const json& args) {
    uint64_t req_id = next_id_++;

    json req = {
        {"jsonrpc", "2.0"},
        {"id", req_id},
        {"method", "tools/call"},
        {"params", {
            {"name", name},
            {"arguments", args}
        }}
    };

    std::future<json> fut;
    {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        fut = pending_requests_[req_id].get_future();
    }

    send_message(req);
    return fut.get();
}

void Client::reader_thread_loop() {
    std::string buffer;
    char chunk[4096];

    while (running_) {
        ssize_t bytes_read = read(read_fd_, chunk, sizeof(chunk));
        if (bytes_read < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // No data available, sleep briefly and try again
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            // Real error
            running_ = false;
            break;
        } else if (bytes_read == 0) {
            // EOF
            running_ = false;
            break;
        }

        buffer.append(chunk, bytes_read);

        // Process message(s) from buffer
        while (true) {
            // Find Content-Length
            size_t cl_pos = buffer.find("Content-Length: ");
            if (cl_pos == std::string::npos) {
                // Check if it's newline delimited JSON (fallback)
                size_t nl_pos = buffer.find('\n');
                if (nl_pos != std::string::npos && buffer[0] == '{') {
                    std::string msg = buffer.substr(0, nl_pos);
                    buffer.erase(0, nl_pos + 1);
                    handle_message(msg);
                    continue;
                }
                break; // Need more data
            }

            size_t header_end = buffer.find("\r\n\r\n", cl_pos);
            if (header_end == std::string::npos) {
                break; // Need more data
            }

            size_t val_start = cl_pos + 16;
            size_t val_end = buffer.find("\r\n", val_start);
            int content_length = std::stoi(buffer.substr(val_start, val_end - val_start));

            size_t message_start = header_end + 4;
            if (buffer.size() < message_start + content_length) {
                break; // Need more data
            }

            std::string_view msg_view(buffer.data() + message_start, content_length);
            handle_message(msg_view);

            buffer.erase(0, message_start + content_length);
        }
    }

    // Clean up pending requests on exit
    std::lock_guard<std::mutex> lock(requests_mutex_);
    for (auto& [id, promise] : pending_requests_) {
        try {
            throw std::runtime_error("Connection closed before response received");
        } catch (...) {
            promise.set_exception(std::current_exception());
        }
    }
    pending_requests_.clear();
}

void Client::handle_message(std::string_view message) {
    try {
        json j = json::parse(message);

        if (j.contains("id") && (j.contains("result") || j.contains("error"))) {
            // Response to a request
            uint64_t id = j["id"].get<uint64_t>();

            std::lock_guard<std::mutex> lock(requests_mutex_);
            auto it = pending_requests_.find(id);
            if (it != pending_requests_.end()) {
                if (j.contains("error")) {
                    try {
                        throw std::runtime_error("RPC Error: " + j["error"].dump());
                    } catch (...) {
                        it->second.set_exception(std::current_exception());
                    }
                } else {
                    it->second.set_value(j["result"]);
                }
                pending_requests_.erase(it);
            }
        }
        // Could also handle server->client requests/notifications here
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse message: " << e.what() << "\nMessage: " << message << std::endl;
    }
}

} // namespace mcp
