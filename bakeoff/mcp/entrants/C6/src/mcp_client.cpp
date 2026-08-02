#include "mcp_client.hpp"

#include <iostream>
#include <stdexcept>
#include <spawn.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>

extern char **environ;

namespace mcp {

// helper to ignore SIGPIPE globally during process init or in a constructor,
// though usually best done once in main or statically.
static struct SigPipeIgnorer {
    SigPipeIgnorer() {
        signal(SIGPIPE, SIG_IGN);
    }
} sigpipe_ignorer;

McpProcess::McpProcess(const std::string& command, const std::vector<std::string>& args) {
    if (pipe(stdin_pipe_) != 0) {
        throw std::runtime_error("Failed to create stdin pipe for McpProcess");
    }
    if (pipe(stdout_pipe_) != 0) {
        close(stdin_pipe_[0]); close(stdin_pipe_[1]);
        throw std::runtime_error("Failed to create stdout pipe for McpProcess");
    }
    if (pipe(stderr_pipe_) != 0) {
        close(stdin_pipe_[0]); close(stdin_pipe_[1]);
        close(stdout_pipe_[0]); close(stdout_pipe_[1]);
        throw std::runtime_error("Failed to create stderr pipe for McpProcess");
    }

    posix_spawn_file_actions_t file_actions;
    posix_spawn_file_actions_init(&file_actions);

    // stdin
    posix_spawn_file_actions_adddup2(&file_actions, stdin_pipe_[0], STDIN_FILENO);
    posix_spawn_file_actions_addclose(&file_actions, stdin_pipe_[1]);

    // stdout
    posix_spawn_file_actions_adddup2(&file_actions, stdout_pipe_[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&file_actions, stdout_pipe_[0]);

    // stderr
    posix_spawn_file_actions_adddup2(&file_actions, stderr_pipe_[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&file_actions, stderr_pipe_[0]);

    std::vector<char*> c_args;
    c_args.push_back(const_cast<char*>(command.c_str()));
    for (const auto& arg : args) {
        c_args.push_back(const_cast<char*>(arg.c_str()));
    }
    c_args.push_back(nullptr);

    int status = posix_spawnp(&pid_, command.c_str(), &file_actions, nullptr, c_args.data(), environ);

    posix_spawn_file_actions_destroy(&file_actions);

    if (status != 0) {
        throw std::runtime_error("posix_spawn failed");
    }

    // Close child ends of pipes in parent
    close(stdin_pipe_[0]);
    close(stdout_pipe_[1]);
    close(stderr_pipe_[1]);
}

McpProcess::~McpProcess() {
    terminate();
}

void McpProcess::terminate() {
    if (pid_ > 0) {
        kill(pid_, SIGTERM);

        int status;
        // Wait briefly for graceful exit, else SIGKILL
        pid_t w = waitpid(pid_, &status, WNOHANG);
        if (w == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            w = waitpid(pid_, &status, WNOHANG);
            if (w == 0) {
                kill(pid_, SIGKILL);
                waitpid(pid_, &status, 0);
            }
        }

        pid_ = -1;
    }

    if (stdin_pipe_[1] != -1) { close(stdin_pipe_[1]); stdin_pipe_[1] = -1; }
    if (stdout_pipe_[0] != -1) { close(stdout_pipe_[0]); stdout_pipe_[0] = -1; }
    if (stderr_pipe_[0] != -1) { close(stderr_pipe_[0]); stderr_pipe_[0] = -1; }
}

std::string McpProcess::read_stdout() {
    std::string result;
    char buffer[4096];
    while (true) {
        ssize_t bytes_read = read(stdout_pipe_[0], buffer, sizeof(buffer));
        if (bytes_read > 0) {
            result.append(buffer, bytes_read);
        } else {
            break;
        }
    }
    return result;
}

std::string McpProcess::read_stderr() {
    std::string result;
    char buffer[4096];
    while (true) {
        ssize_t bytes_read = read(stderr_pipe_[0], buffer, sizeof(buffer));
        if (bytes_read > 0) {
            result.append(buffer, bytes_read);
        } else {
            break;
        }
    }
    return result;
}

ssize_t McpProcess::read_stdout_raw(std::span<char> buf) {
    return read(stdout_pipe_[0], buf.data(), buf.size());
}

bool McpProcess::write_stdin(std::string_view data) {
    size_t total_written = 0;
    while (total_written < data.size()) {
        ssize_t w = write(stdin_pipe_[1], data.data() + total_written, data.size() - total_written);
        if (w < 0) {
            return false;
        }
        total_written += w;
    }
    return true;
}


// McpClient Implementation

McpClient::McpClient(const std::string& command, const std::vector<std::string>& args) {
    process_ = std::make_unique<McpProcess>(command, args);
}

McpClient::~McpClient() {
    if (reader_thread_.joinable()) {
        reader_thread_.request_stop();
        // Closing the stdout pipe will unblock the read call
        process_->terminate();
        reader_thread_.join(); // Explicitly join before member destruction
    }
}

void McpClient::start() {
    reader_thread_ = std::jthread([this](std::stop_token stoken) {
        this->reader_thread_func(stoken);
    });
}

void McpClient::write_message(const json& msg) {
    std::string serialized = msg.dump() + "\n"; // Assuming line-delimited JSON

    std::lock_guard<std::mutex> lock(write_mutex_); // Synchronize writes
    if (!process_->write_stdin(serialized)) {
        throw std::runtime_error("Failed to write to MCP process");
    }
}

std::future<json> McpClient::send_request(const std::string& method, const json& params) {
    std::uint64_t id = next_id_++;

    json req = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"method", method},
    };
    if (!params.empty()) {
        req["params"] = params;
    }

    std::future<json> future;
    {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        future = pending_requests_[id].get_future();
    }

    write_message(req);
    return future;
}

void McpClient::send_notification(const std::string& method, const json& params) {
    json notif = {
        {"jsonrpc", "2.0"},
        {"method", method}
    };
    if (!params.empty()) {
        notif["params"] = params;
    }
    write_message(notif);
}


std::future<json> McpClient::list_tools() {
    return send_request("tools/list");
}

std::future<json> McpClient::call_tool(const std::string& name, const json& arguments) {
    json params = {
        {"name", name},
        {"arguments", arguments}
    };
    return send_request("tools/call", params);
}

void McpClient::reader_thread_func(std::stop_token stoken) {
    std::string buffer;
    char chunk[4096];
    std::span<char> chunk_span(chunk);

    while (!stoken.stop_requested()) {
        ssize_t bytes_read = process_->read_stdout_raw(chunk_span);
        if (bytes_read <= 0) {
            break; // EOF or error
        }

        buffer.append(chunk, bytes_read);

        // Simple line-delimited parsing
        size_t pos;
        while ((pos = buffer.find('\n')) != std::string::npos) {
            std::string line = buffer.substr(0, pos);
            buffer.erase(0, pos + 1);
            if (!line.empty()) {
                handle_message(line);
            }
        }
    }
}

void McpClient::handle_message(const std::string& msg) {
    try {
        json j = json::parse(msg);

        if (j.contains("id")) {
            // Can be int or string, typically int for requests we send
            std::uint64_t id = j["id"].get<std::uint64_t>();

            std::lock_guard<std::mutex> lock(requests_mutex_);
            auto it = pending_requests_.find(id);
            if (it != pending_requests_.end()) {
                it->second.set_value(j);
                pending_requests_.erase(it);
            }
        } else if (j.contains("method")) {
            // Notification or request from server
            // std::cout << "Received notification: " << j["method"] << std::endl;
        }
    } catch (const json::parse_error& e) {
        // std::cerr << "JSON parse error: " << e.what() << std::endl;
    }
}

} // namespace mcp
