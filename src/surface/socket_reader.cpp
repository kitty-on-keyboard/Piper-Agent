#include "src/surface/socket_reader.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "src/surface/worker.hpp"

namespace lmp::surface::worker {

std::string default_socket_path() {
    const char* env = std::getenv("LMP_WORKER_SOCKET");
    if (env && *env) return env;
    const char* home = std::getenv("HOME");
    if (home && *home) {
        return std::string(home) + "/.piper/worker.sock";
    }
    return "/tmp/piper_worker.sock";
}

std::string default_pid_path() {
    const char* home = std::getenv("HOME");
    if (home && *home) {
        return std::string(home) + "/.piper/worker.pid";
    }
    return "/tmp/piper_worker.pid";
}

bool is_daemon_alive(const std::string& socket_path) {
    std::error_code ec;
    if (!std::filesystem::exists(socket_path, ec)) {
        return false;
    }

    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return false;

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (::connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        // Stale socket from a dead daemon: unlink it and its pid file
        ::unlink(socket_path.c_str());
        ::unlink(default_pid_path().c_str());
        return false;
    }

    // Ping check
    std::string ping = "{\"method\":\"ping\"}\n";
    if (::write(fd, ping.data(), ping.size()) <= 0) {
        ::close(fd);
        ::unlink(socket_path.c_str());
        return false;
    }

    struct pollfd pfd{};
    pfd.fd = fd;
    pfd.events = POLLIN;
    if (::poll(&pfd, 1, 2000) <= 0 || !(pfd.revents & POLLIN)) {
        ::close(fd);
        return false;
    }

    char buf[1024];
    ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
    ::close(fd);
    if (n <= 0) return false;
    buf[n] = '\0';

    auto j = nlohmann::json::parse(buf, nullptr, false);
    return !j.is_discarded() && j.value("status", "") == "ok";
}

std::optional<int> forward_to_daemon(
    const std::string& socket_path, const std::string& task_path, bool jsonl,
    bool auto_approve_irreversible, bool auto_approve_all) {
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return std::nullopt;

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (::connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return std::nullopt;
    }

    nlohmann::json req = {
        {"method", "run"},
        {"task", task_path},
        {"jsonl", jsonl},
        {"auto_approve_irreversible", auto_approve_irreversible},
        {"auto_approve_all", auto_approve_all}
    };
    std::string req_str = req.dump() + "\n";
    if (::write(fd, req_str.data(), req_str.size()) <= 0) {
        ::close(fd);
        return std::nullopt;
    }

    std::string accum;
    char buf[4096];
    int exit_code = kExitError;
    bool saw_result = false;

    while (true) {
        ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n <= 0) {
            if (!accum.empty()) {
                auto j = nlohmann::json::parse(accum, nullptr, false);
                if (!j.is_discarded() && j.is_object() && j.contains("exit_code")) {
                    exit_code = j["exit_code"].get<int>();
                    saw_result = true;
                }
            }
            break;
        }
        accum.append(buf, static_cast<size_t>(n));

        size_t nl;
        while ((nl = accum.find('\n')) != std::string::npos) {
            std::string line = accum.substr(0, nl);
            accum.erase(0, nl + 1);
            if (line.empty()) continue;

            auto j = nlohmann::json::parse(line, nullptr, false);
            if (!j.is_discarded() && j.is_object()) {
                if (j.contains("exit_code")) {
                    exit_code = j["exit_code"].get<int>();
                    saw_result = true;
                    break;
                } else if (j.value("kind", "") == "ask") {
                    std::fprintf(stderr,
                                 "piper: awaiting_user.json written (seq %llu). Question: %s\n"
                                 "Waiting for answer.json...\n",
                                 static_cast<unsigned long long>(j.value("seq", 0ULL)),
                                 j.value("question", "").c_str());
                    std::fflush(stderr);
                    if (jsonl) {
                        std::fwrite(line.data(), 1, line.size(), stdout);
                        std::fputc('\n', stdout);
                        std::fflush(stdout);
                    }
                } else if (jsonl) {
                    std::fwrite(line.data(), 1, line.size(), stdout);
                    std::fputc('\n', stdout);
                    std::fflush(stdout);
                }
            }
        }
        if (saw_result) break;
    }

    ::close(fd);
    return saw_result ? std::optional<int>(exit_code) : std::nullopt;
}

DaemonListener::DaemonListener(DaemonConfig config)
    : config_(std::move(config)) {}

DaemonListener::~DaemonListener() {
    stop();
}

bool DaemonListener::start() {
    if (is_daemon_alive(config_.socket_path)) {
        return false;
    }

    std::error_code ec;
    std::filesystem::path parent = std::filesystem::path(config_.socket_path).parent_path();
    std::filesystem::create_directories(parent, ec);

    ::unlink(config_.socket_path.c_str());

    listen_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd_ < 0) return false;
    ::fcntl(listen_fd_, F_SETFD, FD_CLOEXEC);

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, config_.socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (::bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    if (::listen(listen_fd_, 5) != 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
        ::unlink(config_.socket_path.c_str());
        return false;
    }

    // Write PID file
    std::ofstream pid_file(config_.pid_path);
    if (pid_file.is_open()) {
        pid_file << ::getpid() << "\n";
    }

    bound_ = true;
    return true;
}

int DaemonListener::accept_client() {
    if (listen_fd_ < 0) return -1;

    struct pollfd pfd{};
    pfd.fd = listen_fd_;
    pfd.events = POLLIN;

    int timeout_ms = config_.idle_timeout_s > 0
        ? static_cast<int>(config_.idle_timeout_s * 1000.0)
        : -1;

    int ret = ::poll(&pfd, 1, timeout_ms);
    if (ret <= 0 || !(pfd.revents & POLLIN)) {
        return -1;
    }

    int client_fd = ::accept(listen_fd_, nullptr, nullptr);
    if (client_fd >= 0) {
        ::fcntl(client_fd, F_SETFD, FD_CLOEXEC);
    }
    return client_fd;
}

void DaemonListener::stop() {
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    if (bound_) {
        ::unlink(config_.socket_path.c_str());
        ::unlink(config_.pid_path.c_str());
        bound_ = false;
    }
}

} // namespace lmp::surface::worker
