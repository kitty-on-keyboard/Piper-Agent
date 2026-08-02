#include "src/mcp/subprocess.hpp"

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <system_error>
#include <thread>

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace lmp::mcp {

namespace {

// RAII for the two fds of a pipe, so an exception between pipe() and the point where
// ownership is handed to Subprocess cannot leak them.
struct Pipe {
    int fds[2]{-1, -1};

    Pipe() {
        if (::pipe(fds) != 0) {
            throw std::system_error(errno, std::generic_category(), "pipe");
        }
    }
    ~Pipe() {
        if (fds[0] >= 0) { ::close(fds[0]); }
        if (fds[1] >= 0) { ::close(fds[1]); }
    }
    Pipe(const Pipe&) = delete;
    Pipe& operator=(const Pipe&) = delete;

    int release_read() { const int f = fds[0]; fds[0] = -1; return f; }
    int release_write() { const int f = fds[1]; fds[1] = -1; return f; }
};

// posix_spawn wants char* const*, and the strings must outlive the call.
class ArgvBuilder {
public:
    void push(const std::string& s) {
        storage_.push_back(s);
    }
    // Built in a second pass: push() may reallocate storage_ and invalidate pointers
    // taken during the first.
    std::vector<char*> finish() {
        std::vector<char*> out;
        out.reserve(storage_.size() + 1);
        for (auto& s : storage_) {
            out.push_back(s.data());
        }
        out.push_back(nullptr);
        return out;
    }

private:
    std::vector<std::string> storage_;
};

} // namespace

Subprocess::Subprocess(Options options) {
    Pipe in_pipe;
    Pipe out_pipe;
    std::optional<Pipe> err_pipe;
    if (options.stderr_mode == StderrMode::kCapture) {
        err_pipe.emplace();
    }

    posix_spawn_file_actions_t actions;
    if (const int rc = posix_spawn_file_actions_init(&actions); rc != 0) {
        throw std::system_error(rc, std::generic_category(), "posix_spawn_file_actions_init");
    }
    // Destroyed on every path out of the block below.
    struct ActionsGuard {
        posix_spawn_file_actions_t* a;
        ~ActionsGuard() { posix_spawn_file_actions_destroy(a); }
    } guard{&actions};

    // Child: stdin <- read end of in_pipe, stdout -> write end of out_pipe.
    posix_spawn_file_actions_adddup2(&actions, in_pipe.fds[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&actions, out_pipe.fds[1], STDOUT_FILENO);

    if (options.stderr_mode == StderrMode::kCapture) {
        posix_spawn_file_actions_adddup2(&actions, err_pipe->fds[1], STDERR_FILENO);
    } else if (options.stderr_mode == StderrMode::kDiscard) {
        posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    }
    // kInherit: no action. The child keeps our stderr, and there is no pipe to fill.

    // Close the parent-side ends in the child. Leaving them open means the child holds
    // a write end of its own stdout pipe, so the parent never sees EOF when the child
    // dies -- the reader thread then blocks forever on a process that no longer exists.
    posix_spawn_file_actions_addclose(&actions, in_pipe.fds[1]);
    posix_spawn_file_actions_addclose(&actions, out_pipe.fds[0]);
    if (err_pipe.has_value()) {
        posix_spawn_file_actions_addclose(&actions, err_pipe->fds[0]);
    }

    ArgvBuilder argv;
    argv.push(options.program);
    for (const auto& a : options.args) {
        argv.push(a);
    }
    std::vector<char*> argv_ptrs = argv.finish();

    // Environment: the parent's, plus the caller's additions. MCP servers routinely
    // need credentials passed this way, and inheriting a bare environ is not enough.
    ArgvBuilder envp;
    for (char** e = environ; *e != nullptr; ++e) {
        envp.push(*e);
    }
    for (const auto& [k, v] : options.env) {
        envp.push(k + "=" + v);
    }
    std::vector<char*> envp_ptrs = envp.finish();

    const int rc = posix_spawnp(&pid_, options.program.c_str(), &actions, nullptr,
                                argv_ptrs.data(), envp_ptrs.data());
    if (rc != 0) {
        pid_ = -1;
        throw std::system_error(rc, std::generic_category(),
                                "posix_spawnp: " + options.program);
    }

    // Take ownership of the parent-side ends and let Pipe close the child-side ones.
    stdin_fd_ = in_pipe.release_write();
    stdout_fd_ = out_pipe.release_read();
    if (err_pipe.has_value()) {
        stderr_fd_ = err_pipe->release_read();
    }
}

Subprocess::~Subprocess() {
    terminate();
    close_fd(stdin_fd_);
    close_fd(stdout_fd_);
    close_fd(stderr_fd_);
}

void Subprocess::close_fd(int& fd) noexcept {
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
}

bool Subprocess::write_all(std::string_view data) {
    if (stdin_fd_ < 0) {
        return false;
    }
    const char* p = data.data();
    std::size_t remaining = data.size();

    while (remaining > 0) {
        const ssize_t n = ::write(stdin_fd_, p, remaining);
        if (n < 0) {
            if (errno == EINTR) {
                continue; // a signal, not a failure
            }
            return false; // EPIPE: the child is gone
        }
        p += n;
        remaining -= static_cast<std::size_t>(n);
    }
    return true;
}

void Subprocess::close_stdin() noexcept {
    close_fd(stdin_fd_);
}

void Subprocess::terminate(std::chrono::milliseconds grace) noexcept {
    if (pid_ <= 0 || exit_status_.has_value()) {
        return;
    }

    // Closing stdin first gives a well-behaved server the chance to exit on EOF, which
    // is the graceful path the spec describes. Only then do we escalate.
    close_fd(stdin_fd_);

    int status = 0;
    if (::waitpid(pid_, &status, WNOHANG) == pid_) {
        exit_status_ = status;
        return;
    }

    ::kill(pid_, SIGTERM);

    // Poll rather than sleep-then-check: a server that exits in 5 ms should not cost
    // the full grace period on every teardown.
    const auto deadline = std::chrono::steady_clock::now() + grace;
    for (;;) {
        const pid_t r = ::waitpid(pid_, &status, WNOHANG);
        if (r == pid_) {
            exit_status_ = status;
            return;
        }
        if (r < 0) {
            // Already reaped, or never ours to reap.
            exit_status_ = 0;
            return;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    ::kill(pid_, SIGKILL);
    if (::waitpid(pid_, &status, 0) == pid_) {
        exit_status_ = status;
    } else {
        exit_status_ = 0;
    }
}

} // namespace lmp::mcp
