#pragma once
//
// A child process with pipes, as an RAII type.
//
// Kept separate from the protocol layer, which is the one structural idea worth taking
// from the client cook-off (entrant C5): every other entrant interleaved pid_t and
// pipe fds with the JSON-RPC correlation map in one class, and none of them could be
// tested without spawning something.
//
// The stderr policy is the part to read carefully. Cook-off entrant C6 piped the
// child's stderr and never read it, which deadlocks against any server that logs: once
// the 64 KB pipe buffer fills, the child blocks in write(2) forever and stops answering.
// Reproduced in docs/BAKEOFF_MCP.md against a server emitting 150 KB of startup noise --
// C6 hung until killed. So the default here is kInherit, and kCapture is only honoured
// by a transport that actually drains it.
//
#include <chrono>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sys/types.h>

namespace lmp::mcp {

enum class StderrMode {
    kInherit, // child writes to our stderr. Safe: no pipe to fill.
    kCapture, // piped; the owner MUST drain it (SubprocessTransport does).
    kDiscard, // redirected to /dev/null.
};

class Subprocess {
public:
    struct Options {
        std::string program;                                     // resolved via PATH
        std::vector<std::string> args;                           // excluding argv[0]
        std::vector<std::pair<std::string, std::string>> env;    // added to the parent's
        StderrMode stderr_mode = StderrMode::kInherit;
    };

    // Throws std::system_error if the child cannot be spawned.
    explicit Subprocess(Options options);
    ~Subprocess();

    Subprocess(const Subprocess&) = delete;
    Subprocess& operator=(const Subprocess&) = delete;
    Subprocess(Subprocess&&) = delete;
    Subprocess& operator=(Subprocess&&) = delete;

    [[nodiscard]] int stdin_fd() const noexcept { return stdin_fd_; }
    [[nodiscard]] int stdout_fd() const noexcept { return stdout_fd_; }
    [[nodiscard]] int stderr_fd() const noexcept { return stderr_fd_; } // -1 unless kCapture
    [[nodiscard]] pid_t pid() const noexcept { return pid_; }

    // Retries on EINTR and loops on partial writes -- a pipe write is not obliged to
    // take the whole buffer, and a message truncated at 4096 bytes desyncs the peer's
    // framer permanently. Returns false if the pipe is closed (EPIPE).
    bool write_all(std::string_view data);

    // Sending EOF is how a well-behaved MCP server is asked to exit. Idempotent.
    void close_stdin() noexcept;

    // SIGTERM, wait up to `grace`, then SIGKILL. Reaps the child either way, so no
    // zombie survives. Idempotent; safe to call after the child has already exited.
    void terminate(std::chrono::milliseconds grace = std::chrono::milliseconds(2000)) noexcept;

    // Exit status if the child has been reaped, nullopt while it is still running.
    [[nodiscard]] std::optional<int> exit_status() const noexcept { return exit_status_; }

private:
    void close_fd(int& fd) noexcept;

    pid_t pid_ = -1;
    int stdin_fd_ = -1;
    int stdout_fd_ = -1;
    int stderr_fd_ = -1;
    std::optional<int> exit_status_;
};

} // namespace lmp::mcp
