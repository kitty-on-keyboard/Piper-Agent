#include "src/mcp/transport.hpp"

#include <array>
#include <cerrno>
#include <system_error>
#include <utility>

#include <poll.h>
#include <unistd.h>

namespace lmp::mcp {

namespace {

constexpr std::size_t kReadChunk = 65536;

// Retry-on-EINTR write to a raw fd, looping on partial writes. Taken from cook-off
// entrant C2, which was the only client of the seven to get both cases right.
bool write_fd_all(int fd, std::string_view bytes) {
    if (fd < 0) {
        return false;
    }
    const char* p = bytes.data();
    std::size_t remaining = bytes.size();
    while (remaining > 0) {
        const ssize_t n = ::write(fd, p, remaining);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        p += n;
        remaining -= static_cast<std::size_t>(n);
    }
    return true;
}

} // namespace

FdTransport::FdTransport(int read_fd, int write_fd, int stderr_fd)
    : read_fd_(read_fd), write_fd_(write_fd), stderr_fd_(stderr_fd) {
    int wake[2]{-1, -1};
    if (::pipe(wake) != 0) {
        throw std::system_error(errno, std::generic_category(), "pipe (transport wakeup)");
    }
    wake_read_fd_ = wake[0];
    wake_write_fd_ = wake[1];
}

FdTransport::~FdTransport() {
    stop();
    if (wake_read_fd_ >= 0) { ::close(wake_read_fd_); }
    if (wake_write_fd_ >= 0) { ::close(wake_write_fd_); }
}

void FdTransport::start(Handlers handlers) {
    handlers_ = std::move(handlers);
    writable_.store(true, std::memory_order_release);
    reader_ = std::thread([this] { reader_loop(); });
}

bool FdTransport::write_bytes(std::string_view bytes) {
    return write_fd_all(write_fd_, bytes);
}

bool FdTransport::send(const nlohmann::json& message) {
    // Deliberately NOT gated on the reader having finished. A server whose stdin has
    // hit EOF still owes a reply to every request it already accepted, and its stdout
    // is still open to carry them.
    if (!writable_.load(std::memory_order_acquire)) {
        return false;
    }
    const std::string line = encode_line(message);

    // One message per write, under one lock. Two threads interleaving halves of two
    // messages into the same pipe is a desync the peer cannot recover from.
    const std::lock_guard<std::mutex> lock(write_mutex_);
    if (!write_bytes(line)) {
        // The write side is genuinely gone now (EPIPE). Latch it so callers stop.
        writable_.store(false, std::memory_order_release);
        return false;
    }
    return true;
}

void FdTransport::stop() {
    if (stopping_.exchange(true, std::memory_order_acq_rel)) {
        // Already stopping. Still join, so stop() means "finished" on every path.
        if (reader_.joinable() && reader_.get_id() != std::this_thread::get_id()) {
            reader_.join();
        }
        return;
    }

    writable_.store(false, std::memory_order_release);

    // Wake the poll() immediately rather than waiting out a timeout.
    if (wake_write_fd_ >= 0) {
        const char b = 'x';
        const ssize_t ignored = ::write(wake_write_fd_, &b, 1);
        static_cast<void>(ignored);
    }

    if (reader_.joinable() && reader_.get_id() != std::this_thread::get_id()) {
        reader_.join();
    }
    fire_closed();
}

void FdTransport::fire_closed() {
    std::call_once(closed_once_, [this] {
        if (handlers_.on_closed) {
            handlers_.on_closed();
        }
    });
}

void FdTransport::reader_loop() {
    std::array<char, kReadChunk> chunk{};

    for (;;) {
        std::array<pollfd, 3> fds{};
        nfds_t nfds = 0;

        const std::size_t idx_read = nfds;
        fds[nfds].fd = read_fd_;
        fds[nfds].events = POLLIN;
        ++nfds;

        std::size_t idx_wake = static_cast<std::size_t>(-1);
        if (wake_read_fd_ >= 0) {
            idx_wake = nfds;
            fds[nfds].fd = wake_read_fd_;
            fds[nfds].events = POLLIN;
            ++nfds;
        }

        std::size_t idx_err = static_cast<std::size_t>(-1);
        if (stderr_fd_ >= 0) {
            idx_err = nfds;
            fds[nfds].fd = stderr_fd_;
            fds[nfds].events = POLLIN;
            ++nfds;
        }

        const int rc = ::poll(fds.data(), nfds, -1);
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }

        if (idx_wake != static_cast<std::size_t>(-1) && (fds[idx_wake].revents & POLLIN) != 0) {
            break; // stop() was called
        }

        // Drain captured stderr before anything else. This is the whole reason the
        // stderr fd is in the poll set: a pipe nobody reads is a child that blocks in
        // write(2) and stops answering (cook-off entrant C6).
        if (idx_err != static_cast<std::size_t>(-1) &&
            (fds[idx_err].revents & (POLLIN | POLLHUP)) != 0) {
            for (;;) {
                const ssize_t n = ::read(stderr_fd_, chunk.data(), chunk.size());
                if (n > 0) {
                    if (handlers_.on_stderr) {
                        handlers_.on_stderr(std::string_view(chunk.data(), static_cast<std::size_t>(n)));
                    }
                    if (static_cast<std::size_t>(n) < chunk.size()) {
                        break;
                    }
                    continue;
                }
                if (n < 0 && errno == EINTR) {
                    continue;
                }
                if (n == 0) {
                    stderr_fd_ = -1; // EOF; drop it from the poll set
                }
                break;
            }
        }

        // POLLIN is handled BEFORE POLLHUP, deliberately. A peer that writes its last
        // message and exits delivers POLLIN|POLLHUP in the same revents, and a loop
        // that checks the hangup first throws that message away. Cook-off entrant S4
        // does exactly this, and drops replies nondeterministically as a result.
        if ((fds[idx_read].revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
            const ssize_t n = ::read(read_fd_, chunk.data(), chunk.size());
            if (n > 0) {
                framer_.feed(
                    std::string_view(chunk.data(), static_cast<std::size_t>(n)),
                    [this](std::string_view line) {
                        nlohmann::json parsed =
                            nlohmann::json::parse(line, nullptr, /*allow_exceptions=*/false);
                        if (parsed.is_discarded()) {
                            if (handlers_.on_parse_error) {
                                handlers_.on_parse_error(line, "invalid JSON");
                            }
                            return;
                        }
                        if (handlers_.on_message) {
                            handlers_.on_message(parsed);
                        }
                    },
                    [this](std::size_t dropped) {
                        if (handlers_.on_parse_error) {
                            handlers_.on_parse_error(
                                {}, "message exceeded " +
                                        std::to_string(LineFramer::kMaxMessageBytes) +
                                        " bytes; dropped " + std::to_string(dropped));
                        }
                    });
                continue;
            }
            if (n < 0 && errno == EINTR) {
                continue;
            }
            break; // EOF or a hard error: the peer is gone
        }
    }

    // Only the read half has ended. writable_ stays as it is so queued replies can
    // still be written -- run() drains its workers after seeing this.
    reader_done_.store(true, std::memory_order_release);
    fire_closed();
}

// ---------------------------------------------------------------------------

StdioTransport::StdioTransport() : FdTransport(STDIN_FILENO, STDOUT_FILENO, -1) {}

// ---------------------------------------------------------------------------

SubprocessTransport::SubprocessTransport(Subprocess::Options options)
    : SubprocessTransport(std::make_unique<Subprocess>(std::move(options))) {}

SubprocessTransport::SubprocessTransport(std::unique_ptr<Subprocess> proc)
    // The child's stdout is our read side and its stdin is our write side.
    : FdTransport(proc->stdout_fd(), proc->stdin_fd(), proc->stderr_fd()),
      proc_(std::move(proc)) {}

SubprocessTransport::~SubprocessTransport() {
    // Stop the reader before the child dies, so the loop exits on the wakeup pipe
    // rather than on a read error it would report as an unexpected close.
    stop();
    if (proc_) {
        proc_->terminate();
    }
}

bool SubprocessTransport::write_bytes(std::string_view bytes) {
    return proc_ && proc_->write_all(bytes);
}

} // namespace lmp::mcp
