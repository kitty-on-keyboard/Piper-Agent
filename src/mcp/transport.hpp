#pragma once
//
// Transport: the seam between "bytes on a wire" and "MCP messages".
//
// It exists so the protocol layer never sees a file descriptor. The spec defines two
// transports -- stdio and streamable HTTP -- and only stdio is implemented here, but the
// client and server above this interface contain no I/O at all, so adding HTTP is a new
// file rather than a rewrite of two.
//
// Delivery is push-based, on a reader thread owned by the transport. A pull-based
// `receive(timeout)` was the alternative and is worse: every caller then has to poll,
// and server-initiated notifications arrive only when someone happens to ask.
//
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

#include <nlohmann/json.hpp>

#include "src/mcp/framing.hpp"
#include "src/mcp/protocol.hpp"
#include "src/mcp/subprocess.hpp"

namespace lmp::mcp {

class Transport {
public:
    struct Handlers {
        // A well-formed JSON value arrived. Classification into request/response/
        // notification happens above; the transport does not care.
        std::function<void(const nlohmann::json&)> on_message;

        // A line arrived that was not valid JSON. The server turns this into -32700;
        // the client logs it. Either way the stream is not abandoned -- one bad line
        // is not a reason to drop a session.
        std::function<void(std::string_view raw, std::string_view reason)> on_parse_error;

        // The peer's output stream reached EOF, or the transport was stopped. Fired
        // exactly once.
        std::function<void()> on_closed;

        // Captured child stderr, when the transport has any. Never invoked for
        // StderrMode::kInherit.
        std::function<void(std::string_view)> on_stderr;
    };

    virtual ~Transport() = default;

    Transport(const Transport&) = delete;
    Transport& operator=(const Transport&) = delete;

    // Starts the reader thread. Handlers are invoked on that thread, never re-entrantly.
    virtual void start(Handlers handlers) = 0;

    // Serialise and write one message. Thread-safe. Returns false once the peer is gone.
    virtual bool send(const nlohmann::json& message) = 0;

    // Idempotent. Blocks until the reader thread has finished, so no handler can fire
    // after it returns.
    virtual void stop() = 0;

    [[nodiscard]] virtual bool is_open() const = 0;

protected:
    Transport() = default;
};

// ---------------------------------------------------------------------------
// A transport over file descriptors it does not own.
//
// One poll() covers the read fd, the optional stderr fd, and a self-pipe used to wake
// the loop on stop(). The self-pipe is why stop() is immediate rather than "within the
// poll timeout": a 100 ms timeout loop works, but it also means every teardown pays
// 100 ms and every process exit looks slightly hung.
// ---------------------------------------------------------------------------
class FdTransport : public Transport {
public:
    FdTransport(int read_fd, int write_fd, int stderr_fd = -1);
    ~FdTransport() override;

    void start(Handlers handlers) override;
    bool send(const nlohmann::json& message) override;
    void stop() override;
    [[nodiscard]] bool is_open() const override { return writable_.load(std::memory_order_acquire); }

    // True once the peer's output stream has ended. Distinct from is_open(): a peer that
    // closes our stdin has stopped talking, but our stdout is still open and any reply
    // already in flight must still be written. Collapsing the two drops exactly those
    // replies -- which is what this class did until the conformance board caught it.
    [[nodiscard]] bool reader_finished() const { return reader_done_.load(std::memory_order_acquire); }

protected:
    // Overridden by SubprocessTransport so a write goes through Subprocess::write_all.
    virtual bool write_bytes(std::string_view bytes);

private:
    void reader_loop();
    void fire_closed();

    int read_fd_;
    int write_fd_;
    int stderr_fd_;

    int wake_read_fd_ = -1;
    int wake_write_fd_ = -1;

    Handlers handlers_;
    LineFramer framer_;

    std::thread reader_;
    std::mutex write_mutex_;
    std::atomic<bool> writable_{false};   // the write side is usable
    std::atomic<bool> reader_done_{false}; // the peer's output stream ended
    std::atomic<bool> stopping_{false};
    std::once_flag closed_once_;
};

// ---------------------------------------------------------------------------
// Server side: speaks MCP on this process's own stdin/stdout.
//
// The spec's hard rule for this transport is that stdout carries nothing but JSON-RPC.
// A stray printf is not a cosmetic bug -- it desyncs the client's framer. Servers built
// on this should log through Server::log(), which goes to stderr.
// ---------------------------------------------------------------------------
class StdioTransport final : public FdTransport {
public:
    StdioTransport();
};

// ---------------------------------------------------------------------------
// Client side: spawns a server and speaks MCP over its pipes.
//
// Owns the child. Destroying it closes stdin, waits briefly, then escalates -- see
// Subprocess::terminate.
// ---------------------------------------------------------------------------
class SubprocessTransport final : public FdTransport {
public:
    explicit SubprocessTransport(Subprocess::Options options);
    ~SubprocessTransport() override;

    [[nodiscard]] Subprocess& process() noexcept { return *proc_; }

protected:
    bool write_bytes(std::string_view bytes) override;

private:
    // Constructed before the base is given its fds, so the unique_ptr is initialised in
    // a static function called from the delegating constructor.
    explicit SubprocessTransport(std::unique_ptr<Subprocess> proc);

    std::unique_ptr<Subprocess> proc_;
};

} // namespace lmp::mcp
