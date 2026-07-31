#pragma once
//
// Transport -- FRAME IN THE PRODUCER (spec S4.2, S4.3).
//
// One reader thread owns stdin. It reads 64 KB blocks into a growable accumulator,
// splits on newline, and pushes COMPLETE messages into an SPSC queue of owned strings.
//
// This is not a micro-optimisation; it removes a bug class. v1 read 127 bytes at a time
// into a fixed-chunk ring and reassembled in the consumer, so a control-message check
// in the reader operated on an arbitrary byte window and could not see a message that
// straddled two reads. Never let a read boundary be semantically visible.
//
// And the cancel path: v1 did `chunk.find("agent/cancel")` on raw transport bytes,
// which fired on ANY inbound payload containing that literal -- including chat history,
// so discussing cancellation could cancel a running mission. Here the reader parses the
// `method` field of a whole message. NO SUBSTRING MATCHING ON TRANSPORT BYTES, EVER.
//
// The measured cost of the whole outbound transport in v1 was 0.75 ms per run against
// runs of 220-460 s -- 0.0003%. So this is optimised for correctness and clarity, and
// there is deliberately no shared memory anywhere near it (S4.1).
//
#include <atomic>
#include <string>
#include <thread>

#include "src/model/backend.hpp"
#include "src/platform/spsc_channel.hpp"

namespace lmp::surface {

inline constexpr std::size_t kReadBlockBytes = 64 * 1024;

// Extracts the JSON-RPC "method" of a COMPLETE message. Returns empty if absent.
// Operates on a whole message by construction -- it is only ever called with one.
[[nodiscard]] std::string method_of(std::string_view message);

// Extracts a top-level string field. Same contract: whole messages only.
[[nodiscard]] std::string string_field(std::string_view message, std::string_view key);

// Extracts a boolean field. Returns true ONLY for a literal `true`; a missing key, a
// malformed value and an explicit `false` are all false. That asymmetry is deliberate:
// the one caller is the approval reply, where anything we failed to understand must
// read as "not approved" (S7.2).
[[nodiscard]] bool bool_field(std::string_view message, std::string_view key);

// Extracts a numeric field. Returns `fallback` when the key is absent or its value is
// not a bare JSON number -- so a caller passes the value it already intends to use and
// a missing setting keeps it, rather than collapsing to zero. A quoted number is NOT a
// number: "0.6" is a string, and silently coercing it would make a typo in a settings
// file look like a deliberate choice.
[[nodiscard]] double double_field(std::string_view message, std::string_view key,
                                  double fallback);

class StdinReader {
  public:
    StdinReader(platform::SpscChannel<std::string>& out, model::CancelToken& cancel)
        : out_(out), cancel_(cancel) {}
    ~StdinReader();
    StdinReader(const StdinReader&) = delete;
    StdinReader& operator=(const StdinReader&) = delete;

    // Spawns the reader thread. It exits on stdin EOF -- which is how the sidecar
    // notices the parent died and shuts down rather than orphaning a process holding a
    // model in memory (S12.3).
    void start(int fd);
    void join();

    // Test seam: feed bytes as if they arrived from a read, with the SAME framing path.
    // Chunk boundaries may fall anywhere, so a test can put one mid-message and prove
    // the boundary stays invisible.
    void feed_for_test(std::string_view bytes);

    [[nodiscard]] std::size_t cancels_seen() const noexcept {
        return cancels_.load(std::memory_order_acquire);
    }

  private:
    void drain_accumulator();
    void deliver(std::string message);

    platform::SpscChannel<std::string>& out_;
    model::CancelToken& cancel_;
    std::string accumulator_;
    std::thread thread_;
    std::atomic<std::size_t> cancels_{0};
};

} // namespace lmp::surface
