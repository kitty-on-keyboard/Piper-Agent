#pragma once
//
// MCP stdio framing: one JSON object per line, newline-delimited, UTF-8.
//
// There is no length header. Three of the seven cook-off clients wrote LSP-style
// `Content-Length:` preambles -- one of them under a comment reading "For standard MCP"
// -- and the wire trace in docs/BAKEOFF_MCP.md shows the result: a parse error at the
// server for every message sent. They pass their own tests because they ship a mock
// that speaks the same invented dialect.
//
// The framer is a class rather than a getline() loop because a read() boundary falls
// wherever the kernel puts it. Half a message now and half a message in 40ms is the
// normal case, not the edge case, and it is the single thing a hand-rolled loop gets
// wrong first.
//
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace lmp::mcp {

class LineFramer {
public:
    // A message larger than this is treated as a stream desync rather than a message:
    // without a cap, a peer that never sends a newline is an unbounded allocation.
    static constexpr std::size_t kMaxMessageBytes = 32u * 1024u * 1024u;

    using LineFn = std::function<void(std::string_view)>;
    using OverflowFn = std::function<void(std::size_t)>;

    // Append raw bytes and invoke `on_line` once per complete line, in order. Trailing
    // \r is stripped (a peer with CRLF endings is still speaking the protocol) and
    // blank lines are skipped (they carry no message and are not an error).
    //
    // `on_line` sees a view into the internal buffer; it is valid only for the duration
    // of the call.
    void feed(std::string_view bytes, const LineFn& on_line);

    // As feed(), but reports a desync instead of growing without bound. When the
    // buffered partial line exceeds kMaxMessageBytes the buffer is dropped and
    // `on_overflow` is called with the number of bytes discarded.
    void feed(std::string_view bytes, const LineFn& on_line, const OverflowFn& on_overflow);

    // Bytes held in a partial line. Non-zero at EOF means the peer was cut off
    // mid-message, which is worth logging -- it is the signature of a crashed server.
    [[nodiscard]] std::size_t buffered() const noexcept { return buf_.size(); }

    void reset() noexcept { buf_.clear(); }

private:
    std::string buf_;
};

// Serialise one message for the wire: compact, then exactly one '\n'.
//
// nlohmann escapes literal newlines inside strings, so the result cannot contain an
// interior newline and cannot desync the peer's framer. encode_line() asserts that
// rather than assuming it, because the invariant is the whole reason the framing works.
[[nodiscard]] std::string encode_line(const nlohmann::json& message);

} // namespace lmp::mcp
