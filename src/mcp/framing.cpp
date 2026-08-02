#include "src/mcp/framing.hpp"

#include <cassert>

namespace lmp::mcp {

namespace {

std::string_view strip_cr(std::string_view line) noexcept {
    while (!line.empty() && line.back() == '\r') {
        line.remove_suffix(1);
    }
    return line;
}

bool is_blank(std::string_view s) noexcept {
    for (const char c : s) {
        if (c != ' ' && c != '\t') {
            return false;
        }
    }
    return true;
}

} // namespace

void LineFramer::feed(std::string_view bytes, const LineFn& on_line) {
    feed(bytes, on_line, nullptr);
}

void LineFramer::feed(std::string_view bytes, const LineFn& on_line,
                      const OverflowFn& on_overflow) {
    buf_.append(bytes);

    std::size_t start = 0;
    for (;;) {
        const std::size_t nl = buf_.find('\n', start);
        if (nl == std::string::npos) {
            break;
        }

        const std::string_view line =
            strip_cr(std::string_view(buf_).substr(start, nl - start));
        start = nl + 1;

        if (!line.empty() && !is_blank(line)) {
            on_line(line);
        }
    }

    // Erase only what was consumed. What remains is a partial message and must survive
    // until the rest of it arrives.
    if (start > 0) {
        buf_.erase(0, start);
    }

    if (buf_.size() > kMaxMessageBytes) {
        const std::size_t dropped = buf_.size();
        buf_.clear();
        if (on_overflow) {
            on_overflow(dropped);
        }
    }
}

std::string encode_line(const nlohmann::json& message) {
    std::string s = message.dump();

    // The framing's one invariant. nlohmann guarantees it by escaping control characters
    // inside strings; asserting keeps that guarantee from becoming an assumption if the
    // serialiser is ever swapped out.
    assert(s.find('\n') == std::string::npos &&
           "encoded MCP message contains an interior newline; framing would desync");

    s.push_back('\n');
    return s;
}

} // namespace lmp::mcp
