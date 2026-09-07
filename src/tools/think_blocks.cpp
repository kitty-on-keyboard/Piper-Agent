#include "src/tools/think_blocks.hpp"

#include "src/platform/fs.hpp"

namespace lmp::tools {
namespace {

[[nodiscard]] bool line_start(std::string_view t, std::size_t i) noexcept {
    return i == 0 || t[i - 1] == '\n';
}

[[nodiscard]] bool is_space(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\r';
}

[[nodiscard]] bool whitespace_only(std::string_view s) noexcept {
    for (char c : s) {
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::string_view trim_view(std::string_view s) noexcept {
    while (!s.empty() && is_space(s.front())) {
        s.remove_prefix(1);
    }
    while (!s.empty() && is_space(s.back())) {
        s.remove_suffix(1);
    }
    return s;
}

// A line that is exactly ``` plus optional trailing whitespace. `i` is a line start.
[[nodiscard]] bool closing_fence(std::string_view t, std::size_t i,
                                 std::size_t& line_end) noexcept {
    if (i + 3 > t.size() || t[i] != '`' || t[i + 1] != '`' || t[i + 2] != '`') {
        return false;
    }
    std::size_t j = i + 3;
    while (j < t.size() && is_space(t[j])) {
        ++j;
    }
    if (j < t.size() && t[j] != '\n') {
        return false;
    }
    line_end = j < t.size() ? j + 1 : j;
    return true;
}

} // namespace

FenceHarvest extract_fenced_blocks(std::string_view text) {
    FenceHarvest out;
    std::size_t i = 0;
    while (i < text.size()) {
        if (!line_start(text, i) || i + 3 > text.size() || text[i] != '`' ||
            text[i + 1] != '`' || text[i + 2] != '`') {
            ++i;
            continue;
        }
        std::size_t nl = text.find('\n', i + 3);
        if (nl == std::string_view::npos) {
            break; // opener with no body line: unclosed
        }
        const std::string_view lang = trim_view(text.substr(i + 3, nl - (i + 3)));
        const std::size_t body_begin = nl + 1;
        std::size_t j = body_begin;
        bool closed = false;
        std::size_t close_at = 0;
        std::size_t after_close = 0;
        while (j < text.size()) {
            if (line_start(text, j) && closing_fence(text, j, after_close)) {
                close_at = j;
                closed = true;
                break;
            }
            ++j;
        }
        if (!closed) {
            break; // unclosed: drop, do not repair
        }
        const std::string_view body = text.substr(body_begin, close_at - body_begin);
        i = after_close;
        if (body.empty() || whitespace_only(body)) {
            continue;
        }
        if (body.size() > kMaxThinkBlockChars) {
            ++out.dropped_too_large;
            continue;
        }
        if (out.blocks.size() >= kMaxThinkBlocksPerTurn) {
            ++out.dropped_overflow;
            continue;
        }
        ThinkBlock b;
        b.block_id = static_cast<int>(out.blocks.size());
        b.language.assign(lang.data(), lang.size());
        b.content.assign(body.data(), body.size());
        b.sha256 = platform::content_sha256_hex(b.content);
        out.blocks.push_back(std::move(b));
    }
    return out;
}

} // namespace lmp::tools
