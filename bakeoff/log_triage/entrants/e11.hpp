#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <algorithm>

namespace log_triage {

inline std::string compact(std::string_view log_data, size_t max_bytes) {
    if (log_data.size() <= max_bytes) {
        return std::string(log_data);
    }

    struct Line {
        size_t index;
        std::string_view text;
    };

    std::vector<Line> lines;
    size_t start = 0;
    while (start < log_data.size()) {
        size_t end = log_data.find('\n', start);
        if (end == std::string_view::npos) {
            lines.push_back({lines.size(), log_data.substr(start)});
            break;
        } else {
            lines.push_back({lines.size(), log_data.substr(start, end - start + 1)});
            start = end + 1;
        }
    }

    auto is_reverse_match = [](std::string_view s) {
        return s.find("FAIL") != std::string_view::npos ||
               s.find("Caused by:") != std::string_view::npos ||
               s.find("Summary:") != std::string_view::npos;
    };

    auto is_forward_match = [](std::string_view s) {
        return s.find("error:") != std::string_view::npos ||
               s.find("fatal error:") != std::string_view::npos;
    };

    std::vector<size_t> included_bytes(lines.size(), 0);
    size_t current_bytes = 0;

    auto add_line = [&](int idx) -> bool {
        if (idx < 0 || idx >= (int)lines.size() || included_bytes[idx] > 0) return true;
        size_t len = lines[idx].text.size();
        if (current_bytes + len > max_bytes) {
            size_t remaining = max_bytes - current_bytes;
            if (remaining > 0) {
                included_bytes[idx] = remaining;
                current_bytes += remaining;
            }
            return false;
        }
        included_bytes[idx] = len;
        current_bytes += len;
        return true;
    };

    const int CONTEXT = 2;
    auto add_block = [&](int center_idx) -> bool {
        if (!add_line(center_idx)) return false;
        for (int d = 1; d <= CONTEXT; ++d) {
            if (!add_line(center_idx - d)) return false;
            if (!add_line(center_idx + d)) return false;
        }
        return true;
    };

    int rev = (int)lines.size() - 1;
    int fwd = 0;

    while (rev >= 0 || fwd < (int)lines.size()) {
        int found_rev = -1;
        while (rev >= 0) {
            if (is_reverse_match(lines[rev].text)) {
                found_rev = rev;
                rev--;
                break;
            }
            rev--;
        }

        int found_fwd = -1;
        while (fwd < (int)lines.size()) {
            if (is_forward_match(lines[fwd].text)) {
                found_fwd = fwd;
                fwd++;
                break;
            }
            fwd++;
        }

        if (found_rev != -1) {
            if (!add_block(found_rev)) break;
        }
        if (found_fwd != -1) {
            if (!add_block(found_fwd)) break;
        }

        if (found_rev == -1 && found_fwd == -1) break;
    }

    int top = 0;
    int bottom = (int)lines.size() - 1;
    while (top <= bottom && current_bytes < max_bytes) {
        if (!add_line(top++)) break;
        if (top <= bottom && !add_line(bottom--)) break;
    }

    std::string result;
    result.reserve(current_bytes);
    for (size_t i = 0; i < lines.size(); ++i) {
        if (included_bytes[i] > 0) {
            result.append(lines[i].text.data(), included_bytes[i]);
        }
    }

    return result;
}

} // namespace log_triage
