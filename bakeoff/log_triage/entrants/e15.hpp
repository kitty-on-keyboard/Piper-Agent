#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <cmath>
#include <cctype>
#include <algorithm>
#include <sstream>

namespace log_triage {

inline double calculate_entropy(std::string_view s) {
    if (s.empty()) return 0.0;
    int counts[256] = {0};
    for (char c : s) counts[static_cast<unsigned char>(c)]++;
    double ent = 0.0;
    for (int i = 0; i < 256; ++i) {
        if (counts[i] > 0) {
            double p = static_cast<double>(counts[i]) / s.size();
            ent -= p * std::log2(p);
        }
    }
    return ent;
}

inline bool case_insensitive_contains(std::string_view text, std::string_view pattern) {
    auto it = std::search(
        text.begin(), text.end(),
        pattern.begin(), pattern.end(),
        [](unsigned char ch1, unsigned char ch2) {
            return std::tolower(ch1) == std::tolower(ch2);
        }
    );
    return it != text.end();
}

inline bool has_error_indicators(std::string_view s) {
    return case_insensitive_contains(s, "error") ||
           case_insensitive_contains(s, "fail") ||
           case_insensitive_contains(s, "fatal") ||
           case_insensitive_contains(s, "exception") ||
           case_insensitive_contains(s, "panic");
}

inline bool has_warning_indicators(std::string_view s) {
    return case_insensitive_contains(s, "warn");
}

inline bool is_progress_or_test_dots(std::string_view s) {
    if (s.size() < 5) return false;

    int max_char_count = 0;
    int punct_count = 0;
    int counts[256] = {0};

    for (char c : s) {
        counts[static_cast<unsigned char>(c)]++;
        if (std::ispunct(static_cast<unsigned char>(c))) punct_count++;
    }

    for (char c : {'.', '=', '-', '#', '*', '_'}) {
        max_char_count = std::max(max_char_count, counts[static_cast<unsigned char>(c)]);
    }

    double entropy = calculate_entropy(s);

    if (static_cast<double>(max_char_count) / s.size() > 0.4) {
        return true;
    }

    if (entropy < 3.0 && static_cast<double>(punct_count) / s.size() > 0.4) {
        return true;
    }

    return false;
}

inline std::string compact(std::string_view log) {
    if (log.empty()) return "";

    std::vector<std::string_view> lines;
    std::string_view::size_type start = 0;
    while (start < log.size()) {
        auto end = log.find('\n', start);
        if (end == std::string_view::npos) {
            lines.push_back(log.substr(start));
            break;
        }
        lines.push_back(log.substr(start, end - start));
        start = end + 1;
    }

    std::vector<bool> keep(lines.size(), true);
    std::vector<bool> is_err(lines.size(), false);
    std::vector<bool> is_warn(lines.size(), false);

    for (size_t i = 0; i < lines.size(); ++i) {
        is_err[i] = has_error_indicators(lines[i]);
        is_warn[i] = has_warning_indicators(lines[i]);
        if (is_progress_or_test_dots(lines[i])) {
            keep[i] = false;
        }
    }

    const int context_lines = 2; // Lines to keep before and after an error

    for (size_t i = 0; i < lines.size(); ++i) {
        if (is_err[i]) {
            size_t start_ctx = (i >= context_lines) ? (i - context_lines) : 0;
            size_t end_ctx = std::min(lines.size() - 1, i + context_lines);
            for (size_t j = start_ctx; j <= end_ctx; ++j) {
                keep[j] = true;
            }
        }
    }

    // Filter warning floods. Deduplicate identical warnings or limit their count.
    std::string_view last_warn = "";
    int warn_count = 0;
    const int max_warn_duplicates = 3;

    for (size_t i = 0; i < lines.size(); ++i) {
        if (keep[i] && is_warn[i] && !is_err[i]) {
            if (lines[i] == last_warn) {
                warn_count++;
                if (warn_count > max_warn_duplicates) {
                    keep[i] = false;
                }
            } else {
                last_warn = lines[i];
                warn_count = 1;
            }
        } else if (keep[i] && !is_warn[i]) {
             last_warn = "";
             warn_count = 0;
        }
    }

    std::ostringstream oss;
    bool first = true;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (keep[i]) {
            if (!first) oss << '\n';
            oss << lines[i];
            first = false;
        }
    }

    return oss.str();
}

} // namespace log_triage
