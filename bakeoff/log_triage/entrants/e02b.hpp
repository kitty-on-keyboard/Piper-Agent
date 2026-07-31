#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <cctype>
#include <numeric>

namespace log_triage {

inline std::string strip_ansi(std::string_view s) {
    std::string result;
    result.reserve(s.size());
    for (size_t i = 0; i < s.size(); ) {
        if (s[i] == '\x1b' && i + 1 < s.size() && s[i+1] == '[') {
            size_t j = i + 2;
            while (j < s.size() && !(s[j] >= 'a' && s[j] <= 'z') && !(s[j] >= 'A' && s[j] <= 'Z')) {
                j++;
            }
            if (j < s.size()) j++;
            i = j;
        } else {
            result += s[i];
            i++;
        }
    }
    return result;
}

struct SemanticSpan {
    size_t start_idx;
    size_t end_idx;
    int importance_score;
};

inline int evaluate_line_importance(std::string_view line) {
    auto to_lower_ascii = [](unsigned char c) -> unsigned char {
        if (c >= 'A' && c <= 'Z') return c + ('a' - 'A');
        return c;
    };

    auto contains_ignore_case = [&to_lower_ascii](std::string_view haystack, std::string_view needle) {
        auto it = std::search(
            haystack.begin(), haystack.end(),
            needle.begin(), needle.end(),
            [&to_lower_ascii](char ch1, char ch2) {
                return to_lower_ascii(static_cast<unsigned char>(ch1)) ==
                       to_lower_ascii(static_cast<unsigned char>(ch2));
            }
        );
        return it != haystack.end();
    };

    if (contains_ignore_case(line, "error") ||
        contains_ignore_case(line, "fatal") ||
        contains_ignore_case(line, "panic") ||
        contains_ignore_case(line, "segfault")) {
        return 1000;
    } else if (contains_ignore_case(line, "exception") ||
               contains_ignore_case(line, "traceback") ||
               contains_ignore_case(line, "assert")) {
        return 800;
    } else if (contains_ignore_case(line, "warning")) {
        return 500;
    }
    return 10;
}

inline bool is_context_or_continuation(std::string_view line) {
    if (line.empty()) return false;
    char first = line.front();
    if (std::isspace(static_cast<unsigned char>(first))) return true;
    if (first == '|') return true;

    bool only_carets_and_spaces = true;
    bool has_caret = false;
    for (char c : line) {
        if (c == '^' || c == '~') {
            has_caret = true;
        } else if (!std::isspace(static_cast<unsigned char>(c)) && c != '|') {
            only_carets_and_spaces = false;
            break;
        }
    }
    if (has_caret && only_carets_and_spaces) return true;

    auto to_lower_ascii = [](unsigned char c) -> unsigned char {
        if (c >= 'A' && c <= 'Z') return c + ('a' - 'A');
        return c;
    };

    auto starts_with_ignore_case = [&to_lower_ascii](std::string_view haystack, std::string_view needle) {
        if (haystack.size() < needle.size()) return false;
        for (size_t i = 0; i < needle.size(); ++i) {
            if (to_lower_ascii(static_cast<unsigned char>(haystack[i])) !=
                to_lower_ascii(static_cast<unsigned char>(needle[i]))) {
                return false;
            }
        }
        return true;
    };

    if (starts_with_ignore_case(line, "at ") ||
        starts_with_ignore_case(line, "caused by") ||
        starts_with_ignore_case(line, "->")) return true;

    size_t pos = 0;
    while (pos < line.size() && std::isdigit(static_cast<unsigned char>(line[pos]))) {
        pos++;
    }
    if (pos > 0 && pos < line.size()) {
        while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) pos++;
        if (pos < line.size() && line[pos] == '|') return true;
    }

    return false;
}

inline std::string compact(std::string_view log, size_t max_lines) {
    if (log.empty()) return "";

    std::vector<std::string_view> lines;
    size_t start = 0;
    for (size_t i = 0; i <= log.size(); ++i) {
        if (i == log.size() || log[i] == '\n') {
            lines.push_back(log.substr(start, i - start));
            start = i + 1;
        }
    }

    if (lines.size() <= max_lines) {
        std::string result;
        bool first = true;
        for (const auto& line : lines) {
            if (!first) result += '\n';
            result += strip_ansi(line);
            first = false;
        }
        return result;
    }

    size_t head_budget = (max_lines * 15) / 100;
    size_t tail_budget = (max_lines * 15) / 100;
    if (max_lines > 0 && head_budget == 0) head_budget = 1;
    if (max_lines > 1 && tail_budget == 0) tail_budget = 1;

    if (head_budget + tail_budget > max_lines) {
        head_budget = max_lines / 2;
        tail_budget = max_lines - head_budget;
    }
    size_t mid_budget = max_lines - head_budget - tail_budget;

    std::vector<bool> keep(lines.size(), false);
    for (size_t i = 0; i < head_budget; ++i) keep[i] = true;
    for (size_t i = lines.size() - tail_budget; i < lines.size(); ++i) keep[i] = true;

    std::vector<SemanticSpan> spans;
    SemanticSpan current_span{head_budget, head_budget, 0};

    for (size_t i = head_budget; i < lines.size() - tail_budget; ++i) {
        std::string stripped = strip_ansi(lines[i]);
        bool is_continuation = is_context_or_continuation(stripped);
        int score = evaluate_line_importance(stripped);

        if (!is_continuation && current_span.end_idx > current_span.start_idx) {
            spans.push_back(current_span);
            current_span = SemanticSpan{i, i, 0};
        }

        if (current_span.start_idx == current_span.end_idx) {
            current_span.start_idx = i;
        }
        current_span.end_idx = i + 1;
        current_span.importance_score = std::max(current_span.importance_score, score);
    }
    if (current_span.end_idx > current_span.start_idx) {
        spans.push_back(current_span);
    }

    // Importance Graph evaluation
    std::sort(spans.begin(), spans.end(), [](const SemanticSpan& a, const SemanticSpan& b) {
        if (a.importance_score != b.importance_score) return a.importance_score > b.importance_score;
        size_t a_size = a.end_idx - a.start_idx;
        size_t b_size = b.end_idx - b.start_idx;
        if (a_size != b_size) return a_size > b_size; // Larger context is prioritized
        return a.start_idx < b.start_idx; // Stable sort
    });

    size_t mid_lines_kept = 0;
    for (const auto& span : spans) {
        if (mid_lines_kept >= mid_budget) break;

        for (size_t i = span.start_idx; i < span.end_idx; ++i) {
            if (mid_lines_kept < mid_budget) {
                keep[i] = true;
                mid_lines_kept++;
            }
        }
    }

    std::string result;
    bool first = true;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (keep[i]) {
            if (!first) result += '\n';
            result += strip_ansi(lines[i]);
            first = false;
        }
    }

    return result;
}

} // namespace log_triage
