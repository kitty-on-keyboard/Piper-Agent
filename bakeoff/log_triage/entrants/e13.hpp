#pragma once

#include <string_view>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cctype>
#include <cmath>
#include <iostream>

namespace log_triage {

struct Line {
    std::string text;
    size_t index;
    int score;
};

// Pass 1: split input into logical chunks/lines, handle carriage returns and ANSI color escapes
inline std::vector<Line> pass1_split_and_clean(std::string_view input) {
    std::vector<Line> lines;
    size_t start = 0;
    size_t index = 0;

    auto strip_ansi_and_cr = [](std::string_view sv) -> std::string {
        std::string result;
        result.reserve(sv.size());
        for (size_t i = 0; i < sv.size(); ++i) {
            if (sv[i] == '\x1b') {
                // simple ANSI escape code stripping
                if (i + 1 < sv.size() && sv[i+1] == '[') {
                    size_t j = i + 2;
                    while (j < sv.size() && !std::isalpha(static_cast<unsigned char>(sv[j]))) {
                        ++j;
                    }
                    if (j < sv.size()) {
                        i = j; // skip up to the letter
                        continue;
                    }
                }
            } else if (sv[i] == '\r') {
                // handle carriage return: reset the current line output
                // Only reset if \r is NOT the last character in the string view
                // (otherwise it's just a Windows CRLF line ending)
                if (i + 1 < sv.size()) {
                    result.clear();
                }
            } else {
                result += sv[i];
            }
        }
        return result;
    };

    while (start < input.size()) {
        size_t end = input.find('\n', start);
        if (end == std::string_view::npos) {
            end = input.size();
        }
        std::string_view line_sv = input.substr(start, end - start);
        std::string text = strip_ansi_and_cr(line_sv);
        lines.push_back({std::move(text), index++, 0});
        start = end + 1;
    }
    return lines;
}

// Pass 2: Assign priority scores based on signatures and proximity
inline void pass2_score_lines(std::vector<Line>& lines) {
    if (lines.empty()) return;
    int n = static_cast<int>(lines.size());

    // Baseline scores: Head and Outcome lines get a slight boost
    for (int i = 0; i < n; ++i) {
        int score = 10;

        // Head context (first ~10 lines)
        if (i < 10) {
            score = std::max(score, 60 - i * 2);
        }
        // Outcome lines (Tail context, last ~10 lines)
        if (n - 1 - i < 10) {
            score = std::max(score, 60 - (n - 1 - i) * 2);
        }

        lines[i].score = score;
    }

    // Primary diagnostics and signatures
    for (int i = 0; i < n; ++i) {
        const std::string& text = lines[i].text;

        auto contains = [&](std::string_view sub) {
            return text.find(sub) != std::string::npos;
        };

        bool is_error = contains("error:") || contains("fatal error:") ||
                        contains("FAILED:") || contains("Exception:") ||
                        contains("Traceback (");
        bool is_warning = contains("warning:");

        bool is_caret = false;
        if (!text.empty() && contains("^")) {
            is_caret = true;
            for (char c : text) {
                if (c != ' ' && c != '\t' && c != '^' && c != '~' && c != '|') {
                    is_caret = false;
                    break;
                }
            }
        }

        if (is_error) {
            lines[i].score = std::max(lines[i].score, 100);
        } else if (is_caret) {
            lines[i].score = std::max(lines[i].score, 95);
        } else if (is_warning) {
            lines[i].score = std::max(lines[i].score, 80);
        }
    }

    // Proximity boost (context around errors)
    std::vector<int> boosted_scores(n);
    for (int i = 0; i < n; ++i) {
        boosted_scores[i] = lines[i].score;
    }

    for (int i = 0; i < n; ++i) {
        if (lines[i].score == 100) {
            // Boost preceding lines
            if (i >= 1) boosted_scores[i-1] = std::max(boosted_scores[i-1], 90);
            if (i >= 2) boosted_scores[i-2] = std::max(boosted_scores[i-2], 85);
            if (i >= 3) boosted_scores[i-3] = std::max(boosted_scores[i-3], 75);

            // Boost succeeding lines
            if (i + 1 < n) boosted_scores[i+1] = std::max(boosted_scores[i+1], 90);
            if (i + 2 < n) boosted_scores[i+2] = std::max(boosted_scores[i+2], 85);
            if (i + 3 < n) boosted_scores[i+3] = std::max(boosted_scores[i+3], 75);
        }
    }

    for (int i = 0; i < n; ++i) {
        lines[i].score = boosted_scores[i];
    }
}

// Pass 3: Sort/filter lines by priority to fit budget, preserve order, insert elision markers
inline std::string pass3_pack_lines(std::vector<Line>& lines, size_t budget_bytes) {
    if (lines.empty()) return "";

    // Sort lines by priority (descending), then by index (ascending) to stabilize
    std::vector<Line> sorted_lines = lines;
    std::sort(sorted_lines.begin(), sorted_lines.end(), [](const Line& a, const Line& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.index < b.index;
    });

    std::vector<Line> selected_lines;

    // Elision marker constant
    const std::string elision_marker = "\n... (elided) ...\n";

    // Create a boolean array to track selected lines
    std::vector<bool> is_selected(lines.size(), false);

    // Greedily pick lines
    for (const auto& l : sorted_lines) {
        // Find position of the line in the original order
        size_t orig_idx = l.index;
        is_selected[orig_idx] = true;

        // Accurate assembly size calculation without string building
        size_t test_size = 0;
        size_t last_index = SIZE_MAX;
        bool first = true;

        for (size_t i = 0; i < is_selected.size(); ++i) {
            if (is_selected[i]) {
                if (!first) {
                    if (i > last_index + 1) {
                        test_size += elision_marker.size();
                    } else {
                        test_size += 1; // '\n'
                    }
                }
                test_size += lines[i].text.size();
                last_index = i;
                first = false;
            }
        }

        if (test_size > budget_bytes) {
            // Revert insertion
            is_selected[orig_idx] = false;
            break; // Stop adding more lines if we hit the budget boundary
        }
    }

    // Final collection
    for (size_t i = 0; i < is_selected.size(); ++i) {
        if (is_selected[i]) {
            selected_lines.push_back(lines[i]);
        }
    }

    // Final assembly
    std::string result;
    size_t last_index = SIZE_MAX;
    for (size_t i = 0; i < selected_lines.size(); ++i) {
        const auto& sl = selected_lines[i];
        if (i > 0) {
            if (sl.index > last_index + 1) {
                result += elision_marker;
            } else {
                result += '\n';
            }
        }
        result += sl.text;
        last_index = sl.index;
    }

    return result;
}

inline std::string compact(std::string_view input, size_t budget_bytes) {
    auto lines = pass1_split_and_clean(input);
    pass2_score_lines(lines);
    return pass3_pack_lines(lines, budget_bytes);
}

} // namespace log_triage
