#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>

namespace log_triage {

struct Line {
    size_t original_index;
    std::string text;
    double score;
};

inline std::string to_lower(const std::string& s) {
    std::string res = s;
    for (char& c : res) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return res;
}

inline std::string compact(const std::string& input, size_t budget_bytes) {
    if (budget_bytes == 0) return "";

    // 1. Normalization (ANSI strip, progress bar neutralization)
    std::string normalized;
    int ansi_state = 0;
    std::string current_line;

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        if (ansi_state == 0) {
            if (c == '\x1b') {
                ansi_state = 1;
            } else if (c == '\r') {
                // Peek next char to see if it's \n
                if (i + 1 < input.size() && input[i + 1] == '\n') {
                    // It's \r\n, do nothing, the \n will handle the new line
                } else {
                    // It's a standalone \r, neutralizing progress bar: clear line
                    current_line.clear();
                }
            } else if (c == '\n') {
                normalized += current_line + '\n';
                current_line.clear();
            } else {
                current_line.push_back(c);
            }
        } else if (ansi_state == 1) {
            if (c == '[') {
                ansi_state = 2;
            } else {
                ansi_state = 0;
                current_line.push_back(c); // push the unexpected char
            }
        } else if (ansi_state == 2) {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
                ansi_state = 0;
            }
        }
    }
    if (!current_line.empty()) {
        normalized += current_line + '\n';
    }

    // 2. Tokenization & Line Tagging
    std::vector<Line> lines;
    size_t start = 0;
    for (size_t i = 0; i < normalized.size(); ++i) {
        if (normalized[i] == '\n') {
            std::string text = normalized.substr(start, i - start);
            double score = 10.0; // Info default
            std::string lower_text = to_lower(text);
            if (lower_text.find("error") != std::string::npos ||
                lower_text.find("fatal") != std::string::npos ||
                lower_text.find("exception") != std::string::npos ||
                lower_text.find("panic") != std::string::npos ||
                lower_text.find("segfault") != std::string::npos) {
                score = 100.0;
            } else if (lower_text.find("warning") != std::string::npos ||
                       lower_text.find("warn") != std::string::npos) {
                score = 50.0;
            }

            lines.push_back({lines.size(), text, score});
            start = i + 1;
        }
    }

    if (lines.empty()) return "";

    // 3. Score Normalization
    std::vector<double> boosted_scores(lines.size(), 0.0);
    for (size_t i = 0; i < lines.size(); ++i) {
        boosted_scores[i] += lines[i].score;
        boosted_scores[i] += static_cast<double>(i) / static_cast<double>(lines.size() + 1); // break ties safely, prefer later lines slightly

        if (lines[i].score >= 100.0) {
            for (long long j = -5; j <= 5; ++j) {
                if (j == 0) continue;
                long long idx = static_cast<long long>(i) + j;
                if (idx >= 0 && idx < static_cast<long long>(lines.size())) {
                    boosted_scores[idx] += 20.0 / std::abs(j);
                }
            }
        } else if (lines[i].score >= 50.0) {
            for (long long j = -3; j <= 3; ++j) {
                if (j == 0) continue;
                long long idx = static_cast<long long>(i) + j;
                if (idx >= 0 && idx < static_cast<long long>(lines.size())) {
                    boosted_scores[idx] += 10.0 / std::abs(j);
                }
            }
        }
    }

    for (size_t i = 0; i < lines.size(); ++i) {
        lines[i].score = boosted_scores[i];
    }

    // 4. Greedy Knapsack Packing
    std::vector<size_t> sorted_indices(lines.size());
    for (size_t i = 0; i < lines.size(); ++i) sorted_indices[i] = i;
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&](size_t a, size_t b) {
        return lines[a].score > lines[b].score;
    });

    auto calculate_size = [&](const std::vector<bool>& sel) {
        size_t size = 0;
        bool in_gap = false;
        size_t gap_count = 0;
        for (size_t i = 0; i < lines.size(); ++i) {
            if (sel[i]) {
                if (in_gap) {
                    std::string elision = "... (" + std::to_string(gap_count) + " lines elided) ...\n";
                    size += elision.size();
                    in_gap = false;
                    gap_count = 0;
                }
                size += lines[i].text.size() + 1;
            } else {
                in_gap = true;
                gap_count++;
            }
        }
        if (in_gap && gap_count > 0) {
            std::string elision = "... (" + std::to_string(gap_count) + " lines elided) ...\n";
            size += elision.size();
        }
        return size;
    };

    std::vector<bool> selected(lines.size(), false);
    for (size_t idx : sorted_indices) {
        selected[idx] = true;
        if (calculate_size(selected) > budget_bytes) {
            selected[idx] = false;
        }
    }

    // 5. Output generation
    std::string output;
    bool in_gap = false;
    size_t gap_count = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (selected[i]) {
            if (in_gap) {
                output += "... (" + std::to_string(gap_count) + " lines elided) ...\n";
                in_gap = false;
                gap_count = 0;
            }
            output += lines[i].text + '\n';
        } else {
            in_gap = true;
            gap_count++;
        }
    }
    if (in_gap && gap_count > 0) {
        std::string elision = "... (" + std::to_string(gap_count) + " lines elided) ...\n";
        if (output.size() + elision.size() <= budget_bytes) {
            output += elision;
        }
    }

    return output;
}

} // namespace log_triage
