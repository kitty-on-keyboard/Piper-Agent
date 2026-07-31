#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <numeric>

namespace log_triage {

struct TriageConfig {
    size_t n_lines_before;
    size_t m_lines_after;
    std::string_view anchor;
    size_t budget_bytes;
};

inline std::vector<std::string_view> split_lines(std::string_view log) {
    std::vector<std::string_view> lines;
    size_t start = 0;
    while (start < log.size()) {
        size_t end = log.find('\n', start);
        if (end == std::string_view::npos) {
            lines.push_back(log.substr(start));
            break;
        }
        lines.push_back(log.substr(start, end - start + 1));
        start = end + 1;
    }
    return lines;
}

inline std::string compact(std::string_view log, const TriageConfig& config) {
    if (log.empty() || config.budget_bytes == 0) return "";

    auto lines = split_lines(log);

    struct Window {
        size_t start;
        size_t end; // inclusive
    };
    std::vector<Window> windows;

    for (size_t i = 0; i < lines.size(); ++i) {
        if (lines[i].find(config.anchor) != std::string_view::npos) {
            size_t start = (i > config.n_lines_before) ? i - config.n_lines_before : 0;
            size_t end = std::min(i + config.m_lines_after, lines.size() - 1);
            if (!windows.empty() && windows.back().end + 1 >= start) {
                windows.back().end = std::max(windows.back().end, end);
            } else {
                windows.push_back({start, end});
            }
        }
    }

    if (windows.empty()) {
        if (log.size() <= config.budget_bytes) {
            return std::string(log);
        } else {
            return std::string(log.substr(log.size() - config.budget_bytes));
        }
    }

    auto get_window_bytes = [&](const Window& w) {
        size_t bytes = 0;
        for (size_t i = w.start; i <= w.end; ++i) {
            bytes += lines[i].size();
        }
        return bytes;
    };

    size_t total_bytes = 0;
    for (const auto& w : windows) {
        total_bytes += get_window_bytes(w);
    }

    if (total_bytes <= config.budget_bytes) {
        std::string result;
        for (const auto& w : windows) {
            for (size_t i = w.start; i <= w.end; ++i) {
                result += lines[i];
            }
        }
        return result;
    }

    std::string result;
    if (windows.size() == 1) {
        for (size_t i = windows[0].start; i <= windows[0].end; ++i) {
            if (result.size() + lines[i].size() <= config.budget_bytes) {
                result += lines[i];
            } else {
                size_t rem = config.budget_bytes - result.size();
                result += lines[i].substr(0, rem);
                break;
            }
        }
        return result;
    }

    Window first = windows.front();
    Window last = windows.back();

    size_t first_bytes = get_window_bytes(first);
    size_t last_bytes = get_window_bytes(last);

    if (first_bytes + last_bytes <= config.budget_bytes) {
        for (size_t i = first.start; i <= first.end; ++i) {
            result += lines[i];
        }
        size_t remaining_budget = config.budget_bytes - first_bytes - last_bytes;
        std::string middle_result;
        for (size_t w_idx = 1; w_idx < windows.size() - 1; ++w_idx) {
            const auto& w = windows[w_idx];
            for (size_t i = w.start; i <= w.end; ++i) {
                if (middle_result.size() + lines[i].size() <= remaining_budget) {
                    middle_result += lines[i];
                } else {
                    size_t rem = remaining_budget - middle_result.size();
                    middle_result += lines[i].substr(0, rem);
                    remaining_budget = 0;
                    break;
                }
            }
            if (remaining_budget == 0) break;
        }
        result += middle_result;
        for (size_t i = last.start; i <= last.end; ++i) {
            result += lines[i];
        }
        return result;
    } else {
        if (first_bytes >= config.budget_bytes) {
            for (size_t i = first.start; i <= first.end; ++i) {
                if (result.size() + lines[i].size() <= config.budget_bytes) {
                    result += lines[i];
                } else {
                    size_t rem = config.budget_bytes - result.size();
                    result += lines[i].substr(0, rem);
                    break;
                }
            }
            return result;
        } else {
            for (size_t i = first.start; i <= first.end; ++i) {
                result += lines[i];
            }
            size_t remaining_budget = config.budget_bytes - first_bytes;
            std::string last_str;
            for (size_t i = last.start; i <= last.end; ++i) {
                if (last_str.size() + lines[i].size() <= remaining_budget) {
                    last_str += lines[i];
                } else {
                    size_t rem = remaining_budget - last_str.size();
                    last_str += lines[i].substr(0, rem);
                    break;
                }
            }
            result += last_str;
            return result;
        }
    }
}

} // namespace log_triage
