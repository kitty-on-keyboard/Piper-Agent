#ifndef LOG_TRIAGE_HPP
#define LOG_TRIAGE_HPP

#include <string>
#include <string_view>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <cctype>

namespace log_triage {

inline size_t count_tokens(std::string_view s) {
    size_t tokens = 0;
    bool in_token = false;
    for (char c : s) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            in_token = false;
        } else {
            if (!in_token) {
                tokens++;
                in_token = true;
            }
        }
    }
    return tokens;
}

inline bool contains_icase(std::string_view haystack, std::string_view needle) {
    auto it = std::search(
        haystack.begin(), haystack.end(),
        needle.begin(), needle.end(),
        [](char ch1, char ch2) {
            return std::tolower(static_cast<unsigned char>(ch1)) == std::tolower(static_cast<unsigned char>(ch2));
        }
    );
    return it != haystack.end();
}

inline std::string compact(std::string_view raw_log, size_t token_budget) {
    std::vector<std::string_view> lines;
    size_t start = 0;
    for (size_t i = 0; i <= raw_log.size(); ++i) {
        if (i == raw_log.size() || raw_log[i] == '\n') {
            if (i == raw_log.size() && start == i && !raw_log.empty() && raw_log.back() == '\n') {
                break;
            }
            lines.push_back(raw_log.substr(start, i - start));
            start = i + 1;
        }
    }
    if (lines.empty()) return "";

    size_t total_tokens = 0;
    std::vector<size_t> line_tokens(lines.size());
    for (size_t i = 0; i < lines.size(); ++i) {
        line_tokens[i] = count_tokens(lines[i]);
        if (line_tokens[i] == 0) line_tokens[i] = 1; // Base cost for empty lines
        total_tokens += line_tokens[i];
    }

    if (total_tokens <= token_budget) {
        return std::string(raw_log);
    }

    std::vector<bool> is_error(lines.size(), false);
    std::vector<bool> is_warning(lines.size(), false);

    int first_error_idx = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (contains_icase(lines[i], "error")) {
            is_error[i] = true;
            if (first_error_idx == -1) first_error_idx = static_cast<int>(i);
        } else if (contains_icase(lines[i], "warning")) {
            is_warning[i] = true;
        }
    }

    std::vector<bool> assigned(lines.size(), false);
    auto assign = [&](size_t idx, std::vector<size_t>& cat) {
        if (!assigned[idx]) {
            assigned[idx] = true;
            cat.push_back(idx);
        }
    };

    std::vector<size_t> t1_head;
    std::vector<size_t> t2_err_ctx;
    std::vector<size_t> t3_dist_err;
    std::vector<size_t> t4_tail;
    std::vector<size_t> t5_warn;

    size_t head_size = std::min<size_t>(lines.size(), 10);
    size_t tail_size = std::min<size_t>(lines.size(), 10);

    // 1. Command metadata
    for (size_t i = 0; i < head_size; ++i) {
        assign(i, t1_head);
    }

    // 2. First error and context
    if (first_error_idx != -1) {
        int start_ctx = std::max(0, first_error_idx - 5);
        int end_ctx = std::min(static_cast<int>(lines.size()) - 1, first_error_idx + 5);
        for (int i = start_ctx; i <= end_ctx; ++i) {
            assign(i, t2_err_ctx);
        }
    }

    // 3. Distinct errors
    std::unordered_set<std::string_view> seen_errors;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (is_error[i] && assigned[i]) {
            seen_errors.insert(lines[i]);
        }
    }
    for (size_t i = 0; i < lines.size(); ++i) {
        if (is_error[i] && !assigned[i]) {
            if (seen_errors.insert(lines[i]).second) {
                assign(i, t3_dist_err);
            }
        }
    }

    // 4. Outcome summary
    for (size_t i = lines.size() >= tail_size ? lines.size() - tail_size : 0; i < lines.size(); ++i) {
        assign(i, t4_tail);
    }

    // 5. Warning fillers
    for (size_t i = 0; i < lines.size(); ++i) {
        if (is_warning[i] && !assigned[i]) {
            assign(i, t5_warn);
        }
    }

    std::vector<bool> selected(lines.size(), false);
    size_t remaining_budget = token_budget;

    auto process_tier = [&](const std::vector<size_t>& candidates) {
        if (candidates.empty() || remaining_budget == 0) return;

        size_t cost_all = 0;
        for (size_t idx : candidates) cost_all += line_tokens[idx];

        if (cost_all <= remaining_budget) {
            for (size_t idx : candidates) selected[idx] = true;
            remaining_budget -= cost_all;
        } else {
            size_t M = candidates.size();
            for (size_t K = M - 1; K > 0; --K) {
                size_t cost_k = 0;
                std::vector<size_t> subset;
                if (K == 1) {
                    if (candidates.back() == lines.size() - 1) {
                        subset.push_back(M - 1);
                    } else {
                        subset.push_back(0);
                    }
                } else {
                    for (size_t j = 0; j < K; ++j) {
                        size_t idx = j * (M - 1) / (K - 1);
                        subset.push_back(idx);
                    }
                }
                for (size_t sub_idx : subset) {
                    cost_k += line_tokens[candidates[sub_idx]];
                }
                if (cost_k <= remaining_budget) {
                    for (size_t sub_idx : subset) {
                        selected[candidates[sub_idx]] = true;
                    }
                    remaining_budget -= cost_k;
                    break;
                }
            }
        }
    };

    process_tier(t1_head);
    process_tier(t2_err_ctx);
    process_tier(t3_dist_err);
    process_tier(t4_tail);
    process_tier(t5_warn);

    std::string result;
    bool skipped = false;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (selected[i]) {
            if (skipped) {
                result += "...\n";
                skipped = false;
            }
            result += lines[i];
            if (i < lines.size() - 1 || raw_log.back() == '\n') {
                result += "\n";
            }
        } else {
            skipped = true;
        }
    }

    if (skipped && !result.empty()) {
        if (result.back() == '\n') result.pop_back();
        result += "\n...";
        if (raw_log.back() == '\n') result += "\n";
    }

    // Safety check to ensure we don't exceed budget because of added markers
    while (count_tokens(result) > token_budget) {
        // Fallback: If we exceed budget with markers, strip markers
        std::string stripped_result;
        for (size_t i = 0; i < lines.size(); ++i) {
            if (selected[i]) {
                stripped_result += lines[i];
                if (i < lines.size() - 1 || raw_log.back() == '\n') {
                    stripped_result += "\n";
                }
            }
        }

        if (count_tokens(stripped_result) <= token_budget) {
            return stripped_result;
        }

        // If still over budget (e.g. edge cases), drop the last selected line
        bool dropped = false;
        for (size_t i = lines.size(); i > 0; --i) {
            if (selected[i - 1]) {
                selected[i - 1] = false;
                dropped = true;
                break;
            }
        }
        if (!dropped) return ""; // Nothing left to drop

        // Rebuild result
        result = "";
        skipped = false;
        for (size_t i = 0; i < lines.size(); ++i) {
            if (selected[i]) {
                if (skipped) {
                    result += "...\n";
                    skipped = false;
                }
                result += lines[i];
                if (i < lines.size() - 1 || raw_log.back() == '\n') {
                    result += "\n";
                }
            } else {
                skipped = true;
            }
        }
        if (skipped && !result.empty()) {
            if (result.back() == '\n') result.pop_back();
            result += "\n...";
            if (raw_log.back() == '\n') result += "\n";
        }
    }

    return result;
}

} // namespace log_triage

#endif
