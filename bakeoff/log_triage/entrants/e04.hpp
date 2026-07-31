#ifndef LOG_TRIAGE_HPP
#define LOG_TRIAGE_HPP

#include <string>
#include <string_view>
#include <vector>
#include <regex>
#include <map>
#include <algorithm>
#include <numeric>

namespace log_triage {

// Scoring matrix category
enum class Category {
    Error,
    Warning,
    Success,
    TestFailure,
    SourceSnippet,
    TemplateInstantiation,
    Unknown
};

// Cascading penalty scoring configuration
struct PatternRule {
    std::regex pattern;
    Category category;
    int base_score;
    int penalty_step; // Amount to subtract per repeated match
    int min_score;    // Minimum score after penalties
};

// Note: To be thread-safe and avoid global mutable state without sync,
// we define the pattern matrix within a function or construct it locally,
// or as static const (but std::regex is thread-safe for reading after init).
// C++11 guarantees thread-safe initialization of static local variables.

inline const std::vector<PatternRule>& get_pattern_matrix() {
    static const std::vector<PatternRule> matrix = {
        // High priority
        {std::regex(R"(FAIL|FAILED|Assertion failed|Test failed)", std::regex_constants::icase), Category::TestFailure, 100, 0, 100},
        {std::regex(R"(error:|fatal error:|Exception:)", std::regex_constants::icase), Category::Error, 80, 5, 20},

        // Medium priority
        {std::regex(R"(warning:)", std::regex_constants::icase), Category::Warning, 50, 10, 0},
        {std::regex(R"(PASSED|SUCCESS|OK)", std::regex_constants::icase), Category::Success, 30, 2, 10},

        // Specific repetitive patterns (e.g., templates)
        {std::regex(R"(required from|instantiation of)", std::regex_constants::icase), Category::TemplateInstantiation, 40, 15, -10},
        {std::regex(R"(^\s*\d*\s*\||^\s*\^\s*~*)"), Category::SourceSnippet, 40, 5, 10} // E.g. clang/gcc snippet pointing to error
    };
    return matrix;
}

inline std::vector<std::string_view> split_lines(std::string_view text) {
    std::vector<std::string_view> lines;
    size_t start = 0;
    while (start < text.length()) {
        size_t end = text.find('\n', start);
        if (end == std::string_view::npos) {
            lines.push_back(text.substr(start));
            break;
        }
        lines.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    return lines;
}

inline std::string compact(const std::string& log_content, size_t budget_bytes) {
    if (log_content.length() <= budget_bytes) {
        return log_content;
    }

    std::vector<std::string_view> lines = split_lines(log_content);
    if (lines.empty()) return "";

    const size_t num_lines = lines.size();

    // Head and tail rules: always try to keep some head and tail if possible.
    // Let's reserve ~20% of budget for head, ~20% for tail, and 60% for scored middle.
    // Or if lines are long, just keep a fixed number.
    // For simplicity, let's keep first 10 and last 10 lines as "essential",
    // unless they exceed the budget themselves.

    size_t head_count = std::min<size_t>(10, num_lines);
    size_t tail_count = std::min<size_t>(10, num_lines > head_count ? num_lines - head_count : 0);

    struct ScoredLine {
        size_t index;
        std::string_view text;
        int score;
        bool is_head_or_tail;
    };

    std::vector<ScoredLine> scored_lines;
    scored_lines.reserve(num_lines);

    std::map<Category, int> match_counts;
    const auto& matrix = get_pattern_matrix();

    for (size_t i = 0; i < num_lines; ++i) {
        std::string_view line = lines[i];
        bool is_head = (i < head_count);
        bool is_tail = (i >= num_lines - tail_count);
        bool head_tail = is_head || is_tail;

        int line_score = 0;

        if (!head_tail) {
            // Find highest matching score for this line
            int best_match_score = -1000;
            bool matched = false;

            for (const auto& rule : matrix) {
                if (std::regex_search(line.begin(), line.end(), rule.pattern)) {
                    matched = true;
                    int count = match_counts[rule.category]++;
                    int current_score = rule.base_score - (count * rule.penalty_step);
                    current_score = std::max(current_score, rule.min_score);
                    best_match_score = std::max(best_match_score, current_score);
                }
            }

            if (matched) {
                line_score = best_match_score;
            } else {
                line_score = 0; // Default score for unmatched lines
            }
        } else {
            line_score = 10000; // Arbitrary high score for head/tail to ensure they are picked
        }

        scored_lines.push_back({i, line, line_score, head_tail});
    }

    // Sort by score descending
    std::vector<ScoredLine*> sorted_lines;
    for (auto& sl : scored_lines) {
        sorted_lines.push_back(&sl);
    }

    std::stable_sort(sorted_lines.begin(), sorted_lines.end(), [](const ScoredLine* a, const ScoredLine* b) {
        if (a->score != b->score) return a->score > b->score;
        return a->index < b->index; // prefer earlier lines if tie
    });

    std::vector<size_t> selected_indices;
    size_t current_bytes = 0;

    const std::string snip_msg = "\n... [snip] ...\n";
    size_t snip_cost_estimate = snip_msg.length() * 2; // Rough estimate of cost of snips

    for (const auto* sl : sorted_lines) {
        size_t cost = sl->text.length() + 1; // +1 for newline
        if (current_bytes + cost + snip_cost_estimate <= budget_bytes) {
            selected_indices.push_back(sl->index);
            current_bytes += cost;
        }
    }

    // Sort indices back to original order
    std::sort(selected_indices.begin(), selected_indices.end());

    // Reconstruct output
    std::string result;
    size_t last_index = static_cast<size_t>(-1);

    for (size_t idx : selected_indices) {
        if (last_index != static_cast<size_t>(-1) && idx > last_index + 1) {
            result += "... [snip] ...\n";
        }
        result += lines[idx];
        result += '\n';
        last_index = idx;
    }

    // Remove trailing newline if original didn't have one? Wait, split_lines omits trailing newlines but our loop adds them.
    // If the reconstructed string is slightly larger than budget because of snip markers, let's strictly trim.
    if (!result.empty() && log_content.back() != '\n' && result.back() == '\n') {
        result.pop_back();
    }

    // Hard cutoff if we somehow exceeded (though logic above should prevent it)
    if (result.length() > budget_bytes) {
        result.resize(budget_bytes);
    }

    return result;
}

} // namespace log_triage

#endif // LOG_TRIAGE_HPP