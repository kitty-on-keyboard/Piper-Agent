#ifndef LOG_TRIAGE_HPP
#define LOG_TRIAGE_HPP

#include <string>
#include <string_view>
#include <vector>
#include <set>
#include <algorithm>
#include <cctype>

namespace log_triage {

struct Candidate {
    std::size_t index;
    std::size_t offset;
    std::size_t length;
    std::size_t priority;
};

inline bool contains_error(std::string_view line) {
    auto search = [](std::string_view s, std::string_view p) {
        auto it = std::search(s.begin(), s.end(), p.begin(), p.end(),
            [](char c1, char c2) { return std::tolower(static_cast<unsigned char>(c1)) == std::tolower(static_cast<unsigned char>(c2)); });
        return it != s.end();
    };
    return search(line, "error");
}

inline std::string compact(std::string_view log_data, std::size_t budget_bytes) {
    if (log_data.empty() || budget_bytes == 0) {
        return "";
    }

    std::vector<Candidate> candidates;
    std::size_t start = 0;
    std::size_t index = 0;

    // Phase 1: extract candidates
    while (start < log_data.size()) {
        std::size_t end = log_data.find('\n', start);
        std::size_t len = (end == std::string_view::npos) ? log_data.size() - start : end - start + 1;
        candidates.push_back({index, start, len, 0});
        start += len;
        index++;
    }

    if (candidates.empty()) {
        return "";
    }

    bool first_error_found = false;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        auto& c = candidates[i];
        std::string_view line_str = log_data.substr(c.offset, c.length);
        if (!first_error_found && contains_error(line_str)) {
            c.priority = 0;
            first_error_found = true;
        } else if (i == candidates.size() - 1) {
            c.priority = 1;
        } else {
            c.priority = 1000000 + i;
        }
    }

    std::vector<Candidate> sorted_candidates = candidates;
    std::sort(sorted_candidates.begin(), sorted_candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.priority != b.priority) return a.priority < b.priority;
        return a.index < b.index;
    });

    std::set<std::size_t> included;
    std::size_t current_content_size = 0;

    auto calculate_elision_overhead = [&](const std::set<std::size_t>& incl) {
        if (incl.empty()) return (std::size_t)0;
        std::size_t overhead = 0;
        auto it = incl.begin();
        std::size_t prev = *it;
        ++it;
        for (; it != incl.end(); ++it) {
            if (*it > prev + 1) {
                overhead += 4; // "...\n" length
            }
            prev = *it;
        }
        return overhead;
    };

    // Phase 2: sort and filter
    for (const auto& c : sorted_candidates) {
        std::set<std::size_t> next_included = included;
        next_included.insert(c.index);

        std::size_t next_content_size = current_content_size + c.length;
        std::size_t next_overhead = calculate_elision_overhead(next_included);

        if (next_content_size + next_overhead <= budget_bytes) {
            included = next_included;
            current_content_size = next_content_size;
        }
    }

    if (included.empty()) {
        return "";
    }

    std::string result;
    result.reserve(current_content_size + calculate_elision_overhead(included));
    auto it = included.begin();
    std::size_t prev_index = *it;
    result += log_data.substr(candidates[prev_index].offset, candidates[prev_index].length);
    ++it;

    for (; it != included.end(); ++it) {
        if (*it > prev_index + 1) {
            result += "...\n";
        }
        result += log_data.substr(candidates[*it].offset, candidates[*it].length);
        prev_index = *it;
    }

    return result;
}

} // namespace log_triage

#endif // LOG_TRIAGE_HPP
