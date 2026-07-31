#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <cctype>

namespace log_triage {

struct Chunk {
    std::string_view text;
    size_t original_index;
    double score;
};

inline double calculate_score(std::string_view text) {
    std::string lower_text;
    lower_text.reserve(text.size());
    for (char c : text) {
        lower_text.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    // 1. Semantic toolchain patterns
    std::vector<std::string> patterns = {
        "error:", "warning:", "exception", "traceback", "cmake error", "panicked at"
    };

    bool has_pattern = false;
    for (const auto& p : patterns) {
        if (lower_text.find(p) != std::string::npos) {
            has_pattern = true;
            break;
        }
    }

    if (has_pattern) {
        double score = 1000.0;
        for (const auto& p : patterns) {
            size_t pos = 0;
            while ((pos = lower_text.find(p, pos)) != std::string::npos) {
                score += 10.0;
                pos += p.length();
            }
        }
        return score;
    }

    // 2. Fallback heuristic
    std::vector<std::string> keywords = {
        "fail", "error", "exception", "fatal", "critical", "warn", "panic"
    };

    int keyword_count = 0;
    for (const auto& kw : keywords) {
        size_t pos = 0;
        while ((pos = lower_text.find(kw, pos)) != std::string::npos) {
            keyword_count++;
            pos += kw.length();
        }
    }

    int punct_count = 0;
    int digit_count = 0;
    for (char c : text) {
        if (std::ispunct(static_cast<unsigned char>(c))) {
            punct_count++;
        } else if (std::isdigit(static_cast<unsigned char>(c))) {
            digit_count++;
        }
    }

    double density = 0.0;
    if (!text.empty()) {
        density = static_cast<double>(punct_count + digit_count) / text.size();
    }

    return keyword_count * 50.0 + density * 100.0;
}

inline std::string compact(std::string_view log, size_t max_chunks = 5) {
    if (log.empty()) return "";

    std::vector<Chunk> chunks;

    size_t start = 0;
    size_t chunk_idx = 0;

    while (start < log.size()) {
        size_t pos = log.find("\n\n", start);
        size_t pos_rn = log.find("\r\n\r\n", start);

        size_t next_pos = std::min(pos, pos_rn);
        size_t sep_len = (next_pos == pos_rn && pos_rn != std::string_view::npos) ? 4 : 2;

        if (next_pos == std::string_view::npos) {
            next_pos = log.size();
            sep_len = 0;
        }

        std::string_view chunk_text = log.substr(start, next_pos - start);

        size_t trim_start = 0;
        while (trim_start < chunk_text.size() && std::isspace(static_cast<unsigned char>(chunk_text[trim_start]))) {
            trim_start++;
        }
        size_t trim_end = chunk_text.size();
        while (trim_end > trim_start && std::isspace(static_cast<unsigned char>(chunk_text[trim_end - 1]))) {
            trim_end--;
        }

        if (trim_end > trim_start) {
            std::string_view trimmed_chunk = chunk_text.substr(trim_start, trim_end - trim_start);
            chunks.push_back({trimmed_chunk, chunk_idx++, calculate_score(trimmed_chunk)});
        }

        if (next_pos == log.size()) break;
        start = next_pos + sep_len;
    }

    if (chunks.empty()) return "";

    size_t num_selected = std::min(max_chunks, chunks.size());

    std::vector<Chunk> selected_chunks = chunks;
    std::partial_sort(selected_chunks.begin(), selected_chunks.begin() + num_selected, selected_chunks.end(),
        [](const Chunk& a, const Chunk& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.original_index < b.original_index;
        });

    selected_chunks.resize(num_selected);

    std::sort(selected_chunks.begin(), selected_chunks.end(),
        [](const Chunk& a, const Chunk& b) {
            return a.original_index < b.original_index;
        });

    std::string result;
    for (size_t i = 0; i < selected_chunks.size(); ++i) {
        if (i > 0) {
            if (selected_chunks[i].original_index > selected_chunks[i-1].original_index + 1) {
                result += "\n\n...\n\n";
            } else {
                result += "\n\n";
            }
        }
        result += selected_chunks[i].text;
    }

    return result;
}

} // namespace log_triage
