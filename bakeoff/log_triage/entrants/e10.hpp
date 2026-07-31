#ifndef LOG_TRIAGE_HPP
#define LOG_TRIAGE_HPP

#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstddef>

namespace log_triage {

enum class TokenType {
    Noise = 1,
    Snippet = 2,
    CommandHeader = 3,
    DiagnosticMessage = 4,
    TestOutcome = 5
};

struct Token {
    TokenType type;
    std::string text;
    bool elided = false;
    size_t keep_bytes = 0;

    size_t actual_size() const {
        if (!elided) return text.size();
        size_t elided_count = text.size() - keep_bytes;
        return keep_bytes + std::string("\n... [elided " + std::to_string(elided_count) + " bytes] ...\n").size();
    }

    std::string render() const {
        if (!elided) return text;
        size_t elided_count = text.size() - keep_bytes;
        size_t head_len = keep_bytes / 2;
        size_t tail_len = keep_bytes - head_len;
        return text.substr(0, head_len) +
               "\n... [elided " + std::to_string(elided_count) + " bytes] ...\n" +
               text.substr(text.size() - tail_len);
    }
};

inline std::string strip_ansi(std::string_view input) {
    std::string output;
    output.reserve(input.size());
    size_t i = 0;
    while (i < input.size()) {
        if (input[i] == '\x1b') {
            if (i + 1 < input.size() && input[i + 1] == '[') {
                i += 2;
                while (i < input.size() && input[i] >= 0x30 && input[i] <= 0x3F) ++i;
                while (i < input.size() && input[i] >= 0x20 && input[i] <= 0x2F) ++i;
                if (i < input.size() && input[i] >= 0x40 && input[i] <= 0x7E) ++i;
                continue;
            } else if (i + 1 < input.size() && input[i + 1] == ']') {
                i += 2;
                while (i < input.size() && input[i] != '\x07' && input[i] != '\x1b') ++i;
                if (i < input.size() && input[i] == '\x1b') {
                    if (i + 1 < input.size() && input[i+1] == '\\') {
                        i += 2;
                    }
                } else if (i < input.size() && input[i] == '\x07') {
                    i++;
                }
                continue;
            }
            i++;
            continue;
        }
        output.push_back(input[i]);
        i++;
    }
    return output;
}

inline bool contains_ci(std::string_view s, std::string_view sub) {
    auto it = std::search(s.begin(), s.end(), sub.begin(), sub.end(),
        [](char c1, char c2) {
            char l1 = (c1 >= 'A' && c1 <= 'Z') ? c1 - 'A' + 'a' : c1;
            char l2 = (c2 >= 'A' && c2 <= 'Z') ? c2 - 'A' + 'a' : c2;
            return l1 == l2;
        });
    return it != s.end();
}

inline TokenType classify_line(std::string_view line) {
    if (contains_ci(line, "passed") || contains_ci(line, "failed") ||
        contains_ci(line, "test outcome") || contains_ci(line, "tests:") ||
        contains_ci(line, "build success") || contains_ci(line, "build failed")) {
        return TokenType::TestOutcome;
    }

    if (contains_ci(line, "error:") || contains_ci(line, "warning:") ||
        contains_ci(line, "exception") || contains_ci(line, "traceback") ||
        contains_ci(line, "fatal") || contains_ci(line, "panic:")) {
        return TokenType::DiagnosticMessage;
    }

    if (line.starts_with("$ ") || line.starts_with("> ") ||
        line.starts_with("make ") || line.starts_with("cmake ") ||
        line.starts_with("gcc ") || line.starts_with("clang ") ||
        line.starts_with("npm run ")) {
        return TokenType::CommandHeader;
    }

    if (contains_ci(line, " | ") || contains_ci(line, "-->") || contains_ci(line, "~~~") || contains_ci(line, "^~~")) {
        return TokenType::Snippet;
    }

    return TokenType::Noise;
}

inline std::string compact(std::string_view raw_log, std::size_t budget_bytes) {
    std::string clean_log = strip_ansi(raw_log);

    std::vector<Token> tokens;
    size_t start = 0;
    while (start < clean_log.size()) {
        size_t end = clean_log.find('\n', start);
        if (end == std::string::npos) end = clean_log.size();
        else end++;

        std::string_view line(clean_log.data() + start, end - start);
        TokenType type = classify_line(line);

        if (!tokens.empty() && tokens.back().type == type) {
            tokens.back().text.append(line);
        } else {
            tokens.push_back({type, std::string(line), false, 0});
        }

        start = end;
    }

    auto compute_total_size = [&]() {
        size_t total = 0;
        for (const auto& t : tokens) total += t.actual_size();
        return total;
    };

    size_t total_size = compute_total_size();

    bool changed = true;
    while (total_size > budget_bytes && changed) {
        changed = false;
        Token* best_token = nullptr;
        int best_priority = 999;

        for (auto& t : tokens) {
            int prio = static_cast<int>(t.type);
            if (prio <= best_priority) {
                if (!t.elided) {
                    if (t.text.size() > 50) {
                        if (prio < best_priority || (best_token && t.text.size() > best_token->text.size())) {
                            best_priority = prio;
                            best_token = &t;
                        }
                    }
                } else {
                    if (t.keep_bytes > 0) {
                        if (prio < best_priority || (best_token && t.keep_bytes > best_token->keep_bytes)) {
                            best_priority = prio;
                            best_token = &t;
                        }
                    }
                }
            }
        }

        if (best_token) {
            if (!best_token->elided) {
                best_token->elided = true;
                size_t overage = total_size - budget_bytes;
                size_t approx_marker = 40;
                size_t needed_reduction = overage + approx_marker;

                if (best_token->text.size() > needed_reduction) {
                    best_token->keep_bytes = best_token->text.size() - needed_reduction;
                } else {
                    best_token->keep_bytes = 0;
                }
            } else {
                size_t overage = total_size - budget_bytes;
                if (best_token->keep_bytes > overage) {
                    best_token->keep_bytes -= overage;
                } else {
                    best_token->keep_bytes = 0;
                }
            }
            changed = true;
            total_size = compute_total_size();
        }
    }

    std::string result;
    for (const auto& t : tokens) {
        result += t.render();
    }
    return result;
}

} // namespace log_triage

#endif // LOG_TRIAGE_HPP
