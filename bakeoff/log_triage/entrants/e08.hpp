#ifndef LOG_TRIAGE_HPP
#define LOG_TRIAGE_HPP

#include <string>
#include <sstream>
#include <vector>
#include <regex>
#include <unordered_set>
#include <iostream>

namespace log_triage {

enum class State {
    NORMAL,
    BUILDING,
    TEMPLATE_INST,
    ERROR_BLOCK
};

inline std::string compact(const std::string& log) {
    static const std::regex re_build(R"(^(\[\d+/\d+\].*|make\[\d+\].*|-- .*)$)");
    static const std::regex re_template_start(R"(^(.*In instantiation of.*|.*In file included from.*)$)");
    static const std::regex re_template_req(R"(^(.*required from.*|.*required by.*)$)");
    static const std::regex re_error(R"(^.*?(?:error|fatal error):\s*(.*)$)", std::regex::icase);
    static const std::regex re_note(R"(^.*?note:\s*.*$)", std::regex::icase);
    static const std::regex re_warning(R"(^.*?warning:\s*.*$)", std::regex::icase);
    static const std::regex re_outcome(R"(^(?:ninja: build stopped:.*|make: \*\*\*.*|\d+ errors? generated\.|Build (?:failed|succeeded|successful).*)$)", std::regex::icase);

    std::istringstream iss(log);
    std::string line;
    std::ostringstream result;
    std::vector<std::string> outcome_lines;
    std::unordered_set<std::string> seen_errors;

    State state = State::NORMAL;

    int template_depth = 0;
    const int MAX_TEMPLATE_DEPTH = 1; // 1 means we keep 1, elide the rest
    bool template_elided = false;

    bool dampen_current_error = false;

    while (std::getline(iss, line)) {
        if (std::regex_match(line, re_outcome)) {
            outcome_lines.push_back(line);
            continue; // We only push to outcome_lines, do NOT print normally
        }

        // However, some lines after outcome are just random. If we are completely finished with build
        // maybe they should be outcome too? But let's only collect matched outcomes for pinning.

        if (state == State::TEMPLATE_INST) {
            if (std::regex_match(line, re_template_req)) {
                template_depth++;
                if (template_depth <= MAX_TEMPLATE_DEPTH) {
                    result << line << '\n';
                } else if (!template_elided) {
                    result << "    [... template instantiation elided ...]\n";
                    template_elided = true;
                }
                continue;
            } else {
                state = State::NORMAL;
            }
        }

        if (state == State::ERROR_BLOCK) {
            bool is_continuation = std::regex_match(line, re_note) ||
                                   (!line.empty() && (line[0] == ' ' || line[0] == '\t' ||
                                                      line.find(" | ") != std::string::npos ||
                                                      line.find(" |") != std::string::npos ||
                                                      line.find("^") != std::string::npos ||
                                                      line.find("~") != std::string::npos));
            if (is_continuation) {
                if (!dampen_current_error) {
                    result << line << '\n';
                }
                continue;
            } else {
                state = State::NORMAL;
                dampen_current_error = false;
            }
        }

        std::smatch match;

        if (std::regex_match(line, re_template_start)) {
            state = State::TEMPLATE_INST;
            template_depth = 0;
            template_elided = false;
            result << line << '\n';
            continue;
        }

        if (std::regex_match(line, match, re_error)) {
            state = State::ERROR_BLOCK;
            std::string error_msg = match[1].str();

            // Trim
            size_t first = error_msg.find_first_not_of(" \t\r\n");
            if (first != std::string::npos) {
                error_msg = error_msg.substr(first);
            }
            size_t last = error_msg.find_last_not_of(" \t\r\n");
            if (last != std::string::npos) {
                error_msg = error_msg.substr(0, last + 1);
            }

            if (seen_errors.contains(error_msg)) {
                dampen_current_error = true;
                result << "    [... identical error dampened ...]\n";
            } else {
                seen_errors.insert(error_msg);
                dampen_current_error = false;
                result << line << '\n';
            }
            continue;
        }

        if (std::regex_match(line, re_build)) {
            state = State::BUILDING;
            result << line << '\n';
            continue;
        }

        if (std::regex_match(line, re_warning)) {
            state = State::NORMAL;
            result << line << '\n';
            continue;
        }

        // Output normal line
        result << line << '\n';
    }

    for (const auto& oline : outcome_lines) {
        result << oline << '\n';
    }

    std::string res = result.str();
    if (!res.empty() && !log.empty() && log.back() != '\n') {
        if (res.back() == '\n') {
            res.pop_back();
        }
    }

    return res;
}

} // namespace log_triage

#endif // LOG_TRIAGE_HPP
