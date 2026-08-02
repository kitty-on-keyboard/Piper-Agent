#pragma once
//
// How file bytes are presented to the model (spec S6.2, S8.4).
//
// Split out of registry.cpp because it is a presentation decision, not a tool: the same
// two rules apply to every read the model makes, and they are worth stating in one place
// where they can be argued with.
//
//   1. LINE NUMBERS ARE ABSOLUTE. read_slice takes line numbers as arguments and every
//      compiler diagnostic the model will read reports them, so numbering a slice from 1
//      is an off-by-start_line trap. No column padding -- alignment costs tokens and buys
//      nothing.
//
//   2. THE NUMBERS ARE DISPLAY ONLY. A model that pastes them into replace_in_file's
//      old_text gets NoMatch, because graft tokenizes "42" as its own identifier token so
//      the token sequence no longer matches the file -- a refusal with the file untouched,
//      which is that engine's contract. The failure mode is a wasted turn, never a
//      corrupted file. Both tool descriptions say so anyway.
//
#include <cstddef>
#include <string>
#include <string_view>

namespace lmp::tools {

// Lines in `s`, counting the last one whether or not it ends in a newline.
[[nodiscard]] inline std::size_t count_lines(std::string_view s) {
    std::size_t n = 1;
    for (char c : s) {
        n += static_cast<std::size_t>(c == '\n');
    }
    return n;
}

// `body` with each line prefixed by its absolute 1-based number and a tab.
[[nodiscard]] inline std::string number_lines(std::string_view body, long first_line) {
    std::string out;
    out.reserve(body.size() + body.size() / 16 + 16);
    long line = first_line;
    std::size_t at = 0;
    while (at < body.size()) {
        const std::size_t nl = body.find('\n', at);
        const std::size_t stop = nl == std::string_view::npos ? body.size() : nl + 1;
        out += std::to_string(line);
        out += '\t';
        out.append(body, at, stop - at);
        if (nl == std::string_view::npos) {
            break;
        }
        at = stop;
        ++line;
    }
    return out;
}

// Removes the display line numbers from text the model copied back out of a read.
//
// MEASURED, NOT ANTICIPATED. The plan for line numbering said the failure mode was safe --
// graft tokenizes "42" as its own identifier so a numbered old_text returns NoMatch with
// the file untouched -- and left it at a warning in the tool description. The first
// end-to-end run after numbering landed spent FIVE consecutive turns on
// replace_in_file ToolError doing exactly that. Safe is not the same as free: each of
// those turns cost a prefill, a decode and an iteration against the budget.
//
// So the tool accepts what it displayed. Stripping happens only when EVERY non-empty line
// carries the prefix, which is what a copied read looks like and what hand-written
// old_text almost never does -- a single unnumbered line is enough to leave the text
// completely alone.
[[nodiscard]] inline std::string strip_line_numbers(std::string_view s) {
    if (s.empty()) {
        return std::string(s);
    }
    const auto prefix_len = [](std::string_view line) -> std::size_t {
        std::size_t i = 0;
        while (i < line.size() && line[i] >= '0' && line[i] <= '9') {
            ++i;
        }
        return (i > 0 && i < line.size() && line[i] == '\t') ? i + 1 : 0;
    };

    std::size_t at = 0;
    std::size_t numbered = 0;
    while (at < s.size()) {
        const std::size_t nl = s.find('\n', at);
        const std::size_t stop = nl == std::string_view::npos ? s.size() : nl;
        const std::string_view line = s.substr(at, stop - at);
        if (!line.empty()) {
            if (prefix_len(line) == 0) {
                return std::string(s); // one bare line and the whole thing is untouched
            }
            ++numbered;
        }
        at = nl == std::string_view::npos ? s.size() : nl + 1;
    }
    if (numbered == 0) {
        return std::string(s);
    }

    std::string out;
    out.reserve(s.size());
    at = 0;
    while (at < s.size()) {
        const std::size_t nl = s.find('\n', at);
        const std::size_t stop = nl == std::string_view::npos ? s.size() : nl + 1;
        const std::string_view line = s.substr(at, stop - at);
        out.append(line.substr(prefix_len(line)));
        at = stop;
    }
    return out;
}

} // namespace lmp::tools
