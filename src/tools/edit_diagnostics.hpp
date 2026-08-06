#pragma once
//
// Diagnostic-only nearest-region hints for edit refusals (replace_in_file NoMatch).
//
// Similarity NEVER authorizes a write. Callers may attach this text to a ToolError so the
// model can re-author old_text; the graft / apply_patch engines remain exact.
//
#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace edit_diagnostics {

namespace detail {

[[nodiscard]] inline bool is_ws(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

[[nodiscard]] inline bool is_ident(char c) noexcept {
    const unsigned char u = static_cast<unsigned char>(c);
    return (u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') || (u >= '0' && u <= '9') ||
           u == '_' || u >= 0x80;
}

inline std::vector<std::string_view> tokenize(std::string_view s) {
    std::vector<std::string_view> out;
    std::size_t i = 0;
    while (i < s.size()) {
        if (is_ws(s[i])) {
            ++i;
        } else if (is_ident(s[i])) {
            const std::size_t start = i;
            while (i < s.size() && is_ident(s[i])) {
                ++i;
            }
            out.push_back(s.substr(start, i - start));
        } else {
            out.push_back(s.substr(i, 1));
            ++i;
        }
    }
    return out;
}

inline void split_lines(std::string_view text, std::vector<std::string_view>& out) {
    out.clear();
    std::size_t pos = 0;
    while (pos < text.size()) {
        const std::size_t nl = text.find('\n', pos);
        if (nl == std::string_view::npos) {
            out.push_back(text.substr(pos));
            break;
        }
        std::string_view line = text.substr(pos, nl - pos);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        out.push_back(line);
        pos = nl + 1;
    }
}

[[nodiscard]] inline double token_jaccard(const std::vector<std::string_view>& a,
                                          const std::vector<std::string_view>& b) {
    if (a.empty() || b.empty()) {
        return 0.0;
    }
    std::size_t inter = 0;
    for (std::string_view t : a) {
        if (std::find(b.begin(), b.end(), t) != b.end()) {
            ++inter;
        }
    }
    const std::size_t uni = a.size() + b.size() - inter;
    return uni == 0 ? 0.0 : static_cast<double>(inter) / static_cast<double>(uni);
}

} // namespace detail

struct Candidate {
    std::size_t line = 1; // 1-based
    double score = 0.0;
    std::string snippet;
};

// Returns up to `limit` nearest windows whose token overlap with `old_text` is highest.
// Empty when nothing clears a minimal floor — callers still have their own ground-truth
// snippet path.
[[nodiscard]] inline std::vector<Candidate> nearest_regions(std::string_view file,
                                                            std::string_view old_text,
                                                            std::size_t limit = 3) {
    std::vector<Candidate> out;
    if (file.empty() || old_text.empty() || limit == 0) {
        return out;
    }
    const std::vector<std::string_view> want = detail::tokenize(old_text);
    if (want.empty()) {
        return out;
    }

    std::vector<std::string_view> lines;
    detail::split_lines(file, lines);
    if (lines.empty()) {
        return out;
    }

    // Window height tracks the old_text line count, clamped.
    std::size_t old_lines = 1;
    for (char c : old_text) {
        if (c == '\n') {
            ++old_lines;
        }
    }
    const std::size_t win = std::min(std::max(old_lines, std::size_t{1}), std::size_t{12});

    struct Scored {
        std::size_t line;
        double score;
    };
    std::vector<Scored> scored;
    scored.reserve(lines.size());
    for (std::size_t i = 0; i < lines.size(); ++i) {
        std::string window;
        for (std::size_t j = i; j < lines.size() && j < i + win; ++j) {
            if (j > i) {
                window.push_back('\n');
            }
            window.append(lines[j].data(), lines[j].size());
        }
        const double score = detail::token_jaccard(want, detail::tokenize(window));
        if (score >= 0.35) {
            scored.push_back(Scored{i + 1, score});
        }
    }
    std::stable_sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.line < b.line;
    });

    // Suppress overlapping windows: keep the best, skip others within win lines.
    std::vector<std::size_t> taken;
    for (const Scored& s : scored) {
        bool overlap = false;
        for (std::size_t t : taken) {
            const std::size_t lo = t > win ? t - win : 1;
            const std::size_t hi = t + win;
            if (s.line >= lo && s.line <= hi) {
                overlap = true;
                break;
            }
        }
        if (overlap) {
            continue;
        }
        taken.push_back(s.line);
        Candidate c;
        c.line = s.line;
        c.score = s.score;
        const std::size_t begin = s.line;
        const std::size_t end = std::min(lines.size(), begin + win - 1);
        for (std::size_t ln = begin; ln <= end; ++ln) {
            c.snippet += std::to_string(ln);
            c.snippet += '\t';
            c.snippet.append(lines[ln - 1].data(), lines[ln - 1].size());
            c.snippet += '\n';
        }
        out.push_back(std::move(c));
        if (out.size() >= limit) {
            break;
        }
    }
    return out;
}

[[nodiscard]] inline std::string format_nearest(std::string_view path,
                                                std::string_view file,
                                                std::string_view old_text) {
    const std::vector<Candidate> cands = nearest_regions(file, old_text);
    if (cands.empty()) {
        return {};
    }
    std::string out;
    out += "\n\n[Nearest candidate regions in ";
    out.append(path.data(), path.size());
    out += " — diagnostics only; similarity does not authorize a write]:\n";
    for (std::size_t i = 0; i < cands.size(); ++i) {
        out += "candidate ";
        out += std::to_string(i + 1);
        out += " (line ";
        out += std::to_string(cands[i].line);
        out += ", overlap ";
        // one decimal
        const int pct = static_cast<int>(cands[i].score * 100.0 + 0.5);
        out += std::to_string(pct);
        out += "%):\n```\n";
        out += cands[i].snippet;
        out += "```\n";
    }
    return out;
}

} // namespace edit_diagnostics
