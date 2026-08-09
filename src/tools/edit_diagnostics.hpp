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

// --- the applied hunk ------------------------------------------------------------------
//
// WHAT A SUCCESSFUL EDIT ACTUALLY CHANGED, in the same `<line>\t<text>` form read_file
// uses, so the model can anchor its next edit on these numbers without reading the file
// back.
//
// WHY THIS EXISTS. replace_in_file's success path used to return one sentence -- "replaced
// one occurrence in <path>" -- with no line, no diff, and no context, while its FAILURE
// paths returned nearest-match snippets and candidate line numbers. So a model that had
// just edited and wanted to know WHAT it edited had exactly one route: read the whole file
// back and diff it by hand against a copy thousands of tokens up the context.
//
// MEASURED, on the ResMon run that ended at turn 22 of 200 (events.jsonl, run_id 5): a
// model edited DesignTokens.swift, got the one-sentence receipt, and then read that same
// file SIX times, byte-identical every time, writing nothing further. Every re-read was
// detected (`repeat_reread`, prior_count 1..5, unchanged=1) and answered with a note
// asking it to stop. It was not looping for want of being told; it was reconstructing the
// diff the tool declined to give it. Half the run and 25% of the final prompt went on it.
//
// The same lesson is already learned twenty lines away in registry.cpp, where an edit whose
// new_text equals its old_text gets its own message rather than a plain "replaced", with
// the note that reporting it as a replacement "is how a run spends four turns re-applying
// it". This is that fix for the case where the edit DID apply -- to the wrong site.
//
// BOUNDED ON PURPOSE. The failure this is fixing is partly a context-size failure, so a
// receipt that can print a whole file would trade one prompt-bloat bug for another. Three
// context lines, forty changed lines, and a hard byte ceiling; anything past that elides
// with a count, because a model that needs more than forty lines of diff to know where its
// edit landed is better served by reading the file deliberately.
struct AppliedHunk {
    std::size_t line = 0;    // 1-based first changed line; 0 when nothing changed
    std::size_t removed = 0; // lines removed
    std::size_t added = 0;   // lines added
    std::string text;        // rendered hunk, bounded; empty when nothing changed
};

namespace detail {

inline constexpr std::size_t kHunkContextLines = 3;
inline constexpr std::size_t kHunkMaxChangedLines = 40;
inline constexpr std::size_t kHunkMaxLineChars = 200;
inline constexpr std::size_t kHunkMaxChars = 4000;

inline void append_hunk_line(std::string& out, char sign, std::size_t number,
                             std::string_view text) {
    out += sign;
    out += std::to_string(number);
    out += '\t';
    if (text.size() > kHunkMaxLineChars) {
        out.append(text.data(), kHunkMaxLineChars);
        out += "  ... (+";
        out += std::to_string(text.size() - kHunkMaxLineChars);
        out += " chars)";
    } else {
        out.append(text.data(), text.size());
    }
    out += '\n';
}

} // namespace detail

// Line numbers: removed lines carry their number in the PRE-image, added and trailing
// context lines carry theirs in the POST-image -- which is the file as it stands now, and
// therefore the numbering any follow-up edit has to use.
[[nodiscard]] inline AppliedHunk applied_hunk(std::string_view before,
                                              std::string_view after) {
    AppliedHunk h;
    if (before == after) {
        return h;
    }
    std::vector<std::string_view> bl;
    std::vector<std::string_view> al;
    detail::split_lines(before, bl);
    detail::split_lines(after, al);
    const std::size_t bn = bl.size();
    const std::size_t an = al.size();

    std::size_t p = 0;
    while (p < bn && p < an && bl[p] == al[p]) {
        ++p;
    }
    std::size_t s = 0;
    while (s < bn - p && s < an - p && bl[bn - 1 - s] == al[an - 1 - s]) {
        ++s;
    }
    h.removed = bn - p - s;
    h.added = an - p - s;
    if (h.removed == 0 && h.added == 0) {
        // Byte-level difference inside identical lines (a line-ending change, say).
        // Nothing to show as a hunk; the caller still reports the byte count.
        return h;
    }
    h.line = p + 1;

    // Leading context is common to both images, so its numbering is the same in each.
    const std::size_t ctx_begin = p > detail::kHunkContextLines ? p - detail::kHunkContextLines : 0;
    for (std::size_t i = ctx_begin; i < p; ++i) {
        detail::append_hunk_line(h.text, ' ', i + 1, bl[i]);
    }

    // Elide the middle rather than either end: the first changed lines say where the edit
    // landed and the last say where it stopped, and those are the two facts being sought.
    const std::size_t changed = h.removed + h.added;
    const bool elide = changed > detail::kHunkMaxChangedLines;
    const std::size_t head = elide ? detail::kHunkMaxChangedLines / 2 : changed;
    const std::size_t tail = elide ? detail::kHunkMaxChangedLines - head : 0;

    std::size_t emitted = 0;
    const auto want = [&](std::size_t index) {
        return !elide || index < head || index >= changed - tail;
    };
    for (std::size_t i = 0; i < h.removed; ++i) {
        if (want(emitted)) {
            detail::append_hunk_line(h.text, '-', p + i + 1, bl[p + i]);
        } else if (emitted == head) {
            h.text += "   ... (" + std::to_string(changed - head - tail) + " more changed lines)\n";
        }
        ++emitted;
    }
    for (std::size_t i = 0; i < h.added; ++i) {
        if (want(emitted)) {
            detail::append_hunk_line(h.text, '+', p + i + 1, al[p + i]);
        } else if (emitted == head) {
            h.text += "   ... (" + std::to_string(changed - head - tail) + " more changed lines)\n";
        }
        ++emitted;
    }

    for (std::size_t i = 0; i < detail::kHunkContextLines && an - s + i < an; ++i) {
        detail::append_hunk_line(h.text, ' ', an - s + i + 1, al[an - s + i]);
    }

    if (h.text.size() > detail::kHunkMaxChars) {
        h.text.resize(detail::kHunkMaxChars);
        h.text += "\n   ... (hunk truncated)\n";
    }
    return h;
}

// The success-path receipt. Names the file, the line, and what moved -- and says the file
// state is already in hand, because a guarantee the model is not told about buys nothing.
[[nodiscard]] inline std::string format_applied(std::string_view path,
                                                std::string_view before,
                                                std::string_view after) {
    const AppliedHunk h = applied_hunk(before, after);
    std::string out = "edited ";
    out.append(path.data(), path.size());
    if (h.line == 0) {
        return out;
    }
    out += " at line ";
    out += std::to_string(h.line);
    out += " (-";
    out += std::to_string(h.removed);
    out += "/+";
    out += std::to_string(h.added);
    out += " lines). Applied hunk, with the file's CURRENT line numbers -- this is the "
           "whole change, so you do not need to read the file back to see it:\n";
    out += h.text;
    return out;
}

} // namespace edit_diagnostics
