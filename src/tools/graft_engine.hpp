#pragma once
//
// Ported from v1 src/tools/graft_engine.hpp with its provenance comment intact -- like
// the S18 corpora, it is the survivor of a measured cookoff, and its neutral corpus
// lesson ("all ten claimed 0% false applies; nine had them") is the founding story of
// this repo's bakeoff discipline. std-only, unmodified apart from this note.
//
// graft_engine -- whitespace-tolerant, refusal-first application of a model
// edit (old_text -> new_text) to a file.
//
// PROVENANCE. This is a merge of ten independent implementations (the
// "edit-app-engine" cookoff, entrants e01/e03/e05/e06/e07/e08/e09/e10/e14/e15),
// selected by measurement rather than by their self-reported scores. Each
// entrant shipped its own corpus AND its own scorer, so their published
// numbers were mutually incomparable and all ten claimed a 0% false-apply
// rate. Re-scored on one neutral corpus whose ground truth comes from bucket
// semantics rather than from any implementation, nine of the ten had false
// applies. What was kept, and why:
//
//   MATCHER  e08/e14's lexical tokenizer. Identifier runs stay whole and each
//            other byte is its own token, so `a+b` matches `a + b` (same
//            tokens, different gaps) while `foo bar` does NOT match `foobar`
//            (different tokens). It was the only tokenizer that got both of
//            those right. e05's strip-all-whitespace matcher gets the first
//            and fails the second -- it produced 5 of its 10 false applies.
//   REFUSAL  e08's full occurrence count. Counting every hit rather than
//            probing for a second one is what lets an ambiguous edit be
//            reported with all its candidate sites.
//   REPLACER e05's indent-delta re-anchoring (the only entrant to score 15/15
//            on indent_shift) plus e15's line-ending preservation (the only
//            one to score 5/5 on line_endings). Everyone else pasted new_text
//            at the author's indent into a differently-indented site, which
//            silently emits broken code -- and reported Applied for it.
//   BLANKS   e01/e06's insight that blank lines inside a matched span must be
//            accounted for rather than dropped on the floor.
//
// Deliberately NOT taken:
//   e15's five "adaptations for the benchmark buckets" -- post-hoc rewrites of
//     new_text (4-spaces->tab everywhere including inside string literals,
//     appending trailing spaces, doubling newlines). Corpus overfitting that
//     corrupts the model's intended content.
//   e05's `old_text == new_text -> Applied` short circuit, which reports
//     success for an edit never made when old_text is absent (5 false applies).
//   e07's Unicode handling, a hardcoded special case for the single letter
//     "e-acute". The IDEA is sound but the implementation only ever matched
//     its own corpus. Proper NFC/NFD folding needs a real Unicode table; until
//     one is vendored this engine treats normalization variants as NoMatch.
//     See kUnicodeNormalizationUnsupported below.
//   e10's DP alignment with a 30% indel budget, which can match a span the
//     model never wrote, and which returns an EMPTY result string on refusal
//     -- a caller that writes result back unconditionally truncates the file.
//
// CONTRACT.
//   * Applied   -- result is the file with exactly one span replaced. Every
//                  byte outside that span is unchanged.
//   * Ambiguous -- 2+ candidate sites. result is the file, UNMODIFIED.
//   * NoMatch   -- no candidate site. result is the file, UNMODIFIED.
//   The result string is never empty for a non-empty file, in any status.
//
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace graft {

// Normalization-variant text (NFC vs NFD) is reported as NoMatch rather than
// silently matched. Flipping this on requires a real Unicode table, not a
// per-letter special case.
inline constexpr bool kUnicodeNormalizationUnsupported = true;

enum class Status { Applied, Ambiguous, NoMatch };

struct Match {
    std::size_t byte_offset;
    std::size_t line;
    double confidence;
};

struct Result {
    Status status;
    std::string result;
    std::vector<Match> matches;
};

namespace detail {

inline bool is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

inline bool is_hspace(char c) { return c == ' ' || c == '\t'; }

// Identifier bytes. Bytes >= 0x80 count as identifier characters so a UTF-8
// sequence stays inside one token instead of splintering into stray bytes.
inline bool is_ident(char c) {
    const unsigned char u = static_cast<unsigned char>(c);
    return (u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') || (u >= '0' && u <= '9') ||
           u == '_' || u >= 0x80;
}

struct Token {
    std::string_view text;
    std::size_t offset;
};

// Lexical tokenizer. Whitespace is dropped; identifier runs are one token;
// every other byte is its own token. Token IDENTITY is preserved, only the
// gaps between tokens are made flexible -- that is the whole safety property.
inline std::vector<Token> tokenize(std::string_view s) {
    std::vector<Token> out;
    out.reserve(s.size() / 4 + 8);
    std::size_t i = 0;
    while (i < s.size()) {
        if (is_ws(s[i])) {
            ++i;
        } else if (is_ident(s[i])) {
            const std::size_t start = i;
            while (i < s.size() && is_ident(s[i])) ++i;
            out.push_back({s.substr(start, i - start), start});
        } else {
            out.push_back({s.substr(i, 1), i});
            ++i;
        }
    }
    return out;
}

inline std::size_t count_lines(std::string_view s, std::size_t upto) {
    std::size_t line = 1;
    for (std::size_t i = 0; i < upto && i < s.size(); ++i)
        if (s[i] == '\n') ++line;
    return line;
}

inline std::size_t line_start_of(std::string_view s, std::size_t pos) {
    while (pos > 0 && s[pos - 1] != '\n') --pos;
    return pos;
}

// Leading horizontal whitespace of the line containing `pos`.
inline std::string_view indent_at(std::string_view s, std::size_t line_start) {
    std::size_t i = line_start;
    while (i < s.size() && is_hspace(s[i])) ++i;
    return s.substr(line_start, i - line_start);
}

struct IndentUnit {
    char ch = ' ';
    std::size_t width = 4;  // in columns; a tab counts as one unit of `ch`
    bool known = false;
};

inline std::size_t visual_width(std::string_view indent, std::size_t tab_width) {
    std::size_t w = 0;
    for (char c : indent) w += (c == '\t') ? tab_width : 1;
    return w;
}

inline std::size_t gcd_of(std::size_t a, std::size_t b) {
    while (b) { const std::size_t t = a % b; a = b; b = t; }
    return a;
}

// Infer the file's indentation convention: which character it indents with,
// and how wide one level is.
//
// The level width is the GCD of the observed indent widths, not the smallest
// one. A block that happens to sit at columns 8 and 12 has a level width of 4,
// but its smallest indent is 8 -- taking the minimum turns one nested level
// into two and silently doubles the indentation of every nested line.
inline IndentUnit detect_indent_unit(std::string_view text) {
    IndentUnit u;
    std::size_t tabs = 0, spaces = 0, width_gcd = 0;
    std::size_t pos = 0;
    while (pos < text.size()) {
        const std::size_t ls = pos;
        std::size_t i = ls;
        while (i < text.size() && is_hspace(text[i])) ++i;
        // Skip blank lines: their indent says nothing about the convention.
        if (i < text.size() && text[i] != '\n' && text[i] != '\r' && i > ls) {
            if (text[ls] == '\t') {
                ++tabs;
            } else {
                ++spaces;
                width_gcd = gcd_of(width_gcd, i - ls);
            }
        }
        const std::size_t nl = text.find('\n', pos);
        if (nl == std::string_view::npos) break;
        pos = nl + 1;
    }
    if (tabs == 0 && spaces == 0) return u;  // unknown; caller keeps defaults
    u.known = true;
    if (tabs >= spaces) {
        u.ch = '\t';
        u.width = 1;
    } else {
        u.ch = ' ';
        u.width = width_gcd ? width_gcd : 4;
    }
    return u;
}

inline bool file_uses_crlf(std::string_view text) {
    const std::size_t nl = text.find('\n');
    if (nl == std::string_view::npos) return false;
    return nl > 0 && text[nl - 1] == '\r';
}

// Indent of the first non-blank line of `text`.
inline std::string_view first_line_indent(std::string_view text) {
    std::size_t pos = 0;
    while (pos < text.size()) {
        std::size_t i = pos;
        while (i < text.size() && is_hspace(text[i])) ++i;
        if (i < text.size() && text[i] != '\n' && text[i] != '\r')
            return text.substr(pos, i - pos);
        const std::size_t nl = text.find('\n', pos);
        if (nl == std::string_view::npos) break;
        pos = nl + 1;
    }
    return text.substr(0, 0);
}

inline std::string_view trim_ws(std::string_view s) {
    std::size_t b = 0, e = s.size();
    while (b < e && is_ws(s[b])) ++b;
    while (e > b && is_ws(s[e - 1])) --e;
    return s.substr(b, e - b);
}

// Re-anchor `new_text`, authored at `old_indent` using `old_unit`, so it sits
// under `file_indent` using `file_unit`, with `eol` line endings.
//
// This is the step every entrant except e05 skipped, and skipping it is how an
// engine emits a block whose first line is at one indent and whose body is at
// another -- broken code, reported as Applied.
inline std::string reanchor(std::string_view new_text,
                            std::string_view file_indent,
                            std::string_view old_indent,
                            const IndentUnit& file_unit,
                            const IndentUnit& old_unit,
                            bool first_line_bare,
                            std::string_view eol) {
    const std::size_t old_tabw = old_unit.ch == '\t' ? 1 : (old_unit.width ? old_unit.width : 4);
    std::string out;
    out.reserve(new_text.size() + 32);

    std::size_t pos = 0;
    bool first = true;
    while (true) {
        std::size_t nl = new_text.find('\n', pos);
        std::string_view line =
            new_text.substr(pos, (nl == std::string_view::npos ? new_text.size() : nl) - pos);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);

        if (!first) out.append(eol);

        std::size_t lead = 0;
        while (lead < line.size() && is_hspace(line[lead])) ++lead;
        const std::string_view body = line.substr(lead);

        if (body.empty()) {
            // Blank line: emit it blank. Re-indenting nothing produces
            // trailing whitespace, which is a content change we did not ask
            // for.
        } else if (first && first_line_bare) {
            // The file's own indent already precedes the span in the untouched
            // prefix, so the first line must not repeat it.
            out.append(body);
        } else {
            std::string_view rel = line.substr(0, lead);
            if (!old_indent.empty() && rel.size() >= old_indent.size() &&
                rel.substr(0, old_indent.size()) == old_indent) {
                rel = rel.substr(old_indent.size());
            }
            out.append(file_indent);
            const std::size_t rel_w = visual_width(rel, old_tabw);
            const std::size_t step = old_unit.known ? (old_unit.ch == '\t' ? 1 : old_unit.width)
                                                    : 0;
            if (step > 0 && rel_w % step == 0) {
                const std::size_t levels = rel_w / step;
                const std::size_t fstep = file_unit.ch == '\t' ? 1 : file_unit.width;
                out.append(std::string(levels * fstep, file_unit.ch));
            } else {
                out.append(rel);  // not a clean multiple: keep it verbatim
            }
            out.append(body);
        }

        first = false;
        if (nl == std::string_view::npos) break;
        pos = nl + 1;
    }
    return out;
}

inline void splice(std::string& out, std::string_view file, std::size_t begin,
                   std::size_t end, std::string_view replacement) {
    out.clear();
    out.reserve(file.size() - (end - begin) + replacement.size());
    out.append(file.substr(0, begin));
    out.append(replacement);
    out.append(file.substr(end));
}

}  // namespace detail

// Apply `old_text` -> `new_text` to `file`.
//
// Tier 1 is an exact byte search. Tier 2 is a lexical token search that
// tolerates whitespace differences but not token differences. Either tier
// refuses on 2+ candidates rather than guessing.
inline Result apply(std::string_view file, std::string_view old_text,
                    std::string_view new_text) {
    Result res;
    res.status = Status::NoMatch;
    res.result.assign(file);  // refusal leaves the file untouched, never empty

    if (old_text.empty()) return res;

    // ---- Tier 1: exact. Count every occurrence; do not stop at the second.
    {
        std::vector<std::size_t> hits;
        std::size_t p = file.find(old_text);
        while (p != std::string_view::npos) {
            hits.push_back(p);
            p = file.find(old_text, p + old_text.size());
        }
        if (hits.size() == 1) {
            res.status = Status::Applied;
            detail::splice(res.result, file, hits[0], hits[0] + old_text.size(), new_text);
            res.matches.push_back({hits[0], detail::count_lines(file, hits[0]), 1.0});
            return res;
        }
        // Mutation testing showed this branch is status-equivalent to the
        // tier-2 guard below: delete it and 2+ exact hits fall through to the
        // lexical tier, which necessarily also sees 2+ and refuses. It is kept
        // as a fast path (no tokenization of the whole file) and because it
        // reports byte-exact offsets, NOT because it is the thing making
        // duplicate edits safe. If you are hunting the guard that does that,
        // it is `starts.size() > 1` below -- removing THAT one applies an edit
        // to an arbitrary one of several matching sites.
        if (hits.size() > 1) {
            res.status = Status::Ambiguous;
            for (std::size_t h : hits)
                res.matches.push_back({h, detail::count_lines(file, h), 1.0});
            return res;
        }
    }

    // ---- Tier 2: lexical tokens. Whitespace flexes, token identity does not.
    const std::vector<detail::Token> old_toks = detail::tokenize(old_text);
    if (old_toks.empty()) return res;  // old_text was pure whitespace
    const std::vector<detail::Token> file_toks = detail::tokenize(file);
    if (file_toks.size() < old_toks.size()) return res;

    std::vector<std::size_t> starts;
    const std::size_t limit = file_toks.size() - old_toks.size();
    for (std::size_t i = 0; i <= limit; ++i) {
        bool ok = true;
        for (std::size_t j = 0; j < old_toks.size(); ++j) {
            if (file_toks[i + j].text != old_toks[j].text) { ok = false; break; }
        }
        if (ok) starts.push_back(i);
    }

    if (starts.empty()) return res;
    if (starts.size() > 1) {
        res.status = Status::Ambiguous;
        for (std::size_t s : starts) {
            const std::size_t off = file_toks[s].offset;
            res.matches.push_back({off, detail::count_lines(file, off), 1.0});
        }
        return res;
    }

    // ---- Single lexical match: rebuild the replacement for this site.
    const std::size_t s = starts[0];
    const detail::Token& first_tok = file_toks[s];
    const detail::Token& last_tok = file_toks[s + old_toks.size() - 1];
    const std::size_t span_begin = first_tok.offset;
    const std::size_t span_end = last_tok.offset + last_tok.text.size();

    const std::size_t ls = detail::line_start_of(file, span_begin);
    const std::string_view file_indent = detail::indent_at(file, ls);
    // Only treat this as a block edit when the span starts at the first
    // non-blank column; a mid-line match must not drag the line's indent in.
    const bool first_line_bare = (span_begin == ls + file_indent.size());

    const std::string_view old_indent = detail::first_line_indent(old_text);
    const detail::IndentUnit file_unit = detail::detect_indent_unit(file);
    detail::IndentUnit old_unit = detail::detect_indent_unit(old_text);
    if (!old_unit.known) old_unit = file_unit;

    const std::string_view eol = detail::file_uses_crlf(file) ? "\r\n" : "\n";
    const std::string_view new_core = detail::trim_ws(new_text);

    std::string replacement;
    if (new_core.empty()) {
        replacement.clear();  // deletion
    } else {
        replacement = detail::reanchor(new_core,
                                       first_line_bare ? file_indent : std::string_view{},
                                       old_indent, file_unit, old_unit,
                                       first_line_bare, eol);
    }

    res.status = Status::Applied;
    detail::splice(res.result, file, span_begin, span_end, replacement);
    res.matches.push_back({span_begin, detail::count_lines(file, span_begin), 1.0});
    return res;
}

}  // namespace graft
