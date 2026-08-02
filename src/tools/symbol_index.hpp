#pragma once
//
// locate_symbol's ranking (spec S6.2).
//
// WHAT WAS WRONG. One global regex over every extension, `head -60`, no ranking, no
// dedup: whichever definition-shaped line the filesystem walk reached first won, and a
// symbol mentioned in fifty call sites buried its own definition. `rename_across_files` in
// evals/agent is the task that pays for this.
//
// WHAT THIS IS NOT. It is not an index and it is not an AST. Tree-sitter would be correct
// and costs twenty vendored grammars against S2.2's "would a competent team building this
// fresh choose to build it?"; an LSP client would be correct and is sequenced behind the
// post-edit checker that would justify the daemon. This is the cheap middle: the same grep,
// ranked by how definition-shaped each hit is, with the ordering decided here where it can
// be asserted rather than inside a shell pipeline where it cannot.
//
// THE RANKING, in the order that decides ties:
//   3  a definition keyword immediately precedes the symbol  (`def foo`, `class Foo`,
//      `fn foo`, `struct Foo`, `function foo`)
//   2  the symbol opens the line, modulo indentation          (`foo = ...`, `foo() {`)
//   1  anything else the pattern matched
// and within a score, shallower indentation first -- a top-level definition outranks one
// nested in a function, which is nearly always what "where is this defined" means.
//
#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace lmp::tools {

struct SymbolHit {
    std::string path;
    long line = 0;
    std::string text;
    int score = 0;
    std::size_t indent = 0;
};

// The keywords that mean "this line defines something", across the languages this agent
// actually sees. Deliberately a flat list rather than per-language tables: the cost of a
// wrong guess here is a mis-RANKED hit, not a missed one, and a language table that has to
// be kept in step with the extension list is a second thing to get wrong.
inline constexpr std::string_view kDefinitionKeywords[] = {
    "def",   "class",  "struct", "enum",  "fn",    "func",   "function",
    "impl",  "trait",  "type",   "using", "const", "let",    "var",
    "void",  "int",    "bool",   "auto",  "template", "interface",
};

[[nodiscard]] inline std::size_t indent_of(std::string_view text) {
    std::size_t n = 0;
    while (n < text.size() && (text[n] == ' ' || text[n] == '\t')) {
        ++n;
    }
    return n;
}

[[nodiscard]] inline bool is_ident_char(char c) {
    const unsigned char u = static_cast<unsigned char>(c);
    return (u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') || (u >= '0' && u <= '9') ||
           u == '_';
}

[[nodiscard]] inline int score_hit(std::string_view text, std::string_view symbol) {
    const std::size_t at = text.find(symbol);
    if (at == std::string_view::npos) {
        return 1;
    }
    // A keyword immediately before it, separated only by spaces.
    std::size_t back = at;
    while (back > 0 && text[back - 1] == ' ') {
        --back;
    }
    std::size_t word_start = back;
    while (word_start > 0 && is_ident_char(text[word_start - 1])) {
        --word_start;
    }
    const std::string_view word = text.substr(word_start, back - word_start);
    for (std::string_view kw : kDefinitionKeywords) {
        if (word == kw) {
            return 3;
        }
    }
    return at == indent_of(text) ? 2 : 1;
}

// Parses `path:line:text` lines, scores them, deduplicates by (path, line) and returns the
// best `limit` in ranked order. `suppressed` reports how many were dropped, because a
// truncated list that does not say it was truncated is how a model concludes a symbol has
// exactly one definition site.
[[nodiscard]] inline std::vector<SymbolHit> rank_symbol_hits(std::string_view grep_output,
                                                             std::string_view symbol,
                                                             std::size_t limit,
                                                             std::size_t& suppressed) {
    std::vector<SymbolHit> hits;
    std::size_t at = 0;
    while (at < grep_output.size()) {
        const std::size_t nl = grep_output.find('\n', at);
        const std::string_view row =
            grep_output.substr(at, (nl == std::string_view::npos ? grep_output.size() : nl) - at);
        at = nl == std::string_view::npos ? grep_output.size() : nl + 1;
        if (row.empty()) {
            continue;
        }
        const std::size_t c1 = row.find(':');
        if (c1 == std::string_view::npos) {
            continue;
        }
        const std::size_t c2 = row.find(':', c1 + 1);
        if (c2 == std::string_view::npos) {
            continue;
        }
        SymbolHit h;
        h.path = std::string(row.substr(0, c1));
        h.line = std::strtol(std::string(row.substr(c1 + 1, c2 - c1 - 1)).c_str(), nullptr, 10);
        h.text = std::string(row.substr(c2 + 1));
        if (h.line <= 0) {
            continue;
        }
        h.score = score_hit(h.text, symbol);
        h.indent = indent_of(h.text);
        const bool dup = std::any_of(hits.begin(), hits.end(), [&h](const SymbolHit& e) {
            return e.line == h.line && e.path == h.path;
        });
        if (!dup) {
            hits.push_back(std::move(h));
        }
    }
    std::stable_sort(hits.begin(), hits.end(), [](const SymbolHit& a, const SymbolHit& b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.indent < b.indent;
    });
    suppressed = hits.size() > limit ? hits.size() - limit : 0;
    if (hits.size() > limit) {
        hits.resize(limit);
    }
    return hits;
}

} // namespace lmp::tools
