#include "src/pcc/diff.hpp"

#include <algorithm>
#include <cstdint>

namespace lmp::pcc {
namespace {

enum class Tag : std::uint8_t { Equal, Delete, Insert };

struct Op {
    Tag tag;
    std::size_t index; // into `a` for Equal and Delete, into `b` for Insert
};

// Longest common subsequence over the trimmed middle, backtracked into an edit script.
// Classic O(n*m); the caller has already checked the product against the budget.
std::vector<Op> align(const std::vector<std::string>& a, const std::vector<std::string>& b,
                      std::size_t lo, std::size_t a_hi, std::size_t b_hi) {
    const std::size_t n = a_hi - lo;
    const std::size_t m = b_hi - lo;
    std::vector<std::uint32_t> table((n + 1) * (m + 1), 0);
    const auto at = [m](std::size_t i, std::size_t j) { return i * (m + 1) + j; };

    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t j = m; j-- > 0;) {
            table[at(i, j)] = a[lo + i] == b[lo + j]
                                  ? table[at(i + 1, j + 1)] + 1
                                  : std::max(table[at(i + 1, j)], table[at(i, j + 1)]);
        }
    }

    std::vector<Op> ops;
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < n && j < m) {
        if (a[lo + i] == b[lo + j]) {
            ops.push_back({Tag::Equal, lo + i++});
            ++j;
        } else if (table[at(i + 1, j)] >= table[at(i, j + 1)]) {
            ops.push_back({Tag::Delete, lo + i++});
        } else {
            ops.push_back({Tag::Insert, lo + j++});
        }
    }
    for (; i < n; ++i) {
        ops.push_back({Tag::Delete, lo + i});
    }
    for (; j < m; ++j) {
        ops.push_back({Tag::Insert, lo + j});
    }
    return ops;
}

// Everything outside the changed region is identical by construction, so trimming it
// first is not an optimisation detail: it is what keeps the quadratic step off the
// common case, where an agent rewrites four lines of a thousand-line file.
struct Trim {
    std::size_t lo = 0;
    std::size_t a_hi = 0;
    std::size_t b_hi = 0;
};

Trim trim_common(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    Trim t;
    const std::size_t limit = std::min(a.size(), b.size());
    while (t.lo < limit && a[t.lo] == b[t.lo]) {
        ++t.lo;
    }
    t.a_hi = a.size();
    t.b_hi = b.size();
    while (t.a_hi > t.lo && t.b_hi > t.lo && a[t.a_hi - 1] == b[t.b_hi - 1]) {
        --t.a_hi;
        --t.b_hi;
    }
    return t;
}

std::vector<Op> build_script(const std::vector<std::string>& a,
                             const std::vector<std::string>& b, const Trim& t,
                             bool& degraded) {
    std::vector<Op> ops;
    for (std::size_t i = 0; i < t.lo; ++i) {
        ops.push_back({Tag::Equal, i});
    }

    const std::size_t cells = (t.a_hi - t.lo) * (t.b_hi - t.lo);
    if (cells > kDiffCellBudget) {
        degraded = true;
        for (std::size_t i = t.lo; i < t.a_hi; ++i) {
            ops.push_back({Tag::Delete, i});
        }
        for (std::size_t j = t.lo; j < t.b_hi; ++j) {
            ops.push_back({Tag::Insert, j});
        }
    } else {
        const std::vector<Op> middle = align(a, b, t.lo, t.a_hi, t.b_hi);
        ops.insert(ops.end(), middle.begin(), middle.end());
    }

    for (std::size_t k = 0; k < a.size() - t.a_hi; ++k) {
        ops.push_back({Tag::Equal, t.a_hi + k});
    }
    return ops;
}

struct Hunk {
    std::size_t begin = 0; // index into the op script
    std::size_t end = 0;
};

// Runs of change, each padded by `context` equal lines, with overlapping pads merged so
// two edits three lines apart render as one hunk rather than two that repeat lines.
std::vector<Hunk> group_hunks(const std::vector<Op>& ops, std::size_t context) {
    std::vector<Hunk> hunks;
    std::size_t i = 0;
    while (i < ops.size()) {
        if (ops[i].tag == Tag::Equal) {
            ++i;
            continue;
        }
        const std::size_t begin = i >= context ? i - context : 0;
        std::size_t end = i;
        std::size_t equal_run = 0;
        while (end < ops.size() && equal_run <= context * 2) {
            equal_run = ops[end].tag == Tag::Equal ? equal_run + 1 : 0;
            ++end;
        }
        end -= equal_run > context ? equal_run - context : 0;
        if (!hunks.empty() && begin <= hunks.back().end) {
            hunks.back().end = end;
        } else {
            hunks.push_back({begin, end});
        }
        i = end;
    }
    return hunks;
}

void render_hunk(std::string& out, const std::vector<Op>& ops, const Hunk& hunk,
                 const std::vector<std::string>& a, const std::vector<std::string>& b) {
    // Where the hunk starts in EACH file, counted from the top of the script.
    //
    // Both numbers used to be read off the first op's `index`, which is an index into `a`
    // for Equal and Delete and into `b` for Insert -- so one value was printed under two
    // labels. They agree only while nothing has been added or removed above the hunk;
    // after a single insertion every later hunk's `+` line number is short by the number
    // of insertions before it. The diff still READS correctly, because the body lines are
    // right, which is why this survived: it is only the header that lies, and only when
    // the file changed length. A consumer that trusts the header -- `patch`, or anything
    // mapping a hunk back to a line in the new file -- lands in the wrong place.
    std::size_t a_start = 0;
    std::size_t b_start = 0;
    for (std::size_t k = 0; k < hunk.begin; ++k) {
        a_start += ops[k].tag != Tag::Insert ? 1 : 0;
        b_start += ops[k].tag != Tag::Delete ? 1 : 0;
    }
    std::size_t a_count = 0;
    std::size_t b_count = 0;
    for (std::size_t k = hunk.begin; k < hunk.end; ++k) {
        a_count += ops[k].tag != Tag::Insert ? 1 : 0;
        b_count += ops[k].tag != Tag::Delete ? 1 : 0;
    }
    // An empty range is numbered by the line it follows, not by the line after it -- the
    // unified-diff convention, and what makes a pure insertion at the top read `-0,0`.
    const std::size_t a_at = a_count == 0 ? a_start : a_start + 1;
    const std::size_t b_at = b_count == 0 ? b_start : b_start + 1;
    out += "@@ -" + std::to_string(a_at) + "," + std::to_string(a_count) + " +" +
           std::to_string(b_at) + "," + std::to_string(b_count) + " @@\n";
    for (std::size_t k = hunk.begin; k < hunk.end; ++k) {
        const Op& op = ops[k];
        const char sign = op.tag == Tag::Equal ? ' ' : (op.tag == Tag::Delete ? '-' : '+');
        out += sign;
        out += op.tag == Tag::Insert ? b[op.index] : a[op.index];
        out += '\n';
    }
}

} // namespace

std::vector<std::string> split_lines(std::string_view text) {
    std::vector<std::string> lines;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t nl = text.find('\n', start);
        if (nl == std::string_view::npos) {
            if (start < text.size()) {
                lines.emplace_back(text.substr(start));
            }
            break;
        }
        lines.emplace_back(text.substr(start, nl - start));
        start = nl + 1;
    }
    return lines;
}

std::string unified_diff(std::string_view a_text, std::string_view b_text,
                         std::string from_label, std::string to_label, int context) {
    if (a_text == b_text) {
        return {};
    }
    const std::vector<std::string> a = split_lines(a_text);
    const std::vector<std::string> b = split_lines(b_text);

    bool degraded = false;
    const Trim trim = trim_common(a, b);
    const std::vector<Op> ops = build_script(a, b, trim, degraded);
    const std::vector<Hunk> hunks =
        group_hunks(ops, static_cast<std::size_t>(std::max(0, context)));

    std::string out = "--- " + from_label + "\n+++ " + to_label + "\n";
    if (degraded) {
        out += "# alignment budget exceeded; shown as one whole-file replacement\n";
    }
    for (const Hunk& hunk : hunks) {
        render_hunk(out, ops, hunk, a, b);
    }
    return out;
}

} // namespace lmp::pcc
