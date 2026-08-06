#pragma once
//
// Exact structured patch application for the `apply_patch` tool.
//
// Format is a Qwen-friendly freeform block (V4A / *** Begin Patch), meant to travel as
// raw multiline text inside XML tool params — no JSON escaping of the hunk body.
//
// CONTRACT (deliberately stricter than graft):
//   * Exact context/preimage matching only. No whitespace flex, no fuzzy apply.
//   * All hunks for a file apply, or none of them do (working copy is discarded).
//   * CRLF vs LF and a missing final newline are preserved from the file on disk.
//   * Similarity never authorizes a write; failures name the hunk and nearby lines.
//
#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace apply_patch {

enum class FileOpKind { Add, Update, Delete };

enum class Status {
    Applied,
    ParseError,
    NoMatch,
    Ambiguous,
    Conflict, // e.g. Add when the path already exists
};

struct HunkLine {
    char tag = ' '; // ' ' context, '-' remove, '+' add
    std::string text; // without the leading tag; no newline
};

struct Hunk {
    std::vector<HunkLine> lines;
};

struct FileOp {
    FileOpKind kind = FileOpKind::Update;
    std::string path;
    std::vector<Hunk> hunks; // empty for Delete; Add may use one synthetic hunk of + lines
};

struct Failure {
    std::string path;
    std::size_t hunk_index = 0; // 0-based within the file op; npos-ish = whole-file
    std::string reason;
    std::string nearby; // current nearby lines for diagnostics
};

struct FileChange {
    FileOpKind kind = FileOpKind::Update;
    std::string path;
    std::string new_content; // empty for Delete
    bool delete_file = false;
};

struct Result {
    Status status = Status::ParseError;
    std::vector<FileOp> ops;
    std::vector<FileChange> changes; // filled only on Applied
    Failure failure;
};

namespace detail {

[[nodiscard]] inline bool starts_with(std::string_view s, std::string_view p) noexcept {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

[[nodiscard]] inline std::string_view trim_right(std::string_view s) noexcept {
    while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
        s.remove_suffix(1);
    }
    return s;
}

[[nodiscard]] inline std::string_view strip_cr(std::string_view s) noexcept {
    if (!s.empty() && s.back() == '\r') {
        s.remove_suffix(1);
    }
    return s;
}

[[nodiscard]] inline bool file_uses_crlf(std::string_view text) noexcept {
    const std::size_t nl = text.find('\n');
    if (nl == std::string_view::npos || nl == 0) {
        return false;
    }
    return text[nl - 1] == '\r';
}

[[nodiscard]] inline bool ends_with_newline(std::string_view text) noexcept {
    return !text.empty() && text.back() == '\n';
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
        out.push_back(text.substr(pos, nl - pos));
        pos = nl + 1;
    }
}

[[nodiscard]] inline std::size_t count_lines_before(std::string_view text,
                                                    std::size_t byte_off) noexcept {
    std::size_t line = 1;
    for (std::size_t i = 0; i < byte_off && i < text.size(); ++i) {
        if (text[i] == '\n') {
            ++line;
        }
    }
    return line;
}

[[nodiscard]] inline std::string nearby_lines(std::string_view file, std::size_t around_line,
                                              std::size_t radius = 5) {
    std::vector<std::string_view> lines;
    split_lines(file, lines);
    if (lines.empty()) {
        return "(empty file)\n";
    }
    const std::size_t focus = around_line == 0 ? 1 : around_line;
    const std::size_t begin = focus > radius + 1 ? focus - radius : 1;
    const std::size_t end = std::min(lines.size(), focus + radius);
    std::string out;
    for (std::size_t i = begin; i <= end; ++i) {
        out += std::to_string(i);
        out += '\t';
        out.append(strip_cr(lines[i - 1]));
        out += '\n';
    }
    return out;
}

// Build search/replace bodies for one hunk. Search is context + removals; replace is
// context + additions. Bodies use '\n' separators and no trailing newline unless the
// hunk itself encodes one via an empty final tagged line (not used here).
inline bool build_hunk_bodies(const Hunk& hunk, std::string& search, std::string& replace,
                              std::string& err) {
    search.clear();
    replace.clear();
    if (hunk.lines.empty()) {
        err = "empty hunk";
        return false;
    }
    bool first_s = true;
    bool first_r = true;
    for (const HunkLine& hl : hunk.lines) {
        if (hl.tag != ' ' && hl.tag != '-' && hl.tag != '+') {
            err = "hunk line must start with ' ', '-', or '+'";
            return false;
        }
        if (hl.tag == ' ' || hl.tag == '-') {
            if (!first_s) {
                search.push_back('\n');
            }
            search += hl.text;
            first_s = false;
        }
        if (hl.tag == ' ' || hl.tag == '+') {
            if (!first_r) {
                replace.push_back('\n');
            }
            replace += hl.text;
            first_r = false;
        }
    }
    return true;
}

// Adapt a '\n'-joined body to the file's EOL. Does not add a trailing newline.
[[nodiscard]] inline std::string with_eol(std::string_view body, bool crlf) {
    if (!crlf) {
        return std::string(body);
    }
    std::string out;
    out.reserve(body.size() + body.size() / 16);
    for (std::size_t i = 0; i < body.size(); ++i) {
        if (body[i] == '\n') {
            out += "\r\n";
        } else if (body[i] == '\r') {
            // ignore; next \n will emit CRLF
        } else {
            out.push_back(body[i]);
        }
    }
    return out;
}

// Exact locate of `needle` in `hay`. When needle is empty, only valid for pure-addition
// hunks at EOF (caller handles). Counts occurrences; Ambiguous if >1.
struct Locate {
    Status status = Status::NoMatch;
    std::size_t offset = 0;
};

[[nodiscard]] inline Locate locate_exact(std::string_view hay, std::string_view needle) {
    Locate loc;
    if (needle.empty()) {
        loc.status = Status::NoMatch;
        return loc;
    }
    std::size_t count = 0;
    std::size_t first = 0;
    std::size_t p = hay.find(needle);
    while (p != std::string_view::npos) {
        if (count == 0) {
            first = p;
        }
        ++count;
        p = hay.find(needle, p + needle.size());
    }
    if (count == 0) {
        loc.status = Status::NoMatch;
        return loc;
    }
    if (count > 1) {
        loc.status = Status::Ambiguous;
        loc.offset = first;
        return loc;
    }
    loc.status = Status::Applied;
    loc.offset = first;
    return loc;
}

inline void splice(std::string& file, std::size_t begin, std::size_t end,
                   std::string_view replacement) {
    file.replace(begin, end - begin, replacement);
}

[[nodiscard]] inline std::optional<std::string> path_after_prefix(std::string_view line,
                                                                   std::string_view prefix) {
    if (!starts_with(line, prefix)) {
        return std::nullopt;
    }
    std::string_view rest = trim_right(line.substr(prefix.size()));
    while (!rest.empty() && (rest.front() == ' ' || rest.front() == '\t')) {
        rest.remove_prefix(1);
    }
    if (rest.empty()) {
        return std::nullopt;
    }
    return std::string(rest);
}

} // namespace detail

// Parse a freeform patch. Accepts with or without the Begin/End sentinels.
[[nodiscard]] inline Result parse(std::string_view patch) {
    Result res;
    using namespace detail;

    std::vector<std::string_view> raw;
    split_lines(patch, raw);
    // Drop a leading BOM-ish empty and tolerate missing sentinels.
    std::size_t i = 0;
    while (i < raw.size() && strip_cr(raw[i]).empty()) {
        ++i;
    }
    if (i < raw.size() && starts_with(strip_cr(raw[i]), "*** Begin Patch")) {
        ++i;
    }

    FileOp* current = nullptr;
    bool in_hunk = false;
    Hunk* hunk = nullptr;

    auto fail_parse = [&](std::string why) -> Result {
        res.status = Status::ParseError;
        res.failure.reason = std::move(why);
        res.ops.clear();
        return res;
    };

    for (; i < raw.size(); ++i) {
        const std::string_view line = strip_cr(raw[i]);
        if (starts_with(line, "*** End Patch")) {
            break;
        }
        if (starts_with(line, "*** Add File:")) {
            auto path = path_after_prefix(line, "*** Add File:");
            if (!path) {
                return fail_parse("Add File header missing path");
            }
            res.ops.push_back(FileOp{FileOpKind::Add, *path, {}});
            current = &res.ops.back();
            current->hunks.push_back(Hunk{});
            hunk = &current->hunks.back();
            in_hunk = true;
            continue;
        }
        if (starts_with(line, "*** Delete File:")) {
            auto path = path_after_prefix(line, "*** Delete File:");
            if (!path) {
                return fail_parse("Delete File header missing path");
            }
            res.ops.push_back(FileOp{FileOpKind::Delete, *path, {}});
            current = &res.ops.back();
            hunk = nullptr;
            in_hunk = false;
            continue;
        }
        if (starts_with(line, "*** Update File:")) {
            auto path = path_after_prefix(line, "*** Update File:");
            if (!path) {
                return fail_parse("Update File header missing path");
            }
            res.ops.push_back(FileOp{FileOpKind::Update, *path, {}});
            current = &res.ops.back();
            hunk = nullptr;
            in_hunk = false;
            continue;
        }
        if (current == nullptr) {
            if (line.empty()) {
                continue;
            }
            return fail_parse("patch content before any file header");
        }
        if (line == "@@" || starts_with(line, "@@ ")) {
            if (current->kind == FileOpKind::Delete) {
                return fail_parse("Delete File cannot carry hunks");
            }
            current->hunks.push_back(Hunk{});
            hunk = &current->hunks.back();
            in_hunk = true;
            continue;
        }
        if (current->kind == FileOpKind::Delete) {
            if (line.empty()) {
                continue;
            }
            return fail_parse("Delete File cannot carry content lines");
        }
        // Add File without @@: treat every + line as the new file body.
        if (!in_hunk) {
            if (current->kind == FileOpKind::Add &&
                (line.empty() || line.front() == '+' || line.front() == ' ')) {
                if (current->hunks.empty()) {
                    current->hunks.push_back(Hunk{});
                }
                hunk = &current->hunks.back();
                in_hunk = true;
            } else if (line.empty()) {
                continue;
            } else {
                return fail_parse("Update File content before @@ hunk marker at " +
                                  current->path);
            }
        }
        if (hunk == nullptr) {
            return fail_parse("internal: hunk cursor missing");
        }
        if (line.empty()) {
            // Blank line between a finished hunk and the next `***` header is a separator,
            // not an empty context line. Peek ahead.
            std::size_t j = i + 1;
            while (j < raw.size() && strip_cr(raw[j]).empty()) {
                ++j;
            }
            if (j < raw.size() && starts_with(strip_cr(raw[j]), "***")) {
                in_hunk = false;
                hunk = nullptr;
                continue;
            }
            // An empty line inside a hunk is context of an empty line.
            hunk->lines.push_back(HunkLine{' ', std::string()});
            continue;
        }
        const char tag = line.front();
        if (tag != ' ' && tag != '-' && tag != '+') {
            // Bare lines under Add File are treated as additions (model often omits '+').
            if (current->kind == FileOpKind::Add) {
                hunk->lines.push_back(HunkLine{'+', std::string(line)});
                continue;
            }
            return fail_parse("hunk line must start with ' ', '-', or '+' in " +
                              current->path);
        }
        if (current->kind == FileOpKind::Add && tag == '-') {
            return fail_parse("Add File cannot contain '-' lines");
        }
        hunk->lines.push_back(HunkLine{tag, std::string(line.substr(1))});
    }

    if (res.ops.empty()) {
        return fail_parse("patch contains no file operations");
    }
    for (const FileOp& op : res.ops) {
        if (op.path.empty()) {
            return fail_parse("empty path in file operation");
        }
        if (op.kind == FileOpKind::Update && op.hunks.empty()) {
            return fail_parse("Update File has no hunks: " + op.path);
        }
        if (op.kind == FileOpKind::Add && op.hunks.empty()) {
            return fail_parse("Add File has no content: " + op.path);
        }
    }
    res.status = Status::Applied; // parse ok; apply() may still refuse
    return res;
}

// Apply parsed ops against in-memory file contents. `read_file` returns nullopt when the
// path is absent. Does not touch the filesystem.
[[nodiscard]] inline Result apply_to(
    const std::vector<FileOp>& ops,
    const std::function<std::optional<std::string>(const std::string& path)>& read_file) {
    Result res;
    res.ops = ops;
    res.status = Status::Applied;

    for (const FileOp& op : ops) {
        if (op.kind == FileOpKind::Delete) {
            const auto cur = read_file(op.path);
            if (!cur) {
                res.status = Status::NoMatch;
                res.failure.path = op.path;
                res.failure.reason = "Delete File: path not found";
                res.changes.clear();
                return res;
            }
            FileChange ch;
            ch.kind = FileOpKind::Delete;
            ch.path = op.path;
            ch.delete_file = true;
            res.changes.push_back(std::move(ch));
            continue;
        }

        if (op.kind == FileOpKind::Add) {
            const auto cur = read_file(op.path);
            if (cur) {
                res.status = Status::Conflict;
                res.failure.path = op.path;
                res.failure.reason = "Add File: path already exists";
                res.failure.nearby = detail::nearby_lines(*cur, 1);
                res.changes.clear();
                return res;
            }
            std::string body;
            bool first = true;
            for (const Hunk& h : op.hunks) {
                for (const HunkLine& hl : h.lines) {
                    if (hl.tag == '-') {
                        res.status = Status::ParseError;
                        res.failure.path = op.path;
                        res.failure.reason = "Add File cannot contain '-' lines";
                        res.changes.clear();
                        return res;
                    }
                    if (hl.tag != '+' && hl.tag != ' ') {
                        continue;
                    }
                    if (!first) {
                        body.push_back('\n');
                    }
                    body += hl.text;
                    first = false;
                }
            }
            // New files get a trailing newline when they have content — standard for text.
            if (!body.empty()) {
                body.push_back('\n');
            }
            FileChange ch;
            ch.kind = FileOpKind::Add;
            ch.path = op.path;
            ch.new_content = std::move(body);
            res.changes.push_back(std::move(ch));
            continue;
        }

        // Update
        const auto cur = read_file(op.path);
        if (!cur) {
            res.status = Status::NoMatch;
            res.failure.path = op.path;
            res.failure.reason = "Update File: path not found";
            res.changes.clear();
            return res;
        }
        std::string working = *cur;
        const bool crlf = detail::file_uses_crlf(working);
        const bool had_final_nl = detail::ends_with_newline(working);

        for (std::size_t hi = 0; hi < op.hunks.size(); ++hi) {
            std::string search_lf;
            std::string replace_lf;
            std::string err;
            if (!detail::build_hunk_bodies(op.hunks[hi], search_lf, replace_lf, err)) {
                res.status = Status::ParseError;
                res.failure.path = op.path;
                res.failure.hunk_index = hi;
                res.failure.reason = err;
                res.changes.clear();
                return res;
            }
            // Pure addition: empty search means insert replace at end of file (before
            // preserving final-newline rules). Still exact — no fuzzy anchor.
            if (search_lf.empty()) {
                if (replace_lf.empty()) {
                    res.status = Status::ParseError;
                    res.failure.path = op.path;
                    res.failure.hunk_index = hi;
                    res.failure.reason = "hunk removes and adds nothing";
                    res.changes.clear();
                    return res;
                }
                std::string rep = detail::with_eol(replace_lf, crlf);
                if (had_final_nl || working.empty()) {
                    if (!working.empty() && !detail::ends_with_newline(working)) {
                        working += crlf ? "\r\n" : "\n";
                    }
                    working += rep;
                    if (had_final_nl) {
                        working += crlf ? "\r\n" : "\n";
                    }
                } else {
                    working += crlf ? "\r\n" : "\n";
                    working += rep;
                }
                continue;
            }

            const std::string search = detail::with_eol(search_lf, crlf);
            const std::string replace = detail::with_eol(replace_lf, crlf);
            // Also try LF search against a CRLF file's LF-normalized view? No — exact
            // only. But models author LF patches; with_eol already adapted search to CRLF
            // when the file uses CRLF, so find works on the real bytes.

            // If the file has no final newline and the search would include one at EOF,
            // try both with and without a trailing EOL on the needle when the match is
            // at end — still exact candidates, not fuzzy.
            detail::Locate loc = detail::locate_exact(working, search);
            if (loc.status == Status::NoMatch && !detail::ends_with_newline(search)) {
                // Try search + file EOL in case the hunk omitted the final newline but
                // the region is mid-file. Mid-file lines always have EOL in the file
                // between them; build_hunk_bodies joins with \n, and with_eol adapted.
                // Nothing more to try without becoming fuzzy.
            }
            if (loc.status == Status::NoMatch) {
                // Last resort exact variant: if working has no final newline and search
                // is a suffix of working without needing a trailing EOL.
                res.status = Status::NoMatch;
                res.failure.path = op.path;
                res.failure.hunk_index = hi;
                res.failure.reason =
                    "hunk " + std::to_string(hi + 1) + " context/preimage not found exactly";
                // Approximate focus: first non-empty search line.
                std::size_t focus = 1;
                const std::size_t nl = search_lf.find('\n');
                const std::string_view first_line(
                    search_lf.data(),
                    nl == std::string::npos ? search_lf.size() : nl);
                if (!first_line.empty()) {
                    const std::string needle = detail::with_eol(first_line, crlf);
                    const std::size_t at = working.find(needle);
                    if (at != std::string::npos) {
                        focus = detail::count_lines_before(working, at);
                    }
                }
                res.failure.nearby = detail::nearby_lines(working, focus);
                res.changes.clear();
                return res;
            }
            if (loc.status == Status::Ambiguous) {
                res.status = Status::Ambiguous;
                res.failure.path = op.path;
                res.failure.hunk_index = hi;
                res.failure.reason =
                    "hunk " + std::to_string(hi + 1) +
                    " matches more than one site; add more context lines";
                res.failure.nearby =
                    detail::nearby_lines(working, detail::count_lines_before(working, loc.offset));
                res.changes.clear();
                return res;
            }
            detail::splice(working, loc.offset, loc.offset + search.size(), replace);
        }

        // Preserve "no final newline" when the original lacked one and the patch did not
        // clearly introduce a trailing blank through a final context line. If working
        // gained a trailing EOL only from our splice mechanics on a no-final-nl file,
        // strip a single trailing EOL when the original lacked one and the last hunk
        // replace body did not end with an empty line intent.
        if (!had_final_nl && detail::ends_with_newline(working)) {
            // Only strip when the file previously had content; empty stays empty.
            if (crlf && working.size() >= 2 &&
                working[working.size() - 2] == '\r' && working.back() == '\n') {
                // Keep if replace intentionally ended with newline-marked empty — we
                // cannot recover intent perfectly; prefer preserving original no-final-nl
                // when the resulting body (minus one EOL) still contains the last line.
                working.resize(working.size() - 2);
            } else if (!working.empty() && working.back() == '\n') {
                working.pop_back();
            }
        } else if (had_final_nl && !detail::ends_with_newline(working) && !working.empty()) {
            working += crlf ? "\r\n" : "\n";
        }

        FileChange ch;
        ch.kind = FileOpKind::Update;
        ch.path = op.path;
        ch.new_content = std::move(working);
        res.changes.push_back(std::move(ch));
    }
    return res;
}

[[nodiscard]] inline Result apply(
    std::string_view patch,
    const std::function<std::optional<std::string>(const std::string& path)>& read_file) {
    Result parsed = parse(patch);
    if (parsed.status != Status::Applied) {
        return parsed;
    }
    return apply_to(parsed.ops, read_file);
}

[[nodiscard]] inline std::vector<std::string> paths_in_patch(std::string_view patch) {
    Result parsed = parse(patch);
    std::vector<std::string> out;
    out.reserve(parsed.ops.size());
    for (const FileOp& op : parsed.ops) {
        out.push_back(op.path);
    }
    return out;
}

} // namespace apply_patch
