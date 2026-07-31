#ifndef TOOLS_LOG_TRIAGE_HPP
#define TOOLS_LOG_TRIAGE_HPP
//
// Compacts captured command output so that what an agent needs to ACT survives a byte cap.
//
// This is the consolidated engine from the log-triage cookoff (round 1, 2026-07-30, 15
// entrants at cat-collector-king/log-triage-engine). What was taken from whom, and what was
// rejected, is recorded at the bottom of this file. The benchmark it is measured on lives at
// bakeoff/log_triage/ and its answer key was written by the compiler, not by us.
//
// The problem, stated once. A build fails. The shell tool captures 12 MB of output. The
// model gets 8 KB. If the 40 bytes saying WHICH LINE OF ITS OWN CODE to edit are not in that
// 8 KB, the agent cannot fix the build -- and it will not say so, it will guess. That is the
// harness-blindness failure: "the model could not fix the build" when the compiler error
// never reached it.
//
// Three principles, each earned from a case in the corpus:
//
//   1. AN ANCHOR BEATS ITS CONTEXT. Every distinct diagnostic gets its locator and message
//      before any diagnostic gets its source-and-caret block. bare_error_limit has 19
//      diagnostics and a 2048-byte cap; spending it all on the first three is the wrong
//      answer, and it is the answer every context-first implementation gives.
//
//   2. A DIAGNOSTIC IN YOUR OWN CODE BEATS ONE IN A SYSTEM HEADER, WHATEVER ITS LEVEL.
//      build_template_deep is 68 KB of libc++ backtrace for one error. Every `error:` line
//      is inside libc++. The only line naming a file the agent can edit is a
//      `note: in instantiation of ... requested here`. An errors-only matcher keeps the
//      message and throws away the address.
//
//   3. CONTEXT IS PROXIMITY, NOT PATTERN. The lines under a diagnostic are worth keeping
//      because they are NEAR it, not because they match something. That is the one idea
//      worth taking whole from the cookoff (e05): it keeps clang's caret block, Swift's
//      numbered source window, rustc's `|` gutter and Python's indented frame source
//      without knowing anything about any of them.
//
#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace log_triage {

namespace detail {

// --- scoring constants -----------------------------------------------------
// Ratios, not absolutes: the packer only ever compares them to each other.
inline constexpr int kLocalAnchor = 1000;  // a diagnostic naming a file we can edit
inline constexpr int kAnchor = 600;        // a diagnostic naming a system file, or none
inline constexpr int kOutcome = 300;       // "3 errors generated", "make: *** Error 1"
inline constexpr int kSystemNote = 320;    // a `note:` inside a file the agent cannot edit
inline constexpr int kWarning = 40;        // kept only if the budget is generous
inline constexpr int kOrdinary = 10;
inline constexpr int kNoise = 0;           // compiler invocations, progress percentages

// How far a diagnostic's importance reaches, and how fast it decays, for lines that are
// not part of its rendered block.
inline constexpr int kProximityReach = 6;
inline constexpr int kProximityWeight = 180;

[[nodiscard]] inline bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }

// A line belonging to a diagnostic's rendered block: the numbered-source and caret gutter
// that clang, swiftc and rustc all draw as `NNN | source` and `    | ^~~~`, or a bare run
// of carets and tildes.
//
// A contiguous run of these attached to an anchor inherits the anchor's importance in FULL,
// however long it is, rather than decaying with distance. The distance rule alone was
// calibrated on clang and Swift and a comment here claimed 6 lines "covers the widest block
// any tool prints" -- the held-out set falsified that: rustc draws a nine-line gutter with
// the type annotations, the `expected due to this` note and a `help:` suggestion all inside
// it, and the tail of the block was scoring like ordinary text. Structure is the right
// boundary for a block; a radius is a guess about one.
[[nodiscard]] inline bool is_block_line(std::string_view line) noexcept {
    const std::size_t first = line.find_first_not_of(" \t");
    if (first == std::string_view::npos) {
        return false;
    }
    std::size_t i = first;
    while (i < line.size() && is_digit(line[i])) {
        ++i;
    }
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) {
        ++i;
    }
    if (i < line.size() && line[i] == '|') {
        return true;
    }
    // A bare caret row: clang without line numbers, and rustc's `help:` underlines.
    const std::string_view rest = line.substr(first);
    return rest.find_first_not_of("~^+- ") == std::string_view::npos;
}

// Byte reservation for one elision marker, so the packer's running total is an UPPER bound
// on the rendered size -- which is what lets the budget check be O(1) per line instead of
// e05's O(n) rebuild.
//
// It must be TIGHT, not merely safe. A first cut reserved a flat 48 bytes against a marker
// that renders as 23 + digits, and the slack compounded: a selection with 30 gaps reserved
// 700 bytes it would never spend, which on a 2048-byte budget stopped the packer with a
// third of the budget unused and cost bare_error_limit every one of its 38 context lines.
// An over-cautious bound is not free.
inline constexpr std::size_t kMarkerFixed =
    sizeof("[... ") - 1 + sizeof(" lines elided ...]\n") - 1;

[[nodiscard]] inline std::size_t digits(std::size_t n) noexcept {
    std::size_t d = 1;
    while (n >= 10) {
        n /= 10;
        ++d;
    }
    return d;
}

[[nodiscard]] inline bool starts_with(std::string_view s, std::string_view p) noexcept {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

[[nodiscard]] inline bool contains(std::string_view s, std::string_view n) noexcept {
    return s.find(n) != std::string_view::npos;
}

// Case-insensitive search, ASCII only. Compiler output is ASCII in its structural parts;
// identifiers may not be, and are not searched for.
[[nodiscard]] inline bool contains_ci(std::string_view s, std::string_view needle) noexcept {
    if (needle.empty() || s.size() < needle.size()) {
        return false;
    }
    const auto lower = [](char c) noexcept -> char {
        return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    };
    for (std::size_t i = 0; i + needle.size() <= s.size(); ++i) {
        std::size_t j = 0;
        while (j < needle.size() && lower(s[i + j]) == lower(needle[j])) {
            ++j;
        }
        if (j == needle.size()) {
            return true;
        }
    }
    return false;
}

// --- locators --------------------------------------------------------------
//
// A locator is the part of a diagnostic that says WHERE. Four shapes cover every tool in
// the corpus, and they are matched structurally rather than by tool name, so a tool nobody
// has run yet gets the same treatment:
//
//   clang / swiftc / javac   path:LINE:COL:   and   path:LINE:
//   rustc                     --> path:LINE:COL      (on its own line, under the message)
//   python                   File "path", line N
//   pytest                   path:LINE: message
//
// Returns the offset just past the locator, or npos.
[[nodiscard]] inline std::size_t find_locator_end(std::string_view line) noexcept {
    // ` --> path:LINE:COL` -- rustc puts this on the line AFTER its message, which is why
    // an implementation that keeps only the line its matcher fired on loses the address.
    const std::size_t arrow = line.find("--> ");
    if (arrow != std::string_view::npos && line.find_first_not_of(" \t") == arrow) {
        return line.size();
    }
    // `File "path", line N` -- Python. Its own shape entirely; no colons involved.
    const std::size_t file_kw = line.find("File \"");
    if (file_kw != std::string_view::npos && contains(line.substr(file_kw), "\", line ")) {
        return line.size();
    }
    // `path:LINE:` or `path:LINE:COL:`. Scan for a colon followed by digits followed by a
    // colon. Requiring the digits stops this firing on a bare `http://` or a Windows drive.
    for (std::size_t i = 0; i + 2 < line.size(); ++i) {
        if (line[i] != ':' || !is_digit(line[i + 1])) {
            continue;
        }
        std::size_t j = i + 1;
        while (j < line.size() && is_digit(line[j])) {
            ++j;
        }
        if (j >= line.size() || line[j] != ':') {
            continue;
        }
        // Optional second number (the column).
        std::size_t k = j + 1;
        while (k < line.size() && is_digit(line[k])) {
            ++k;
        }
        if (k > j + 1 && k < line.size() && line[k] == ':') {
            return k + 1;
        }
        return j + 1;
    }
    return std::string_view::npos;
}

// Is this locator's path one the agent can edit?
//
// Textual, and deliberately so: nothing is stat'ed, because the log may have been captured
// on another machine or in a container. A path is treated as the toolchain's when it sits
// under a well-known toolchain root. Everything else -- including every relative path,
// which is what a build prints for its own sources -- is treated as the agent's.
//
// Being wrong in the agent's favour costs a little budget. Being wrong the other way loses
// the only line that says where to type, so the default leans local.
[[nodiscard]] inline bool locator_is_local(std::string_view line) noexcept {
    static constexpr std::string_view toolchain_roots[] = {
        "/Applications/Xcode", "/Library/Developer", "/usr/include", "/usr/lib",
        "/usr/local/include", "/System/", "/opt/homebrew/include", "/include/c++/",
        "/.rustup/", "/toolchains/", "/site-packages/", "/lib/python3",
        "/Toolchains/XcodeDefault", "/SDKs/MacOSX"};
    for (std::string_view root : toolchain_roots) {
        if (contains(line, root)) {
            return false;
        }
    }
    return true;
}

// --- line classification ---------------------------------------------------

// A line the build printed to describe what it is DOING, not what went wrong: a compiler
// invocation, a progress percentage, a make recursion notice. These are the bulk of a real
// build log and they carry no information the agent can act on -- but they routinely
// contain the substring "error" (`-Werror`, a path like `src/error_handling.cpp`), so they
// must be recognised BEFORE severity keywords are looked for.
[[nodiscard]] inline bool is_build_noise(std::string_view line) noexcept {
    const std::size_t first = line.find_first_not_of(" \t");
    if (first == std::string_view::npos) {
        return true; // blank
    }
    const std::string_view t = line.substr(first);
    if (t[0] == '[' && contains(t.substr(0, 8), "%]")) {
        return true; // "[ 42%] Building CXX object ..."
    }
    // clang's include stack. It names a file and a line, so find_locator_end fires on it and
    // it was being scored as a full anchor -- kLocalAnchor when the header happens to sit in
    // the project. These lines are 150-200 bytes of SDK path each and a template blow-up
    // emits dozens: on build_template_deep at 2048 they consumed the whole budget and cost
    // the case its primary locator, its message and all four context lines. They announce
    // nothing wrong; the diagnostic they precede does. e08 was the only entrant to recognise
    // them (as its TEMPLATE_INST state) and the round-1 merge did not take it.
    if (starts_with(t, "In file included from ")) {
        return true;
    }
    if (starts_with(t, "make[") || starts_with(t, "make ") ||
        starts_with(t, "cd /") || starts_with(t, "Running `") ||
        starts_with(t, "-- ") || starts_with(t, "Compiling ") ||
        starts_with(t, "Building for") || starts_with(t, "collected ") ||
        starts_with(t, "cachedir:") || starts_with(t, "rootdir:") ||
        starts_with(t, "platform ") || starts_with(t, "plugins:")) {
        return true;
    }
    // A full compiler command line: an absolute path to a tool, with -c and -o in it.
    if (t[0] == '/' && contains(t, " -o ")) {
        return true;
    }
    // "Test #12: pass_3 ... Passed" and "[ 12%] Built target x"
    if (contains(t, "...   Passed") || contains(t, "Built target")) {
        return true;
    }
    return false;
}

// The build's verdict. Not a diagnostic -- it names no location -- but it is what tells the
// agent whether the thing failed at all, and it is cheap.
[[nodiscard]] inline bool is_outcome(std::string_view line) noexcept {
    return contains(line, "errors generated") || contains(line, "error generated") ||
           contains(line, "Error 1") || contains(line, "*** [") ||
           contains(line, "FAILED") || contains(line, "failed,") ||
           contains(line, "aborting due to") || contains(line, "BUILD FAILED") ||
           contains(line, "tests passed") || contains(line, "Errors while running");
}

// Does this line announce something WRONG? Severity keywords, plus the markers of tools
// that do not use the word "error" at all -- the linker, Python, ctest, pytest.
[[nodiscard]] inline bool has_severity(std::string_view line) noexcept {
    static constexpr std::string_view markers[] = {
        "error", "fatal", "Undefined symbol", "undefined reference", "Traceback",
        "Assertion failed", "panic", "Exception", "exception", "SIGSEGV", "Segmentation",
        "cannot find", "ld: ", "E   ", "(Failed)"};
    for (std::string_view m : markers) {
        if (contains(line, m)) {
            return true;
        }
    }
    return contains_ci(line, "error");
}

[[nodiscard]] inline bool is_warning(std::string_view line) noexcept {
    return contains(line, "warning:") || contains(line, "Warning:") ||
           contains(line, "WARNING");
}

// --- normalisation ---------------------------------------------------------

// Strip ANSI escapes and collapse carriage-return progress bars, keeping only each line's
// final state. Both give budget back rather than spending it on bytes the model cannot use:
// a 28 KB colourised log is ~1 KB of escape sequences, and a progress bar that redrew 2000
// times is 2000 copies of one line.
//
// Only ever applied on the COMPACTION path. A log that already fits is returned untouched.
inline void normalise(std::string_view in, std::string& out) {
    out.clear();
    out.reserve(in.size());
    std::size_t line_start = 0;
    for (std::size_t i = 0; i < in.size();) {
        const char c = in[i];
        if (c == '\x1b' && i + 1 < in.size() && in[i + 1] == '[') {
            std::size_t j = i + 2;
            while (j < in.size() && ((in[j] >= '0' && in[j] <= '9') || in[j] == ';')) {
                ++j;
            }
            i = (j < in.size()) ? j + 1 : j;
            continue;
        }
        if (c == '\r') {
            if (i + 1 < in.size() && in[i + 1] == '\n') {
                ++i; // \r\n: let the \n do the work
                continue;
            }
            out.resize(line_start); // a redrawn line replaces what came before it
            ++i;
            continue;
        }
        out.push_back(c);
        if (c == '\n') {
            line_start = out.size();
        }
        ++i;
    }
}

struct Line {
    std::string_view text;
    int score = 0;
    bool anchor = false; // carries a locator or a severity marker: kept before any context
    bool diagnostic = false; // an anchor for a real diagnostic, not a warning's address
    bool duplicate = false; // an anchor repeating a message already kept
    bool selected = false;
};

// The severity a bare-locator line inherits from the diagnostic it belongs to.
//
// Walks back over the block gutter to the first line that says what kind of thing this is.
// A warning's address is worth a warning; only an error's address is worth an anchor. When
// nothing is found within a few lines -- a locator with no visible owner -- the cautious
// reading wins and it is treated as a diagnostic, because the cost of demoting a real error
// is losing the only line that says where to type.
[[nodiscard]] inline int inherited_score(const std::vector<Line>& lines, std::size_t i,
                                         bool local) {
    constexpr std::size_t kLookback = 4;
    for (std::size_t back = 1; back <= kLookback && back <= i; ++back) {
        const std::string_view parent = lines[i - back].text;
        if (is_block_line(parent)) {
            continue;
        }
        if (is_warning(parent)) {
            return kWarning;
        }
        if (has_severity(parent)) {
            break;
        }
        break;
    }
    return local ? kLocalAnchor : kAnchor;
}

} // namespace detail

// Compact `full` to at most `budget_bytes`.
//
// Pure and thread-safe: no globals, no I/O, no static mutable state. Linear in the size of
// the input apart from one sort -- which matters, because this runs on every shell tool
// result and a runaway build emits megabytes. The cookoff's highest-scoring entrant took
// 19 seconds on a 12 MB log because its packer rebuilt the whole output for every candidate
// line; this one takes hundredths.
[[nodiscard]] inline std::string compact(std::string_view full, std::size_t budget_bytes) {
    using namespace detail;

    // A log that already fits comes back byte for byte. Rewriting it destroys information
    // for no gain, and callers do compare.
    if (full.size() <= budget_bytes) {
        return std::string(full);
    }
    if (budget_bytes == 0) {
        return {};
    }

    std::string normalised;
    normalise(full, normalised);
    if (normalised.size() <= budget_bytes) {
        return normalised;
    }

    // --- split ---------------------------------------------------------------
    std::vector<Line> lines;
    lines.reserve(normalised.size() / 48 + 8);
    for (std::size_t pos = 0; pos < normalised.size();) {
        const std::size_t nl = normalised.find('\n', pos);
        const std::size_t end = (nl == std::string::npos) ? normalised.size() : nl;
        // Named member only: every other field has a default member initialiser, and the
        // positional form silently re-bound its trailing `false` to a different member the
        // moment `diagnostic` was inserted into the struct.
        lines.push_back(Line{.text = std::string_view(normalised).substr(pos, end - pos)});
        pos = (nl == std::string::npos) ? normalised.size() : nl + 1;
    }
    if (lines.empty()) {
        return {};
    }

    // --- classify ------------------------------------------------------------
    for (Line& ln : lines) {
        if (is_build_noise(ln.text)) {
            ln.score = kNoise;
            continue;
        }
        const bool located = find_locator_end(ln.text) != std::string_view::npos;
        const bool severe = has_severity(ln.text);
        if (located && (severe || is_warning(ln.text))) {
            // A located warning is still only a warning: it is an anchor for proximity
            // purposes but must never outrank a real diagnostic for budget.
            if (is_warning(ln.text) && !severe) {
                ln.score = kWarning;
                continue;
            }
            ln.anchor = true;
            ln.score = locator_is_local(ln.text) ? kLocalAnchor : kAnchor;
        } else if (located) {
            // A bare locator with no severity word: rustc's ` --> path:L:C` and Python's
            // `File "x.py", line N`. It IS the address, but it says nothing about how bad
            // the thing at that address is -- its severity belongs to the line ABOVE it.
            //
            // Scoring these as top-priority anchors on sight is wrong, and expensively so:
            // rustc prints the same ` --> path:L:C` under a `warning:` as under an
            // `error:`, so a crate with 120 unused-variable warnings produced 120 lines
            // that outranked the real errors' source blocks. On the held-out set that
            // filled 16 KB with warning addresses and cost the case ten of its fourteen
            // context lines. A continuation line inherits; it does not assert.
            ln.anchor = true;
            const bool local = locator_is_local(ln.text);
            // A `note:` in a file the agent cannot edit ranks BELOW a real diagnostic in one.
            // Principle 2 ranks by locality; within the system tier it did not rank by
            // severity at all, so libc++'s instantiation backtrace -- twenty
            // `note: in instantiation of ... requested here` lines, 200 bytes of SDK path
            // each -- tied with the one `error:` at kAnchor and crowded it out of
            // build_template_deep entirely. Local notes are untouched: they are what solves
            // build_no_matching_ctor, where every candidate signature is a note.
            if (!local && contains(ln.text, "note:")) {
                ln.score = kSystemNote;
            } else {
                ln.score = inherited_score(
                    lines, static_cast<std::size_t>(&ln - lines.data()), local);
            }
        } else if (severe) {
            ln.anchor = true;
            ln.score = kAnchor;
        } else if (is_outcome(ln.text)) {
            ln.score = kOutcome;
        } else if (is_warning(ln.text)) {
            ln.score = kWarning;
        } else {
            ln.score = kOrdinary;
        }
        // Phase 1 packs anchors ahead of all context, so anchor-ness -- not score -- is what
        // actually spends the budget there. Demoting a warning's bare locator to kWarning
        // (see inherited_score) therefore did NOT stop it being packed first: on
        // ho_rustc_no_cargo, 240 warnings each contribute a distinct ` --> shard.rs:N:C`
        // line, all still anchors, and phase 1 filled 2048 bytes with them before the three
        // real errors' caret blocks were considered. The score floor is what makes the
        // demotion bite.
        ln.diagnostic = ln.anchor && ln.score >= kAnchor;
    }

    // --- proximity -----------------------------------------------------------
    // Principle 3. Every line near an anchor inherits a share of its importance, falling
    // off with distance. This is what keeps a caret block, a numbered source window or a
    // traceback frame without recognising any of them.
    std::vector<int> boost(lines.size(), 0);
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (!lines[i].anchor) {
            continue;
        }
        // Scaled by the anchor's OWN importance, so principle 2 reaches the context too.
        // Without this, the caret block under the one `note:` naming the agent's file and
        // the caret block under a libc++ error score identically, and in a template blow-up
        // there are twenty of the latter -- build_template_deep lost all four of its
        // context lines at every budget, including one with 16 KB it had not spent.
        const int reach = (lines[i].score * kProximityWeight) / kAnchor;

        // The anchor's own rendered block, in full and without decay. Bounded by structure
        // rather than by a radius: walk while the lines still look like gutter.
        for (std::size_t j = i + 1; j < lines.size() && is_block_line(lines[j].text); ++j) {
            boost[j] += reach;
        }
        for (std::size_t j = i; j-- > 0 && is_block_line(lines[j].text);) {
            boost[j] += reach;
        }

        for (int d = 1; d <= kProximityReach; ++d) {
            const std::size_t before = i - static_cast<std::size_t>(d);
            const std::size_t after = i + static_cast<std::size_t>(d);
            const int share = reach / d;
            if (i >= static_cast<std::size_t>(d)) {
                boost[before] += share;
            }
            if (after < lines.size()) {
                boost[after] += share;
            }
        }
    }
    for (std::size_t i = 0; i < lines.size(); ++i) {
        // Build noise stays noise however close to an error it sits: the compiler command
        // line directly above a diagnostic is the single most common thing to waste budget
        // on, because it is both adjacent and enormous.
        if (lines[i].score != kNoise) {
            lines[i].score += boost[i];
        }
    }

    // --- pack ----------------------------------------------------------------
    // Two phases, and the order between them IS principle 1: every distinct anchor before
    // any context.
    //
    // Budget accounting is incremental and conservative. `reserved` is an UPPER bound on
    // the rendered size: each unselected run costs at most one marker, and the real marker
    // is always shorter than kMarkerReserve. So a selection that fits the estimate always
    // fits the budget, and the check is O(1) per line rather than a rebuild.
    std::size_t used = 0;
    std::size_t gaps = 1; // the whole log is one unselected run to start with
    // No gap can elide more lines than the log has, so this bounds every marker.
    const std::size_t marker_reserve = kMarkerFixed + digits(lines.size());

    const auto fits = [&](std::size_t line_bytes, std::size_t new_gaps) noexcept {
        return used + line_bytes + new_gaps * marker_reserve <= budget_bytes;
    };
    const auto select = [&](std::size_t i) {
        const bool left_gap = (i > 0) && !lines[i - 1].selected;
        const bool right_gap = (i + 1 < lines.size()) && !lines[i + 1].selected;
        const std::size_t new_gaps =
            gaps - 1 + static_cast<std::size_t>(left_gap) + static_cast<std::size_t>(right_gap);
        if (!fits(lines[i].text.size() + 1, new_gaps)) {
            return false;
        }
        lines[i].selected = true;
        used += lines[i].text.size() + 1;
        gaps = new_gaps;
        return true;
    };

    std::vector<std::size_t> order(lines.size());
    for (std::size_t i = 0; i < lines.size(); ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&lines](std::size_t a, std::size_t b) {
        return lines[a].score > lines[b].score;
    });

    // Phase 1: anchors, best first, DEDUPLICATED BY MESSAGE rather than by whole line.
    //
    // The message is what is left after the locator, so two complaints that differ only in
    // where they happened collapse to one. That is the difference between reporting a
    // template instantiation failure and reporting it twenty times: build_template_deep's
    // twenty libc++ diagnostics are four distinct messages repeated across sort.h, and the
    // sixteen duplicates were consuming three kilobytes that the agent's own `note:` and
    // its caret block needed. SwiftPM's habit of printing every diagnostic twice, once from
    // emit-module and once from the compile job, collapses the same way.
    //
    // The cost, stated: the same mistake made in three files reports once, and the elision
    // marker is the only sign the other two existed. That is the right trade against a
    // cascade, and local anchors are considered first, so the copy that survives is the one
    // in code the agent can edit.
    std::vector<std::string_view> seen;
    for (std::size_t i : order) {
        if (!lines[i].diagnostic) {
            continue;
        }
        // When the locator consumes the whole line -- rustc's ` --> path:L:C` and Python's
        // `File "path", line N` -- the location IS the distinguishing content and there is
        // no message to compare. Deduplicating on the empty remainder collapsed every
        // traceback frame and every rustc address to a single line, which cost the `other`
        // family 4 locators and 6 messages the moment message-dedup was introduced.
        const std::size_t loc_end = find_locator_end(lines[i].text);
        const std::string_view message =
            (loc_end == std::string_view::npos || loc_end >= lines[i].text.size())
                ? lines[i].text
                : lines[i].text.substr(loc_end);
        if (std::find(seen.begin(), seen.end(), message) != seen.end()) {
            lines[i].duplicate = true;
            continue;
        }
        if (select(i)) {
            seen.push_back(message);
        }
    }

    // Phase 2: everything else by score -- context first, because proximity put it there.
    //
    // `duplicate` is load-bearing here, not bookkeeping. Phase 1 skips a repeated message,
    // but a skipped anchor still scores like an anchor, so without this flag phase 2 puts
    // every copy straight back: build_template_deep came back with the SAME
    // `note: in instantiation of ...` line four times while the source line it refers to
    // did not fit.
    for (std::size_t i : order) {
        if (!lines[i].selected && !lines[i].duplicate && lines[i].score > kNoise) {
            select(i);
        }
    }

    // Phase 3: spend whatever is left, head first and then tail.
    //
    // Only reached when everything the engine considers informative already fits, which on
    // a log that is mostly progress output is most of the budget: a 12 MB build log of
    // "[ 42%] Building ..." lines with one error in it left 8 KB of allowance and used 128
    // bytes of it. Returning unused budget is not a virtue -- the head carries the command
    // that was run and the tail carries the verdict, and neither is worth displacing a
    // diagnostic for, which is exactly why this runs last.
    for (std::size_t i = 0; i < lines.size() && !lines[i].selected; ++i) {
        if (!select(i)) {
            break;
        }
    }
    for (std::size_t i = lines.size(); i-- > 0 && !lines[i].selected;) {
        if (!select(i)) {
            break;
        }
    }

    // --- render --------------------------------------------------------------
    std::string out;
    out.reserve(budget_bytes);
    std::size_t elided = 0;
    const auto flush_gap = [&out, &elided]() {
        if (elided == 0) {
            return;
        }
        out.append("[... ");
        out.append(std::to_string(elided));
        out.append(" lines elided ...]\n");
        elided = 0;
    };
    for (const Line& ln : lines) {
        if (ln.selected) {
            flush_gap();
            out.append(ln.text.data(), ln.text.size());
            out.push_back('\n');
        } else {
            ++elided;
        }
    }
    flush_gap();

    // The estimate is an upper bound on the SELECTED lines, so this rarely fires -- but it
    // is not dead code, and it was itself off by one until 2026-07-31. When no line can be
    // afforded at all, the whole log is one gap and the output is a lone elision marker
    // whose own length the packer never checked: at budget 26 a 800-line log rendered
    // "[... 800 lines elided ...]\n", 27 bytes.
    //
    // The clamp did not save it. `rfind('\n', budget_bytes)` searches positions <= the
    // budget, so it can return the newline sitting exactly AT budget_bytes, and resizing to
    // cut + 1 then leaves the string one byte over -- a no-op that looked like a truncation.
    // Cutting at budget_bytes - 1 is what makes cut + 1 <= budget_bytes hold. `budget_bytes`
    // is at least 1 here; zero returned above.
    if (out.size() > budget_bytes) {
        const std::size_t cut = out.rfind('\n', budget_bytes - 1);
        out.resize(cut == std::string::npos ? budget_bytes : cut + 1);
    }
    return out;
}

} // namespace log_triage

// ---------------------------------------------------------------------------
// Provenance: log-triage cookoff round 1, 2026-07-30. 15 entrants, all shipped code.
// Benchmark and full results at bakeoff/log_triage/README.md.
//
// TAKEN
//   e05  The proximity boost, whole. Scoring lines by keyword and then boosting neighbours
//        by 1/distance is the best idea in the round: it retains clang's caret block,
//        Swift's numbered window, rustc's gutter and Python's frame source without knowing
//        any of their formats. Also its up-front normalisation (ANSI + \r collapse).
//   e06  Detection breadth. It had the best diagnostic recall of any entrant -- 5 locator
//        and 2 message misses out of 177 and 195 -- by looking for markers beyond `error:`.
//   e10  Treating `note:` lines as first-class. It and e02 are the only implementations
//        that solve build_no_matching_ctor, where the candidate signatures are all notes.
//   e12  Locators that are not on the same line as their message. The only entrant that
//        solves cargo_two_errors, where rustc prints ` --> path:L:C` under the message.
//   e08  (round 2) Recognising clang's "In file included from ..." include stack as
//        structure rather than diagnosis -- its TEMPLATE_INST state. Round 1 rejected e08
//        wholesale because it takes no budget, and threw this out with it. The line carries
//        `path:line:`, so the locator matcher fires and it scored as a full anchor; a
//        template blow-up emits dozens at 150-200 bytes of SDK path each. Rejecting an
//        entrant's INTERFACE is not a reason to reject everything it noticed.
//
// REJECTED
//   e05's packer. A greedy knapsack that recomputes the full rendered size for every
//        candidate line: O(n^2), and 19.1 SECONDS on a 12 MB log where e06 took 0.06 s.
//        The corpus never caught it -- its largest case is 118 KB -- so it was found by
//        timing the candidates on a synthetic 200k-line log before adopting any of them.
//        Replaced with incremental accounting against a conservative upper bound.
//   e01's severity table. It matches uppercase FATAL / CRITICAL / ERROR / WARN, which no
//        compiler emits, and returns the EMPTY STRING on all 75 scoring points. It was
//        written for syslog-style application logs.
//   e08 and e15's signatures. Neither takes a budget at all.
//   e02 and e09's ranking. Both look strong only because the bakeoff adapter binary-searches
//        their non-byte budget against the byte cap -- an oracle production does not have.
//        Estimating instead, e02 goes 78 -> 516 weighted and exceeds the budget on 31 of 75
//        points. Their SELECTION was good; their interface was not.
//
// WRITTEN HERE, because no entrant did it
//   Anchors before context (principle 1). bare_error_limit has 19 diagnostics in a 2048-byte
//        cap and every context-first implementation spends it on the first three.
//   Local-over-system ranking (principle 2). build_template_deep's only actionable line is a
//        `note:` in the project, under 20 `error:` lines inside libc++.
//   Build-noise suppression before severity matching. A compiler command line contains
//        "error" whenever the build uses -Werror or compiles a file with "error" in its
//        name, and it is both adjacent to real diagnostics and very long.
//   Anchor de-duplication. SwiftPM emits every diagnostic twice.
//
// WRITTEN HERE IN ROUND 2 (2026-07-31), all three fixing something round 1 got wrong
//   Selecting phase 1 on `diagnostic` rather than `anchor`. Round 1 demoted a warning's bare
//        locator to kWarning and believed that fixed it; phase 1 packs by anchor-ness and
//        never reads the score, so the demotion did nothing. A fix to a RANKING is not a fix
//        when the code in question does not rank.
//   kSystemNote. Principle 2 ranked by locality and, inside the system tier, not by severity
//        at all -- so libc++'s twenty instantiation notes tied with the one real error.
//        Applied only when the locator is NOT local, which is what keeps e10's contribution.
//   The one-byte budget overrun in the final clamp (see the render section). Round 1 exceeded
//        its budget on 32 of 137,408 (log, budget) points, all at the single budget per log
//        where a lone elision marker's newline lands exactly on the cap. Unreachable in
//        production and invisible to all 75 scoring points; found by a contract test.
//
// WHAT ROUND 2 SAYS ABOUT THE METHOD
//   Every one of these had been measured, written up and shipped. What found them was
//   re-reading the OUTPUT on cases the scorer had already gone quiet on -- and for the
//   clamp, writing the test this file had been claimed to have since round 1. A scoreboard
//   that has stopped moving is not the same thing as an engine that is right.
// ---------------------------------------------------------------------------

#endif // TOOLS_LOG_TRIAGE_HPP
