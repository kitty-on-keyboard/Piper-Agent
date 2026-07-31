#pragma once
//
// The bar the cookoff had to clear: SubprocessVerifier::compact_command_output and
// line_is_diagnostic AS THEY SHIPPED BEFORE THIS CHANGE, frozen.
//
// This was a wrapper that INCLUDED the production header and called the real function, so
// the baseline could never drift from what the agent actually got. It cannot be any more:
// production now calls log_triage::compact, and both would define the same symbol in this
// translation unit. The code below is therefore a verbatim copy, extracted from the commit
// that removed it, and a copy is safe here for the one reason that usually makes copies
// dangerous -- the original no longer exists, so there is nothing left for it to drift from.
//
// Recover it with:
//   git log --diff-filter=M -- src/tools/subprocess_verifier.hpp
//
// Seven substrings, errors-only, tuned by hand against clang and swift. Measured on the
// 25-case corpus: 261 weighted misses, 50 of 75 points exact, 34 of 177 locators lost.
// Genuinely better than "seven substrings" sounds, because most of a real build log has
// exactly one diagnostic in clang's format -- and that is exactly the shape it was tuned on.
//
#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>

namespace log_triage {

namespace incumbent_detail {

inline bool line_is_diagnostic(std::string_view line) noexcept {
    static constexpr std::string_view markers[] = {
        "error:", "error generated", "fatal error", "FAILED",
        "Undefined symbol", "ld: ", "Command failed"};
    for (std::string_view m : markers) {
        if (line.find(m) != std::string_view::npos) {
            return true;
        }
    }
    return false;
}

inline void compact_command_output(std::string_view full, std::size_t max_bytes,
                                   std::string& out,
                                   std::string_view spool_ref = {}) noexcept {
    if (full.size() <= max_bytes) {
        out.append(full.data(), full.size());
        return;
    }
    const std::size_t head_cap = std::min<std::size_t>(768, max_bytes / 6);
    const std::size_t tail_cap = std::min<std::size_t>(2048, max_bytes / 4);
    const std::size_t diag_budget =
        (max_bytes > head_cap + tail_cap + 96) ? (max_bytes - head_cap - tail_cap - 96) : 0;
    const std::size_t tail_start = full.size() - tail_cap;

    out.append(full.data(), head_cap);
    out.append("\n[... ");
    out.append(std::to_string(full.size() - max_bytes));
    out.append(" bytes condensed — head + error lines + tail ...]\n");
    if (!spool_ref.empty()) {
        out.append("[FULL OUTPUT SAVED: ");
        out.append(spool_ref.data(), spool_ref.size());
        out.append(" — ");
        out.append(std::to_string(full.size()));
        out.append(" bytes. If what you need is not shown below, search_files (literal "
                   "substring) or read_file_slice that file instead of re-running the "
                   "command.]\n");
    }

    std::size_t pos = head_cap;
    std::size_t emitted = 0;
    // Context lines to keep after a diagnostic. Compilers print the offending
    // SOURCE line and a caret under the error:
    //     File.swift:50:48: error: instance member 'huntId' cannot be used...
    //         @Query(filter: #Predicate<Hunt> { $0.id == huntId }, ...)
    //                                                    ^
    // Those follow-on lines contain no "error:" marker, so keeping only matched
    // lines handed the model the message WITHOUT the code it refers to — and it
    // then edited the wrong line (observed live: it renamed a closure param on
    // line 35 instead of fixing the @Query on line 50).
    constexpr int kContextLinesAfterDiagnostic = 2;
    int context_left = 0;
    while (pos < tail_start && emitted < diag_budget) {
        const std::size_t nl = full.find('\n', pos);
        const std::size_t end = (nl == std::string_view::npos) ? full.size() : nl;
        const std::string_view line = full.substr(pos, end - pos);
        const bool is_diag = line_is_diagnostic(line);
        if (is_diag) {
            context_left = kContextLinesAfterDiagnostic;
        }
        if (is_diag || context_left > 0) {
            const std::size_t take = std::min(line.size(), diag_budget - emitted);
            out.append(line.data(), take);
            out.push_back('\n');
            emitted += take + 1;
            if (!is_diag) {
                --context_left;
            }
        }
        pos = (nl == std::string_view::npos) ? full.size() : nl + 1;
    }

    out.append("[... tail ...]\n");
    out.append(full.data() + tail_start, tail_cap);
}

} // namespace incumbent_detail

[[nodiscard]] inline std::string compact(std::string_view full, std::size_t budget_bytes) {
    std::string out;
    incumbent_detail::compact_command_output(full, budget_bytes, out);
    return out;
}

} // namespace log_triage
