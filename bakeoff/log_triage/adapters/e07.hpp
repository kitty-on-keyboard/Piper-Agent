#pragma once
// e07 takes its own TriageConfig{n_lines_before, m_lines_after, ANCHOR, budget_bytes}.
// The byte budget goes straight through; the other three are knobs it invented, and the
// adapter has to choose them, so both choices are stated:
//
//   anchor "error"   e07's own tests pass "ERROR", which matches nothing clang, swiftc,
//                    rustc or pytest ever prints. Scoring it on a value that cannot fire
//                    would be scoring an unanswerable case. "error" is the lowercase
//                    substring every one of those tools does emit, and e07 matches it
//                    literally with find(), so this is the value that lets it work at all.
//   1 before / 2 after
//                    2-after is what production found necessary (subprocess_verifier.hpp
//                    :416) to carry clang's source-and-caret block. e07's own tests use
//                    1/1, which would cut every caret line; the generous reading is used.
//
// The finding is not the adapter, it is that e07 pushed the triage decision onto the
// caller: hand it the wrong anchor and it returns the tail of the log.
#include <string>
#include <string_view>

namespace log_triage_bridge {

const int kAdapterKind = 0;
const char* const kAdapterNote =
    "direct budget, but the adapter must supply the anchor: "
    "TriageConfig{1, 2, \"error\", budget}";

std::string compact_entrant(std::string_view full, std::size_t budget_bytes) {
    const lt_entrant::TriageConfig config{1, 2, "error", budget_bytes};
    return lt_entrant::compact(full, config);
}

} // namespace log_triage_bridge
