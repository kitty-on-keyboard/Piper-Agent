#pragma once
// e08 TAKES NO BUDGET. compact(const std::string&) -- there is no parameter through which
// the caller's cap can be expressed, so there is nothing an adapter can do. It is called
// as shipped and its output is scored as it comes, which means it is over budget whenever
// its own heuristics happen to leave more than the cap.
//
// That is not a scoring artefact. compact_command_output exists to fit max_bytes; an
// implementation that cannot see max_bytes has not implemented it.
#include <string>
#include <string_view>

namespace log_triage_bridge {

const int kAdapterKind = 2;
const char* const kAdapterNote = "NO BUDGET PARAMETER: compact(const std::string&)";

std::string compact_entrant(std::string_view full, std::size_t) {
    return lt_entrant::compact(std::string(full));
}

} // namespace log_triage_bridge
