#pragma once
// e15 TAKES NO BUDGET. compact(std::string_view) -- same as e08. See that adapter.
#include <string>
#include <string_view>

namespace log_triage_bridge {

const int kAdapterKind = 2;
const char* const kAdapterNote = "NO BUDGET PARAMETER: compact(std::string_view)";

std::string compact_entrant(std::string_view full, std::size_t) {
    return lt_entrant::compact(full);
}

} // namespace log_triage_bridge
