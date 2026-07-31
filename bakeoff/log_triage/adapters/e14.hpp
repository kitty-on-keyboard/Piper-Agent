#pragma once
// e14 ships an istream overload and a string_view one, both taking a byte budget.
// The string_view overload is the contract.
#include <string>
#include <string_view>

namespace log_triage_bridge {

const int kAdapterKind = 0;
const char* const kAdapterNote = "direct: compact(std::string_view, byte budget)";

std::string compact_entrant(std::string_view full, std::size_t budget_bytes) {
    return lt_entrant::compact(full, budget_bytes);
}

} // namespace log_triage_bridge
