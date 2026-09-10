#pragma once
//
// The adapter for an entrant whose signature IS the contract: compact(text, byte budget).
// Nothing is accommodated, nothing is searched.
//
#include <string>
#include <string_view>

namespace log_triage_bridge {

const int kAdapterKind = 0;
const char* const kAdapterNote = "direct: compact(text, byte budget)";

std::string compact_entrant(std::string_view full, std::size_t budget_bytes) {
    return lt_entrant::compact(full, budget_bytes);
}

} // namespace log_triage_bridge
