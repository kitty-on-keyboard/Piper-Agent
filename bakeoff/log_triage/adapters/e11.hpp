#pragma once
// e11 takes (text, byte budget) -- the contract. Nothing is accommodated.
#include <string>
#include <string_view>

namespace log_triage_bridge {

const int kAdapterKind = 0;
const char* const kAdapterNote = "direct: compact(text, byte budget)";

std::string compact_entrant(std::string_view full, std::size_t budget_bytes) {
    return lt_entrant::compact(full, budget_bytes);
}

} // namespace log_triage_bridge
