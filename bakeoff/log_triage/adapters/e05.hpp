#pragma once
// e05 takes (const std::string&, byte budget). The budget is the contract's; only the
// parameter type differs, so the sole accommodation is a copy the real caller would also
// have to make.
#include <string>
#include <string_view>

namespace log_triage_bridge {

const int kAdapterKind = 0;
const char* const kAdapterNote = "direct: compact(const std::string&, byte budget)";

std::string compact_entrant(std::string_view full, std::size_t budget_bytes) {
    return lt_entrant::compact(std::string(full), budget_bytes);
}

} // namespace log_triage_bridge
