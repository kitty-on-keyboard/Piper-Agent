#pragma once
// e01 ships two overloads -- an istream one and a std::string one -- both taking
// (text, byte budget, context window = 5, priority function = its own default). The
// budget is the contract's, so the trailing defaults are left as the entrant chose them.
#include <string>
#include <string_view>

namespace log_triage_bridge {

const int kAdapterKind = 0;
const char* const kAdapterNote =
    "direct: compact(const std::string&, byte budget) [entrant's own defaults for its "
    "context-window and priority-function parameters]";

std::string compact_entrant(std::string_view full, std::size_t budget_bytes) {
    return lt_entrant::compact(std::string(full), budget_bytes);
}

} // namespace log_triage_bridge
