#pragma once
// e02's budget is a LINE COUNT, not a byte count. It cannot be handed the caller's cap.
// The adapter searches for the largest line count that still fits -- see
// search_unit_budget: strictly more than the real caller could do, and marked as such.
#include <string>
#include <string_view>

namespace log_triage_bridge {

const int kAdapterKind = 1;
const char* const kAdapterNote =
    "unit-searched: compact(text, MAX_LINES) -- largest line count that fits the budget";

std::string compact_entrant(std::string_view full, std::size_t budget_bytes) {
    // A line cannot be shorter than one byte, so the byte budget bounds the line count.
    return search_unit_budget(budget_bytes, budget_bytes,
                              [full](std::size_t lines) {
                                  return lt_entrant::compact(full, lines);
                              });
}

} // namespace log_triage_bridge
