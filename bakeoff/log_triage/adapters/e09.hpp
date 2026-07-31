#pragma once
// e09's budget is a TOKEN COUNT, counted by its own count_tokens(). Searched for the
// largest token budget that fits in bytes. The search bound is the byte budget itself,
// since no tokenisation makes a token smaller than a byte.
#include <string>
#include <string_view>

namespace log_triage_bridge {

const int kAdapterKind = 1;
const char* const kAdapterNote =
    "unit-searched: compact(text, TOKEN_BUDGET) -- largest token budget that fits";

std::string compact_entrant(std::string_view full, std::size_t budget_bytes) {
    return search_unit_budget(budget_bytes, budget_bytes,
                              [full](std::size_t tokens) {
                                  return lt_entrant::compact(full, tokens);
                              });
}

} // namespace log_triage_bridge
