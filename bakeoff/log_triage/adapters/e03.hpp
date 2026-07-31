#pragma once
// e03's budget is a CHUNK COUNT with a default of 5, not a byte count. Searched for the
// largest chunk count that fits. Its chunks are paragraphs, so the count is small; the
// search bound is generous rather than tight.
#include <string>
#include <string_view>

namespace log_triage_bridge {

const int kAdapterKind = 1;
const char* const kAdapterNote =
    "unit-searched: compact(text, MAX_CHUNKS=5) -- largest chunk count that fits";

std::string compact_entrant(std::string_view full, std::size_t budget_bytes) {
    return search_unit_budget(budget_bytes, 4096,
                              [full](std::size_t chunks) {
                                  return lt_entrant::compact(full, chunks);
                              });
}

} // namespace log_triage_bridge
