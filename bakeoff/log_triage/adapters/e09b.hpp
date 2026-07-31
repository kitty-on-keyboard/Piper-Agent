#pragma once
// e09 again, byte for byte, but WITHOUT the oracle.
//
// e09's budget is a token count, counted by its own whitespace-splitting count_tokens().
// adapters/e09.hpp binary-searches for the largest token budget that fits in bytes --
// help the real caller cannot give. This adapter ESTIMATES instead, at the ratio MEASURED
// over the corpus with e09's own definition of a token: 14.6 bytes per token across
// 971,544 bytes of real build output.
//
// The gap between e09 and e09b is what the search is worth, and only the e09b number is
// comparable with an entrant that took a byte budget in the first place.
#include <string>
#include <string_view>

namespace log_triage_bridge {

const int kAdapterKind = 0;
const char* const kAdapterNote =
    "e09 with NO unit search: token budget ESTIMATED as byte_budget / 14.6";

std::string compact_entrant(std::string_view full, std::size_t budget_bytes) {
    const std::size_t tokens = static_cast<std::size_t>(budget_bytes / 14.6);
    return lt_entrant::compact(full, tokens ? tokens : 1);
}

} // namespace log_triage_bridge
