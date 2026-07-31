#pragma once
// e02 again, byte for byte, but WITHOUT the oracle.
//
// e02's budget is a line count. adapters/e02.hpp binary-searches for the largest line
// count that fits the byte budget, which is help the real caller cannot give: production
// would have to ESTIMATE. This adapter does what production would -- divide the byte
// budget by a mean line length -- so the difference between e02 and e02b is exactly what
// the search is worth, and the headline number can be reported without it.
//
// 130 bytes/line is MEASURED over the corpus, not guessed: 25 logs, 971,544 bytes,
// 7,464 lines. An estimator that guesses low emits too many lines and blows the cap,
// so the measured mean is the fairest value available to a caller with no oracle.
#include <string>
#include <string_view>

namespace log_triage_bridge {

const int kAdapterKind = 0;
const char* const kAdapterNote =
    "e02 with NO unit search: line budget ESTIMATED as byte_budget / 130";

std::string compact_entrant(std::string_view full, std::size_t budget_bytes) {
    const std::size_t lines = budget_bytes / 130;
    return lt_entrant::compact(full, lines ? lines : 1);
}

} // namespace log_triage_bridge
