#pragma once
//
// The seam between an entrant and the scorer.
//
// Every entrant is compiled ALONE, in entrant_tu.cpp, with its namespace renamed out of
// the way; the scorer never sees the entrant's own declarations. In the blast-radius round
// the reason was that all eleven entrants re-declared the contract instead of including it.
// Here it is worse and simpler: there WAS no contract to include. The entrant repository's
// main branch held a nineteen-byte README, so fifteen entrants invented fifteen signatures
// and the budget is not the same quantity in all of them -- bytes in six, lines in one,
// chunks in one, tokens in one, a private config struct in one, and absent in two.
//
// So the call itself has to be adapted per entrant, and that adaptation lives in
// adapters/eNN.hpp -- a file WE own and can review -- while entrants/eNN.hpp stays byte
// for byte as shipped. Each adapter is a handful of lines and its content is reported
// alongside the score, because "how much accommodation did this need" is a real result.
//
// The rule the adapters follow, written down once:
//
//   * A byte budget goes straight through.
//   * A budget in some other unit (lines, chunks, tokens) is BINARY-SEARCHED for the
//     largest value whose output fits budget_bytes. That is more generous than the real
//     caller could ever be -- the caller has no oracle to search with -- so the result is
//     the entrant's SELECTION quality with its unit problem forgiven. Adapters that do
//     this are marked, and the count is reported.
//   * An entrant with no budget parameter is called as shipped and its output is scored as
//     it comes. It cannot honour a byte cap and that is not something an adapter can fix.
//
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>

namespace log_triage_bridge {

// Find the largest value of an entrant's own budget unit whose output still fits
// `budget_bytes`.
//
// This is strictly MORE than the real caller can do -- SubprocessVerifier has no oracle to
// search with, and an entrant that takes a line count genuinely cannot be handed a byte
// cap. Searching is the generous reading: it forgives the unit mismatch entirely and
// measures only what the entrant SELECTED. Entrants scored this way are marked
// `unit-searched` in the scoreboard so the help is never invisible.
//
// Monotonicity is assumed (more lines -> more bytes) and not required to be exact: the
// result is verified to fit, and the empty string is returned if nothing does.
[[nodiscard]] inline std::string search_unit_budget(
    std::size_t budget_bytes, std::size_t hi,
    const std::function<std::string(std::size_t)>& call) {
    std::size_t lo = 0;
    std::string best;
    while (lo <= hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        std::string out = call(mid);
        if (out.size() <= budget_bytes) {
            best = std::move(out);
            lo = mid + 1;
        } else {
            if (mid == 0) {
                break;
            }
            hi = mid - 1;
        }
    }
    return best;
}

// Implemented in entrant_tu.cpp against exactly one entrant.
std::string compact_entrant(std::string_view full, std::size_t budget_bytes);

// What accommodation this entrant needed, reported with its score.
//   0  called directly with a byte budget, as the contract says
//   1  budget searched in the entrant's own unit (lines / chunks / tokens)
//   2  entrant takes no budget at all; output scored as it comes
extern const int kAdapterKind;
extern const char* const kAdapterNote;

} // namespace log_triage_bridge
