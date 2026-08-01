#pragma once
//
// moetrace -- expert-routing trace analysis.
//
// Amalgamated from the 6-entrant cook-off. All six agreed to four decimal places on a
// 1M-line trace and all six passed every known-answer anchor, so correctness was settled;
// what differed was speed (313 ms to 1239 ms), malformed-line detection (5/6 vs 6/6), and
// whether the output actually answered a question.
//
// Two changes over the best entrant (B5):
//
//   1. PARSING. B5 was the fastest at 313 ms via getline + from_chars, but getline copies
//      every line into a std::string. This reads 256 KB blocks and scans them in place, so
//      a 1M-line trace costs no per-line allocation at all.
//
//   2. THE ANSWER, NOT THE INGREDIENTS. Every entrant reported unique_ratio, which is the
//      raw material. None reported the number the tool exists to produce: how many drafted
//      tokens must be ACCEPTED before speculating k of them is cheaper than decoding them
//      one at a time. That is a two-line derivation from unique_ratio and it is the whole
//      decision, so it belongs in the output rather than in someone's head.
//
// The derivation, so the field can be checked rather than trusted. Decoding one token
// costs 8 expert loads per layer. Verifying a k-token draft in one pass costs
// unique(k) = unique_ratio[k] * 8k loads. A draft with `a` accepted tokens advances a+1
// positions, which sequentially would have cost 8(a+1). Speculation therefore pays when
// 8(a+1) >= unique_ratio[k] * 8k, i.e. when a >= unique_ratio[k]*k - 1. A value <= 0 means
// speculating k tokens is cheaper even if EVERY draft token is rejected.
//
#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace moetrace {

struct LayerStats {
    int layer = 0;
    double gini = 0.0;
    double mean_consecutive_overlap = 0.0; // [0,1]
    std::array<double, 4> unique_ratio{};  // for k = 1,2,4,8
    std::size_t cold_experts = 0;
    std::size_t selections = 0;

    // Accepted tokens required for a k-token draft to break even, for k = 1,2,4,8.
    // <= 0 means speculation at that depth is free on expert bandwidth alone.
    std::array<double, 4> breakeven_accepted{};
};

struct TraceStats {
    std::vector<LayerStats> per_layer;
    LayerStats aggregate;
    std::size_t steps = 0;
    std::size_t malformed_lines = 0;
};

[[nodiscard]] TraceStats analyse_file(const std::string& path, int num_experts = 256);
[[nodiscard]] TraceStats analyse_lines(std::span<const std::string> lines,
                                       int num_experts = 256);

} // namespace moetrace
