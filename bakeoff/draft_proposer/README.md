# draft_proposer

`SuffixProposer`: model-free draft proposal for speculative decoding. Amalgamated from the
5-entrant model-free-draft-proposer cook-off (2026-08-01).

Storage is the winning entrant's (compact array-backed trie, 32-bit node indices). The
DECISION is rebuilt: minimum support so a context seen once is not treated as evidence, a
stop rule on CUMULATIVE rather than step probability, and binary search for the longest
supported match (support is monotonically non-increasing in match length, which no entrant
exploited).

Scored by `scoreboard.cpp` on a shared harness, same corpus for every implementation:

| | accepted/call | waste/call | precision | p50 |
|---|---|---|---|---|
| **this** | 4.174 | **0.060** | **98.6%** | 1.21 us |
| best entrant | 4.214 | 2.660 | 61.3% | 0.04 us |
| next best | 4.391 | 3.324 | 56.9% | 0.29 us |

`draft_cost_ratio` defaults to 0.60 from a sweep on synthetic data. Real routing data
(`docs/MOE_ROUTING_FINDINGS.md`) independently puts the correct band at 0.56-0.81.
