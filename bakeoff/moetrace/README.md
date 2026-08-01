# moetrace

Expert-routing trace analysis. Amalgamated from the 6-entrant
`cat-collector-king/expert-router-trace` cook-off (2026-08-01).

All six entrants passed every known-answer anchor and agreed to four decimal places, so
correctness was settled; they differed on speed (313-1239 ms per 1M lines), malformed-line
detection (5/6 vs 6/6), and whether the output answered a question. This one reads 256 KB
blocks scanned in place rather than copying every line, and reports `breakeven_accepted` --
how many drafted tokens must be accepted before speculating k of them is cheaper than
decoding them one at a time -- which is the decision the tool exists to inform.

1M lines: **149 ms**, against 285 ms for the fastest entrant. Anchors in `scoreboard.cpp`.

Consumes traces written by `src/model/mlx/moe_trace.hpp` (`LMP_MOE_TRACE=<path>`).
Results in `docs/MOE_ROUTING_FINDINGS.md`.
