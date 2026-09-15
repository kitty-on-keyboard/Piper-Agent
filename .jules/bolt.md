## 2024-03-24 - One precise optimization
**Learning:** Avoid shotgun-style micro-optimizations across multiple files that reduce readability (like manually unrolling arrays or rewriting DOM queries to more verbose forms) unless they solve a measurable bottleneck.
**Action:** When asked for ONE small optimization, pick the single highest-impact and lowest-risk change, such as avoiding quadratic allocations during high-frequency parsing.
