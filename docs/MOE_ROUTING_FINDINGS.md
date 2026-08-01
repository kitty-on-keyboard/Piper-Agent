# Expert routing in Qwen 3.6 35B A3B: measurements, and what they decide

Measured 2026-08-01 on the production checkpoint
(`Qwen3.6-35B-A3B-MLX-4bit`), captured with `LMP_MOE_TRACE` from
`src/model/mlx/moe_trace.hpp`. **16,914 real decode steps, 40 MoE layers, 8 of 256 experts
per layer per token, 676,560 routing records, 0 malformed.** Analysis by the amalgamated
`moetrace` built from the `expert-router-trace` cook-off.

Capture forces a mid-graph sync, so the traced run decodes at 52 tok/s against 85 untraced.
Routing is unaffected by when it is read; **no throughput number from a traced run means
anything.**

---

## 1. The headline routing numbers (nobody had these for this model)

| | |
|---|---|
| consecutive-step expert overlap | **0.3562** |
| Gini of expert selection | 0.1476 (mildly skewed) |
| experts never selected in 16,914 steps | 113 of 10,240 (40x256) |
| per-layer overlap, min | **0.0500** (layer 0) |
| per-layer overlap, max | **0.4496** (layer 26) |

Adjacent tokens reuse about a third of their experts. The **9x spread between layers** is
the structural surprise: layer 0 re-routes almost completely every token, deep-middle layers
are nearly half stable.

Cost of verifying a k-token draft, as a fraction of decoding those k tokens one at a time,
and the accepted-token count at which speculation breaks even:

| k | unique/(8k) | accepted tokens needed to break even |
|---|---|---|
| 1 | 1.0000 | 0.00 |
| 2 | 0.8219 | 0.64 |
| 4 | 0.6804 | 1.72 |
| 8 | 0.5540 | 3.43 |

---

## 2. Routing is TOKEN-determined, not context-determined

The same token id, at unrelated positions in the stream, reuses far more of its experts than
two unrelated tokens do.

| | overlap |
|---|---|
| same token, different positions | **0.4923** |
| different tokens (baseline) | 0.0956 |
| adjacent tokens in the stream | 0.3562 |
| **lift** | **5.15x** |

Per layer the lift is 10.6x at layer 0 and 1, settling to a stable 4–5x through the entire
depth. Note that same-token overlap (0.49) **exceeds** adjacent-token overlap (0.36): a
token's routing is more like its own past than like its neighbour.

**Held out**, this predicts. Signatures built from the first 8,457 steps, tested on the
second 8,457:

| predictor | overlap with actual expert set |
|---|---|
| the token's 8 most frequent experts | **0.5434** (covers 86.7% of records) |
| the layer's 8 most frequent experts (token-blind) | 0.1886 |
| lift | **2.88x** |

So more than half of a token's routing is knowable from its id alone, before the model runs.

---

## 3. The negative result: expert-aware drafting does not pay

Section 2 makes an obvious-looking optimisation available. If a draft's expert cost is
predictable, steer drafting toward tokens whose experts are already resident — this is the
direction the 2026 MoE speculative-decoding literature is pushing from the model side
(expert budgeting, cost-aware routing). **Tested three ways on this trace, it does not
work.**

- **Draft *selection*.** Among the top-6 candidate continuations, choose by predicted expert
  cost instead of frequency: **0.4% cheaper per accepted token.** Nothing. The candidates
  cost nearly the same, because unioning across 40 layers washes out per-token differences.
- **Draft *length* gating on marginal predicted cost.** Never beat the plain
  cumulative-probability rule at matched draft length.
- **The apparent winner was an artefact.** The best-scoring expert-aware setting
  (`tau=4.0`, 1.481x) turned out to produce a draft-length histogram of `1:4786` — it is
  *exactly* fixed-length-1, byte for byte identical in accepted tokens and loads. The
  expert-awareness contributed nothing; it collapsed to a trivial policy that a one-line
  rule expresses better.

Acceptance probability dominates expert cost. Predicting routing is real (section 2) and
still not actionable for drafting.

---

## 4. The decision-grade result: naive speculative decoding LOSES on this model

Expert-bandwidth speedup, where one sequentially decoded token costs 320 expert loads
(40 layers x 8 experts), and a draft that gets `a` tokens accepted advances `a+1` positions:

| policy | accepted/draft | loads/draft | speedup |
|---|---|---|---|
| fixed length 1 | 0.481 | 320.0 | **1.481x** |
| fixed length 2 | 0.788 | 579.0 | 0.988x |
| fixed length 4 | 1.169 | 1023.6 | **0.678x** |
| fixed length 8 | 1.473 | 1742.4 | **0.454x** |
| cumulative-probability, tau=0.8 | 1.659 | 680.8 | 1.250x |
| cumulative-probability, tau=0.9 | 1.646 | 675.4 | 1.254x |

**Fixed-length drafting is a net loss.** At k=4 it costs 1.47x what sequential decoding
costs; at k=8, 2.2x. This is the MoE objection, quantified on the real model: expert
scattering eats the entire speculative gain and then some.

Only two things win: very short drafts, or **stopping on cumulative acceptance
probability** — which is exactly the rule in the amalgamated `SuffixProposer`.

### The law worth remembering

Marginal cost of one more draft position, from the fixed-length rows:

| position | extra loads | as a fraction of a sequential token |
|---|---|---|
| 1 -> 2 | 259.0 | 0.81 |
| 2 -> 4 | 222.3 /token | 0.69 |
| 4 -> 8 | 179.7 /token | 0.56 |

Extending the draft pays only while the probability of acceptance at that position exceeds
that fraction — so **extend while cumulative acceptance probability is above ~0.65**,
falling toward ~0.56 for deeper positions.

The `SuffixProposer` default `draft_cost_ratio` was set to **0.60** by a sweep on synthetic
data, before any of this was measured. The real routing data puts the correct value at
0.56–0.81. The synthetic tuning landed inside the measured band by accident, and it is right
for the reason the sweep could not see.

---

## 5. What this changes

- **Speculative decoding is still worth doing here, but only adaptively.** A fixed draft
  length — the default in most implementations — makes this model *slower*. Anyone porting
  a stock speculative decoder to a MoE of this shape should expect a regression.
- **The three model-layer blockers in `PLAN_PARALLELISM.md` are unchanged** (last-position
  logits, no KV tail rollback, append-only ledger). What has changed is that the payoff is
  now measured rather than hoped for, and the drafting policy is settled.
- **Do not spend on expert-aware drafting.** Section 3 is the reason.
- Caveat on scope: one model, one workload (agent/tool-use plus expository prose), 16,914
  steps. The token-determination result is strong enough to survive a different workload;
  the exact speedup numbers are not, and should be re-measured against real drafts from the
  `SuffixProposer` rather than the trigram stand-in used here.
