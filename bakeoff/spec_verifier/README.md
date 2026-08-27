# SpecVerifier cook-off (Jules round 2, Brief C)

The acceptance rule for speculative decoding: which drafted tokens to keep. One PR per
entrant, **5 of 5 landed**.

## Standing, 2026-08-01 — complete

```
                tv_det_good  tv_det_bad | tv_soft_brief  tv_soft_target | floor  perfect  det  degen  accept_rate
e1 (PR 1)          0.0020      0.0012   |    0.0027         0.0597      |   0       1      1     0       0.58
e2 (PR 2)          0.0020      0.0012   |    0.0027         0.0597      |   0       1      1     0       0.58
e3 (PR 3)          0.0020      0.0012   |    0.0027         0.0597      |   0       1      1     0       0.58
e4 (PR 4)          0.0020      0.0012   |    0.0027         0.0597      |   0       1      1     0       0.58
e5 (PR 5)          0.0020      0.0012   |    0.0027         0.0597      |   0       1      1     0       0.58
--- falsifiers ---
naive_argmax       0.8557      0.0022   |    0.1269         0.1043      |   0       1      1     0       1.18
no_residual        0.1239      0.0062   |    0.0040         0.0625      |   0       1      1     0       0.58
```

**All five are correct, and identical.** 0.0020 and 0.0012 sit at the sampling-noise floor
for this histogram (N = 400,000, V = 12: expected TV from noise alone ≈ 0.0026), so the
entrants are indistinguishable from exact.

Byte-identical outputs across five independent implementations looked like a broken harness
and is not. The brief's header pins the RNG state as a single `std::uint64_t rng_`, and all
five independently reached for splitmix64 with the canonical constants
(`0xbf58476d1ce4e5b9`, `0x94d049bb133111eb`), seeded directly from `seed`. Same stream, same
draws, same histogram. Verified by reading all five RNGs, and by the falsifiers below
producing different numbers through the same harness.

Adoption is therefore a choice among five equally correct implementations on secondary
grounds (`e3` is the most compact at 97 lines), not a correctness ranking.

## The brief contradicts itself, and it matters

Found by exact enumeration while building the harness — no sampling involved. Brief C asks
for two things that cannot both hold:

1. **A procedure.** On rejection, "subtract `q` from the rejected token's mass only, clamp
   negatives to zero, renormalise, sample."
2. **A property.** "The histogram must converge to the target distribution."

Textbook speculative sampling draws the replacement from `norm((p - q)_+)`, which needs the
drafter's whole row. The brief supplies only the scalar `q(t)`, so it prescribes the
reduction in (1) — and that reduction is not distribution-preserving in general:

| rule | worst TV from target, 20,000 random (p, q) |
|---|---|
| textbook full-row residual | 5.6e-17 (exact) |
| Brief C scalar reduction | **2.3e-01** |

Checkable by hand: `p = (.5, .3, .2)`, `q = (.8, .1, .1)`. The full-row residual commits
`(.5, .3, .2)`. The scalar reduction commits `(.5, .28, .22)`.

**The mitigation, and why this is not fatal.** The two coincide exactly when the drafter is
deterministic — `q(proposed) = 1` — which is exactly this project's regime: `SuffixProposer`
proposes a concrete continuation from matched history, it is not a sampling model.

| drafter concentration `q(proposed)` | worst TV from target |
|---|---|
| 1.00 | 8.3e-17 (exact) |
| 0.99 | 6.6e-03 |
| 0.95 | 3.1e-02 |
| 0.90 | 5.5e-02 |
| 0.75 | 1.2e-01 |
| 0.50 | 1.7e-01 |

**Consequence for the wiring: pass `draft_probs[i] = 1.0`.** Treat the proposer's output as
the deterministic proposal it is. Acceptance then reduces to `u < p(t)` — accept the drafted
token with the target's own probability for it — and the whole procedure becomes exactly
distribution-preserving. Passing a confidence score in that slot instead would silently buy
the bias in the table above, in exchange for nothing.

This is why the scoreboard reports two regimes. `tv_det_*` is the deterministic one, where
the brief is self-consistent and any deviation is the entrant's fault; it decides adoption.
`tv_soft_*` is reported for information only — an entrant near zero on `tv_soft_target`
implemented the full-row residual (deviating from the brief, arguably correctly), one near
`tv_soft_brief` implemented the brief as written. All five did the latter.

## Falsifiers — and a correction to the brief's test order

`falsifiers/` holds two deliberately broken verifiers, built on every run.

`naive_argmax` is the exact failure the brief says the task exists to catch. Note where it
goes red: **`tv_det_good = 0.8557` while `tv_det_bad = 0.0022`.** It passes the floor,
determinism, perfect-drafter and degenerate tests, and it passes the BAD-drafter histogram
cleanly — because a drafter proposing the least likely token is never the argmax, so the
naive rule never accepts, falls through to sampling the target row, and converges perfectly.

The brief says: *"Run tens of thousands of verifications with a deliberately BAD drafter ...
The histogram must converge. Then repeat with a GOOD drafter."* On this evidence that order
is backwards — **the bad-drafter test does not catch the bug the task exists to catch.** The
good-drafter histogram is the discriminator, and should be stated first in any reissue of
this brief.

`no_residual` keeps the correct acceptance test and samples the replacement from the target
row rather than the residual — the subtler mistake, and the one a code review is most likely
to pass. It scores 0.1239 on `tv_det_good`, again with every structural test green.

## Is there an amalgamation to build here? Not from the implementations — from the interface

Unlike Brief D, there is nothing to win by rewriting the body. All five produce identical
output, all five are at the sampling-noise floor, and the two micro-optimisations available
were measured and are not worth taking:

- **Every entrant allocates and copies the whole row on rejection** to build the residual
  (`std::vector<float> residual(target_rows[i].begin(), ...)`). At this model's vocabulary of
  248,320 that is **10.6 us per rejection**, against an ~11 ms forward pass. 0.1%. The
  allocation is avoidable — the residual is just `p` with one coordinate reduced, so it can
  be CDF-walked in place with no copy — but it is not worth a rewrite on its own.
- **Every entrant sums and CDF-walks in `float`.** At V = 248,320 the float row sum comes to
  0.999824 rather than 1.0, and a float walk picks a different token from a double walk on
  **63% of draws** — which sounds alarming and is not. Almost all of it is neighbour-swapping
  among near-identical tail tokens: the induced sampling law sits **TV = 6.5e-4** from the
  true row, two orders of magnitude below the ~4e-2 the batched forward pass already
  contributes. Measured before claiming, and the claim does not survive the measurement.

**The real gap is that the brief's interface is wrong for this system, and no entrant could
have known.** Brief C hands the verifier "proper distributions ... non-negative, sums to ~1",
so all five sample from raw probability rows. `MlxBackend::generate` never samples from a raw
row. `Sampler::sample` applies, in order: repetition penalty, **the grammar mask**,
temperature, top-k, top-p, min-p. Two consequences:

1. Verifying against raw softmax while the loop samples from the transformed distribution
   means the committed tokens follow neither law. The "identical to ordinary sampling"
   guarantee is void in exactly the way that matters, and it would be invisible in testing —
   the text stays fluent.
2. Worse, speculation could **accept a token the grammar forbids**. The mask is a hard
   constraint: `sampler.cpp` treats an all-masked row as `no_legal_token`, "a build defect".
   A speculative path that commits unmasked tokens can emit malformed tool-call JSON.

So the adoption shape is: take any entrant's body (`e3` is the most compact at 97 lines), and
wrap it in an adapter that builds each target row by running **the same `Sampler` transform**
the non-speculative path runs, before verification — and that filters the drafter's proposals
through the grammar mask, since a masked-out proposal has p = 0 and is rejected with
certainty, wasting a draft slot.

Combined with `draft_probs[i] = 1.0` from the section above, the acceptance test collapses to
`u < p(t)` on the post-mask distribution — no division, and the residual is exactly `p` with
the drafted coordinate zeroed. Simpler than what any entrant wrote, and the only version that
is correct *in this loop*.
