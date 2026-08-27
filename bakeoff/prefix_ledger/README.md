# PrefixLedger cook-off (Jules round 2, Brief D)

The bookkeeping half of KV rollback: append, prefix comparison, truncation, fingerprint.
Adoption
target is `KvCacheLedger` (`src/model/kv_cache.hpp`), which today offers `append` and
`clear` and nothing between — speculative decoding needs the thing between.

One PR per entrant.

## Standing, 2026-08-01 — COMPLETE, 5 of 5, plus an amalgamation that beats all of them

```
            agree  semantics  fp_collide  fp_constructed  fp_path  rollback   p50      bulk_trunc
e1  (PR 1)    0        0          0          1024 <-        0         1      12.71us    49.88us
e2  (PR 2)    0        0          0          1024 <-        0         1       9.54us    50.49us
e3  (PR 3)    0        0          0          1024 <-        0         1       6.71us    58.98us
e4  (PR 4)    0        0          0          8192 <-        0         1      10.29us    51.51us
e5  (PR 5)    0        0          0          8192 <-        0         1       8.12us    52.09us
amalgam       0        0          0             0           0         1       7.46us     0.02us
```

**All five entrants are correct**, and on every column the brief names they are
indistinguishable. Two columns separate them from `amalgam/`, and the first is a defect none
of their own tests could have found.

e4 shipped its tree one level deeper than the brief specified
(`prefix_ledger/include/...`). That is a layout deviation, not a defect — the harness was
pointed at the nested directory rather than the entrant being marked down for a folder.

### `fp_constructed`: every entrant's fingerprint collides by construction

The field split into two designs, and both are breakable:

- **e1, e2, e3** — polynomial rolling hash `h = h*P + token` mod 2^64, un-rolled with P's
  modular inverse. **Collides at 1024 tokens.**
- **e4, e5** — FNV-1a style `h = (h ^ token) * PRIME`, un-rolled with the prime's inverse.
  A better fold, and 8x more resistant: **collides at 8192 tokens.** Still collidable.

Both are forced into the same corner by the same requirement: O(removed) truncation from a
single scalar of state needs an *invertible* fold, and invertible-and-cheap means enough
algebraic structure to build a collision on. For a Thue-Morse word and its complement the
hash difference is divisible by a product whose 2-adic valuation grows quadratically, so it
exceeds 64 at a short length. Measured against each entrant's actual constants.

1024 tokens is a short prompt, and token ids are partly user text. This is defence-in-depth
rather than a live exploit — `KvCacheLedger` still confirms id-by-id, so a collision alone
does not reuse stale context — but the fingerprint is the part callers are invited to trust
*cheaply*, and one collidable this easily is not worth trusting.

**This was a hole in the harness first.** The original `fp_collide` corpus was random
near-misses, which do not find a structured collision: all five scored 0. The
`fp_constructed` column was added afterwards, and it is the column that discriminates.

### The fix removes the constraint that caused it, and makes truncation O(1)

`amalgam/` keeps a running hash **per position** (`hashes_[i]` is the fingerprint of the
first i tokens). Three consequences:

1. Truncation stops un-rolling anything — a resize and a read. **`bulk_trunc` measures
   0.02 us against 50-59 us** for dropping 50,000 tokens from a 100,000-token ledger.
2. **The fold no longer has to be invertible**, which is the constraint that forced both
   designs above. So it is splitmix64's finaliser, and the Thue-Morse identity — an
   algebraic fact — does not apply. No collision at any length up to 2^16.
3. `fingerprint_at(k)` becomes O(1). That is what the consuming project actually wants:
   `plan_reuse` returns `reusable = k`, and the caller then needs that prefix's fingerprint
   to key cached state. Every entrant makes it O(k).

The cost is 8 bytes per token — 800 KB at 100,000 tokens, against a 19 GB model.

**Adopted into `src/model/kv_cache.hpp`.**

### What was taken from the entrants

The comparison loop is **e3's `std::mismatch`**, kept verbatim — e1 hand-rolled a 256-token
`memcmp` block loop and e2 a binary search over `memcmp`, and both are beaten by letting the
compiler vectorise the obvious thing. e3's additive constant is kept in spirit: a token id
of 0 must not fold as an identity. `amalgam` and e3 share a `plan_reuse` body, so any p50
gap between them is measurement noise.

## The harness

`scoreboard.cpp` is neutral: written before any entrant was read, graded against a
brute-force `std::vector<TokenId>` reference, and using only the public API the brief
specified. `score.sh` compiles it once per entrant — N implementations of one symbol must
never meet in a translation unit.

Columns, and what each is for, are documented at the top of `scoreboard.cpp`. The one worth
naming here is `fp_collide`: a fingerprint that is a sum, an xor, or a polynomial over too
small a modulus passes every functional test in the brief and fails only that column — and
in production it fails as silent stale-context reuse, which is the exact bug the ledger
exists to prevent.

## Falsifiers — the harness is shown red before it is believed

Every entrant passing everything is not evidence until the scoreboard has been shown capable
of failing. `falsifiers/` holds two deliberately broken ledgers, built by `score.sh` on every
run:

| falsifier | planted defect | fires |
|---|---|---|
| `broken_divergence` | `divergent = (reusable != candidate.size())` | `agree_fail=2597`, `semantics_fail=2`, `fp_collide=13705`, p50 39 us, `trunc_ratio=93.7` |
| `broken_stale_hash` | running hash updated on append, not on truncation | `fp_path_fail=200` — and nothing else |

The second row is the more important one. It fires exactly one column and leaves the other
six clean, which is what says the scoreboard discriminates rather than just alarms. The
first row's `semantics_fail` names both directions of the asymmetry: it reports a strict
prefix as non-divergent and a strict extension as divergent, the single most likely way to
get this component wrong.
