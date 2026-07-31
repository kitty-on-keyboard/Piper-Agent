# parsephony — synthesis design

Goal: one parser that beats all 11 cookoff entries, aimed at LM-Pipe's real workload
(Qwen streams tokens → we need tool-call fields ASAP, guaranteed well-formed, low alloc).

## What each entry contributed

| PR | Entry | Idea taken | Flaw fixed here |
|----|-------|-----------|-----------------|
| #1 | arena-zerocopy | Arena bump alloc, zero-copy `string_view` | used a temp `std::vector` per container (heap churn); fixed arena could `bad_alloc` |
| #2 | lazy | **Flat tape** (type/offset/len) + lazy value decode | no skip pointers → field lookup was O(n) subtree walk |
| #3 | feat-impl | strict UTF-8 validation | naive byte loop |
| #4 | typed-sink | zero-DOM typed sink (parse straight into struct) | didn't compile (namespace bug) |
| #5 | schema-codegen | compile-time schema specialization | required a Python codegen build step; never decoded escapes |
| #6 | typed-schema | correctness bar: 318/318 JSONTestSuite, surrogates, depth caps | string-heavy, slowest path |
| #7 | bake-cache | persistent parse-plan cache | marginal for 200-byte payloads; dropped |
| #8 | perfect-hash | perfect hash key → KeyId in O(1) | Python codegen step; skipped escapes entirely |
| #9 | round2-a | resumable SAX, cross-chunk surrogate correctness | byte-at-a-time `while(!processed)` loop |
| #10 | round2-d | token-ID fast path (skip UTF-8 materialization) | toy 22-token vocab; `map<vector<int>>` **allocates per lookup** |
| #11 | round2-b | push-down automaton, grammar/parser "duality" | never actually emitted a decoding mask |

## Architecture — one grammar, three consumers

```
            ┌────────────────────────────────────┐
            │  PDA core (resumable, no recursion)│   ← #11 automaton + #9 resumability
            └────────────────────────────────────┘
               │              │               │
        ┌──────▼─────┐  ┌─────▼──────┐  ┌─────▼─────────┐
        │ Tape       │  │ Typed sink │  │ allowed_next()│
        │ builder    │  │ (constexpr │  │ decode mask   │
        │ (lazy)     │  │  key hash) │  │   ** NEW **   │
        └────────────┘  └────────────┘  └───────────────┘
```

Plus a **one-shot bulk path** that skips the PDA for complete buffers
(SWAR scan + arena + tape) — that's where raw throughput comes from.

## Six things no single entry had

1. **Tape with skip pointers** — O(1) subtree skip; pull 2 fields out of a tool call
   without materializing a DOM at all. (#2 had the tape, no skip; #1 had zero-copy, but built a DOM.)
2. **constexpr perfect hash** — compile-time, no Python build step (#5/#8 both shelled out to Python).
3. **SWAR string scanning** — 8 bytes/iteration. No entry did any bulk scanning.
4. **Real token DFA** over the actual vocab, zero allocation (#10's idea, but its
   `std::map<std::vector<int>,…>` allocated a vector on every single lookup).
5. **Actual constrained-decoding mask** — `allowed_next()` a sampler can use to mask
   logits, so Qwen *cannot* emit invalid JSON. (#11 named this "duality" but never built it.)
6. **Correctness at #6's bar on the fast path** — 318/318 JSONTestSuite, not just on the slow one.

## The honest performance thesis

simdjson beat every entry by 10–40× — but it's built for MB-scale documents and pays
fixed setup cost (padded buffer alloc, stage1 structural index, stage2 tape build).
LM-Pipe's payloads are **100–400 byte tool calls**. In that regime the setup cost dominates
and a lean scalar parser can genuinely win. That is the claim to measure — not
"we out-SIMD simdjson on 1 MB of Twitter JSON," which would be false.

Measured claims required:
- throughput vs nlohmann + simdjson on a shared fixture set, same harness
- TTFF for the streaming path
- JSONTestSuite pass rate
- normalized speedup-vs-nlohmann against all 11 entries (each entry benchmarked
  itself against nlohmann, so that ratio is the common yardstick)
