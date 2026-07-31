# Phase 0 — repo, gate, ratchets, event log, arenas, SPSC channel

**Exit criterion (spec §17): "Gate runs and is verified to run the expected test count."**
Met, and verified by intervention rather than by reading the output — see
[Falsification](#falsification) below.

## What landed

| | |
|---|---|
| Build | CMake 3.24+, C++20, `-Wall -Wextra -Wpedantic -Werror` plus `-Wshadow -Wconversion -Wsign-conversion -Wold-style-cast`. Apple Silicon is a `FATAL_ERROR`, not a fallback. |
| Assertions | `cmake/LmpAssertions.cmake` strips `-DNDEBUG` from **every** configuration, including Release. |
| L0 platform | `arena.hpp`, `clock.hpp`, `spsc_channel.hpp`, `event_log.{hpp,cpp}`, `fs.{hpp,cpp}` |
| Gate | 11 tests, **0.6-1.7 s** against a 5-minute budget. Contents pinned by count *and* name. |
| Ratchets | 6 gates, 3 live and self-falsifying, 3 dormant and failing-on-activation. |
| §18 corpora | Ported byte-identical, loading in the gate, scoreboards building. |
| CI | `.github/workflows/gate.yml` — gate + ASan/UBSan. No write permission, no commit step. |

## Measurements

Everything below was measured in this repo, not carried from a document.

```
gate                     11 tests, 0.57 s warm / 1.74 s cold   (budget: 5 min)
gate under ASan+UBSan    11 tests, 2.34 s   clean
checks executed          549 across 9 binaries (+2 script gates)
blast_radius corpus      187 cases (179 headline + 8 contested) + 42 held out
log_triage corpus        25 trees / 971,544 bytes + 7 held out / 274,221 bytes
diagnostics in the key   61, of which 61 locators appear verbatim in their own log
```

### Re-measurements that contradict the spec

§19.6 says to re-measure before acting on any document, including the spec. Three results:

| Spec §  | Spec says | Measured | Effect |
|---|---|---|---|
| §1 preamble | v1 is 44,104 non-test lines | **40,471** | none; the point stands |
| §18.5 | "the 11 cookoff entrant implementations" | 11 for blast_radius, **17** for log_triage | all 28 carried |
| §18 ⚠ | `.gitignore`'s `*.log` will swallow the log corpus | v1's `.gitignore` **already had** the negation | carried forward, plus a CI check and a byte-count assertion |

The largest v1 file is 5,196 lines (`src/agent/react_loop.hpp`), exactly as stated.

## Falsification

Nothing here is trusted because it printed green. Each mechanism was made to fail first.

**The gate manifest.** Bumping the pinned count 11 → 12 fails on the count; renaming
`test_arena` in the manifest while leaving the count alone fails on the names, in both
directions ("manifest lists X but -L gate does not select it" *and* "-L gate selects Y
which is not in the manifest"). Restoring returns green. A bogus label is asserted to
select zero while `gate` selects eleven — the assertion v1 never made.

**The ratchets.** `--self-test` plants a violation per live gate in a throwaway copy and
requires it to fire. This immediately found a defect in itself: the dead-code probe used a
literal symbol name, `scripts/` is a scanned source root, so the name appearing in the
planting script counted as a reference and the planted dead symbol looked alive. The gate
was right; the probe was contaminated. Fixed by generating the symbol name — not by
excluding `scripts/` from the scan, which would have hidden a directory from every gate to
fix one probe.

**The check framework.** `EXPECT_FAILING_CHECKS` runs checks that must fail and asserts the
framework counted exactly them. `test_main.cpp` fails any binary that reports zero checks.

**Assertions.** Two proofs: a `#error` if `NDEBUG` is defined, and a forked child that must
die of `SIGABRT` on a failing `assert`. Checking the macro checks a proxy; killing a
process checks the thing.

## Findings about the ported corpora

Both are recorded in
[bakeoff/blast_radius/KEY_CORRECTIONS.md](../bakeoff/blast_radius/KEY_CORRECTIONS.md); the
verbatim v1 READMEs are not edited.

1. **The three §18 key defects are real and are fixed.** `benign_ctest`,
   `look_cargo_offline` and `chain_mkdir_cmake` were labelled `parsed` while the same key
   labels `make clean`, `make install`, `npm run build`, `npm run format` and `npm install`
   `partial` under the same rule 7. All three run project-supplied code whose bytes are not
   in the string. Status moved to `partial`; flags unchanged.

2. **v1's results table was already stale on arrival.** Scoring `e12` against a
   reconstructed pre-correction key gives 169 / 8 alarms / 7 status-misses / 106 exact,
   where the README claims 169 / 9 / 7 / 105. The port is byte-identical, so the
   disagreement is between v1's README and v1's own corpus: the table was measured *before*
   the `effective_truth()` fix that the same README describes. Only the primary metric,
   weighted misses, survives — so the ranking stands and every other column must be
   re-scored before it is quoted.

## Carried forward, deliberately

- `bakeoff/blast_radius/entrants/{incumbent,e00_merged}.hpp` do not build. Both are thin
  adapters over v1 source that v2 does not carry. The other eleven — the actual blind
  submissions — are self-contained and still score. Re-wire in phase 5 and **re-measure**;
  do not quote the old baseline.
- Two dead-code exemptions, each stating when it expires: `SystemClock` (no `main()` until
  phase 4) and `base64_decode` (no `ReplayBackend` until phase 2).

## What phase 1 may assume

A green gate that has been proven capable of red; an event log whose serialisation is
byte-faithful for input that is *not* valid UTF-8 — which is precisely what the 944
byte-fragment tokens in §5.3 will hand it; an arena, a clock seam, and an SPSC channel of
owned whole messages for the §4.2 framing reader.

**Phase 1 must not start by adding a tokenizer to `src/model/`** without also adding
`src/model` to the layer map's live subjects — it is already in `scripts/ratchets.json`, so
the layer gate will pick it up automatically, but `min_subjects` for `layers` is pinned at
5 and should rise with it.
