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

---

# Phases 1–9

All nine phases landed. Below is what is real, what is measured, and what is not done.

## Measured on this machine (2026-07-30)

```
gate                     19 tests, 5.7 s          (budget: 5 min)
gate reliability         5/5 = 100%, median 5.55 s, spread 0.09 s
real model               Qwen3.6-35B-A3B-MLX-4bit, in-process via MLX
  end-to-end turn        prompt -> prefill -> masked decode -> grammar accept
  decode                 22.9 tok/s     ttft 1775 ms     300 tokens to a complete turn
blast_radius engine      0 wmiss / 179-of-179 corpus;  15 wmiss / 34-of-42 HOLDOUT
log_triage engine        34 weighted, 71-of-75 exact on the 25-tree corpus
mutation testing         3/8 killed, 5 survivors (below)
ratchets                 6 of 6 live and green; each proven capable of red
```

The real-model turn is the claim that matters: the model emitted `</think>`, answered
"Two plus two equals four.", and the grammar **accepted** on `<|im_end|>` — stopping
exactly there, by grammar state, with no text matching anywhere in the path.

## Where the spec was overruled, with the argument

**§5.6 pins the JSON tool-call syntax.** Qwen 3.6's own `chat_template.jinja` specifies
XML (`<function=name>` / `<parameter=x>`), which v1 verified and parsephone re-verified.
§19.6 says re-measure before acting on any document *including the spec*, so the model's
template wins: enforcing JSON with a mask would push the model off its trained
distribution on every call. `TurnGrammar` enforces the XML form via
`parsephony::ToolCallGuard`, schema-aware, byte by byte.

**§18.5 says "the 11 cookoff entrant implementations".** 11 for blast_radius, 17 for
log_triage. All 28 carried.

## Reused rather than rebuilt, and why

§2.2's test is "would a competent team building this fresh choose to build it?" — four
components pass it, so they were vendored instead of rewritten:

| | |
|---|---|
| `frankentok` | Qwen 3.6 tokenizer. HF-parity encode over 15,045 cases, streaming decode byte-identical to batch, binary cache. A hand-rolled replacement was written first in this session and **deleted**: its pretokenizer approximated the Split regex and skipped NFC, both live encode bugs frankentok had already found and fixed. |
| `parsephony` | JSON PDA + Qwen XML `ToolCallGuard` + token mask engine. 1000/1000 valid constrained generations, 17.7 ns/step. |
| `graft_engine` | Whitespace-tolerant edit application, cookoff-merged, refusal-first. |
| `src/model/mlx/` | v1's debugged forward pass for this exact checkpoint: hybrid linear/full attention, 256-expert switch MoE, gated q-projection, partial RoPE. Model math, not harness. |

## Findings

**The mutation harness was measuring itself.** Its first run reported a perfect 6/8
killed. Three separate defects made every copy fail for reasons unrelated to any
mutation: `ignore_patterns("build*")` also matched `simdjson/builder.h`; excluding
`node_modules` dropped a gate test and broke the pinned manifest; and `copytree`
dereferenced npm's `.bin` symlinks. Fixed, and a **null mutant** now runs first — an
unmutated copy must build and pass before any kill is believed. The honest score is
3/8.

**Surviving mutations, unfixed and named** (§11.4: a survivor is a finding about the
suite):

| site | why it survived |
|---|---|
| `grammar.cpp:105` | `permitted()`'s tool-call branch is only reached by real-model tests, which are excluded from the gate. Needs a small committed vocab fixture. |
| `sidecar.cpp:85` | The sidecar's dispatch loop has no test; it needs a spawn-and-drive harness. |
| `agent.cpp:168`, `agent.cpp:287` | `Agent::step` is not driven end-to-end in the gate — same vocab-fixture blocker. |
| `verification.cpp:86` | **Fixed.** `tests/loop/test_verifier.cpp` was written for it. |

The single highest-value next task is a committed miniature Qwen-shaped `tokenizer.json`
fixture; it unblocks gate-level testing of the grammar and the whole loop, and would
close three of the four remaining survivors.

**The sandbox tests found two real holes** when first run: `/tmp` is a symlink so
Seatbelt subpaths need `realpath`, and a blanket `/private/var/folders` allowance
covered the user's entire temp tree. Both fixed; sandboxed processes now get scratch
space via `TMPDIR` pointed inside the jail.

## Not done, stated plainly

- **T2 containers.** `SandboxTier::T2_Container` **refuses** rather than downgrading to
  T1. §7.2 requires T2 for unattended runs, so unattended runs are not available.
- **Speculative decoding.** The config seam (`draft_model_dir`) exists; the
  implementation does not.
- **`lmp/edit` through the extension.** §12.4 wants workspace edits applied via VS
  Code's edit API for undo and diff review. The sidecar writes files directly today; the
  notification type exists and is exempted in `ratchets.json` with that reason.
- **Approval round-trip.** The HITL router, risk scoring and approval cards are built,
  but the sidecar passes a null approver, so escalation currently denies. Deny-by-default
  is the honest behaviour with nobody to ask; wiring the UI approver back is a small job.
- **The `.vsix` is not packaged**, and `bin/lmp_sidecar` is not copied into the
  extension. `cmake --build --preset dev` produces the binary at
  `build/src/surface/lmp_sidecar`.
