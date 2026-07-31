# log-triage — the corpus, the scorer, and round 1

The answer key for the `log-triage-engine` cookoff, and the permanent home of the benchmark
afterwards. The target was `SubprocessVerifier::compact_command_output` / `line_is_diagnostic`
— seven substrings tuned for clang and swift. That matcher is now **deleted**; production
calls `log_triage::compact` (`src/tools/log_triage.hpp`), the consolidated engine scored at
the bottom of this file.

**This directory never ships to an entrant.** Their own corpora, their own scorers and
their self-reported numbers are not used for anything.

## The key is written by the compiler

This is the strongest key we have had, and the reason is that we did not write it.

Every case is a real source tree with a real defect, built by the real
cmake / clang / swiftc / cargo / pytest / python / ctest on the machine that generated it.
Each case is run **twice**:

| run | flags | what it is |
|---|---|---|
| **full** | everything on — carets, template backtraces, include stacks, `VERBOSE=1` | the corpus **input**, byte for byte: what `SubprocessVerifier` actually captures |
| **key** | the tool's own quiet mode — `-fno-caret-diagnostics`, `-ftemplate-backtrace-limit=1`, `--error-format=short`, `--tb=line` | the **key**: the tool's own answer to "what are the errors" |

The noise around each diagnostic is real build output too — full compiler command lines,
include paths, progress percentages — because the builds are real. `generate_corpus.py` is
the whole story; nothing in `corpus.jsonl` was labelled by hand.

The one judgement we make is *which* of the tool's diagnostics an agent needs, and it is
made mechanically from two things the generator already knows:

- **`primary`** — the FIRST error the tool emitted. Later errors are usually cascade.
- **`local`** — every error, fatal error or **note** whose location is a file inside the
  project. This is what tells the agent *which line of its own code to edit*. In a template
  blow-up there is no error in the project at all and the only local diagnostic is a
  `note: in instantiation of ... requested here`.

**Warnings are deliberately not in the key.** A failing build emits hundreds and they are
exactly what a compactor should drop; scoring their retention would pay an entrant for
crowding the error out.

> ⚠️ **A locator must be a substring the tool actually printed.** Three times while building
> this, a key entry was synthesised in a shape no tool emits — `path:line:col` for a Python
> traceback (which prints `File "x.py", line N`), rustc's `--error-format=short` message
> (which folds the caret label onto the end), pytest's absolute path (its normal output is
> relative). Each one made a case **nobody could answer**, which is the same defect the
> entrants found in the blast-radius key. `tests/bakeoff/test_corpus_loads.cpp` asserts every
> key entry is findable in its own log; it caught all three.

## Layout

| File | What |
|---|---|
| `log_triage.hpp` | The contract. **It did not exist during round 1** — see below. |
| `generate_corpus.py` | Builds the corpus by running the real toolchain. |
| `logs/*.log` | 25 real build logs, 971 KB, committed and **frozen**. |
| `corpus.jsonl` | The key: per case, the primary and local diagnostics with their context. |
| `corpus.hpp` | Loading and scoring. One implementation, shared by scorer and test. |
| `score.cpp` | Per-entrant scoreboard. `-v` per case, `-q` one line, `-j` per point. |
| `entrants/eNN.hpp` | Round-1 submissions, byte for byte, named by PR number. |
| `entrants/incumbent.hpp` | The pre-change `compact_command_output`, frozen. See below. |
| `holdout_logs/`, `holdout.jsonl` | 7 cases written AFTER the engine. Scored once. |
| `adapters/eNN.hpp` | **Ours.** The call, adapted to the entrant's signature. |

The corpus is validated by `tests/bakeoff/test_corpus_loads.cpp` (9 cases, 65 checks across
both bakeoffs) and the engine is pinned by `tests/bakeoff/test_log_triage_engine.cpp`
(8 cases, 38 checks), both labelled `gate` and both named in `tests/gate/gate_manifest.txt`.

> ⚠️ **Until 2026-07-31 the second of those did not exist, and this file said it did.** The
> claim here was that `src/testing/test_log_triage_corpus.cpp` (341 checks) and
> `src/testing/test_log_triage_engine.cpp` (13 checks) validated and pinned the engine in the
> standard gate. `src/testing/` has never existed, neither name has ever appeared in the gate
> manifest, and **nothing tested `log_triage::compact` at all**. The round-2 patch below then
> changed line selection on nearly every case in the corpus and the entire gate stayed green,
> because there was nothing for it to turn red. Corpus validation was real all along under a
> different path and name; the engine pin was not, and is now written.

```bash
cmake --build build --target log_triage_score_e05 && ./build/log_triage_score_e05 -v
```

Regenerating the corpus on a different Xcode WILL change the logs and break the pins. That
is intended: it should be a deliberate act, not a side effect of upgrading a laptop.

## The metrics

Scored at **three budgets — 2048, 8192, 16384** — because compaction only means anything
relative to a cap. 8192 is `SubprocessVerifier::execute`'s default and 16384 is what the
shell tool passes; 2048 is the pressure test. 25 cases × 3 budgets = **75 scoring points**.

- **Locator miss, weight 3.** Without `file:line` the agent does not know where to type, and
  cannot recover it by reading a file because it does not know which file.
- **Message miss, weight 2.** It does not know what is wrong — but it has the locator, so it
  can read the source and often see it. Strictly less bad, hence 2 and not 3.
- **Context miss, weight 1.** The source and caret lines. Losing them is what made the agent
  edit line 35 instead of line 50 in the incident recorded at `subprocess_verifier.hpp:407`.
- **Pass-through violation, weight 3.** A log that already fits must come back byte for byte.
- **Over budget scores the case as if it retained nothing.** Not a punishment tariff — the
  arithmetic. The caller hard-truncates anything past the cap, so those bytes were never
  delivered. Without this rule the best score would go to `return std::string(full)`.
- **Noise is reported and NOT weighted.** Some retained non-key text (the command echo, the
  final status) is legitimately useful and this corpus does not try to adjudicate which.

Both sides of every comparison are ANSI-stripped, so an entrant that strips escapes and one
that passes them through are judged on what they *kept*. The escape bytes still count
against the budget, which is the honest cost of not stripping.

## ⚠️ Round 1 had no contract, and it changed the ranking

The entrant repository's `main` was a **19-byte README**. The task lived only in a prompt, so
nothing had to compile against anything. Fifteen entrants produced fifteen signatures, and
the budget — the entire point of the function — is not the same *quantity* across them:

| budget parameter | entrants |
|---|---|
| `size_t` bytes | e04 e05 e06 e10 e11 e12 e13 e14 e01 |
| `size_t` **lines** | e02 |
| `size_t` **chunks**, defaulted to 5 | e03 |
| `size_t` **tokens** (its own `count_tokens`) | e09 |
| an entrant-defined `TriageConfig` | e07 |
| **none at all** | **e08, e15** |

So each entrant needs an adapter, and the adapters are **ours** — short, in `adapters/`, and
reviewable — while `entrants/` stays byte for byte. Three kinds:

- **direct** — the signature is the contract. Nothing is accommodated.
- **unit-searched** — the budget is in some other unit, so the adapter **binary-searches**
  for the largest value that fits. That is *more* than the real caller can do: production
  has no oracle. Marked in every scoreboard.
- **NO BUDGET** — e08 and e15 have no parameter through which the cap can be expressed.
  Nothing an adapter can fix; called as shipped.

**How much the search is worth was measured, not assumed.** `e02b` and `e09b` are the same
entrant bytes with the search replaced by an *estimate* — a byte-budget division at the mean
line length (130 B) and mean token length (14.6 B) measured over this corpus:

| | weighted | over budget | exact |
|---|---|---|---|
| e02 **with** the search | **78** | 0 | 64/75 |
| e02b, estimating instead | **516** | **31** of 75 | 31/75 |
| e09 **with** the search | **182** | 0 | 55/75 |
| e09b, estimating instead | **302** | **10** of 75 | 43/75 |

**e02's first place is the adapter, not e02.** Given a byte cap it cannot honour, it
overshoots on 31 of 75 points. That is the number to quote.

## Round 1 results (2026-07-30)

15 submissions at `cat-collector-king/log-triage-engine`. **All fifteen shipped real code** —
no empty commits this round, unlike blast-radius's 3 of 14. **None of the fifteen PR
descriptions contains a blind-spots section or a challenge to the spec.**

| | weighted | loc miss/177 | msg miss/195 | ctx miss/384 | over | passthru | exact/75 |
|---|---|---|---|---|---|---|---|
| e05 | **99** | 9 | 8 | 56 | 0 | 0 | 63 |
| e10 | 133 | 16 | 11 | 63 | **5** | 0 | 61 |
| e06 | 167 | **5** | **2** | 148 | 0 | 0 | 39 |
| e13 | 202 | 13 | 19 | 80 | 0 | **15** | 33 |
| e11 | 236 | 29 | 32 | 85 | 0 | 0 | 50 |
| e04 | 244 | 35 | 36 | 67 | 0 | 0 | 46 |
| e07 | 261 | 31 | 37 | 85 | 0 | 3 | 46 |
| **incumbent** | **261** | 34 | 37 | 85 | 0 | 0 | 50 |
| e12 | 270 | 38 | 40 | 76 | 0 | 0 | 42 |
| e14 | 310 | 28 | 28 | 170 | 0 | 0 | 38 |
| e03 | 866 | 109 | 119 | 256 | 0 | 15 | 3 |
| e08 | 866 | 114 | 126 | 269 | **60** | 1 | 14 |
| e15 | 908 | 114 | 126 | 269 | **60** | 15 | 0 |
| e01 | **1350** | 177 | 195 | 384 | 0 | 15 | 3 |
| *e02 (oracle-assisted)* | *78* | *8* | *11* | *32* | *0* | *0* | *64* |
| *e09 (oracle-assisted)* | *182* | *19* | *14* | *97* | *0* | *0* | *55* |

**e05 is the honest winner**, 62% better than the incumbent with no accommodation at all.

### The incumbent is better than "seven substrings" sounds

261 weighted, **50 of 75 exact**, 81% locator recall, and it **never exceeds its budget**.
Most of a real build log has exactly one diagnostic in clang's format, and that is the case
it was built for. It loses where a log is not shaped like that, and the per-case numbers say
where: `build_no_matching_ctor` 63, `cargo_two_errors` 44, `python_assert_midway` 42,
`bare_error_limit` 26, `build_three_files_fail` 24.

### e01 returns the empty string on every single case

`default_priority_func` matches `FATAL`, `CRITICAL`, `ERROR`, `WARN` — **uppercase**. No
compiler emits any of them. Every line scores 0, no block is ever pushed, and the output is
empty: 177 of 177 locators missed, 0 bytes of noise. It was written for syslog-style
application logs. Checked before being believed, because a 100% miss is normally a harness
bug.

### e08 and e15 blow the cap on 60 of 75 points

Neither takes a budget. `compact_command_output` exists to fit `max_bytes`; an
implementation that cannot see `max_bytes` has not implemented it.

## ⚠️ Merging pays here, and it did not last round

The per-point oracle, computed from `score.cpp -j` **before** any merge was planned:

| | weighted |
|---|---|
| best single entrant (e05, no oracle) | 99 |
| best single entrant (e02, oracle-assisted) | 78 |
| oracle picking the best whole ANSWER per point | **22** |
| oracle picking the best COMPONENT per point | **17** |

**72 of 75 points are already solved exactly by at least one entrant.** That is the opposite
of the blast-radius round, where the component ceiling was only ~18% better than the best
entrant and most of the work had to be written from scratch. Here the entrants are
genuinely complementary and the job really is a merge.

The strengths to take are visible per family:

- **e06** has the best diagnostic recall of anyone — 5 locator and 2 message misses out of
  177 and 195 — and the worst context retention, 148 of 384. It finds the error lines and
  drops the caret blocks.
- **e05** is the only entrant exact on all of `swift` (9/9) and it never overshoots.
- **e02** is exact on 17 of 25 cases, more than any other, but see the adapter warning.
- **e12** is the ONLY implementation that solves `cargo_two_errors`, where rustc puts the
  locator on a different line from the message.
- **e10** and **e02** are the only ones that solve `build_no_matching_ctor`, where the
  candidate signatures live in `note:` lines.

### What nobody solves

Only three of 75 points resist every implementation:

- **`build_template_deep` at 2048 and 8192.** 68 KB of libc++ backtrace for one error, and
  the only line naming an editable file is a `note:`. Best weighted 7, by e02/e07/e09/e11.
- **`bare_error_limit` at 2048.** 19 diagnostics and a 2048-byte cap: a genuine capacity
  limit, not a triage failure. Best weighted 8.

## The consolidated engine

`src/tools/log_triage.hpp`. `line_is_diagnostic` is **deleted**, not kept alongside — two
implementations of "is this line important" is how the caller ends up asking the wrong one.
The frozen pre-change code lives at `entrants/incumbent.hpp` so the baseline stays measurable.

**Where it is actually called.** One site: `src/tools/registry.cpp:488`, in the shell tool
handler, on every command result. There is no `SubprocessVerifier` in the tree — this file
said the engine was "wired into `SubprocessVerifier::compact_command_output`" until
2026-07-31, and that class and method are v1 names that did not survive the port.

That one site is the whole path, which is worth stating because the absence of a `compact`
call in `src/loop/verification.cpp` reads like a hole and is not one: `Verifier` does not run
subprocesses itself. `run_and_record` and `prove_falsifiable` both go through
`registry_.execute("shell", ...)`, so `VerificationRecord::detail` is already compacted
output. Verification is triaged, by the shell handler, one layer down.

| | corpus, round 1 | corpus, **round 2** | holdout, round 1 | holdout, **round 2** |
|---|---|---|---|---|
| weighted | 34 | **15** | 20 | **0** |
| locators lost | 1 / 177 | **0** | 0 / 39 | **0** |
| messages lost | 1 / 195 | **0** | 0 / 48 | **0** |
| context lost | 29 / 384 | **15** | 20 / 117 | **0** |
| exact | 71 / 75 | **73 / 75** | 19 / 21 | **21 / 21** |
| over budget | 0 | 0 | 0 | 0 |

**Quote the holdout column.** The corpus number is a memory test — the engine was written
with that corpus open, iterating until the scorer went quiet.

⚠️ **And for round 2, do not quote the holdout column either.** See "one holdout case is
burned" below: `ho_rustc_no_cargo` is now burned twice over, and it is where the entire
20 → 0 comes from.

### Round 2 (2026-07-31): three defects in the shipped engine

Found by re-reading the merged engine against the entrants, not by a new corpus.

1. **Phase 1 selected by anchor-ness, so round 1's warning fix never bit.** Round 1 demoted a
   warning's bare ` --> path:L:C` to `kWarning` in `inherited_score`, which fixed the *score* —
   but phase 1's only test was `if (!lines[i].anchor)` and it never read the score. On
   `ho_rustc_no_cargo` all 240 warning locators were still anchors, each a distinct line so
   message-dedup did not collapse them, and phase 1 filled the whole 2048-byte budget with
   them before the three real errors' caret blocks were considered. A `diagnostic` flag
   (anchor **and** score ≥ `kAnchor`) is what makes the demotion bite. Worth the holdout
   20 → 0 on its own, and it also solves `cargo_two_errors` on the corpus.
2. **clang's include stack scored as a full anchor.** `In file included from …/vector:312:`
   carries `path:line:`, so `find_locator_end` fired and nothing else claimed the line — it
   became an anchor at `kAnchor`, or `kLocalAnchor` when the header happened to sit in the
   project. Each is 150–200 bytes of SDK path and a template blow-up emits dozens. **e08 was
   the only entrant to recognise these** (its `TEMPLATE_INST` state) and the round-1 merge
   did not take it.
3. **Locality was ranked; severity within locality was not.** Principle 2 ranks own-code over
   system headers, but inside the system tier a `note:` and an `error:` both scored `kAnchor`,
   so libc++'s twenty `note: in instantiation of …` lines tied with the one real `error:` and
   crowded it out of `build_template_deep`. `kSystemNote = 320` applies **only** when the
   locator is not local, so local notes are untouched — which is what preserves e10's
   `build_no_matching_ctor` win. Takes `build_template_deep` at 8192 from w=4 to solved and at
   2048 from w=9 to w=2; the round-1 table above records w=7 as the best any implementation
   had managed on that case.

Speed is unchanged — all three are O(1) per line, no new passes. Re-timed on a 15 MB,
233k-line synthetic build log, best of 5: **0.045 s before, 0.040 s after.**

#### A fourth defect, found by writing the missing test

`compact` could return **one byte over budget**. When no line at all is affordable the whole
log is a single gap and the output is a lone `[... N lines elided ...]\n` marker, whose own
length the packer never checks — 27 bytes at `N=800`. The final clamp did not save it:
`rfind('\n', budget_bytes)` searches positions ≤ the budget, so it returns the newline
sitting exactly *at* `budget_bytes` and `resize(cut + 1)` is a no-op that looks like a
truncation. Cutting from `budget_bytes - 1` is the fix.

Round 1 violated its budget on **32 of 137,408** (log, budget) points — one per log, at the
single budget where that newline lands on the cap. Round 2 violates none. It was never
reachable in production, where the cap is 8192 or 16384, and no scoring point is at a budget
that small, which is exactly why only a contract test could find it.

**The remaining 15 is almost all `bare_error_limit` at 2048 (w=13), and it is capacity, not
triage.** 19 diagnostics ≈ 1064 B plus 38 context lines ≈ 1254 B against a 2048 B cap; the
output uses 2013 of 2048, the 13 gap markers cost 325 B and the next context pair needs 65 B.
There is nothing to reclaim without shortening the marker format.

### The comparison is not one-sided, and the honest reading is this

*Round 1's reading, kept as written. Round 2 takes the engine's holdout weighted to 0, which
settles the aggregate in the engine's favour — but on a case that was already burned, so the
tension below is resolved by a number that is not blind, and the reasoning still stands.*

On the held-out set the frozen incumbent scores **18 weighted to the engine's 20**. The
weights were published before anything was scored and are not being changed now that the
result is in. What the two numbers are made of:

| | locators lost | messages lost | context lost | exact |
|---|---|---|---|---|
| engine | **0** | **0** | 20 | **19/21** |
| incumbent | 2 | 1 | 10 | 16/21 |

The engine never loses an address or a message on blind data; the incumbent loses two
addresses and a message. Losing an address is the failure that makes an agent edit the wrong
line and cannot be recovered by reading a file, because it does not know which file. Losing a
caret line costs one cheap `read_file`. The weighted total puts those at 3:1 and the engine's
whole 20 is context on a single Rust case — so the aggregate favours the incumbent while
every component the aggregate calls important favours the engine.

That is a real tension and it is why the engine ships: on the corpus it is 34 against 261 and
71/75 against 50/75, on the holdout it is exact on more points, and it never once loses an
address. But the ratio between locator and context weight was a judgement made before any
data existed, and it is the thing to revisit first with a fresh set.

### ⚠️ One holdout case is burned — twice over

`ho_rustc_no_cargo` exposed two genuine defects in round 1 and both were fixed with its
output in front of me:

1. **A bare locator line asserted maximum severity instead of inheriting it.** rustc prints
   the same ` --> path:L:C` under a `warning:` as under an `error:`, so a crate with 120
   unused-variable warnings produced 120 lines that outranked the real errors' source blocks
   and filled 16 KB with warning addresses.
2. **Context was bounded by a radius, not by structure.** A comment in the engine claimed six
   lines "covers the widest block any tool prints"; rustc's gutter is nine.

**Round 2 burned it again.** Defect 1 above was found by reading this case's output; defect 1
of round 2 — that the fix never actually bit, because phase 1 packs by anchor-ness and never
reads the score — was found by reading *the same case's output again*, after it had been
declared fixed. So the holdout **20 → 0 is not a blind result and must not be quoted as one.**

The defensible claims are narrower, and these are the ones to use:

- The other six holdout cases were perfect before round 2 and are perfect after it. They
  remain blind; nothing in round 2 was driven by them.
- Round 2's defects 2 and 3 were found on **corpus** cases (`build_template_deep`) with the
  holdout untouched. Ablated — each fix applied alone to the round-1 engine, measured, not
  reasoned about:

  | | corpus w | corpus exact | holdout w | holdout exact |
  |---|---|---|---|---|
  | round-1 engine | 34 | 71/75 | 20 | 19/21 |
  | fix 1 alone (the `diagnostic` flag) | 26 | 72/75 | **0** | **21/21** |
  | fixes 2+3 alone (include stack, `kSystemNote`) | 27 | 71/75 | 20 | 19/21 |
  | **all three** | **15** | **73/75** | **0** | **21/21** |

  **These do not add up, and that is the finding, not a rounding error.** Separately the
  fixes are worth 8 and 7 weighted; together they are worth 19. Fix 1 stops 240 warning
  addresses eating the budget, and fixes 2 and 3 are what spend the freed budget well —
  neither is worth much without the other. Do not quote any row here as a fix's "cost" or
  "contribution"; an ablation measures a removal, not a share.

  The honest split on blindness: **fixes 2 and 3 move the holdout not at all.** Every bit of
  the 20 → 0 is fix 1, and fix 1 came off the burned case.
- One case has now supplied three of the five defects ever found in this engine. That is a
  statement about the case, not about the engine's quality on unseen logs.

`tests/bakeoff/test_corpus_loads.cpp` asserts the holdout stays harder than the tuned set —
though only **structurally**, comparing mean log bytes per case, and its comment still says
"the engine does not exist yet". Round 1 described this as a per-point weighted tripwire at
0.95 vs 0.45; it has never measured that, and with the holdout now at 0 weighted a per-point
version of it would fail. **The next round needs a fresh set**, ideally one whose cases do
not come from this generator at all, and that is now the blocking item for this benchmark
rather than a nice-to-have.

### Speed was a disqualification no score would have shown

Timed on a 12 MB, 200k-line build log — a runaway build, which is exactly when compaction
matters most:

| | time |
|---|---|
| incumbent | 0.01 s |
| **engine** | **0.03 s** |
| e06 | 0.06 s |
| e10 | 0.10 s |
| **e05 (the honest cookoff winner)** | **19.13 s** |

e05's packer recomputes the whole rendered output for every candidate line. The corpus never
caught it because its largest case is 118 KB. **Score alone would have shipped a 19-second
stall into a function that runs on every shell tool result.** The candidates were timed
before any of their code was adopted, and that check belongs in the method.

## Known limits of this corpus, stated

1. **`build_ok_verbose` cannot discriminate on recall.** A successful build has no
   diagnostics, so an implementation returning the empty string scores "exact" on it. e01
   gets 3 of its 3 exact points that way. Budget compliance and pass-through are the only
   things it measures.
2. **One machine, one toolchain.** Apple clang 21, Swift 6.3.3, rustc stable, pytest 8.4.2.
   A different Xcode renders diagnostics differently.
3. **The holdout comes from the same generator**, so it inherits the generator's blind
   spots — no Windows toolchain, no Java, no Go, no CI-runner output, and no log that
   interleaves two processes' stdout. A genuinely independent set would come from a different
   author. One of its seven cases is burned (see above), leaving six blind.
4. **Noise is unweighted**, so an implementation that fills the budget with correct content
   and an implementation that fills it with correct content plus filler score the same.
5. **The `other` family is 7 of 25 cases** and carries most of the discrimination. A skew
   worth knowing when reading the totals.
