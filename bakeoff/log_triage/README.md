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
> entrants found in the blast-radius key. `test_log_triage_corpus` now asserts every key
> entry is findable in its own log; it caught all three.

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

Validated by `src/testing/test_log_triage_corpus.cpp` (341 checks) and pinned by
`src/testing/test_log_triage_engine.cpp` (13 checks), both in the standard gate.

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

`src/tools/log_triage.hpp`, wired into `SubprocessVerifier::compact_command_output`.
`line_is_diagnostic` is **deleted**, not kept alongside — two implementations of "is this
line important" is how the caller ends up asking the wrong one. The frozen pre-change code
lives at `entrants/incumbent.hpp` so the baseline stays measurable.

| | corpus (tuned against) | **holdout (blind)** |
|---|---|---|
| weighted | 34 | **20** |
| locators lost | 1 / 177 | **0 / 39** |
| messages lost | 1 / 195 | **0 / 48** |
| context lost | 29 / 384 | 20 / 117 |
| exact | 71 / 75 | **19 / 21** |
| over budget | 0 | 0 |

**Quote the holdout column.** The corpus number is a memory test — the engine was written
with that corpus open, iterating until the scorer went quiet.

### The comparison is not one-sided, and the honest reading is this

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

### ⚠️ One holdout case is burned

`ho_rustc_no_cargo` exposed two genuine defects and both were fixed with its output in front
of me:

1. **A bare locator line asserted maximum severity instead of inheriting it.** rustc prints
   the same ` --> path:L:C` under a `warning:` as under an `error:`, so a crate with 120
   unused-variable warnings produced 120 lines that outranked the real errors' source blocks
   and filled 16 KB with warning addresses.
2. **Context was bounded by a radius, not by structure.** A comment in the engine claimed six
   lines "covers the widest block any tool prints"; rustc's gutter is nine.

The other six holdout cases were perfect before and after those fixes and remain blind.
`test_log_triage_engine` asserts the held-out set stays harder **per point** than the corpus
(0.95 vs 0.45) — a tripwire on the process, not a quality bar. **The next round needs a fresh
set**, ideally one whose cases do not come from this generator at all.

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
