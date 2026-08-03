# Handoff: close the remaining agent/harness gaps

Paste this whole file as the opening prompt of a fresh session in
`/Users/dev/Desktop/seans_projects_local/LM_Pipe_2`.

## The one-sentence version

The harness is sound and the loop is instrumented, but a real 45-turn mission on
2026-08-03 could not have succeeded **at any budget** — it was verifying against a
contract that could never pass, and it spent two thirds of its turns re-reading files it
already had in context. Fix those two and the agent stops losing runs to itself.

## The goal, stated so it is not lost

Sean's framing: a coding agent that beats Cursor and Claude Code. That is not a feature
list — those agents win on **not wasting the run**. They do not re-read a file they read
four turns ago, they do not verify against a command that cannot pass, and they do not
end 45 turns in with nothing. Every gap below is one of those.

## Read this first: the measured status, not the intended one

All numbers below are from one real run: `~/Desktop/Agent_testing/ResMon`, a 5-task Swift
6 / SwiftUI mission, ended 2026-08-03. Reproduce the evidence with:

```bash
sqlite3 ~/Desktop/Agent_testing/ResMon/.lmp-context.db "select title,count(*) c from item where title!='' group by title order by c desc;"
```

That store is readable because PCC was wired the same day; before that this diagnosis was
impossible. `~/Library/Logs/LM_Pipe/events.jsonl` is the other half — 24 runs, split on
`policy` events, one per run.

**What the run did:** 45 turns, 2,508 s, ended `budget_exhausted` (now
`wall_clock_exhausted`). 65 turn rows. 23 `read_file`, 11 `read_slice`, 10 `list_dir`,
**11 `write_file`**. `HostStatsService.swift` read **11 times** at ~12.5 KB each.
`break_repeat` fired 11 times. Peak prompt 47,220 tokens against a 96,000 budget, so
**compaction never ran** — every one of those re-reads was still in context.

---

## Gap 1 — a verify contract that can never pass, and nothing notices (HIGHEST)

**This is the one that decided the run.** The mission's contract was:

```
xcodebuild build -scheme ResMon -destination 'platform=macOS'
```

The project's only scheme is **`Untitled Project`**. Evidence, from the event log:

```json
{"kind":"plan","items":"5","open":"5","verify_with":"xcodebuild build -scheme ResMon ..."}
{"kind":"baseline_check","contract":"xcodebuild build -scheme ResMon ...","passed":"0"}
{"kind":"verification","contract":"xcodebuild build -scheme ResMon ...","ran":"1","passed":"0","falsifiable":"0"}
{"kind":"plan","items":"5","open":"4","verify_with":"xcodebuild build -scheme ResMon ..."}
{"kind":"verification","contract":"xcodebuild build -scheme ResMon ...","ran":"1","passed":"0","falsifiable":"1"}
```

The agent **discovered** the real scheme mid-run (`xcodebuild -list` on turn ~30, visible
in the store), acted on it by rewriting `Package.swift` — and then re-stated `plan` with
**the same unsatisfiable `verify_with`**. `evaluate_completion()` requires that contract to
pass, so the run was unable to complete from turn 1 onward. Budget was never the binding
constraint; the goal was unreachable.

**What to build.** A contract that has failed repeatedly for a reason that is not the code
must be surfaced as a *contract* problem, not a code problem. Concretely:

- Count consecutive failures of the *same canonical contract* that produced no change in
  the failure text. `src/loop/verification.cpp` already canonicalises contracts (`canon`)
  and already has `unfalsifiable_reason()` for a related job.
- After N (2 is probably right — it re-planned twice), inject an observation naming the
  contract and telling the run to re-derive it, and consider making `plan` the only
  callable tool for one turn — the same MECHANISM the stale-plan path already uses
  (`plan_is_stale()` in `src/context/context.hpp`, enforced in the loop). Prose will not
  do it; this repo's whole `prose_correctives` ratchet exists because of that.
- The failure text is the signal. "scheme ResMon not found" is stable across attempts and
  never mentions a source file. Do **not** try to classify build errors semantically —
  match on *the failure being byte-identical across attempts while the workspace changed*,
  which is decidable and cheap.

**Do not** let `plan` silently re-assert an operator contract — `ContractSource::Operator`
already wins and must keep winning. This is only about model-chosen contracts.

## Gap 2 — falsifiability is certified by a red that proves nothing

`src/loop/verification.cpp:221` marks a contract falsifiable when
`v.ran && !v.passed && v.contract == canon`. `ran` is false only for `Refused` or
`ToolResult::never_executed()` (exit 126/127). `xcodebuild` with a nonexistent scheme
exits non-zero but **not** 127, so it counts as "ran", and the run above certified its
contract `falsifiable: 1` on the strength of a failure that had nothing to do with the
code.

This is the same shape as the bug `never_executed()` was added for — "a red that proves a
check capable of failing" — one level up. A red proves falsifiability only if it is a red
**about the thing being checked**. Suggested rule: a red whose output does not change when
the workspace changes is not evidence of falsifiability. Related memory:
`agent-blames-the-harness`.

## Gap 3 — the agent re-reads what is already in its context

44 of 65 turns were `read_file`/`read_slice`/`list_dir` against 11 writes. One file 11
times. **Compaction never ran**, so this is not a memory problem and `context_recall` will
not fix it — the content was verbatim in the prompt when it re-read.

`break_repeat` fired 11 times, so the repeat detector saw it, but detection is
exact-match on (tool, params): `read_file(x)` then `read_slice(x, 1, 50)` then
`read_file(x)` evades it. Look at `RepeatDetector` in `src/loop/` and
`Agent::can_run_in_parallel`'s neighbours.

**Suggested direction, to be validated not assumed:**

- Track *paths read this run* and their turn index. On a re-read of an unchanged file
  (no write to it since), return a short refusal-shaped observation pointing at the turn
  that already has it, rather than 12.5 KB again. The `run_wrote_` set in
  `src/loop/approval.cpp` is the precedent for per-run path tracking, including the
  `platform::lexically_normal` normalisation it needs.
- Make it a *typed* result, not prose. A silent or empty result is forbidden here — see
  the comment in `Registry::execute`.
- Measure before and after on the same mission and seed. If prompt tokens and turn count
  do not both fall, it did not work.

**Trap:** a file the run *wrote* since reading it must be re-readable. Key the suppression
on "unchanged since last read", not "read before".

## Gap 4 — approval residue

Three items, all verified this session:

1. **`run_wrote_` is per-Agent.** Files LM_Pipe created in an *earlier* run still raise an
   overwrite card. A run resuming its own previous work asks about its own output.
2. **The command allowlist only takes effect on the next run.** "Always allow" writes to
   editor settings, which reach the sidecar at the next `lmp/start`. Within the run that
   asked, the same command asks again. The write gate got run-scoped consent
   (`allow_writes_for_run` on `lmp/approve`, latched in `RunInbox`) — the command gate
   needs the same treatment, and the plumbing now exists to copy.
3. **`replace_in_file` is ungated by design** and is the better tool for editing existing
   files. The model reaches for whole-file `write_file` anyway. Consider whether the tool
   descriptions in `src/tools/registry.cpp` steer hard enough.

## Gap 5 — recall quality (small, measured)

- `context_recall` returns **the current session's own mission row** when the query echoes
  the mission text — ~60 tokens spent handing back something already in the prompt
  verbatim. Filter the current session's mission, or exclude `first_event == 0` rows of
  the live session.
- Against an **empty** store the model re-calls `context_recall` with an identical query.
  I rewrote the empty-result message to name the fallback (`search`, then `read_file`),
  rebuilt and re-ran: **it did not help** — still 3 recalls before falling back, 2 of them
  byte-identical. Do not assume better wording fixes this; it was tried. A mechanism
  (suppress a repeated zero-result query) is likely needed.
- `remember` appends to PCC as `kFact` with **no key**, so it cannot supersede. Adding a
  `key` parameter needs `.lmp-memory.md` taught to *replace* rather than append, or the
  prompt carries both the stale and current line — the exact failure PCC exists to end.
- Existing workspaces have a populated `.lmp-memory.md` and no `kFact` history. A backfill
  keyed on a hash of each line (`Store::remember` is idempotent on identical bodies) would
  make old notes searchable. Not done.

## Gap 6 — the compacted span line doubles the tool name

`ContextStore::compact_oldest()` in `src/context/context.cpp` builds
`"- " + t.tool_name + "(" + t.tool_args_summary + ")"`, but `tool_args_summary` comes from
`preview_of()` (`src/loop/approval.cpp:93`) which **already** names the tool. Every
compacted span line in the prompt reads `- read_file(read_file(path=x)) -> ...`.

The identical bug in the journal was fixed this session — copy the fix from `turn_body()`
in `src/surface/context_journal.cpp`, which tests
`tool_args_summary.rfind(tool_name + "(", 0) == 0`. This one is prompt-facing, so it costs
tokens in every run that trims. Check `tests/context/` for span-format assertions.

---

## What is already done — do not redo it

Landed and verified 2026-08-03, both presets green:

- **PCC is wired.** `ContextStore::TurnSink` writes every turn as it is recorded;
  compaction contributes the `kSpan` summary; the mission row is written at journal open.
  Native `context_recall` / `context_rehydrate` (`src/tools/context_tools.cpp`), declared
  from `sidecar.cpp` only when a journal opened. Measured: 9 turns / 23.9k prompt tokens
  with a populated store vs 12 turns / 39.3k and **no answer** with an empty one.
- **`budget_exhausted` split** into `wall_clock_exhausted` / `turn_budget_exhausted`;
  `halt_on_budget` carries `limit`, `max_iterations`, `wall_clock_seconds`. Both branches
  proven.
- **Approval decisions are logged.** An `approval` event on both gates, including
  auto-approvals which previously emitted nothing. The gate reports `escalated`; the
  approver reports `card`/`answer` — deliberately different words, because one cut of this
  logged `asked` from both and produced two events with opposite values for one call.
- **Run-scoped write consent.** `allow_writes_for_run` on `lmp/approve`. Measured 3 cards
  → 1. Not persisted, by design.
- **The tool-honesty ratchet** now scans `context_tools.cpp` as well as `registry.cpp`,
  proven to fire on a planted ghost tool.

Three artifacts disagreed about one rule before this session (the allowlist beating
irreversibility was implemented in `approval.cpp` but contradicted by
`protocol/schema.json` and the card UI). **If you change a gate rule, grep the schema
comments and `extension/src/webview.ts` too.**

## Verification — required, and there are traps

**Run BOTH presets.** `-DLMP_MLX_PYTHON=/usr/bin/false` is mandatory: CI has no MLX and
compiles the `#else` half of `src/model/mlx_backend.cpp` a local build never touches.

```bash
cmake --preset dev  -B /tmp/nomlx -DLMP_MLX_PYTHON=/usr/bin/false
cmake --build /tmp/nomlx -j8 && ctest --test-dir /tmp/nomlx -L gate

cmake --preset asan -B /tmp/asan  -DLMP_MLX_PYTHON=/usr/bin/false
cmake --build /tmp/asan -j8 && ctest --test-dir /tmp/asan -L gate
```

`tests/gate/gate_manifest.txt` pins the count (**41**) and the names; adding a gate test is
two edits.

**Traps, all of which have cost real time here:**

- ASan catches lifetime bugs the plain build reads as freed-but-plausible memory.
  `const auto& x = store.current(k)->field;` binds into a temporary `optional` and does not
  extend the subobject's lifetime — if you write that, you wrote the bug.
- **Never run two MLX processes.** One checkpoint is ~19 GB on a 48 GB host; doing it took
  the machine down. Run real missions one at a time, in the foreground.
- `scripts/drive.py` puts `model_dir` and `workspace_root` **inside** `settings`, with
  sampling in a nested `sampling` object, and sends `lmp/start` only after `lmp/ready`. A
  hand-rolled client that gets this wrong hangs with the sidecar idle at ~7 MB RSS and no
  error.
- The protocol is generated: edit `protocol/schema.json` and run
  `python3 scripts/gen_protocol.py`. The `protocol` ratchet regenerates and diffs.

**Prove it end to end.** The failure this work exists to fix is "the unit tests pass and
the agent still loses the run". Required evidence for Gaps 1 and 3:

1. Re-run the ResMon mission (or an equivalent scaffolded multi-file Swift/Python project
   whose build command is *initially wrong*) through `scripts/drive.py`.
2. Show from the event log that the run either **fixed its contract** or **stopped and said
   the contract was wrong** — rather than failing the same check to the end of its budget.
3. Show from `.lmp-context.db` that no file is read more than twice unchanged, and that
   read turns are no longer the majority.

Numbers, before and after, same mission and seed. A change that does not move them did not
work.

## Ground rules

- **Do not re-litigate the no-embedding decision** in PCC. BM25 over FTS5 with RRF at
  k=60, by design; a real embedder needs a model this component deliberately does not link,
  and one 19 GB model on a 48 GB host means there cannot be a second MLX process.
  `recall()` fuses *rank lists* precisely so an embedder drops in later as a third list.
- **The code is the truth; a doc that disagrees is a bug.** This handoff included — verify
  before building on any claim in it.
- Comments here explain *why* and cite the measured failure that motivated the code.
  Several of the sharpest bugs in this repo were found because a comment recorded what had
  already gone wrong.
- No size limits. Enforce design by review, never by a line count — the size ratchet was
  removed 2026-08-02 and is not coming back.
- `git log` carries reasoning worth reading before changing anything it touches.

## Suggested order

1. Gap 1 (unsatisfiable contract) — it is the difference between a run that fails and a run
   that could never have succeeded.
2. Gap 3 (re-reading) — the largest consumer of a run's time and context.
3. Gap 2 (falsifiability) — small, and it makes Gap 1's evidence trustworthy.
4. Gaps 4–6 — real, bounded, independent.
