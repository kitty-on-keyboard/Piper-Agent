# Handoff: from "it works" to competitive

> **HISTORICAL (2026-08).** References below to `verification.cpp`, `prove_falsifiable()`,
> and related verification-ledger APIs are obsolete — those files were removed. Treat this
> document as a measured gap list from its write date, not as an API map. Current product
> claim: Mac-local one-model Qwen/MLX, no subagents.

Paste this whole file as the opening prompt of a fresh session in
`/Users/dev/Desktop/seans_projects_local/LM_Pipe_2`.

Written 2026-08-02, immediately after the session that fixed the harness bugs which made
the agent unable to finish a real task at all. That session's job was triage. This
document is about the gap that remains: LM_Pipe now **completes** a real mission, and it
is still roughly **3x slower in turns** than it should be, for reasons that are measured
below rather than guessed at.

Read `HANDOFF_REMAINING.md` alongside this — its standing rules, its ranked list (C1, V1,
D1, D2 …) and its "how to run things" are all still current. This file does not repeat
them. It supersedes nothing; it adds the agent-quality workstream that the triage session
turned up evidence for.

---

## STOP — do these two things first

**1. Twenty files are uncommitted on `main`.** The previous session left the tree dirty:
eight harness fixes, three new test files' worth of cases, and the raised budget defaults.
All of it is verified (both gate configs 40/40, ratchets 6/6, ten real-model runs), none
of it is committed. Review the diff, branch, and open a PR before doing anything else —
the repo's convention is a branch per change with a PR onto `main`.

**2. `agent_eval.py` was never run against those changes.** This violates a standing rule
("`python3 scripts/agent_eval.py run` before and after any loop change"). It was attempted
and killed by a 10-minute tool timeout, and the session ran out of budget to retry it.
The changes touch verification semantics, the completion gate, tool-result text and error
classification — exactly the surfaces the pins measure. **Run it, in the background, before
you trust the pins.** Expect movement; decide deliberately whether it is an improvement to
re-pin or a regression to fix.

```bash
nohup python3 scripts/agent_eval.py run > /tmp/eval.log 2>&1 &
```

Pins as of the last deliberate measurement (2026-08-02, *before* these changes):
corpus 5/6 solved · 2 completed · 2 verified · 119 turns;
holdout 2/4 solved · 2 completed · 3 verified · 66 turns.

---

## Where things actually stand

```
ctest -L gate         40/40, both preset configs        (was 34/34)
ratchets              6/6
agent_eval            NOT RUN against current tree      <-- see above
extension             installed into Antigravity + Cursor, sidecar sha verified
real-model E2E        completes the KV-store mission: 48 turns, 53 tests passing
unload                measured: frees 18.04 of 18.07 GB (was 0.00)
```

**What the triage session fixed**, all with regression tests, all measured on the real
model — do not re-open these:

- `write_file` could not create parent directories, and failed on the atomic write's
  *temp* file. This alone burned 20 of 40 turns in the original failure.
- Tools could return empty observations, which `context.cpp` drops from the prompt
  entirely — a byte-identical next prompt, and at a fixed seed a deterministic loop.
- Deterministic write failures were flagged `retryable`, so `BreakRepeat` could never fire.
- A contract ending `| tail -20` exits with *tail's* status: it could never fail. See
  `executable_form()` vs `canonicalize_check()` in `verification.cpp` — the form that RUNS
  and the form that IDENTIFIES are now deliberately different functions.
- Exit 126/127 counted as a red, proving falsifiability for a check that never executed.
- The completion gate compared the raw contract against a canonically-keyed ledger.
- The workspace root was absent from the prompt; the model guessed `/home/user`.
- `~MlxBackend` never called `mx::clear_cache()`, so unload freed nothing.

**New instrumentation you should use.** `LMP_TRACE_TEXT=1` adds `turn_text` (reasoning +
answer) and `tool_call` (full arguments) events. `verification` and `not_complete` events
are always on — `not_complete` names which completion gate is still shut, which is the
single most useful line when a run works and won't say it finished. The original 40-turn
failure was undiagnosable precisely because the log recorded what the harness DID and
nothing the model SAID.

---

## The thesis

The agent is no longer *broken*. It is *wasteful*. It took **48 turns** to produce a
191-line implementation and a 308-line test suite. A competitive 2026 agent does that in
roughly 15. The waste is not diffuse — it concentrates in one causal chain, and A1 below
is most of it.

Measured across the two clean end-to-end runs (89 turns total):

| tool | calls | failures | rate |
|---|---|---|---|
| `write_file` | 28 | 0 | 0% |
| `read_file` | 19 | 0 | 0% |
| `replace_in_file` | 14 | 4 | **29%** |
| `plan` | 12 | 0 | 0% |
| `shell` | 8 | 4 | 50% |
| `list_dir` | 2 | 0 | 0% |

---

## Ranked

| # | Item | Size | Risk | Why here |
|---|---|---|---|---|
| **A1** | Surgical edits are unreliable, so the model rewrites whole files | M | **high** | One chain causes ~20% of all wasted turns |
| **A2** | The falsifiability proof is improvised, and has corrupted the workspace | M | **high** | `prove_falsifiable()` exists and is unreachable |
| **A3** | Nothing batches tool calls | S | medium | 83/83 calls were index 0; the machinery is dead weight |
| **A4** | Test output is parsed as prose | M | medium | The loop cannot tell "3 of 26 failing" from "all failing" |
| **A5** | No turn-efficiency metric | S | low | You cannot optimise what agent_eval does not score |

---

## A1 — Surgical edits are unreliable, so the model rewrites whole files

**This is the highest-leverage change in the document.** Three symptoms, one cause.

`replace_in_file` failed **4 of 14 times (29%)**, and every single failure was the same
message:

```
old_text not found in src/kv_store.py; re-read the file and try again
```

So the model learned — correctly, within a run — that `write_file` is the reliable tool.
It called it **twice as often (28 vs 14)**. But `write_file` means emitting the entire
file, and the generation cap is 4096 tokens (`AgentConfig::max_new_tokens`,
`src/loop/agent.hpp`). A 13 KB test file is ~3,500–4,000 tokens of content before any
reasoning. So **7–12% of all turns ended `LengthCapped`**: the model hit the cap
mid-`write_file`, no tool call was parsed, and the entire turn produced nothing.

Edit-tool unreliability → whole-file rewrites → token-cap deaths. One chain.

**What to do, in order:**

1. **Make failure informative.** "re-read the file and try again" costs a turn and teaches
   nothing. Return the *nearest* candidate region with line numbers, and the diff between
   what was asked for and what is there. `graft_engine.hpp` already computes match
   candidates for the ambiguous case — the no-match case should use the same machinery to
   show the closest span.
2. **Add a line-anchored edit.** `replace_in_file` is content-addressed, which is why a
   stale mental model of the file breaks it. `read_file` already returns 1-based line
   numbers and `read_slice` already takes them. An edit that takes `(start_line, end_line,
   new_text)` closes the loop the display format already opened. Note the existing
   asymmetric `strip_line_numbers` logic in `registry.cpp` and do not regress it.
3. **Only then raise `max_new_tokens`.** There is a queued task for this (4096 → 8192).
   Do it *after* 1 and 2, because if surgical edits work the model stops needing giant
   writes and the cap stops mattering. Raising it first hides the real defect.

**Falsify this properly:** the claim is that edit reliability, not the token cap, is the
root. So the pass condition is `write_file`/`replace_in_file` call ratio *inverting*, not
just the `LengthCapped` rate dropping.

---

## A2 — The falsifiability proof is improvised, and has corrupted the workspace

`Verifier::prove_falsifiable()` exists in `verification.cpp`. It takes `breaker` and
`restore` callbacks and enforces green → red → green. **Nothing calls it.** Grep the tree:
it is referenced only by its own definition and `test_verifier.cpp`. It is unreachable
from the loop and unregistered as a tool.

So a run that must prove its check can fail (S10.2) improvises: it hand-edits its own
source to inject a bug, runs the check, and hand-edits it back. That works when it works.
**Twice in this session it did not** — the budget ended mid-proof and the workspace was
left with the injected bug still in it. Once with 9 failing tests, once with the module
not even importable. The harness's own evidence rule damaged the deliverable it exists to
protect.

The triage session added `Corrective::BudgetNearlyGone` (fires
`kBudgetWarningTurns` = 8 turns out, telling the run to restore anything it broke). That is
a mitigation, not a fix — it is still the model's job to remember.

**What to do:** expose the proof as a first-class tool that owns the restore. The model
supplies the file, the span to break, and the check; the harness snapshots, breaks, runs,
and **restores unconditionally, including on abort, cancellation and budget exhaustion**.
Restoration must be in a destructor or equivalent, not on the happy path — the whole point
is that the dangerous window survives the run ending badly.

This also makes the proof cheap enough to stop dominating the endgame. Both clean runs
spent their last ~8 turns on it.

---

## A3 — Nothing batches tool calls

`src/loop/parallel_calls.cpp` and the `run_calls_concurrently` path in `agent.cpp` are
real, tested, and **never exercised by the model**: across 89 turns, all 83 tool calls
were at index 0. The grammar permits multiple calls per turn and the model never emits
them.

Worth an hour of investigation before any code: is it the grammar, the tool descriptions,
or simply that nothing in the prompt suggests batching is possible? Reading four files is
four full prefill+decode round-trips today, which the parallel machinery was built to
avoid. Either make it happen or delete the machinery — dead concurrency is worse than none,
because it looks like a solved problem.

---

## A4 — Test output is parsed as prose

The loop knows a check passed or failed, and nothing else. `log_triage::compact` shapes the
text for the prompt, but no structure is extracted. So "3 of 26 failing" and "26 of 26
failing" are the same fact to the harness, and the loop cannot tell progress from
thrashing, cannot target the failing case, and cannot notice that a fix broke two other
tests.

A small structured extractor for the common runners (pytest, ctest, cargo) — counts,
plus the names of failing cases — makes the checklist and the correctives far better
informed. Keep it strictly observational: it must not become a second path to completion
(S10.1), the same rule the syntax checker already follows.

---

## A5 — No turn-efficiency metric

`agent_eval.py` scores `solved`, `completed`, `verified`, `turns`, `intact`. `turns` is
recorded but is an aggregate across the split, so a change that solves the same tasks in
half the turns is invisible unless someone reads the raw number and remembers the old one.

Given that turn efficiency is now the main gap, promote it: score per-task turns, and
`turns_to_first_green` separately from total turns. Those two diverging is exactly the
signature of the A2 endgame problem.

---

## Deliberately not in scope

- **Sub-agents / delegation.** One 19 GB model on a 48 GB host, one process. The
  concurrency story is A3, not a second agent.
- **Replacing MLX.** Closed — see `MOE_ROUTING_FINDINGS.md`.
- **T2 containers** (D3 in `HANDOFF_REMAINING.md`). Still do not start without asking Sean.

---

## Working agreements carried forward

- **One MLX process at a time.** 19 GB on a 48 GB host; two at once has taken this machine
  down. Run model work in the foreground, serially. Background it only with `nohup` when
  you must, and never start a second.
- **Both gate configs before pushing.** CI has no MLX and compiles the `#else` half a local
  build never touches: `-DLMP_MLX_PYTHON=/usr/bin/false`.
- **A new test bumps both the count and the names** in `tests/gate/gate_manifest.txt`.
- **Verify with your own eyes.** The agent reporting "26 tests pass" is not evidence; run
  the suite yourself. That distinction is the entire point of the completion gate, and it
  applies to whoever is holding the keyboard too.
