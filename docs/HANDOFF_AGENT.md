# Handoff: make LM_Pipe an agent Sean can actually drive

Paste this whole file as the opening prompt of a fresh session in
`/Users/dev/Desktop/seans_projects_local/LM_Pipe_2` (branch `perf/mask-and-scan`).

## The goal has changed. Read this before doing anything.

**Stop performance work.** Six passes went into making decode and prefill beat LM Studio.
Both criteria pass. `docs/HANDOFF_PERF.md` is the complete record and it is closed —
consult it if you touch `src/model/`, but do not open a seventh pass, do not re-run the
sweeps, and do not chase the remaining open items in it. They are documented precisely so
nobody has to hold them in their head.

**The goal is a working agent, end to end, that Sean can run against real work and form
opinions about.** Scaffolded the way a competent 2026 agent is scaffolded. Refinement
comes after real-world use tells us what is actually wrong, not before.

The failure mode to avoid is the one the last session fell into: picking one component,
instrumenting it deeply, and optimising it while the product as a whole still cannot be
used. Breadth first. If something is slow but works, leave it slow and write it down.

## What actually exists

All nine phases in `docs/PHASES.md` landed. This is not a skeleton — it is a nearly
complete agent with a few disconnected wires.

| layer | state |
|---|---|
| `src/platform/` | arenas, clock, SPSC channel, event log, fs. Solid. |
| `src/model/` | Qwen3.6-35B-A3B 4-bit in-process via MLX. Prefill + masked decode + grammar. **Fast and finished.** |
| `src/model/grammar.*`, `token_mask.hpp` | constrained decode, XML tool-call syntax enforced byte by byte via parsephony |
| `src/loop/` | `Agent::step`, turn state, verification. The ReAct loop. |
| `src/tools/` | registry, Seatbelt sandbox, graft (edit application), log_triage, blast_radius |
| `src/context/` | context assembly |
| `src/surface/` | sidecar binary + JSON protocol + VS Code extension (TypeScript) |

A real end-to-end turn works today: prompt → prefill → masked decode → grammar accept, on
the real model, stopping by grammar state rather than string matching.

## What blocks it from being usable, in priority order

These are the wires that are cut. Each is small; together they are the difference between
"the pieces pass their tests" and "Sean can use this."

1. **The approval round-trip is not connected.** `src/surface/sidecar.cpp:195` calls
   `agent.set_approver(nullptr)`, so `Agent::step` denies every escalation
   (`src/loop/agent.cpp:168` requires a non-null `approver_`). The HITL router, the risk
   scoring and the approval cards are all built. Nothing asks the human. Until this is
   wired, any tool call that scores above the auto-allow threshold simply fails, which
   will look like the agent being broken.

2. **The extension is not packaged and does not ship the binary.**
   `extension/package.json` has only `compile` and `watch` — no `vscode:prepublish`, no
   `package` script, no `vsce`. `main` is `./out/extension.js`, which requires `tsc -p .`
   first. `bin/lmp_sidecar` is not copied in; the binary builds to
   `build/src/surface/lmp_sidecar`. There is no install path for a human.

3. **T2 containers refuse rather than downgrade** (`SandboxTier::T2_Container`), and
   §7.2 requires T2 for unattended runs. So unattended runs are unavailable. Decide
   deliberately whether the first usable version needs them — attended-only may be
   entirely fine for real-world testing, in which case write that down and move on rather
   than building container support nobody asked for yet.

4. **`lmp/edit` writes files directly** instead of going through VS Code's edit API, so
   there is no undo and no diff review for anything the agent changes. That is a real
   trust problem the first time it edits something wrong. The notification type exists and
   is exempted in `ratchets.json` with that reason.

5. **A committed miniature Qwen-shaped `tokenizer.json` fixture.** `docs/PHASES.md` calls
   this the single highest-value next task and it is still right: it unblocks gate-level
   testing of the grammar and of `Agent::step`, and closes three of the four surviving
   mutations. It is the one piece of test infrastructure worth building before shipping,
   because without it the loop itself is only covered by tests excluded from the gate.

## What the tests are, and which ones matter to you

Sean asked, fairly, what the tests do and whether they mean anything for the end goal.
Plainly:

| | what it is | does it matter to the goal? |
|---|---|---|
| `ctest --preset gate` (20) | Fast unit tests — arenas, channels, fs, tokenizer, grammar, transport, sandbox. Runs in 6 s. | **Yes.** This is the "did I break something" check. Run it after any change. |
| `ctest --preset realmodel` (2) | Loads the real 19 GB model and runs a full turn. ~5 s. | **Yes**, and it is the closest thing to a smoke test of the product. |
| `scripts/run_ratchets.py` (6) | Repo hygiene gates — file size, layering, dead code, protocol sync, and two prose gates that check documentation honesty. | Mostly. It stops rot; it says nothing about whether the agent is good. |
| `scripts/eval.py score` | Scores the `blast_radius` engine (risk classification of shell commands) against a corpus + holdout. | **Yes, indirectly** — blast_radius is what decides whether a tool call needs approval. Its holdout is 34/42. |
| `lmp_diag *` | Performance attribution for the model. | **No.** Not for this goal. Leave it alone. |
| mutation testing | Plants bugs and checks tests catch them. Honest score 3/8. | Diagnostic only. The 5 survivors are named in `docs/PHASES.md`. |

If you only run two things, run `ctest --preset gate` and `ctest --preset realmodel`.

## Suggested shape for the next session

Aim to end the session with **Sean able to install the extension and drive a real task.**
That probably means, roughly in order: wire the approver round-trip (1), then packaging
and install (2), then take a real task end to end yourself and fix what breaks. Items 4
and 5 are strong candidates if that goes quickly. Do not start item 3 without asking.

Report what broke when you drove it. That list is worth more than any test result.

## Working agreements

- **Never run two MLX processes at once.** One loaded checkpoint is 19 GB resident and
  peaks at 21 GB; this machine has 48 GB and normally has three IDEs open. Doing it
  crashed the machine on 2026-07-31. Anything that loads a model — `lmp_diag`,
  `ctest --preset realmodel`, `scripts/mlxlm_reference.py`,
  `scripts/graph_histogram.py --reference` — runs alone, in the foreground, to completion.
- **Breadth before depth.** If a component works but is ugly or slow, write it down and
  keep going. Do not open an investigation without saying what decision it will change.
- **Do not measure performance.** That work is done. If something is unusably slow during
  real use, say so and stop; do not start attributing it.
- **This machine drifts ~9% in throughput** with background load, so any before/after
  comparison spanning more than one sitting is meaningless. If you must compare, use a
  same-session A/B on one binary.

## Do not re-litigate

Settled, with arguments recorded in `docs/PHASES.md`: Apple Silicon only, MLX in-process,
Qwen3 only, XML tool-call syntax (the model's own chat template specifies it; JSON would
push it off its trained distribution). Speculative decoding is out of scope. The
`gated_delta_update_ops` reference path and the equivalence tests in
`tests/model/test_grammar.cpp` stay — they are what make the fast paths falsifiable.
