# Handoff: make LM_Pipe an agent Sean can actually drive

Paste this whole file as the opening prompt of a fresh session in
`/Users/dev/Desktop/seans_projects_local/LM_Pipe_2` (branch `perf/mask-and-scan`).

## Read this first

The goal is **a fast, local, competitive 2026 coding agent** that Sean can drive against
real work. Single model, single process, Apple Silicon, Qwen3 via MLX in-process.

**Performance work is closed.** Six passes beat LM Studio on decode and prefill.
`docs/HANDOFF_PERF.md` is the record — consult it if you touch `src/model/`, but do not
open a seventh pass and do not re-run the sweeps.

**The core agent loop is now good. The product surface around it is thin.** That is the
whole job for this session, and it is two things: **the agent cannot be talked to**, and
**there is no way to measure whether it is any good**. Everything else can wait.

> **Nothing is committed.** The previous session left ~2,950 insertions across 25 files in
> the working tree, all building and green. Read `git diff` before you change anything,
> and commit early so you have a floor to fall back to.

## Priority 1 — the agent cannot be talked to

The protocol is `lmp/start`, `lmp/cancel`, `lmp/approve`, `lmp/shutdown`. That is all.
**There is no way to send a message to a running agent, and no way to follow up on a
finished one.** Every run is a fresh one-shot mission with no history.

You can abort it. You cannot say "no, use the other approach", "keep going", or "now do
the same to the other file". Sean's goal is an agent he can *drive*; today it is a batch
job he launches and watches. For a 2026 harness, conversational steering is table stakes.

What this needs, roughly:

- **`lmp/message`** — deliver user text into a run that is already in flight. The
  transport already frames whole messages on a reader thread and the run loop already
  drains the inbox mid-run for approvals (`ApprovalBridge` in `src/surface/sidecar.cpp`
  is the working pattern to copy — it blocks the run and drains the same SPSC channel).
- **Run history** — a follow-up must continue the conversation rather than restart it.
  `context::ContextStore` is constructed per run in `run_mission` and thrown away at the
  end; it needs to outlive a single mission, with the mission itself becoming one of
  several user turns rather than an immutable singleton (`mission_` is `const` today).
- **Steering, not just aborting** — an injected message should reach the model at the next
  turn boundary. Cancel already sets a token mid-generation; steering can be gentler.
- **Protocol is generated.** Edit `protocol/schema.json`, run `scripts/gen_protocol.py`,
  and both the C++ header and the TypeScript interfaces regenerate. The `protocol`
  ratchet regenerates and diffs, so drift fails the build. Then wire the extension
  (`extension/src/client.ts`, `sidebar.ts`) — the sidebar needs an input box.

## Priority 2 — nothing measures whether the agent is good

`scripts/eval.py` scores `blast_radius`, one component, against a corpus and a held-out
set. **Nothing measures the agent end to end.**

This is not theoretical. The previous session made roughly ten behavioural changes —
compaction strategy, prompt ordering, plan-forcing, multi-call, stall detection, persona,
baseline verification — and validated them against **one task**. Their aggregate effect is
genuinely unknown, and some plausibly hurt: forcing `plan` as the first call costs a turn
on trivial missions, and the grammar restriction that enforces it is a blunt instrument.

Build a small task suite before touching the loop again. Even ten fixture workspaces with
known-good outcomes — a failing test to fix, a build error, a rename across files, a task
that should be refused — scored on did-it-finish, did-it-verify, how many turns. Pin the
scores the way `eval.py` already pins its corpus and holdout, and make the holdout the
harder set (S11.3).

Without this, every future loop change is an argument instead of a measurement.

## Priority 3 — a decision only Sean can make

**`completed: true` is still not reached, and the last gate is a judgement call.**

Everything evidential works and was verified on a real run: the agent plans, fixes the
bug, the declared contract goes red at 4.9 s and green at 19.3 s, and the pass is recorded
`falsifiable=True`. What remains is `evaluate_completion` (`src/loop/turn.cpp`) also
requiring **every checklist item ticked** — and the model narrates that it is done instead
of restating the checklist. Restricting the callable tools to `plan` does not compel it,
because a text-only turn selects no tool at all; compelling it needs the grammar to reject
a turn that ends without a call.

Before building that, question the gate. "All items ticked" is the model's **self-report**
— the same prose-trust the rest of this design refuses (S10.4). The **evidential** gates
are a recorded deliverable and a falsifiable passing verification of the declared
contract, and both are observed facts. Requiring the model to also *agree* with the
evidence may simply be the wrong gate.

Changing it changes what "complete" means. **Ask Sean; do not decide it yourself.**

## Everything else, ranked. Do not start these before 1 and 2.

| gap | note |
|---|---|
| No cross-session memory | Nothing persists between runs. |
| Nothing retries | `ToolResult.retryable` is set by tools and read by nobody. |
| No resume | A crashed run is lost. The event log is rich enough to replay and `ReplayBackend` exists — but only for tests, not for recovering a live session. |
| Code intelligence is grep | `locate_symbol` shells out to `grep -rnE`. No LSP, no clangd, no semantic index — on a large repo, the difference between the right symbol and 200 matches. This repo already runs clangd. |
| No git write | It can read its own diff now but cannot commit or branch, so it cannot hand work off. |
| No editor-API edits | `lmp/edit` writes files directly: no undo, no diff review. Exempted in `ratchets.json` with that reason. |
| T2 containers refuse | So unattended runs are unavailable (S7.2). Deliberate. **Do not start without asking.** |
| Miniature `tokenizer.json` fixture | Would unblock gate-level testing of the grammar and the loop, and close three surviving mutations. |

## What already works — verified on the real model, not assumed

- Install: `cmake --build build --target lmp_sidecar`, then `npm run install-local` from
  `extension/`. **The editor is Cursor, not VS Code** (`~/.cursor/extensions/`; the CLI is
  at `/Applications/Cursor.app/Contents/Resources/app/bin/cursor` and is not on `$PATH`).
- Approval round-trip, deny-by-default. Wired, unit-tested — but **never yet fired on a
  real run**, because every command the model chose scored below the 0.35 threshold.
- Sandboxed shell. `RLIMIT_NPROC` is per-UID on macOS, so a fixed `max_processes` made
  every `fork()` fail; it is now a headroom over current usage.
- Planning (`plan` tool, grammar-enforced as the first call), deliverable ledger,
  verification routed through the `Verifier`, falsifiability from a captured red
  (FAIL_TO_PASS baseline run at declaration time).
- Context: compaction on the real token budget; mutable state renders **last** so ledger
  updates stop diverging the KV prefix.
- Parallel tool calls, up to 4 per turn. Confirmed batching two reads into one round-trip.
- Project conventions (AGENTS.md / CLAUDE.md / .cursorrules), read-only git tools,
  the Piper persona, stall detection after 3 consecutive text-only turns.

**Genuinely strong, leave alone:** grammar-constrained tool calls (syntax enforced
byte-by-byte, stops on grammar state not string match), verified KV prefix reuse with
divergence detection, the event log, and the safety model (grants unforgeable from risk
hints, egress denied in the Seatbelt profile itself).

## How to run things

```bash
ctest --preset gate && python3 scripts/run_ratchets.py
```

Gate is 20 tests in ~6 s and six ratchets. `ctest --preset realmodel` loads the real model.
To drive a mission without the editor:

```bash
python3 scripts/drive.py --workspace /path/to/ws --mission "fix the failing test"
```

It speaks the real protocol, auto-approves cards and prints them, and is how every bug
above was found. `--mode plan` and `--deny` are useful.

Known flake: `test_sandbox`'s spin test races the wall-clock killer against `RLIMIT_CPU`,
both at 1 s. It passes on rerun. Pre-existing.

## Working agreements

- **Never run two MLX processes at once.** One checkpoint is 19 GB resident and peaks at
  21 GB on a 48 GB machine that normally has three IDEs open. It crashed the machine on
  2026-07-31. Anything that loads a model runs alone, in the foreground, to completion.
- **Breadth before depth.** If something works but is ugly, write it down and keep going.
  Do not open an investigation without saying what decision it will change.
- **Do not measure performance.** That work is done. If something is unusably slow in real
  use, say so and stop.
- **This machine drifts ~9% in throughput** under background load, so any before/after
  spanning more than one sitting is meaningless.
- **Drive it yourself and report what broke.** That list has been worth more than any test
  result in every session so far.

## Do not re-litigate

Settled, with arguments in `docs/PHASES.md`: Apple Silicon only, MLX in-process, Qwen3
only, XML tool-call syntax (the model's own chat template specifies it). Speculative
decoding is out of scope. **Subagents are out** — one local model, one process; that is
the whole point of this agent. The `gated_delta_update_ops` reference path and the
equivalence tests in `tests/model/test_grammar.cpp` stay: they are what make the fast
paths falsifiable.
