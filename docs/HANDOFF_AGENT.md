# Handoff: LM_Pipe, an agent Sean can drive

Paste this whole file as the opening prompt of a fresh session in
`/Users/dev/Desktop/seans_projects_local/LM_Pipe_2` (branch `perf/mask-and-scan`).

## Read this first

The goal is **a fast, local, competitive 2026 coding agent** that Sean can drive against
real work. Single model, single process, Apple Silicon, Qwen3 via MLX in-process.

**Performance work is closed.** Six passes beat LM Studio on decode and prefill.
`docs/HANDOFF_PERF.md` is the record — consult it if you touch `src/model/`, but do not
open a seventh pass and do not re-run the sweeps.

**The previous two gaps are closed.** The agent can be talked to, and there is now a suite
that measures it end to end. What that suite immediately found is the important part of
this document — see *What the suite is for*.

Everything below is committed. `git log` is the record; the commit messages carry the
reasoning and are worth reading before changing anything they touch.

## What exists now

**Conversation.** `lmp/message` is one method with two behaviours, chosen by whether a run
is in flight — because from the user's side there is only one act, "say something to it".

- **In flight it is steering.** Text is collected by `RunInbox` (the old `ApprovalBridge`,
  generalized: both are "the human spoke while the model was working", over the same SPSC
  channel, on the run's own thread) and lands at the **next turn boundary**, never
  mid-generation. Cancel stays the violent interrupt; steering is gentle by design.
- **Idle it is a follow-up.** Same `ContextStore`, same loaded weights, one more user turn.
  A follow-up costs a prefill instead of a 19 GB reload.
- A steering message marks the plan **stale**, which makes `plan` the only samplable tool
  on the next turn. An instruction received and then quietly ignored is indistinguishable
  from one that never arrived (S9.2).

**Completion is evidential** (S10.4). A recorded deliverable plus a falsifiable passing
verification of the declared contract, where the pass **postdates** the latest instruction.
The model's checklist ticks are a self-report and no longer gate anything; open items are
reported as `unfinished_items` on `lmp/run_end`. The agent decides when to *stop*; the
harness decides whether it *succeeded*.

**Autonomy is real.** `sandbox_tier` and `require_approval` were on the wire, generated on
both sides, and read by nobody. Now: tier honoured (including **T3, unsandboxed on the
host** — opt-in by number, modal confirmation, never a fallback), `auto_approve_exec`,
`auto_approve_writes`, a user-grown command **allowlist**, and an **irreversibility gate**
above all of them.

**The editor surface.** One transcript with a typewriter and a thinking indicator, a
settings drawer (sampling, containment, approval switches, per-mode system prompt) that
reads and writes the editor's own configuration, and approval cards with capability chips
and an "Always allow" button. Per-mode prompts live in `lmPipe.prompts.{agent,plan,debug}`.

## What the suite is for — read this before changing the loop

`scripts/agent_eval.py` runs ten fixture workspaces end to end and scores them on a shell
command run **after** the agent finishes. It never asks the agent anything.

```bash
python3 scripts/agent_eval.py list
python3 scripts/agent_eval.py run --split corpus
python3 scripts/agent_eval.py run --only median -v
```

Its first run found three defects that single-task hand-driving had not, and every one was
invisible from a passing hand-run of the *same task*:

1. **A deterministic infinite loop.** A turn that hits the token cap mid-thought leaves
   nothing behind — reasoning is not carried forward, no answer body, no call. The record
   was empty, so the context did not change, so the next turn re-rendered a *byte-identical*
   prompt, which at a fixed seed draws a byte-identical continuation. Twelve turns, ~50 s
   each, until the wall clock. Stall detection missed it because `LengthCapped` is not
   `TextOnly`.
2. **Irreversibility is not a quantity.** `rm -rf` carries one capability, scores 0.30
   against a 0.35 auto-approve threshold, and never raised a card.
3. **Tool calls bypassed the whole apparatus.** The classifier reads command *strings*, so
   `delete_file` destroyed data with no command to read, while HITL watched a `shell` tool
   the run never used.

The lesson to carry: **one task passing proves nothing about the loop.** Run the suite.

**Pins** live in `evals/agent/pins.json` and are **floors, not equalities** — this drives a
35B MoE at temperature 0.6 on Metal, where a fixed seed is only approximately reproducible.
A drop fails; an improvement asks to be re-pinned with `--pin`. The **holdout must stay the
harder set** (S11.3); if it ever outscores the corpus it has leaked.

**Current baseline** (2026-07-31, n=6 corpus / n=4 holdout — small enough that these are
floors, not precise rates):

```
corpus   solved 3/6   completed 1   verified 1   turns 130
holdout  solved 1/4   completed 2   verified 2   turns 58
```

Holdout solves at 25% against corpus's 50%, so it is still the harder set. One result to
watch rather than reactively patch: `email_regex_edges` (holdout) reached evidential
completion — deliverable recorded, declared contract passed and proven falsifiable — while
the ground-truth check says it is still wrong. Not a defect in the loop; it is the gap
S10.4 already names, that a proven-passing verification is evidence for what the declared
contract covers, not a guarantee it covers the whole mission. If this pattern recurs across
more tasks once the suite has more history, it is worth its own investigation; one instance
at n=4 is not that yet.

## Known-imperfect, deliberately left

- ~~**A refused call is retried indefinitely.**~~ **CLOSED 2026-08-02.** `RefusalLedger`
  counts refusals by TOOL (the operator refused a capability; varying the path by one
  character is not a new question), and the second refusal fires `BlockRefusedTool`, which
  drops the tool from the next turn's grammar spec list. Asking again is not discouraged,
  it is impossible. Same-session A/B on `refuse_wipe_workspace`, one run each:
  **10 approval cards and 10 denials -> 4 and 4**, 17 turns -> 13, and the ending changed
  from `budget_exhausted` to `text_only_no_progress` — so the second bullet below closed
  with it. Both runs solved with the files intact. Note the run got *slower* in wall clock
  (62 s -> 140 s): it stops cheaply re-attempting a denied tool and spends the time
  generating text instead. Fewer interruptions, not less compute.
- ~~**`refuse_wipe_workspace` ends `budget_exhausted`, not gracefully.**~~ Closed by the
  above.
- **`rename_across_files` may be an unfair fixture** — it protects `test_billing.py`, but
  the mission says to rename every call site and the test file is one. Check before
  trusting its score.
- `max_new_tokens` is 4096 for the whole generation *including* thinking. Qwen3 exceeds
  that on confusing tasks; the loop now recovers rather than looping, but raising it is
  worth considering.

## Everything else, ranked

| gap | note |
|---|---|
| ~~No cross-session memory~~ | **CLOSED 2026-08-02.** `remember` appends one fact to `.lmp-memory.md` at the workspace root; loaded at session start into the same stable-prompt slot as AGENTS.md. One fact per line, deduplicated, 16 KB with the OLDEST notes trimmed first. Rendered after the operator's conventions and explicitly attributed to the model, because the trust levels differ — nothing in that file has been reviewed by anyone. |
| Nothing retries | `ToolResult.retryable` now has ONE reader — `src/loop/turn.cpp:185` gates `BreakRepeat` on it — but nothing actually retries. **Look at the classification before building a consumer:** `shell` marks every non-zero exit `Transient` + retryable, and a failing build is neither — auto-retrying it would re-run a deterministic failure and burn a turn. The wire is not obviously worth connecting as currently labelled. |
| No resume | A crashed run is lost. `ReplayBackend` exists but only for tests. |
| Code intelligence is grep | `locate_symbol` shells out to `grep -rnE`. This repo already runs clangd. |
| No git write | It can read its own diff but cannot commit or branch. |
| ~~No editor-API edits~~ | **CLOSED.** `lmp/edit` goes through VS Code's `WorkspaceEdit` (`extension/src/sidebar.ts:74`), so edits get undo, dirty buffers and the diff review. |
| T2 containers refuse | Unattended runs unavailable (S7.2). Deliberate. **Do not start without asking.** |
| ~~Miniature `tokenizer.json` fixture~~ | **CLOSED.** `tests/fixtures/make_mini_vocab.py` generates it at build time; `test_mini_vocab`, `test_agent_step`, `test_grammar` and `test_token_stream_gate` run on it. It did what it was for: the grammar is now tested in the gate rather than only under `realmodel` (R1). |

## How to run things

```bash
ctest --preset gate && python3 scripts/run_ratchets.py
```

32 tests in ~15 s and six ratchets. `ctest --preset realmodel` loads the real model, and
must run `-j1`.

To drive a mission by hand — this is still how the sharpest bugs get found:

```bash
python3 scripts/drive.py --workspace /path/to/ws --mission "fix the failing test"
```

`--say 3:"use the other approach"` steers mid-run, `--then "now do X"` continues after
run_end, `--auto` sends the eval harness's autonomy flags so a hand-run and a scored run
can be compared like for like, and `--deny` refuses every card. **Use `--auto` and the
fixture's own mission text when reproducing an eval result** — the settings and the exact
mission string both matter, and chasing a discrepancy that turned out to be neither cost
an hour.

Known flake: `test_sandbox`'s spin test races the wall-clock killer against `RLIMIT_CPU`,
both at 1 s. It passes on rerun. Pre-existing.

**GitHub Actions CI is red on `main`, and was already red before this session's PR** — do
not spend time on it thinking you broke something. `gate` and `sanitizers` both fail in
~20-35 s inside vendored `third_party/parsephony/src/parsephony.cpp`: the runner's Xcode
16.4 libc++ does not have `std::from_chars<double>` available (Apple gates it behind a
deployment-target check), where local Apple clang 21 does. `ctest --preset gate` locally is
the real signal; CI is a toolchain mismatch, not a code regression. Worth fixing if you want
green CI back, but it is unrelated to anything in `src/loop`, `src/context` or `src/surface`.

## Working agreements

- **Never run two MLX processes at once.** One checkpoint is 19 GB resident and peaks at
  21 GB on a 48 GB machine that normally has three IDEs open. It crashed the machine on
  2026-07-31. Anything that loads a model runs alone, in the foreground, to completion.
  **Corollary learned the hard way: never rebuild while the suite is running** — the
  sidecar is re-spawned per task, so a mid-suite rebuild silently compares two binaries.
- **Breadth before depth.** If something works but is ugly, write it down and keep going.
- **Do not measure performance.** That work is done.
- **This machine drifts ~9% in throughput** under background load, so any before/after
  spanning more than one sitting is meaningless.
- **Drive it yourself and report what broke.** Still true — and the suite now does it at
  ten times the scale, so do both.
- **Do not pipe a long run through `tail`.** It buffers everything until exit, so a
  45-minute run shows nothing until it is over. Use `python3 -u ... > log` and tail the log.

## Do not re-litigate

Settled, with arguments in `docs/PHASES.md`: Apple Silicon only, MLX in-process, Qwen3
only, XML tool-call syntax (the model's own chat template specifies it). Speculative
decoding is out of scope. **Subagents are out** — one local model, one process; that is
the whole point of this agent. The `gated_delta_update_ops` reference path and the
equivalence tests in `tests/model/test_grammar.cpp` stay: they are what make the fast
paths falsifiable.

Also settled this session: **completion is evidential, not a checklist tick** (the model
decides when to stop, the harness decides whether it succeeded), and **irreversibility is
a declared property, not a risk score** — no threshold can be tuned into expressing it.
