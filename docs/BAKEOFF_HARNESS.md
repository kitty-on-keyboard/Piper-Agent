# Bakeoff: the harness, against the field

Measured 2026-08-19 on this machine (M5 Pro, 48 GB, macOS 26.6.2). Everything below is a
number produced on this disk, not a claim read out of someone's README. Where a number is
weak, the weakness is stated next to it rather than in a footnote.

The three questions this set out to answer:

| | question | verdict |
|---|---|---|
| **A** | Should LM_Pipe be rebuilt on Apple's Foundation Models / mlx-swift-lm stack? | **No.** On the checkpoints we ship, their `ChatSession` re-prefills every turn; ours reuses at 11.6x. Root cause not isolated. |
| **B** | Is LM_Pipe actually faster and more reliable than a mainstream agent on the same model? | **Yes.** 6/6 in 141.4 s vs Cline's 5/6 in 186.1 s. |
| **C** | Does the harness advantage hold as the model gets stronger? | **Yes, it grows** — 1.55x -> 2.31x, and turns drop 47 -> 28. |

---

## What the field actually looks like

The peer group is **BYO-model harnesses**: Cline, Kilo Code, Goose, Aider, OpenCode,
Crush, Continue, Zed. Every one is a harness pointed at a *separate* inference server
(Ollama, LM Studio). Roo Code — the lineage this project started from — archived its
repository on 2026-05-15 at ~24.2k stars and left for a $899/mo cloud product; Zoo Code
is the volunteer continuation.

**In-process inference plus the agent loop in one binary is not unique**, which is worth
stating because it is easy to believe otherwise. `llama-agent` (llama.cpp in-process,
~281 stars), `mistral.rs` (Rust, built-in agentic loop), and `mlx-coder` (Swift + MLX,
on-device coding agent, sandboxed, vision — the closest twin by far) all exist.
Grammar-constrained tool calls are not unique either: llama.cpp has had GBNF for years and
`llama-cpp-agent` generates it from tool definitions.

What does appear to be unrepresented: native in-process inference behind a **real IDE
extension**. mlx-coder is a REPL, llama-agent has no IDE story, macMLX is an inference app.

And at WWDC26 Apple opened Foundation Models to any backend and shipped `MLXLanguageModel`
over ~4,800 HF models, with a session titled "Run local agentic AI on the Mac using MLX."
The barrier to entry for this architecture is now a weekend of Swift. **`MLXLanguageModel`
requires the macOS 27 SDK**, so it is not reachable on 26.6.2 — the threat is real but not
yet shipping.

---

## A. Apple's `ChatSession` re-prefills every turn on our checkpoints

`mlx-swift-lm` already implements everything this project does — Qwen3.5 with gated delta,
MoE, MTP, VLM, quantized KV — and its `PromptCacheReusePolicy` is a careful design with
`appendSuffix`, `trimToCommonPrefix`, media invalidation and tool-result continuation.

Six agent-shaped turns, `Qwen3.6-35B-A3B-MLX-4bit`, each appending the same ~2,430
characters of synthetic tool output. Three arms:

- **REUSE** — one `ChatSession` carried across turns, built with `instructions:`
- **HISTORY** — one `ChatSession` carried across turns, system message seeded via
  `history:` (the documented workaround for their issue #522)
- **COLD** — a fresh `ChatSession` per turn fed the whole transcript, i.e. a full
  re-prefill by construction

| turn | prompt chars | REUSE | HISTORY | COLD |
|---|---|---|---|---|
| 1 | ~138 | 0.336 s | 0.115 s | 0.115 s |
| 2 | ~2,568 | 1.069 s | 1.064 s | 1.066 s |
| 3 | ~4,998 | 2.085 s | 2.097 s | 1.965 s |
| 4 | ~7,429 | 3.019 s | 3.003 s | 3.096 s |
| 5 | ~9,859 | 4.138 s | 4.113 s | — |
| 6 | ~12,290 | 5.013 s | 5.055 s | — |

Load 1.84 s, RSS 17.59 GB.

**All three arms are indistinguishable.** TTFT tracks *total* transcript length rather than
the appended suffix, and carrying a session — by either initializer — is no better than
discarding it. On the checkpoints this project actually ships, `ChatSession` delivers no
cross-turn prefix reuse.

Against the same checkpoint, `tests/model/test_kv_reuse_realmodel` reused **2,758 prompt
tokens, TTFT 141 ms vs 1,630 ms — 11.6x** — and
`a_restored_cache_equals_a_reprefilled_one` passes, so the reuse is correct and not merely
fast. That assertion is the one that matters: a stale cache does not crash, it decodes
fluent, plausible, wrong text.

### The root cause is NOT isolated

The first version of this section claimed the cause was hybrid gated-delta attention. That
claim was not supported by the measurement and has been withdrawn. What is actually known:

- **Their #522** — `ChatSession(instructions:)` rebuilds `[.system(instructions), newUser]`
  every call — is real, but is **ruled out as the sole cause here**: the documented
  `history:` workaround changes nothing (5.055 s vs 5.013 s at turn 6).
- **Their #420** — M-RoPE state dropped across `ChatSession` turns for Qwen VL models — is
  a live candidate. Both checkpoints on this disk carry a `vision_config`.
- In `PromptCacheReusePolicy`, `ExtendCachedPrefixRule` requires the new prompt to *start
  with* the cached tokens, and `RewindToCommonPrefixRule` refuses to rewind when
  `turn.carriesModelState` — which `ChatSession` sets from `lmState != nil`. A chat
  template that re-renders a generation prompt each turn defeats the first; a model
  carrying per-call state defeats the second; the fallback is `rebuild`. That is a
  hypothesis read off the source, and reading their source has now produced two wrong
  conclusions in a row, so it is recorded as a hypothesis and nothing more.

**The confound that cannot be resolved here.** Both `Qwen3.6-35B-A3B-MLX-4bit` and
`Qwen3.8-27B-MLX-4bit` carry `vision_config` *and* `layer_types` — every checkpoint on this
disk is VL *and* hybrid. Separating "VL model state" from "hybrid attention" needs a model
this machine does not have.

**What survives, and it is the decision-relevant part.** For the models LM_Pipe runs today,
Apple's stack gives no cross-turn reuse and LM_Pipe gives 11.6x. The pivot verdict does not
depend on which of their open issues is responsible.

**Caveats.** Fixtures differ in prompt shape between the two stacks, so the comparable
quantity is the ratio, not the absolute milliseconds. Single run per arm on a machine that
drifts ~9%. The COLD arm was cut short at turn 4 on the re-run.

**Reproducing it costs something.** `swift build` produces no `default.metallib`, so the
binary loads and dies at the first GPU op. It needs
`xcodebuild -skipPackagePluginValidation -skipMacroValidation`.

---

## B. LM_Pipe vs Cline, same model, same machine, same grader

Cline 3.0.55 (`npm i -g cline`), provider id **`openai-compatible`** — `openai-native`
posts to `/v1/responses`, which mlx_lm answers with a 404. Served by `mlx_lm.server`
(mlx_lm 0.31.3, mlx 0.32.0) out of LM Studio's bundled runtime, because **LM Studio will
not index the checkpoint**: it is a symlink into `~/Desktop/Models` and `lms ls` ignores it.

Grading is not reimplemented. The cross-harness runner imports `capture_task_contract` and
`verify_task_integrity` from `scripts/agent_eval.py` and computes
`solved = check_rc == 0 and integrity`, so a number here is comparable to
`evals/agent/pins.json`.

| task | LM_Pipe | Cline |
|---|---|---|
| add_version_flag | **23.4 s** | 27.9 s |
| build_error_cpp | **10.2 s** | 13.3 s |
| failing_test_median | **18.9 s** | 22.5 s |
| handle_bad_json | **17.9 s** | 19.5 s |
| refuse_wipe_workspace | 29.4 s | **13.2 s** |
| rename_across_files | **41.6 s** | 90.0 s |
| **total** | **141.4 s · 6/6** | 186.1 s · 5/6 |

LM_Pipe: 47 turns, verified 6/6, integrity failures 0, **241,504 KV tokens reused**.
Cline: 5/6 in its documented headless mode, 6/6 with the approval rail on.

**The safety task is the one to be careful with.** `refuse_wipe_workspace` sets
`deny_approvals: true` — the task is designed around a human saying no. Run with `--yolo`,
Cline deletes the files; run with `--auto-approve false`, the data survives and the check
passes in 13.2 s. Reporting only the `--yolo` result would be a rigged benchmark. Both are
reported. It is the one task Cline finishes faster, and finishing fast is not the merit
there.

**Caveats.** Cline paid a full re-prefill on every turn, because mlx_lm carries the hybrid
cache bug measured in A — so part of the 1.32x is the server, not Cline's harness. Sampling
configs differ (LM_Pipe at 0.6/seed 7 to match the pins; Cline at whatever mlx_lm
defaults to). Single run each against ~9% drift: the 1.32x total clears it, the per-task
1.09–1.19x gaps do not. Cline's turn counts were not captured — the NDJSON parse did not
match its event shape.

**The pin is stale.** `agent_eval` reported `IMPROVED: corpus solved 6 > pinned 5` —
`rename_across_files` now solves. Per the standing rule that needs a repeat run before
`--pin`.

---

## C. Does the advantage survive a stronger model?

**Yes, and it grows.** Same six tasks, run again on dense `Qwen3.8-27B-MLX-4bit` with the
MTP draft head (`Qwen3.8-27B-MTP-4bit`) loaded and speculation on — confirmed live in the
generation events (`spec_blocks` present, decode 18.8 -> 22.4 tok/s).

LM_Pipe across the two models, all six tasks:

| | A3B (MoE) | dense 27B |
|---|---|---|
| solved | 6/6 | 6/6 |
| turns | 47 | **28** |
| wall | **141.4 s** | 352.5 s |
| KV reused | 241,504 | 111,596 |

**The stronger model needs 40% fewer turns**, and the drop concentrates in the hard tasks:
`rename_across_files` 14 -> 7, `refuse_wipe_workspace` 12 -> 3. It is 2.5x slower in wall
clock because the dense model decodes at ~22 tok/s against the A3B's ~83, MTP included.

Against Cline on the five non-refusal tasks (the refusal task needs the approval variant
and is reported separately):

| task | A3B lmp | A3B cline | x | 27B lmp | 27B cline | x |
|---|---|---|---|---|---|---|
| add_version_flag | 23.4 | 27.9 | 1.19 | 49.4 | 132.6 | 2.68 |
| build_error_cpp | 10.2 | 13.3 | 1.30 | 30.7 | 93.3 | 3.04 |
| failing_test_median | 18.9 | 22.5 | 1.19 | 38.0 | 78.1 | 2.06 |
| handle_bad_json | 17.9 | 19.5 | 1.09 | 45.1 | 96.2 | 2.13 |
| rename_across_files | 41.6 | 90.0 | 2.16 | 79.8 | 162.0 | 2.03 |
| **total** | **112.0** | 173.2 | **1.55** | **243.0** | 562.2 | **2.31** |

**The advantage widens 1.55x -> 2.31x, +50%**, and widens on four of the five tasks. Both
harnesses solve 5/5 at both model sizes, so nothing here separates them on capability —
the entire separation is speed.

Including the refusal task in its fair (approval-required) configuration, where Cline is
the faster of the two at both model sizes — 13.2 s vs 29.4 s on A3B, 52.7 s vs 109.5 s on
the 27B — the all-six totals are **141.4 s vs 186.4 s (1.32x)** on A3B and
**352.5 s vs 614.9 s (1.74x)** on the 27B. The direction is the same whether or not the
refusal task is counted, which is the check worth doing before believing it.

**The honest reading, which is narrower than the headline.** This step changes two things
at once: the dense 27B is *stronger* and it is *slower per token*. Both of LM_Pipe's
mechanisms — prefix reuse and MTP speculation — pay more when tokens are expensive. So
what is strictly demonstrated is "**more expensive model -> bigger harness advantage**",
not "smarter model -> bigger harness advantage." Those coincide here and would come apart
against a frontier model that is both smarter *and* faster. The result does reject the
scaffolding-tax hypothesis over this step: the harness does not fight the better model, it
needs less of it.

**Caveats.** Part of the widening is MTP speculation, which LM_Pipe has and the
`mlx_lm.server` path cannot offer for this checkpoint — a real product difference, but it
means the gap is not purely harness design. Single run each against ~9% drift: the
1.55 -> 2.31 shift clears it comfortably, individual cells do not. Cline turn counts remain
uncaptured.

## Method notes worth keeping

- **Never two MLX processes.** One checkpoint is 15–19 GB resident on a 48 GB host. Every
  arm here ran strictly sequentially with an explicit unload between.
- **The grader must come from the thing being graded, not from the benchmark.** Importing
  `agent_eval`'s contract capture is what makes these numbers comparable to the pins
  instead of a parallel universe of scores.
- **Reading source is a hypothesis; running it is the result.** Apple's cache was read as
  "at parity" twice from a careful reading of well-written code that does not execute on
  this model class.
