# Handoff: SWE-bench, and the Apple cache question

Paste this whole file as the opening prompt of a fresh session in
`/Users/dev/Desktop/seans_projects_local/LM_Pipe_2`.

Written 2026-08-19, at the end of the session that produced
[BAKEOFF_HARNESS.md](BAKEOFF_HARNESS.md). Read that first — it is the measurement this
one continues, including the claim it had to withdraw.

---

## The goal, stated plainly

**Establish, on a public benchmark, whether LM_Pipe is the best local coding agent — and
publish the number whether or not it is.**

The problem this fixes: every claim about this project so far is first-person. The
2026-08-19 session produced the first external comparison (LM_Pipe 6/6 in 141.4 s vs
Cline 5/6 in 186.1 s, same weights, same machine), but on **our own six tasks**. The
obvious objection is that we wrote the benchmark we win. A public benchmark answers it.

Two things make this worth doing rather than defensive:

- The field concedes the premise. Harness variance is **10–20 percentage points on
  identical model weights**, and "the harness is half the score." A same-model harness
  comparison is a recognised measurement, not a rigged frame.
- There is a **published anchor**: `mini-swe-agent` + Qwen3-Coder-30B-A3B scores **18.8%**
  on SWE-bench Verified restricted to Bash. Reproducing something near that on this machine
  calibrates our absolute scale and makes every other number here readable by a stranger.

**Success is not "we win."** Success is a number produced by someone else's benchmark,
with the arms held constant, that we publish either way. A losing number honestly reported
is worth more than the six-task win, because it is the first result that could have gone
the other way in public.

---

## Why SWE-bench and not Terminal-Bench

Terminal-Bench v2 (89 tasks) is the better *conceptual* fit — its leaderboard scores
agent+model pairs, which is exactly our claim shape. **It is the wrong first move.**
Harbor, its official harness, installs the agent *inside* the Linux container via a Jinja
template. LM_Pipe is a native macOS binary with MLX in-process and cannot run there. The
`BaseAgent` path (agent on the host, driving the container remotely) requires our shell
tool to target a container — which is exactly the T2 container work `PLAN_GAP_CLOSURE.md`
lists as unfinished. Do not start there.

SWE-bench fits the architecture: **the agent runs on the host against a checked-out repo
and emits a patch; evaluation happens separately in Docker.** Nothing of ours runs in a
container.

Practical notes gathered but **not yet verified on this machine**:

- Epoch AI publishes native **arm64** images covering ~1,819 of 2,294 instances; ~496
  instances still need x86 because of conda binaries (scikit-learn, matplotlib, xarray).
  Native images avoid the QEMU tax — a 500-instance run drops from ~14 h to ~2–3 h *on the
  test-running side*.
- **Inference dominates, not evaluation.** A local 35B doing hundreds of multi-turn tasks
  is days of wall clock, not hours. Start with SWE-bench **Lite (300)** or budget a
  multi-day background run for Verified (500). Do not invent a subset — a non-standard
  sample re-opens the exact objection this exercise exists to close.

**Arms to run**, all on the same checkpoint, strictly sequentially:

1. LM_Pipe (needs a patch-extraction path: run, then `git diff` the workspace)
2. `mini-swe-agent` — the calibration anchor against the published 18.8%
3. Cline via `scripts/xharness.py` — carries forward the existing comparison

---

## The Apple cache question, in full

This is a real, measured, **unexplained** finding. It is worth resolving properly, and it
is the most publishable thing here — but only once isolated.

### What is measured and solid

On `Qwen3.6-35B-A3B-MLX-4bit`, `mlx-swift-lm`'s `ChatSession` **re-prefills the entire
transcript on every turn**. Six agent-shaped turns, each appending ~2,430 characters, three
arms:

| turn | prompt chars | REUSE (`instructions:`) | HISTORY (`history:`) | COLD (fresh session) |
|---|---|---|---|---|
| 2 | ~2,568 | 1.069 s | 1.064 s | 1.066 s |
| 4 | ~7,429 | 3.019 s | 3.003 s | 3.096 s |
| 6 | ~12,290 | 5.013 s | 5.055 s | — |

All three indistinguishable. TTFT tracks *total* prompt length, not the appended suffix.
LM_Pipe on the same checkpoint reuses 2,758 tokens for **141 ms vs 1,630 ms (11.6x)**, and
`a_restored_cache_equals_a_reprefilled_one` passes, so ours is correct as well as fast.

### What was claimed, and withdrawn

The first writeup blamed **hybrid gated-delta attention**. That was not isolated and is
withdrawn. Do not repeat it.

### Candidates, and what is ruled out

- **[#522](https://github.com/ml-explore/mlx-swift-lm/issues/522)** — `ChatSession(instructions:)`
  rebuilds `[.system(instructions), newUser]` every call. Real, but **ruled out as the sole
  cause**: its documented `history:` workaround changes nothing here.
- **[#420](https://github.com/ml-explore/mlx-swift-lm/issues/420)** — M-RoPE state dropped
  across `ChatSession` turns for Qwen VL models. **Live candidate.** Both our checkpoints
  carry a `vision_config`. Note #420 says #399 fixed this for Qwen3.5/3.6, so if it *is*
  the cause, the fix does not cover our path.
- **Source-reading hypothesis, not evidence.** In `PromptCacheReusePolicy`,
  `ExtendCachedPrefixRule` requires the new prompt to *start with* the cached tokens, and
  `RewindToCommonPrefixRule` refuses when `turn.carriesModelState` (set from
  `lmState != nil`). A chat template that re-renders a generation prompt each turn defeats
  the first; a VL model defeats the second; the fallback is `rebuild`. **Reading their
  source produced two wrong conclusions in this session. Treat this as a lead, not a
  finding.**

### The confound, and the experiment that breaks it

`Qwen3.6-35B-A3B-MLX-4bit` and `Qwen3.8-27B-MLX-4bit` **both** carry `vision_config` *and*
`layer_types` — every checkpoint on this disk is VL *and* hybrid. To separate them, fetch a
**text-only, full-attention** MLX checkpoint (their registry has `qwen3_5_2b_4bit`) and
re-run `fmbench`. If a text-only model shows flat TTFT, the cause is VL/model-state and
#420 covers it. If it also re-prefills, the finding is broader and worth filing.

**Do not file anything until that experiment runs.** An unisolated report on an Apple repo,
probably duplicating #420, is a bad first public contribution. Check `gh` account choice
with the operator first — two are authenticated (`dyrr72xjsd-jpg` active,
`cat-collector-king`), and "main" was never confirmed.

---

## Reproduction facts that cost hours to find

- **`swift build` cannot build mlx-swift-lm usefully.** It emits no `default.metallib`, so
  the binary loads and dies at the first GPU op. Use
  `xcodebuild -scheme <target> -configuration Release -destination 'platform=macOS,arch=arm64' -derivedDataPath <dd> -skipPackagePluginValidation -skipMacroValidation build`.
  Without the two skip flags it fails validating the `CudaBuild` plugin.
- `mlx-swift-lm` does **not** depend on swift-transformers; the tokenizer is the consumer's
  job. Add `huggingface/swift-transformers` **from 1.3.0** (0.1.x predates the macro's API)
  and depend on the product named **`Transformers`**, not `Tokenizers`.
- The benchmark source is `Sources/fmbench/Bench.swift` in the cloned tree. It is in the
  session scratchpad and **will be gone**; re-create it from the table above if needed.
- **Cline**: `npm i -g cline` (3.0.55 tested). Provider id is **`openai-compatible`** —
  `openai-native` posts to `/v1/responses`, which mlx_lm answers with 404. Use
  `--data-dir` to keep benchmark state out of `~/.cline`.
- **LM Studio will not index our checkpoints** — they are symlinks into `~/Desktop/Models`
  and `lms ls` ignores them. Serve with `mlx_lm.server` from LM Studio's bundled runtime:
  `~/.lmstudio/extensions/backends/vendor/_amphibian/app-mlx-generate-mac26-arm64@33/bin/python -m mlx_lm server --model <dir> --port 8080`
  (mlx_lm 0.31.3, mlx 0.32.0).
- `timeout` does not exist on macOS. Use the harness's own timeouts.

---

## Standing rules for any arm

- **Never two MLX processes.** 15–19 GB resident each on a 48 GB host; doing it has taken
  this machine down before. Every arm runs sequentially with an explicit unload between,
  and `pgrep -f 'mlx_lm|lmp_sidecar|fmbench'` before starting the next.
- **This machine drifts ~9%.** Single runs settle 1.5x effects, not 1.1x ones. Anything
  that matters gets n>=3.
- **Import the grader, do not reimplement it.** `scripts/xharness.py` runs any agent over
  our eval tasks using `agent_eval`'s own `capture_task_contract` and
  `verify_task_integrity`, so its numbers are comparable to `evals/agent/pins.json`. Extend
  that file for new arms rather than writing a parallel scorer.
- **Fair configuration or the benchmark is rigged.** `refuse_wipe_workspace` sets
  `deny_approvals: true`; running Cline under `--yolo` removes its safety rail and it
  deletes the workspace, while `--auto-approve false` passes. Report both, always.

---

## Uncommitted state left by the 2026-08-19 session

- `docs/BAKEOFF_HARNESS.md` — new, the full writeup.
- `docs/HANDOFF_SWEBENCH.md` — this file.
- `scripts/xharness.py` — new, the cross-harness runner.
- `scripts/agent_eval.py` — `build_start_request` now reads **`LMP_DRAFT_DIR`** and adds
  `draft_model_dir` to the start request, which is how MTP speculation became reachable
  from the evaluator. Inert when the variable is unset, so existing pins stay comparable.

**The corpus pin is stale.** `agent_eval` reported `IMPROVED: corpus solved 6 > pinned 5` on
*both* checkpoints — `rename_across_files` now solves. The standing rule is a repeat run
before `--pin`. Do that early; the stale pin makes every future comparison read wrong.
