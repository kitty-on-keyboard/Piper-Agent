# Dense MTP dual-path — handoff for Grok (Cursor)

You are implementing this for Sean's C++ MLX sidecar (`LM_Pipe_2`). Be conservative: **MoE / A3B path stays the bakeoff reference and must not regress.** Dense 27B is the path that needs to close the gap with specialized MTP runtimes while keeping Piper's agent control plane.

## 0. Non-negotiables

- Repo: `/Users/dev/Desktop/seans_projects_local/LM_Pipe_2`.
- Branch: `fix/greenfield-write-caps`.
- Working tree is dirty (many files + untracked `docs/QWEN4EXP_SIDECAR_HANDOFF.md`). HEAD ≈ `2a568b8`.
- **Do not clean, stash, reset, checkout, or overwrite unrelated work.**
- Hardware: M5 Pro, **48 GB** unified memory. Never load two large models at once. One MLX holder at a time.
- Models already on disk (do not re-download unless Sean says so):
  - Target: `/Users/dev/Desktop/Models/Qwen3.8-27B-MLX-4bit` (~15G)
  - MTP draft head: `/Users/dev/Desktop/Models/Qwen3.8-27B-MTP-4bit` (~253M, `model_type: qwen3_5_mtp`, `block_size: 3`)
  - MoE bakeoff baseline: `/Users/dev/Desktop/Models/Qwen3.6-35B-A3B-MLX-4bit` (~19G) — leave alone unless a regression check needs it
- Enable dense MTP today via `draft_model_dir` / env `LMP_DRAFT_DIR` (see `scripts/agent_eval.py` ~386–390 and `mlx_backend.cpp` load path). Loading a draft head **auto-enables** speculation; `LMP_SPECULATIVE=0` forces plain AR with the head still loaded (A/B control).
- MTP draft is **dense-only** by design (`is_dense()` gate; MoE+MTP refused — upstream mlx-vlm crash class). Do not remove that gate to “try MoE MTP.”
- Correctness: speculative path must preserve sampling distribution (rejection sampling / residual correction — same contract as Leviathan–Chen). Do not ship greedy draft acceptance that changes answers.
- Agent hooks that must keep working on the dense path: grammar / tool masks, `LMP_SPEC_IN_TOOLCALLS`, live KV reuse (`Extend`/`Restore`/`Reset`), cancel, and the existing `SpecPhases` accounting.
- Do **not** “solve” this by HTTP-proxying to MTPLX/oMLX as the production path. That is explicitly the rejected easy seam (loses Piper control). Study those stacks; implement the wins **inside** this sidecar.
- Do **not** vendor or copy proprietary MTPLX binary guts. Use public docs, HF packaging notes, open MLX / llama.cpp draft-MTP ideas, and measurements on this machine.
- Related but separate: `piper-bench/HARDWARE_SQUEEZE.md` (suffix prefill, quantized KV, shadow-swap). Do not expand this handoff into a full MLX fork or a faster generic matmul.
- Known post-bakeoff issue (do not “fix” opportunistically mid-port): Piper ToolError / malformed tool-call grammar loops (see `piper-bench/MISS_LIST.md`). If you touch mask/spec interaction, keep that failure mode in mind; do not start a ToolError rewrite unless Sean asks.

## 1. Product goal

Sean wants **both**:

1. The advantages Piper already has (constrained tool decode, speculation under masks, live KV, agent loop), **and**
2. Dense 27B decode in the **MTPLX / oMLX class** of absolute tok/s on Apple Silicon — not the ~25 tok/s the sidecar currently gets with MTP.

Architecture intent (dual path by model shape):

| Shape | Runtime path | Status |
|---|---|---|
| MoE (A3B / 35B-A3B) | Current sidecar MLX path | Keep; do not regress |
| Dense (Qwen3.8-27B) | Same sidecar control plane + **much better** native MTP / verify / Metal residency | This work |

Community context (order-of-magnitude, not a contract):

- Sidecar plain AR dense 27B on this Mac: ~16–17 tok/s (fair quiet).
- Sidecar with existing MTP: Sean measured ~**25** tok/s (~1.5×). That uplift is **normal**; the miss is absolute tok/s vs specialized engines.
- MTPLX / oMLX published M5-class dense 27B numbers often sit ~**50–65** tok/s on coding prompts (engine + packaging, not “magic acceptance”).
- Early sidecar MTP once **lost** to plain AR (~16.2 vs ~17.4) despite ~65% acceptance — `SpecPhases` exists because verify/checkpoint overhead lied. Any speed claim must print phases + acceptance.

Success is **not** “we turned MTP on.” Success is dense decode that is clearly in another league while Piper still owns the tokens.

## 2. Goal & success criteria

### Primary (v1)

- Quiet-machine sustained dense decode with `LMP_DRAFT_DIR` pointing at the MTP head reaches **≥ 40 tok/s** median on a fixed prompt/completion length (document the lengths), with acceptance rate and `SpecPhases` printed every run.
- Stretch target: **≥ 50 tok/s** on the same harness (MTPLX-neighborhood). If stuck in the high-20s / low-30s after the milestones below, stop and report what’s left (kernel vs overhead vs acceptance) — do not thrash.

### Must not break

- MoE load + plain generate smoke still works (no MTP dir).
- Dense + MTP: grammar / tool-call speculation still runs (`LMP_SPEC_IN_TOOLCALLS` default on). Abandoned-block rate must not explode.
- `LMP_SPECULATIVE=0` with draft loaded still equals plain AR speed/quality control.
- Sampling equivalence: temperature / top-p path does not silently become greedy-draft.

### Explicit non-goals (v1)

- Porting `qwen4_exp` / Flash-Next (see `docs/QWEN4EXP_SIDECAR_HANDOFF.md`).
- MoE + MTP.
- Full MLX fork / handwritten GEMM.
- HTTP dual-backend as the shipped solution.
- Full Aider 49 JS bakeoff on 27B until tok/s + TTFT are fixed (prior 27B bakeoff was wall/TTFT-suspect — see `piper-bench/results/piper-js-27b-run1-seed7/SUSPECT_WALL_TTFT.txt`).

## 3. What already exists (do not rebuild)

| Capability | Where |
|---|---|
| Draft load + dense-only gate + auto-enable spec | `src/model/mlx_backend.cpp` (~233–267) |
| Env override A/B | `LMP_SPECULATIVE` via `speculative_env_override` |
| Speculative decode loop (separate from plain `generate`) | `decode_speculative` in `mlx_backend.cpp` |
| MTP block size from head; history proposer vs MTP drafting | `src/model/speculative.hpp` (`SpecConfig::mtp_block_size`) |
| `mtp_step` / `mtp_logits` / trim / reset via `MlxSpecForward` | `mlx_backend.cpp` |
| Phase timers (checkpoint / restore / verify / mtp_step / mtp_logits / hidden) | `SpecPhases` in `mlx_backend.cpp` — **use these** |
| Tool-call speculation opt-out | `LMP_SPEC_IN_TOOLCALLS` |
| `load_mtp` / `mtp_forward` / `has_mtp` / `is_dense` | `src/model/mlx/qwen35_moe_model.hpp` |
| Agent eval draft wiring | `scripts/agent_eval.py` (`LMP_DRAFT_DIR` → `draft_model_dir`) |
| Diag / verify helpers | `tests/model/diag_main.cpp` (`lmp_diag`) |

Today’s honest story: MTP **works**, uplift ~1.5× is fine, absolute tok/s lags specialized MTP engines. Close the **engine** gap; don’t rewrite the proposer algebra from scratch unless measurements say the algebra is the bottleneck.

## 4. What to investigate / build (ordered)

Treat these as hypotheses. Measure first; implement what the phases prove.

1. **Verify cost** — batched k-position verify vs k× single-token; Metal sync / eval granularity.
2. **Checkpoint / restore tax** — earlier MTP regression was overhead-dominated; drive `checkpoint_ms`/`restore_ms` toward noise on the hot path.
3. **Draft head path** — `mtp_step` / `mtp_logits` residency; avoid host round-trips per draft token; keep BF16/4bit quirks documented.
4. **Acceptance** — depth 3 head; if acceptance ≪ ~0.8 on coding prompts, fix drafting/sampling before chasing kernels. MTPLX coding reports often cite high per-depth acceptance (~0.95/0.88/0.80 class).
5. **Prefill / TTFT** — dense cold TTFT was previously catastrophic under bakeoff pressure (100–900s class). Speed work that ignores TTFT still fails the agent wall. Prefer not regressing prefill while accelerating decode.
6. **Packaging learnings (public)** — how MTPLX/oMLX ship “Optimized Speed” quants + runtime depth defaults (`mtplx_runtime.json` on HF cards). Replicate *ideas* (depth 3, sustained profile), not their closed code.
7. **Optional later** — if MLX cannot express a critical verify/append pattern, a **narrow** custom Metal op in the sidecar (HARDWARE_SQUEEZE rule). Not a wholesale MLX fork.

## 5. Milestones (ordered, with go/no-go)

### M0 — Fair baseline on this Mac (NO architectural rewrite)

Quiet machine, one model.

1. Plain AR dense 27B: median decode tok/s + TTFT (`LMP_SPECULATIVE=0`, no draft or draft loaded with spec forced off).
2. Current MTP: `LMP_DRAFT_DIR=/Users/dev/Desktop/Models/Qwen3.8-27B-MTP-4bit`, print acceptance + `SpecPhases` + tok/s.
3. Same prompt length / max new tokens for both; document exact command (`lmp_diag` or sidecar generate helper — prefer existing diag).

**GO** if you can reproduce ~15–20 AR and ~25 MTP (±20%) and have phase breakdown.  
**NO-GO** if MTP ≤ AR or acceptance/phases missing — fix measurement / enablement before any “optimization.”

Done when: numbers + phases are recorded in a short note under `docs/` or `piper-bench/results/` (prefer `piper-bench/results/dense-mtp-baseline.md`).

### M1 — Name the tax (still minimal code)

From M0 phases, attribute the gap to verify vs draft vs checkpoint/restore vs something else. Write the attribution before changing kernels.

Done when: one paragraph “we are leaving X ms/block on Y” with evidence.

### M2 — Kill the dominant tax (control plane preserved)

Implement the smallest change that attacks M1’s dominant bucket. Keep MoE path untouched. Keep mask/spec behavior.

Done when: same M0 harness shows a clear jump (target band toward ≥40 tok/s) **or** a written explanation why the remaining gap is Metal/MLX ceiling and needs a custom op (M3).

### M3 — Custom op only if required

If M2 plateaus below ~40 and evidence says MLX graph/sync cannot express the needed verify/append, add a **narrow** sidecar Metal op. Prove numeric parity on a tiny case before claiming speed.

### M4 — Agent smoke (not full bakeoff)

One constrained tool-call heavy turn + one plain completion on dense+MTP. Confirm abandonment rate sane, answers coherent, KV reuse still works. Only then consider a short Aider smoke (single exercise), not a 49-task run.

## 6. Suggested measurement recipe

- Always: fans/power normal, no second MLX process, no bakeoff running.
- Print: prefill tok/s or TTFT, decode tok/s, tokens generated, wall, acceptance, drafted/accepted, phase ms/block.
- A/B: `LMP_SPECULATIVE=0` vs default with `LMP_DRAFT_DIR` set.
- Optional: `LMP_SPEC_IN_TOOLCALLS=0` once, to price tool-call speculation — do not leave it off for “speed” in the agent product path.

## 7. Reference links (external, for ideas only)

- MTPLX project: https://github.com/youssofal/MTPLX
- Example HF speed packaging: `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed` (and M3 Max Q4G64 variants) — read README / `mtplx_runtime.json` for depth defaults and how they report tok/s (decode-only vs e2e vs rolling peak).
- llama.cpp draft-MTP / community Qwen3.8 MTP notes (GPU-oriented, but acceptance + n_max=3 hazards transfer).

Do not treat published peak UI tok/s as the M0 bar; match their **sustained single-stream decode** methodology as closely as possible.

## 8. How to report back to Sean

1. M0 table (AR vs MTP, lengths, acceptance, phases).
2. M1 one-paragraph tax attribution.
3. Diff summary of what changed (files + why).
4. New tok/s vs ≥40 / ≥50 targets.
5. Explicit statement: MoE path untouched / regression smoke result.
6. What you deliberately did **not** do (HTTP backend, MoE+MTP, MLX fork, full bakeoff).

## 9. Why this lines up with Sean

He already chose dual backends by model shape in principle, rejected “MLX is wrong,” and rejected giving up Piper’s agent path for raw speed. He is the person who picks the hard seam: **bring the dense MTP engine up to the specialized class inside the sidecar.** Cursor Grok implements; this doc is the brief. A3B bakeoff numbers remain the public story until dense is both fast and agent-correct.
