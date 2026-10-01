# Piper improvement report

Prepared 2026-10-01 for a coding agent to implement. It covers two areas:

- The cloud-orchestration path: `piper packet` → `piper dispatch` / `piper worker` → the review card and `result.json`.
- The Godoer → `piper distill` crash-analysis integration, and new features in the same style: deterministic first, the local model only on failure, nothing extra on green runs.

Out of scope: the VS Code extension, and inference or model performance (inference is stable).

Line numbers refer to Piper commit `101b162` and Godoer commit `d2d08e6` (`kitty-on-keyboard/Godoer`). If a line has moved, search for the quoted code.

## How this was made

1. 13 finder agents each read one area and proposed changes, citing file:line evidence.
2. A merge step combined duplicates into 34 tasks. It dropped 48 reports, mostly duplicates ([dropped.md](dropped.md)).
3. One adversarial skeptic per task opened every cited line, tried to refute the claim, and judged whether the fix addresses the root cause. "Reproduced" means it ran code: Piper's Python harness and Godoer directly, and parts of the C++ worker compiled on Linux against stubs.
4. Verification stopped early to save credits:
   - 16 tasks are verified (16 of them reproduced).
   - 18 are not verified.
   - 0 were rejected.

   The reproduction scripts were not kept. Each task file's "Skeptic verdict" section says how the problem was reproduced.

## Status

- **VERIFIED**: the skeptic confirmed the problem and judged the fix worth shipping. Its corrected problem and fix replace the finder's. Implement from "Fix to ship".
- **UNVERIFIED**: found with file:line evidence but not checked by a skeptic. Confirm every cited line first, then implement, reshape, or drop it.

## Rules for every task

- **Hard constraints:**
  - Apple Silicon only.
  - One in-process model: no second inference server and no second model load.
  - No subagents.
  - The agent's tools never run git. The worker harness may, as `collect_git` already does.
  - C++20 with `-Werror`.
  - `ctest --preset gate` stays green.
- **Measured engines:** the ones under `src/` (blast-radius, log triage, draft proposer, MCP, PCC) may be reused freely. Re-score one before rewriting it.
- **Root cause and tests:** fix the root cause, and give every behavior change a regression test:
  - C++: under `tests/`.
  - Harness: `scripts/test_*.py` or `piper self-test`.
  - Godoer: its `tests/`.
- **Build and test:**
  - Piper: `cmake --preset dev && cmake --build --preset dev -j8 && ctest --preset gate`.
  - Godoer: `python3 -m pytest tests/test_telemetry_distiller.py -q`, plus the suites for the files you touch.
- **Docs:** keep the contract docs in step with the code. That means `PIPER.md`, `docs/ORCHESTRATOR_WORKER_VISION.md`, `docs/AGENT_WAKE.md`, the `piper-orchestration` skills under `.agents/` and `.cursor/`, `.piper/skills/godoer/SKILL.md`, and the `--help` text.
- **Commits:** one task per commit, or one tight group where the order below pairs tasks.

## Recommended order

### Phase 1: make the result trustworthy (Piper)

1. **[C02](C02-decide-final-status-in-one-tested.md):** one `finalize_run()` decides status, exit code and wake kind. A green check then promotes `max_turns`/`stalled`, and never promotes a crash, a timeout or an unanswered ask.
2. **[C01](C01-make-piper-dispatch-attached-by-construction.md):** `piper dispatch` is attached by construction, with no stdin/stdout probing. Today the probe misfires under agent tool runners such as Claude Code's Bash tool.
3. **[C06](C06-stop-promoting-a-working-mode-text.md):** stop turning a working-mode text turn into a blocking `ask_user`.
4. **[C04](C04-make-dispatch-output-a-compact-complete.md) with [G04](G04-red-check-cards-carry-2000-raw.md):** make the card compact and complete.
   - Worker telemetry goes to a per-slice log.
   - Failure evidence goes on the card.
   - The card says UNVERIFIED when no check ran.
   - Red check output goes through `log_triage`.
   - The test block gains `timed_out` and `could_not_run`. [C02](C02-decide-final-status-in-one-tested.md)'s policy reads them, so timeouts are never promoted.
5. **[C05](C05-bind-ask-answer-status-to-the.md):** bind the ask and answer files to the run. An attached dispatch returns on an ask nobody can answer, and stale ask files get cleaned up.

### Phase 2: scope (Piper)

6. **[C03](C03-scope-diff-stat-git-diff-and.md), the same task as [G06](G06-card-diff-and-files-describe-all.md):** take a baseline snapshot tree at slice start. Compute `diff_stat`, `git.diff` and `files_touched` against it in both runners: the C++ worker and its Python copy.
   - The two files hold compatible designs. Both use a temporary index plus a separate object directory, so nothing is written to the user's index, refs or object store.
   - Implement it once.
7. **[G08](G08-parse-the-brief-s-edit-create.md):** a `piper packet --protect PATH` contract. A slice is solved only when the check is green and every protected path is intact.
   - Phase 1 of [G08](G08-parse-the-brief-s-edit-create.md) is independent.
   - Its card `scope:` line needs step 6.
   - [C09](C09-enforce-the-brief-s-scope-and.md) (unverified) has extra enforcement ideas.

### Phase 3: distill and the daemon (Piper and Godoer)

8. **[G07](G07-godoer-s-distill-client-re-derives.md) and [G10](G10-distill-has-no-budget-or-ownership.md):** Python-only quick wins.
   - [G07](G07-godoer-s-distill-client-re-derives.md): delete Godoer's copies of Piper's model, Python and path discovery. Never pass `--model-dir`. Tie each diagnosis to the run that produced it.
   - [G10](G10-distill-has-no-budget-or-ownership.md): give distill a deadline and an incremental report, so a caller timeout keeps the diagnoses that already finished.
   - If [G01](G01-replace-distill-s-fake-coding-mission.md) follows soon, fold [G10](G10-distill-has-no-budget-or-ownership.md)'s deadline and report contract into it.
9. **[G02](G02-a-busy-daemon-reads-as-offline.md):** one three-state daemon probe (Absent, Idle, Busy), shared by the C++ worker, `piper_worker.py` and Godoer. The daemon answers `ping` and `status` while busy. A busy daemon must never read as offline.
10. **[G03](G03-enforce-one-model-where-the-weights.md):** enforce one model.
    - A kernel lock is held for the model's lifetime.
    - Delete the `mlx_lm` cold path and its hardcoded `/Users/dev` paths.
    - "Cold" now means starting the one daemon.
    - Follow the ship order inside [G03](G03-enforce-one-model-where-the-weights.md), which interleaves with [G01](G01-replace-distill-s-fake-coding-mission.md).
11. **[G01](G01-replace-distill-s-fake-coding-mission.md) with [G05](G05-make-the-diagnosis-card-structured-and.md):** a daemon `analyze` job replaces distill's fake coding task.
    - It runs one constrained turn on the resident model, with no tools and no workspace.
    - Output goes through a grammar-enforced report tool whose incident IDs are limited to an enum.
    - A deterministic grounding check validates the output, and the report is versioned with a status per incident.
    - Ship [G01](G01-replace-distill-s-fake-coding-mission.md) and [G05](G05-make-the-diagnosis-card-structured-and.md) together. [G05](G05-make-the-diagnosis-card-structured-and.md) is [G01](G01-replace-distill-s-fake-coding-mission.md)'s output contract.
12. **[G09](G09-crash-and-parser-gap-card-diagnose.md):** Godoer only, independent of the rest. Add a deterministic `ENGINE_CRASH` parser for the engine fork's crash block. The skeptic replaced the proposed LLM card with this parser, because a parser is enough.

### Phase 4: unverified

Confirm each one first, then implement it or drop it.

In rough order of expected value: [C07](C07-stop-search-tools-walking-piper-harness.md), [C11](C11-make-apply-patch-match-whole-lines.md), [C09](C09-enforce-the-brief-s-scope-and.md) (with [G08](G08-parse-the-brief-s-edit-create.md)), [C10](C10-run-the-operator-check-through-one.md), [C12](C12-key-the-in-loop-check-to.md), [C16](C16-keep-the-failing-check-s-diagnostics.md), [C15](C15-give-worker-runs-a-worker-prompt.md), [C17](C17-build-edit-feedback-from-a-real.md), [C13](C13-make-keep-warm-the-default-launch.md), [C08](C08-guarantee-one-truthful-terminal-result-per.md), [C14](C14-test-and-measure-the-shipped-orchestration.md), [G12](G12-deterministic-run-digest-on-stalled-fail.md), [G11](G11-godoer-inside-a-piper-slice-write.md), [G13](G13-move-godoer-s-telemetry-hook-out.md), [G14](G14-assertion-failure-card-for-probe-evaluate.md), [G15](G15-route-stall-card-link-the-stalled.md), [G16](G16-watch-note-grounding-card-resolve-a.md), [C18](C18-keep-the-slice-invariant-prefix-warm.md).

- **[C14](C14-test-and-measure-the-shipped-orchestration.md)** adds an end-to-end orchestration eval and gates the C++ worker. Move it to the front if you want numbers before and after the other changes.
- **[C18](C18-keep-the-slice-invariant-prefix-warm.md)** is inference-side KV reuse. It is last on purpose.
- **[G11](G11-godoer-inside-a-piper-slice-write.md)–[G16](G16-watch-note-grounding-card-resolve-a.md)** are new Godoer features. [G14](G14-assertion-failure-card-for-probe-evaluate.md)–[G16](G16-watch-note-grounding-card-resolve-a.md) should reuse the [G01](G01-replace-distill-s-fake-coding-mission.md)/[G05](G05-make-the-diagnosis-card-structured-and.md) analyze path rather than each building its own.

## Overlaps

- **[C03](C03-scope-diff-stat-git-diff-and.md) and [G06](G06-card-diff-and-files-describe-all.md)** are the same work.
- **[C04](C04-make-dispatch-output-a-compact-complete.md) and [G04](G04-red-check-cards-carry-2000-raw.md)** both change the card's failure evidence. [G04](G04-red-check-cards-carry-2000-raw.md)'s test-block fields feed [C02](C02-decide-final-status-in-one-tested.md) and overlap [C10](C10-run-the-operator-check-through-one.md).
- **[C10](C10-run-the-operator-check-through-one.md), [C12](C12-key-the-in-loop-check-to.md) and [G04](G04-red-check-cards-carry-2000-raw.md)** all touch the operator check. Settle the single check executor ([C10](C10-run-the-operator-check-through-one.md)) before [C12](C12-key-the-in-loop-check-to.md)'s finish-time check.
- **[G08](G08-parse-the-brief-s-edit-create.md) and [C09](C09-enforce-the-brief-s-scope-and.md)** both protect the files a check depends on. [G08](G08-parse-the-brief-s-edit-create.md)'s `--protect` contract is the verified design.
- **[G02](G02-a-busy-daemon-reads-as-offline.md) and [G03](G03-enforce-one-model-where-the-weights.md)** must land before or with [G01](G01-replace-distill-s-fake-coding-mission.md). Otherwise the analyze job inherits the busy-reads-as-offline bug and the risk of a second model load.

## All tasks

| ID | Status | Repo | Impact/Effort | Title |
|---|---|---|---|---|
| [C01](C01-make-piper-dispatch-attached-by-construction.md) | VERIFIED | piper | high/S | Make `piper dispatch` attached by construction instead of inferring detach from stdin/stdout |
| [C02](C02-decide-final-status-in-one-tested.md) | VERIFIED | piper | high/M | Decide final status in one tested function so a green check promotes max_turns and never promotes a crash |
| [C03](C03-scope-diff-stat-git-diff-and.md) | VERIFIED | piper | high/M | Scope diff_stat, git.diff and files_touched to the slice with a baseline tree taken at slice start |
| [C04](C04-make-dispatch-output-a-compact-complete.md) | VERIFIED | piper | high/S | Make dispatch output a compact, complete card: telemetry to a log file, failure evidence on the card, UNVERIFIED without a check |
| [C05](C05-bind-ask-answer-status-to-the.md) | VERIFIED | piper | high/M | Bind ask/answer/status to the active slice and the current run, and stop attached dispatch waiting on asks nobody can answer |
| [C06](C06-stop-promoting-a-working-mode-text.md) | VERIFIED | piper | medium/S | Stop promoting a working-mode text summary into a blocking ask_user |
| [C07](C07-stop-search-tools-walking-piper-harness.md) | UNVERIFIED | piper | high/S | Stop search tools walking .piper/ harness state and aborting silently on one oversized line |
| [C08](C08-guarantee-one-truthful-terminal-result-per.md) | UNVERIFIED | piper | high/L | Guarantee one truthful terminal result per run: a single lifecycle owner, visible running/died states, one cancel path for signals and deadlines |
| [C09](C09-enforce-the-brief-s-scope-and.md) | UNVERIFIED | piper | high/M | Enforce the brief's scope and protect lists so a green check cannot be self-graded |
| [C10](C10-run-the-operator-check-through-one.md) | UNVERIFIED | piper | medium/M | Run the operator check through one executor with one timeout, report timeouts as timeouts, and reuse an unchanged reading |
| [C11](C11-make-apply-patch-match-whole-lines.md) | UNVERIFIED | piper | high/M | Make apply_patch match whole lines, honor @@ anchors, and apply all-or-nothing |
| [C12](C12-key-the-in-loop-check-to.md) | UNVERIFIED | piper | high/M | Key the in-loop check to real workspace changes and take a current reading at finish, with one bounded retry on red |
| [C13](C13-make-keep-warm-the-default-launch.md) | UNVERIFIED | piper | high/M | Make keep-warm the default launch path and carry every per-run option with the run |
| [C14](C14-test-and-measure-the-shipped-orchestration.md) | UNVERIFIED | piper | medium/L | Test and measure the shipped orchestration path end to end: retire the Python mission fork, gate the C++ worker, add a real-model orchestration eval with per-slice timing |
| [C15](C15-give-worker-runs-a-worker-prompt.md) | UNVERIFIED | piper | medium/M | Give worker runs a worker prompt profile: state the acceptance check, drop IDE-only ask/sidebar guidance, cross-session recall and remember |
| [C16](C16-keep-the-failing-check-s-diagnostics.md) | UNVERIFIED | piper | medium/M | Keep the failing check's diagnostics through compaction and make context_rehydrate work |
| [C17](C17-build-edit-feedback-from-a-real.md) | UNVERIFIED | piper | medium/M | Build edit feedback from a real line alignment: NoMatch diagnostics that name the divergent line, receipts that list every changed region |
| [C18](C18-keep-the-slice-invariant-prefix-warm.md) | UNVERIFIED | piper | medium/L | Keep the slice-invariant prefix warm: anchor KV checkpoints and a byte-stable system message |
| [G01](G01-replace-distill-s-fake-coding-mission.md) | VERIFIED | piper | high/M | Replace distill's fake coding mission with a daemon analyze job: one constrained turn on the resident model, no tools, no workspace, no model swap |
| [G02](G02-a-busy-daemon-reads-as-offline.md) | VERIFIED | both | high/M | A busy daemon reads as offline: answer status from a control thread and replace the three hand-written probes with one tri-state client |
| [G03](G03-enforce-one-model-where-the-weights.md) | VERIFIED | both | high/M | Enforce one model where the weights live: delete the mlx_lm cold path, make cold mean starting the one daemon, and hold a kernel lock for the model's lifetime |
| [G04](G04-red-check-cards-carry-2000-raw.md) | VERIFIED | piper | medium/M | Red check cards carry 2000 raw tail bytes: run the check output through the measured log_triage engine and put the reason on the card |
| [G05](G05-make-the-diagnosis-card-structured-and.md) | VERIFIED | both | high/M | Make the diagnosis card structured and grounded: one grammar-enforced report tool with enum incident ids, a deterministic grounding check, and a versioned report with per-incident status |
| [G06](G06-card-diff-and-files-describe-all.md) | VERIFIED | piper | high/M | Card diff and files describe all uncommitted work, not this slice: snapshot a baseline tree at dispatch and diff against it |
| [G07](G07-godoer-s-distill-client-re-derives.md) | VERIFIED | both | high/M | Godoer's distill client re-derives Piper's internals and trusts any file on disk: successes read as errors, skips and stale cards read as ok, failed runs stale evidence |
| [G08](G08-parse-the-brief-s-edit-create.md) | VERIFIED | piper | medium/M | Parse the brief's EDIT/CREATE/DO NOT TOUCH once at piper packet: lint it before dispatch and print a deterministic scope line on the card |
| [G09](G09-crash-and-parser-gap-card-diagnose.md) | VERIFIED | both | medium/S | Crash and parser-gap card: diagnose the failed runs where Godoer today says 'read the raw output' |
| [G10](G10-distill-has-no-budget-or-ownership.md) | VERIFIED | both | medium/M | Distill has no budget or ownership contract: a caller timeout discards finished diagnoses while the daemon keeps running abandoned work and starts requests whose client already left |
| [G11](G11-godoer-inside-a-piper-slice-write.md) | UNVERIFIED | both | medium/M | Godoer inside a Piper slice: write incidents into the slice and show them on the review card instead of calling the busy model |
| [G12](G12-deterministic-run-digest-on-stalled-fail.md) | UNVERIFIED | piper | medium/S | Deterministic run digest on STALLED/FAIL cards from events.jsonl and loop_metrics |
| [G13](G13-move-godoer-s-telemetry-hook-out.md) | UNVERIFIED | godoer | medium/L | Move Godoer's telemetry hook out of the renderers: one post-grade hook emitting typed, content-addressed failure packets, with cards delivered asynchronously on every surface including --json |
| [G14](G14-assertion-failure-card-for-probe-evaluate.md) | UNVERIFIED | both | medium/M | Assertion-failure card for probe/evaluate/improve: join the failing check to its probe line, the trace's delivered signals and the handler code |
| [G15](G15-route-stall-card-link-the-stalled.md) | UNVERIFIED | both | medium/M | Route-stall card: link the stalled stage's action to control, connect, handler and change_scene call site, with cited lines |
| [G16](G16-watch-note-grounding-card-resolve-a.md) | UNVERIFIED | both | medium/M | Watch-note grounding card: resolve a person's 'that tree is too big' to node path, .tscn line and property before the agent's next command |
