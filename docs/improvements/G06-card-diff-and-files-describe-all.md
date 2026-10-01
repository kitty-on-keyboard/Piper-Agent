# G06: Card diff and files describe all uncommitted work, not this slice: snapshot a baseline tree at dispatch and diff against it

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| VERIFIED (reproduced) | piper | bug | high | M |

> **Verified.** The problem and fix below are the skeptic's corrected versions. They replace the finder's original claim where the two differ.

Merged from these finder reports:

- ideas-orchestration#2 Per-slice baseline snapshot

## Problem

The core claim is true. collect_git (worker.cpp:943-1039) and collect_git_changed_paths (:809-872), and their Python copies (piper_worker.py:268-357), report the worktree against the index plus untracked files. Nothing records the workspace at dispatch. So from the second uncommitted slice on, diff_stat, git.diff and the git-path part of files_touched include earlier slices' edits. The git paths are merged into files_touched for every trust_mcp slice, and for any slice with an empty ledger and a remote write (sidecar.cpp:2304-2306).

What I reproduced:
- After slice 2 (creates c.py, edits core.py), the C++ logic reports +4 -2 across 4 files: a.py, core.py, b.py, c.py.
- The vision doc's own demo path (slice 1 creates index.html, slice 2 adds a Restart button) gives +305 -0 with the whole file shown as new on the C++ card, and +0 -0 (0 files) from the Python copy. A baseline diff gives 5 0 index.html.

One sub-claim is wrong. "They miss files the parent had staged" is not a per-slice defect. Staging before dispatch is the only thing that makes today's diff per-slice; I reproduced that. The real defect is that what the diff means depends on what the parent staged, and nothing documents it.

The candidate missed several more bugs in the same collectors. They affect slice 1 too, and the same fix removes them:
1. `git status --porcelain` shows a new untracked directory as one line, `?? pkg/`. That is not a regular file, so the code skips it, and files in a new directory never reach diff_stat, git.diff or the git paths.
2. Untracked binary files are read line by line into git.diff as '+' text.
3. When cwd is a subdirectory of the repo, porcelain paths are relative to the repo root. Untracked files in cwd are skipped, and edits outside cwd are counted.
4. With status.showUntrackedFiles=all, the .piper/slices artifacts are counted.
5. The Python copy never counts untracked files in diff_stat.

The ledger gaps are real but secondary. Shell writes are missed. Path arguments from any tool_call (worker.cpp:755-767), read_file included, are counted only under LMP_TRACE_TEXT=1, a diagnostic mode that is off by default.

## Why it matters

diff_stat and git.diff are on every card, and files_touched feeds the scope rubric. Across a multi-slice horizon without per-slice commits, which the documented loop never asks for, they are wrong from slice 2 on. The cloud either learns to ignore them or reads N slices of history to review one. This is also the prerequisite for any automatic scope check (G08).

## Fix to ship

Replace the worktree-vs-index collectors with a diff between two snapshot trees, in both runners.

1. worker.cpp: add an env-overlay parameter to run_cmd that sets variables in the child, so paths need no shell quoting. Add `snapshot_worktree(cwd, out_index, seed_index) -> optional<tree>`:
   - Return nullopt unless `git rev-parse --is-inside-work-tree` prints true and `git check-ignore -q .` fails.
   - Find the real index and object store with `git rev-parse --path-format=absolute --git-path index|objects`.
   - Copy the seed index to out_index.
   - Run with env GIT_INDEX_FILE=out_index, GIT_OBJECT_DIRECTORY=<cwd>/.piper/slices/.objects (one store shared by all slices, so unchanged content is stored once) and GIT_ALTERNATE_OBJECT_DIRECTORIES=<real store>.
   - Run `git add -A -- . ':(exclude).piper/slices' ':(exclude).piper/active.json' ':(exclude).piper/progress.log' ':(exclude).piper/orch_webhook'`, then `git write-tree`.
   - Best effort: never fail a run.

2. sidecar.cpp execute_task_packet:
   - Take the baseline right after archive_prior_events (:1937), seeded from the real index, into <result_dir>/snap/base.index.
   - Replace :2298-2307, still before run_check, with an after-snapshot seeded from base.index.
   - Run `git diff --relative --no-renames --numstat | --name-status | -p <base> <after>` with the same env. That gives diff_stat, git.diff and files_changed=[{path, status A/M/D}].
   - result.json gains baseline_tree, after_tree, files_changed and diff_scope ('slice', or 'unavailable: <reason>').
   - If no snapshot can be taken, report no diff. Never fall back to the old worktree-vs-index diff.

3. Delete collect_git_changed_paths, the untracked-file loop in collect_git, the trust_mcp / log_has_remote_tool_write union, and the tool_call argument harvest in collect_files_touched. agent.cpp:2381-2400 already emits kind=write for mutating remote tools. Set files_touched = files_changed paths ∪ ledger write paths git cannot see (ignored or outside cwd), so a write to an ignored .env still reaches the scope check.

4. piper_worker.py: mirror this in run_mission (after :1565, replacing :1630-1635). In format_review_card, print 'M index.html' and '+5 -0 (1 file, since dispatch)'.

5. Tests:
   - Python self-test: the repo has a staged file and an untracked earlier-slice file, and a tracked file already dirty. The fake sidecar edits that dirty file again, creates pkg/new.py in a new directory, and writes a binary. Assert exactly that change set, an unchanged staged set and index bytes, and no new files in .git/objects.
   - C++: a real temp-repo test in tests/surface/test_worker.cpp replacing the mock-git collect_git tests.

6. Docs: PIPER.md:99, AGENT_WAKE.md:99 and SKILL.md section 4 should say that files and diff mean this slice since dispatch.

Follow-up this makes possible: `piper revert --task`, which restores only the slice's changed paths from baseline_tree. That is a safe way to do the skill's 'Revert the surprise'.

## Evidence (file:line)

- piper:src/surface/worker.cpp:943-1000 collect_git runs `git diff --numstat` and `git diff` (worktree vs index), then appends every '??' file from `git status --porcelain` (:992) as new. There is no baseline
- piper:src/surface/worker.cpp:808-851 collect_git_changed_paths: `git diff --name-only` plus untracked files
- piper:src/surface/sidecar.cpp:2298-2307 these paths are merged into files_touched whenever trust_mcp is set; collection runs before run_check (:2323)
- piper:src/surface/worker.cpp:753-767 collect_files_touched harvests every path-like argument of any tool_call event, with no tool filter
- piper:src/surface/sidecar.cpp:1934-1937 result_dir setup and archive_prior_events, the natural pre-run hook
- piper:scripts/piper_worker.py:268,313 the Python mirror (collect_git_changed_paths / collect_git) has the same no-baseline logic
- piper: a grep for commit|git add|stage over PIPER.md, .cursor/skills/piper-orchestration/SKILL.md and docs/ORCHESTRATOR_WORKER_VISION.md finds nothing. PIPER.md:99 and SKILL.md:106 make 'files_touched inside the brief' and 'proportional diff' rubric checks

## Skeptic verdict

real=True, reproduced=True, keep=True, root-cause fix=True, already handled=False, violates constraint=False

EVIDENCE. I opened every cited line.
- The sidecar.cpp:2298-2307 merge, the order before run_check (:2323) and the hook at :1934-1937 are all as described.
- Dispatch goes through main(['run']) and runs `lmp_sidecar --worker --task`, which calls execute_task_packet. The daemon uses the same function. The Python run_mission is only a fallback.
- No baseline, snapshot, write-tree or GIT_INDEX_FILE logic exists in either repo.
- PIPER.md, SKILL.md and the vision doc never tell the parent to commit or stage. README says only that staging and commits are the user's.
- Both repos are untouched (`git status` empty).

Scripts, all in scratchpad/g06:
- repro.py runs the real Python collectors plus a line-for-line port of the C++ ones over 6 scenarios.
- proto.py and proto_private_odb.py prototype the fix.
- The Piper self-test runs with 30 scenarios and 1 failure that has nothing to do with this.
- pytest is not installed, so Godoer's tests did not run. They are not relevant here.

VALUE. Distill gets nothing from this. Orchestration gets a lot:
- The card is the rubric (PIPER.md:99). SKILL.md section 4 says "files_touched ⊆ brief, on failure: Revert the surprise" and "Diff proportional", and tells the parent not to re-read when the card is green.
- With polluted data the cloud either rejects good slices or, on Godoer slices, reverts earlier accepted work.
- git.diff grows by a slice every dispatch, which wastes cloud input tokens.
- The vision doc lists "reliable git_diff / diff_stat" as a build-out not yet done (ORCHESTRATOR_WORKER_VISION.md:115).

The doc-only alternative (tell the parent to `git add -A` or commit between slices) is a band-aid. It pushes a git workflow onto the user, does nothing for failed slices left in the tree, and leaves bugs 1-5 in place.

No hard constraint is hit. The harness already runs git and already writes .git/info/exclude (context_journal.cpp:73). The model is not involved, so local_model_can_do_it does not apply and is set false.

PROTOTYPE RESULTS (temp-index snapshot). It gave the exact per-slice set for:
- a file dirty before the slice and edited again (only the slice's 1/1 delta shows)
- a deleted earlier-slice file
- a new directory, a binary file and an ignored build/ file
- an unborn HEAD, a linked worktree, and a subdirectory cwd (with --relative)

The real index was byte-identical afterwards. Each snapshot took 8-60 ms on clones of Piper and Godoer.

Corrections to the candidate's design:
- Copying ".git/index" breaks in linked worktrees, where .git is a file. Use `git rev-parse --git-path index`.
- Excluding all of .piper would be wrong: .piper/skills is tracked user content (skills.cpp:49). Exclude only the harness-owned paths.
- The distill packet's cwd is `<project>/.godoer`, which is gitignored. `git add -A -- .` exits 1 there (reproduced), so skip the snapshot when cwd is ignored.
- Writing loose objects into the user's .git is avoidable. A private object store with the real object store as an alternate leaves .git with no new files at all (verified), and keeps the trees diffable for as long as the store lives.

Known limits:
- Edits inside submodules or embedded repos appear only as a commit-pointer change.
- Large untracked assets that are not ignored are hashed into the store. Godoer's scaffolded .gitignore covers .godot/, .godoer/, capture/ and evidence/.

Back to the [index](README.md).
