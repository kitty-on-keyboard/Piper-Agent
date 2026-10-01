# C03: Scope diff_stat, git.diff and files_touched to the slice with a baseline tree taken at slice start

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| VERIFIED (reproduced) | piper | false-success | high | M |

> **Verified.** The problem and fix below are the skeptic's corrected versions. They replace the finder's original claim where the two differ.

Merged from these finder reports:

- Scope diff_stat, git.diff and files_touched to this slice using a git baseline snapshot
- Compute diff_stat, git.diff and MCP files_touched against a per-slice baseline
- Scope result.json diff and files_touched to this slice, using a baseline tree taken at slice start

## Problem

The production path writes result.json through C++ `execute_task_packet`, and `piper dispatch`, `piper worker run` and `piper worker serve` all reach it. At sidecar.cpp:2298-2307 it runs collect_git, which computes diff_stat and git.diff as worktree-vs-index plus untracked regular files from `git status --porcelain`. There is no reference point taken at slice start. Nothing in the loop commits: `piper progress` only appends a log line, and PIPER.md, the vision doc and SKILL.md never say to commit.

I compiled the real worker.cpp unmodified on Linux and ran all five sub-claims:

(1) Earlier slices count against later ones.
- After an uncommitted 40-line slice, a slice that writes 2 lines reports +42 (2 files).
- If this slice's file is in a new directory, the card reports +40 (1 file). That is only the previous slice's work and none of this slice's.
- An untracked prompt.md at the root is counted on every slice.
- With trust_mcp, earlier-slice paths and prompt.md are also merged into files_touched.

(2) A slice that only creates tests/unit/test_x.py sees porcelain `?? tests/`. The result is +0 -0 (0 files) and no git.diff.

(3) With cwd=app/:
- app/new.py is dropped, because its root-relative path gets joined to cwd.
- util.py at the root, outside cwd, is counted.
- collect_git_changed_paths returns root-relative util.py, mixed in with the cwd-relative ledger paths.

(4) A shell edit only emits workspace_freshness why=shell (agent.cpp:2295-2301), so it never reaches files_touched. It is still counted in diff_stat's file count and shown in git.diff. The card's file list and file count therefore disagree with nothing flagging it, and the rubric check "files_touched inside the brief" passes. Example: npm install rewriting a lockfile the brief says not to touch.

(5) A 4 KB random untracked binary becomes +22 "insertions" and 4.2 KB of raw bytes in git.diff.

Corrections to the claim:
- The cited `{'insertions': 40, 'files': 1}, b.py missing` figure comes from the Python fallback, whose collect_git ignores untracked files entirely. I re-ran it on the candidate's own r3 repo. On the C++ path a new root-level file IS counted (+41, 2 files); it only goes missing inside a new directory.
- The Python mirrors run only in run_mission, the fallback used when LMP_USE_CPP_WORKER=0 or the sidecar is a .py fake in self-test. They are not on the production path, and they have already drifted from C++: for the same tree, Python reports +0 where C++ reports +2.
- Sub-claim (4)'s "passes review" is accurate for the files_touched check. A careful reader could still spot the file-count mismatch.

## Why it matters

Priority 1: diff_stat and files_touched are two of the four checks on the card. They go wrong in both directions (phantom work, missing work, hidden out-of-scope edits) in four cases: from the second uncommitted slice on, on any slice that creates a directory, in subdirectory workspaces, and for every shell edit. Priority 3: git.diff stays the size of one slice instead of growing with every uncommitted slice. This is also a prerequisite for C09's scope/protect enforcement and for the diff in C08's died result.

## Fix to ship

1. In src/surface/worker.{hpp,cpp}, add a function that snapshots the workspace into a git tree without touching the user's repo.
- Signature: `std::optional<SliceSnapshot> snapshot_workspace(const std::string& cwd, const std::filesystem::path& snap_dir, const SliceSnapshot* seed, const std::vector<std::string>& excludes, double timeout_s)`. SliceSnapshot holds the index path, object dir, alternates and tree.
- Resolve the real index and objects directories with `git rev-parse --path-format=absolute --git-path index` and `--git-path objects`.
- Copy the seed into `<snap_dir>/{base|end}.index`: the real index for the base snapshot, base.index for the end snapshot.
- Run `GIT_INDEX_FILE=<idx> GIT_OBJECT_DIRECTORY=<snap_dir>/objects GIT_ALTERNATE_OBJECT_DIRECTORIES=<real objects> git add -A -- . <excludes>`, then `git write-tree`. Shell-quote the paths, since run_cmd goes through `sh -c`.
- The excludes are `:(exclude).piper`, `:(exclude)<result_dir relative to cwd>` when result_dir is inside cwd, `:(exclude).lmp-context.db*`, `:(exclude).lmp_spool` and `:(exclude).lmp_tmp`.
- Nothing is written to the user's index, refs, HEAD, worktree or object store.

2. Replace collect_git and collect_git_changed_paths with `collect_slice_diff(cwd, base, end, out_dir, RunResult&)`. Run it with the same env:
- `git diff --relative --name-status -z B E -- .` gives touched paths with an A/M/D/R status.
- `git diff --relative --numstat -z B E -- .` gives diff_stat. Binary files show as `-`: count them as a file with 0 lines.
- `git diff --relative B E -- .` gives git.diff.
- Delete all porcelain parsing and the getline dump of untracked files.

3. In src/surface/sidecar.cpp execute_task_packet:
- Take the base snapshot immediately before start_mission (~line 2104), so the daemon path gets one per slice.
- Take the end snapshot where lines 2298-2307 are now, before compose_result_message and run_check, so files the check itself writes are never counted.
- Set files_touched to the union of the ledger and the slice paths, always. Delete the trust_mcp/remote_tool special case and log_has_remote_tool_write.
- Remove the snap dir afterwards; git.diff is the artifact.

4. In RunResult, write_result and format_review_card:
- Add `diff_scope`: "slice", "worktree" or "none". Also record `base_tree` and `end_tree`.
- When no baseline is possible (not a repo, a git failure, or a timeout budget of about 10 s), keep files_touched from the ledger only and set diff_scope accordingly. The card prints diff_scope whenever it is not "slice", so an unscoped diff is never shown as slice-scoped.

5. In scripts/piper_worker.py, either port the same snapshot (about 40 lines) into the run_mission fallback or retire that fallback. Either way, delete collect_git, collect_git_changed_paths and log_has_remote_tool_write there so the two implementations cannot drift.

6. In tests/surface/test_worker.cpp, use real git repos, not the mock-git script. Cover:
- earlier uncommitted dirt plus an untracked prompt.md
- a new nested directory
- cwd set to a subdirectory
- a shell edit with an empty ledger
- a delete plus rename
- an untracked binary
- harness files in result_dir
In each case, assert that the real index, HEAD and `git count-objects` are unchanged.

7. Docs: add one line each to the PIPER.md step-5 rubric and SKILL.md §4 saying files and diff are scoped to the slice (see diff_scope).

## Evidence (file:line)

- src/surface/worker.cpp:964 runs `git diff --numstat` and 984 runs `git diff`: worktree against index, whole repo, no baseline
- src/surface/worker.cpp:986-1003: `??` entries are joined to cwd and kept only if is_regular_file, so `?? pkg/` is skipped and root-relative paths break when cwd is a subdirectory. 1004-1021: untracked files are read with getline and prefixed with '+', so binaries are dumped as text
- src/surface/worker.cpp:846-866: collect_git_changed_paths uses the same porcelain logic (`if (!is_regular_file(full, ec)) continue;`)
- src/surface/worker.cpp:718-785: collect_files_touched only collects write events and path-like args
- src/surface/sidecar.cpp:2298-2307: git paths are merged into files_touched only with trust_mcp, or when the ledger is empty and a remote_tool bump exists
- src/loop/agent.cpp:2295-2301: shell mutations only bump freshness (why=shell) and never reach files_touched
- scripts/piper_worker.py:268-310 and 313-357: Python mirrors with the same logic; the Python collect_git ignores untracked files entirely
- PIPER.md:42,54 and SKILL.md:64-65: the cloud writes prompt.md inside the workspace, and no doc says to commit between slices
- Repros (scratchpad r3, r5, cpp_collect_git_mirror.py): an earlier uncommitted 40-line edit plus a new b.py gives {'insertions': 40, 'files': 1} with b.py missing; a new tests/unit/test_x.py gives porcelain '?? tests/' and no changed paths; cwd=app/ drops app/new.py and counts util.py, which is outside cwd
- Prototypes (scratchpad r6, baseline_proto.py, a scratch repo): a temp GIT_INDEX_FILE plus `git add -A -- . ':(exclude).piper'` plus `git write-tree`, taken before and after, gives exactly this slice's paths (e.g. 'M test_a.py', 'A tests/unit/test_new.py') for both cwd=. and cwd=app, and leaves the real index untouched

## Skeptic verdict

real=True, reproduced=True, keep=True, root-cause fix=True, already handled=False, violates constraint=False

Evidence is in <scratch, not kept>/c03v/:
- driver.cpp links the real src/surface/worker.cpp, with a stub mach-o/dyld.h.
- scen.sh holds scenarios S1-S5 against the current code.
- proto.py holds my own baseline prototype.

Ways I tried to refute it, all of which failed:
- No doc or helper commits or stages between slices.
- No existing slice-scoped snapshot exists. PCC's CAS stores only the agent's own artifact revisions; blast-radius is per tool call.
- Existing tests (test_worker.cpp:1453-1581) use a mock git script and cover none of these cases.
- Nothing in the prompt steers the model away from shell edits.
- The trust_mcp union at sidecar.cpp:2304 was the fix for stress-log issue P1 (docs/AGENT_LOOP_WINS_BRANCH_PLAN.md:317). It carries the same baseline and porcelain bugs into files_touched for every Godoer slice after the first.
- The vision doc still lists "reliable git_diff / diff_stat" as unfinished (docs/ORCHESTRATOR_WORKER_VISION.md:115).

Why it matters: 3 of the 5 rubric items (files, diff, git_diff) are wrong in both directions under the default loop. They show phantom work, hide real work, and let out-of-scope shell edits through. A wrong +400 on a 20-line slice also pushes the cloud to read an ever-growing cumulative git.diff, which costs cloud tokens.

The design holds up. In my prototype, using a temp GIT_INDEX_FILE plus `add -A` plus `write-tree` before and after, then `diff --relative B E -- .`, gave exactly this slice's changes in all 7 cases:
- earlier dirt
- a new nested directory
- a cwd subdirectory
- `sed -i`
- a delete plus rename (R100)
- a binary (counted as 1 file, 0 lines, "Binary files differ")
- harness files under .piper

The real index, `git status` and HEAD stayed byte-identical. An early "changed" reading came from my own `git status` refreshing the index; `--no-optional-locks` removed it.

Cost on a clone of this repo (947 tracked files, plus 200 untracked .py and a 5 MB binary):
- base snapshot: 0.17 s cold
- end snapshot: 0.05 s
- the diff itself: 0.005 s

That is negligible next to model time.

Scoping to cwd matches what the slice can write. Agent mode runs at sandbox tier 1 (turn.cpp:111), and the Seatbelt profile denies writes outside the workspace root (sandbox.cpp:128-129). The one exception is unsandboxed MCP processes.

Refinements over the candidate's fix:
(a) Don't write loose objects into the user's .git. Set GIT_OBJECT_DIRECTORY=<result_dir>/snap/objects and GIT_ALTERNATE_OBJECT_DIRECTORIES=<real objects dir>. I verified that `git count-objects -v` stays unchanged and the diffs still resolve. This avoids .git bloat and auto-gc "unreachable loose objects" noise on repos with large untracked content, which matches the project's ethos in context_journal.cpp.
(b) Seed end.index from base.index, not from the real index, so untracked files hashed at the base keep their stat cache.
(c) Exclude result_dir itself, relative to cwd, not just a hardcoded `.piper`. `--result-path` or `--out` can place events.jsonl, live.jsonl and orch.jsonl inside cwd, and they would then be counted.

Leave the Python decision explicit. Deleting the mirrors means either porting the ~40-line snapshot to run_mission or retiring that fallback. worker.hpp already calls the Python driver superseded. Either way, update the self-test.

Remaining edge cases are acceptable and handled by the timeout/fallback path:
- Edits inside submodules are recorded as gitlinks only.
- git-lfs clean filters still write to .git/lfs.
- Concurrent operator edits during a run get attributed to the slice.

No hard constraint is touched. The harness runs git, not the agent; refs, HEAD, the real index and the worktree are untouched. worker.cpp's collect_git is not a measured engine.

Back to the [index](README.md).
