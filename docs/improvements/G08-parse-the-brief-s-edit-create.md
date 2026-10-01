# G08: Parse the brief's EDIT/CREATE/DO NOT TOUCH once at piper packet: lint it before dispatch and print a deterministic scope line on the card

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| VERIFIED (reproduced) | piper | feature | medium | M |

> **Verified.** The problem and fix below are the skeptic's corrected versions. They replace the finder's original claim where the two differ.

Merged from these finder reports:

- ideas-orchestration#3 Brief scope ledger
- ideas-orchestration#6 Pre-dispatch brief linter

## Problem

The gap is real, but it is narrower and more serious than the candidate says.

What is true:
- `piper packet` validates nothing in the brief. I reproduced the finder's defective brief: exit 0, task.json written, no warnings.
- The card prints `files_touched` with nothing to compare it against.
- Nothing in Piper or Godoer parses EDIT/CREATE/DO NOT TOUCH.
- The emitter has no `--check-timeout-s`. The C++ worker honors `check_timeout_s` (default 60 s, worker.hpp:55), and a timeout sets exit -1, which makes the status error.

What is misleading:
- piper_worker.py:1591 `protected=()` is in the Python fallback `run_mission`. Production goes through the C++ `execute_task_packet` (`LMP_USE_CPP_WORKER` defaults on, :2561). That `protected` field is the eval's byte-integrity contract, which `run_mission` never checks, so it is not an unwired production hook.

The real hole:
- Production PASS means only that the check went green. The repo's own eval scores `solved = check==0 and protected files intact` (agent_eval.py:815), and 16 eval tasks protect their test files.
- `files_touched` (sidecar.cpp:2298-2307, worker.cpp:718-785) only sees native write tools and MCP path arguments.
- `shell` is `executes_commands`, not `mutates_workspace` (registry.cpp:2105-2113). agent.cpp:2295-2302 itself says shell writes bypass the ledger, and `log_has_remote_tool_write` ignores `why=shell`.
- So a model that runs `sed -i` on the acceptance test or a DO NOT TOUCH file gets PASS, the `done` webhook, and an empty trace.

Why the candidate's post-run check is unsound today:
- It is built on that `files_touched`. Repro A: a shell edit to a protected test reads as "scope: ok".
- Repro B: with trust_mcp (the Godoer path), files from earlier uncommitted slices show up as "outside".
- So "on scope ok the cloud can skip git.diff" is unsafe until G06 lands.

Why the emit-time ERROR rules are unsafe: they misfire often enough to block good work.
- 7 of 11 correct briefs were blocked with exit 3:
  - `res://` paths (Godoer uses `res://` in 446 places);
  - a dotted symbol in backticks;
  - "anything except `x`";
  - "`tests/` (except …)";
  - checks starting with `cd`, `source`, or `VAR=`. Checks run through `/bin/sh -c` (worker.cpp:76), and the PATH at emit time is not the daemon's PATH.
- 3 briefs parsed to nothing or to 'unparsed'. Two of those then flag the EDIT target itself as "outside" after a perfect run.

## Why it matters

This turns a manual rubric check done on every slice into one deterministic line, and it catches doomed briefs before any compute runs. On a green card with scope ok, the cloud can skip git.diff, which is what SKILL.md anti-pattern 6 asks for. It is deterministic: one parse at emit and one set comparison after the run, with nothing on the model path.

## Fix to ship

Keep it only in a reshaped form. Do not parse prompt.md. Ship a structured protect contract that mirrors the eval's rule `solved = check==0 and intact`.

PHASE 1 (independent of G06)
1. Emitter (piper_worker.py: build_parser, cmd_packet, emit_task_packet)
   - Add `--protect PATH`, repeatable, for a file or a directory.
   - Paths are relative to cwd. Normalize `res://` when cwd contains project.godot. Reject absolute paths, `..` and symlink escapes, using `agent_eval.safe_relative_path` / `read_regular_under` semantics.
   - A protect path that does not exist is exit 3. That check is exact, not a guess.
   - Write `protect:[...]` into task.json.
   - Append one deterministic prompt line, "Harness-verified DO NOT TOUCH: `a`, `b`", so the prose and the contract cannot diverge.
   - Add `--check-timeout-s` → `check_timeout_s`.
2. C++
   - worker.hpp/.cpp `load_packet`: parse an optional `protect` string array of contained relative paths; anything else is an invalid packet.
   - sidecar.cpp `execute_task_packet`: take a sha256 manifest of the protected regular files just before the mission starts. Walk directories, skipping .piper, .godot and .git.
   - Verify after the loop ends and before `run_check`.
   - If a file was modified or deleted: status "error", error "protected path modified: …", exit kExitError. The webhook follows the status, so it sends "stalled".
   - result.json gets `protect:{intact, modified[], deleted[], added_under_protect[]}`.
   - New files under a protected directory are reported but do not change the verdict, so Godot .uid/.import companions cannot cause a false FAIL.
3. Card (piper_worker.py)
   - `format_review_card` prints `protect: intact (N files)` or `protect: MODIFIED tests/test_x.py`.
   - `review_verdict` returns FAIL when modified or deleted is non-empty, and `--json` carries the field.
   - Mirror the same check in `run_mission` (:1589) so self-tests exercise it.
4. Docs: PIPER.md step 3, both SKILL.md copies, and the `piper init` text say to pass DO NOT TOUCH paths and the acceptance test to `--protect`.

Phase 1 tests, runnable here:
- a missing protect path exits 3;
- task.json carries `protect`;
- a fake sidecar whose shell edits a protected file gives FAIL and MODIFIED;
- a clean run reports intact;
- a cwd that is not a git repo works.

PHASE 2 (only after G06 provides a per-slice changed set that includes shell writes)
- Optional `--edit`/`--create` flags with exact emit checks:
  - a missing EDIT target is exit 3 with a difflib suggestion;
  - an existing CREATE target is a warning;
  - more than 3 files, or no `--check`, is a warning only.
- A card `scope:` line (outside / missing CREATE) computed from G06's set, never from today's `files_touched`.

PHASE 3 (measured before enabling)
- Pass `protect` into `lmp/start` so the write gate refuses native writes to protected paths.
- At tier T1, add `(deny file-write* (subpath …))` to the Seatbelt profile (sandbox.cpp:124-129). This prevents the edit instead of detecting it afterwards.
- Score it on evals/agent first; 16 fixtures already declare `protect`.

DROP
- regex parsing of prompt.md;
- the ERROR rule on the check's first word being on PATH (it misfires on `sh -c` builtins and `VAR=` prefixes, and the emit-time PATH is not the daemon's).

## Evidence (file:line)

- piper:PIPER.md:42-45 the brief must list 'Files: EDIT, CREATE, and DO NOT TOUCH'; :99 a pass requires 'files_touched inside the brief'
- piper:.cursor/skills/piper-orchestration/SKILL.md:22-23 1-3 files per slice, and enumerate EDIT/CREATE/DO NOT TOUCH; :36-38 the fixed template lines; :106 the rubric row 'files_touched ⊆ the brief'; :129-130 anti-patterns 6 and 7 (re-reading when green; trusting 'done' with no --check)
- piper:scripts/piper_worker.py:685-729 emit_task_packet writes only id/cwd/prompt/model_dir/flags/check/result_path/trust_mcp; :788-819 cmd_packet does no content validation
- piper:scripts/piper_worker.py:2344-2354 the packet flags have no check timeout; src/surface/worker.cpp:347-349 load_packet parses check_timeout_s
- piper:scripts/piper_worker.py:980-1018 the card prints files as a joined list with no reference set; :1591 the worker passes protected=()
- Finder repros (scratchpad/scope_proto.py, scratchpad/brief.md): a DO NOT TOUCH edit plus an unplanned file render a normal card; a defective brief (typo'd EDIT, 4 EDITs plus a CREATE of an existing file, DO NOT TOUCH 'src/', no --check) emits with exit 0 and no warnings

## Skeptic verdict

real=True, reproduced=True, keep=True, root-cause fix=False, already handled=False, violates constraint=False

Evidence I checked myself:
- These citations are accurate: PIPER.md:42-45 and :99; SKILL.md:22-23, 36-38, 106, 129-130; piper_worker.py:685-729, 788-819, 980-1018 and 2344-2354; worker.cpp:347-349.
- `execute_task_packet` is the correct C++ hook. It is the single entry for both the CLI worker (sidecar.cpp:2731) and the daemon socket `run` (sidecar.cpp:2580).
- "600 s per defect" is overstated for a typo'd EDIT, since the model often finds the right file. It is plausible for a check that cannot run, because `verify_contract` thrashes until max_turns.

Value for the user's two uses:
- Orchestration only. Zero value for `piper distill` or Godoer crash analysis; Godoer never calls packet or dispatch.
- No model is involved, so it fits the user's taste for deterministic, zero-overhead features.
- How much it adds on top of G06: once G06 gives an accurate per-slice changed set, the cloud could catch a DO NOT TOUCH edit by eye. What remains is:
  - a deterministic verdict;
  - workspaces that are not git repos, where G06's git baseline does nothing;
  - robustness when the cloud skims the card or loses the brief to compaction.
- The eval pins show no recorded tampering, so the gain is closing a false-PASS hole, not fixing an observed rate.

Design verdict: parsing LLM-written prose to recover a contract that can block emission is a band-aid. The harness already owns the packet shape, so scope should be structured packet data.

Repros (scratchpad/g08v/):
- `postrun_repro.py`: real collectors; shell tamper reads as ok; trust_mcp reports cumulative dirt as outside.
- `lint_repro.py`: 11 correct briefs, 7 blocked.
- `protect_proto.py`: a run-start sha256 snapshot catches the shell tamper in a non-git workspace. It took 0.25 ms for 2 files and 48 ms for a 2000-file directory in Python.

The existing `piper_worker.py self-test` runs here with a fake sidecar (30 scenarios, 1 failure that already existed), so phase 1's Python side can be verified here. The C++ side needs the Mac.

Back to the [index](README.md).
