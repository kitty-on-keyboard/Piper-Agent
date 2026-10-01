# C09: Enforce the brief's scope and protect lists so a green check cannot be self-graded

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| UNVERIFIED | piper | false-success | high | M |

> **Not yet verified.** Open every cited line below and confirm the problem still holds before you implement. Drop or reshape the task if it does not.

Merged from these finder reports:

- Enforce the brief's DO NOT TOUCH list and protect the files the check depends on, so a green check cannot be self-graded

## Problem

The EDIT/CREATE/DO NOT TOUCH scope exists only as prose in prompt.md, and nothing enforces it. status=ok and PASS depend only on the check's exit code. Writes are auto-approved, so the model can make the check pass by editing the test file the check runs, or by editing a DO NOT TOUCH file, and the slice still reports ok/PASS. The skill's own example check, `--check "pytest tests/test_slice.py"`, names a test file the model can rewrite. The eval harness already counts this case as unsolved through its `protect` integrity check, but the product packet has nothing equivalent. Checking 'files_touched ⊆ the brief' is left to the cloud remembering a brief it wrote several turns earlier. The docs also say irreversible tools pause by default, but `piper packet` auto-approves them, so the guardrail the cloud believes is on is actually off.

## Why it matters

Priority 1: this closes the main false-PASS path, where the model edits the test or a DO NOT TOUCH file and the check goes green. It also turns the 'files ⊆ brief' judgement into a deterministic field. Priority 3: the cloud no longer has to re-read its brief and compare file lists by hand on every slice.

## Proposed fix

Add two repeatable flags to `piper packet`: `--scope GLOB` for the EDIT/CREATE allowlist, and `--protect GLOB` for DO NOT TOUCH, which should name the check's test files. Write both into task.json and parse them in load_packet. After the run, compute `protected_modified` and `out_of_scope` from the slice delta in C03, which is the only complete list of changed paths (it includes shell and MCP edits). Write both fields to result.json. If protected_modified is non-empty, set status error with 'protected path modified: …', and never let a green check promote it; this matches the eval's `intact`. If out_of_scope is non-empty, the card verdict is 'FAIL (out of scope: …)'. The C++ worker remains the only writer of result.json, and the card just renders the fields. Correct PIPER.md:88 and the skill so they state the real irreversible default. Optional follow-up: pass `protect` into lmp/start so native write tools refuse at write time, which gives the model faster feedback. The post-run delta stays authoritative, because shell writes bypass the write ledger.

## Evidence (file:line)

- scripts/agent_eval.py:30-31: 'A run that "fixes" a failing test by editing the test has not fixed anything'; verify_task_integrity is at 271-307; evals/agent/tasks/*/task.json:10 declare protect
- scripts/piper_worker.py:685-733: emit_task_packet has no protect or scope field, hard-codes auto_approve_writes to True, and defaults auto_approve_irreversible=True (687). src/surface/worker.hpp:33-68: TaskPacket has no protect or scope field
- src/surface/worker.cpp:1064-1079: in run_check, exit 0 keeps or promotes ok, and nothing else is consulted
- scripts/piper_worker.py:959-968: PASS iff status is ok and exit is 0. 980-1018: the card prints files_touched without comparing it to anything
- PIPER.md:99 and SKILL.md:106 make 'files_touched ⊆ the brief' a pass criterion, but the only source of the brief's file list is prose (PIPER.md:45, SKILL.md:23,38). SKILL.md:65's example check names a test file
- PIPER.md:88 says 'Irreversible tools pause unless auto-approve was set', but packets auto-approve by default and only --no-auto-approve-irreversible opts out (scripts/piper_worker.py:2353); src/surface/sidecar.cpp:2028 approves by task policy

## How the finder suggested verifying it

The Python parts can be checked here: packet emission writes scope and protect, and a synthetic result with protected_modified renders FAIL. Enforcement needs the Mac: add a test_worker.cpp case where a protected test file is modified and the check exits 0, and require status error. Then run the orchestration eval (C14) with the fixtures' protect lists passed as --protect and assert false_PASS == 0.

Back to the [index](README.md).
