# C11: Make apply_patch match whole lines, honor @@ anchors, and apply all-or-nothing

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| UNVERIFIED | piper | false-success | high | M |

> **Not yet verified.** Open every cited line below and confirm the problem still holds before you implement. Drop or reshape the task if it does not.

Merged from these finder reports:

- Make apply_patch match whole lines and honor @@ anchors (it currently false-applies)
- Make apply_patch all-or-nothing across sections and files, and record writes that did land

## Problem

apply_patch can report a result that does not match what landed on disk. There are three causes.
(1) It claims exact matching, but it finds each hunk with std::string::find over the whole file, on hunk lines joined with newlines and with no leading or trailing newline. The first search line can therefore match the tail of a different line, and the last search line can match the head of one. A stale or hallucinated `-x = 1` silently edits `    max = 1` and is reported as Applied.
(2) It throws away the V4A `@@ <text>` anchor. An add-only hunk is appended at end of file regardless of the anchor the model gave, and an anchor cannot pick between repeated context. The approval gate leaves apply_patch updates ungated on the premise that it is 'exact match or refuse'.
(3) It computes each '*** Update File' section from the original snapshot, so two sections for one path each produce a whole file. The first commits and erases the read ledger. The second fails with Conflict, and the tool returns an error that does not say half the patch landed. Retrying duplicates the first hunk every time. Any commit failure partway through a multi-file patch behaves the same way. Because the result is an error, the agent records no write event, so the landed file is missing from files_touched and gets no syntax check.
Separately, the pre-edit 'was it already broken' snapshot keys on a `path` param that apply_patch does not have, so a patch to an already-failing file is reported as having broken it.

## Why it matters

Priority 1. Two failure modes reach the cloud looking like success or a plain error. One is a silent wrong edit reported ok, with the file listed in files_touched. The other is a partial apply reported as a plain error, with the landed files missing from files_touched. The cloud catches either only if the check exercises that line or it reads the diff closely, and the protocol tells it not to by default. Priority 2: retries stop corrupting files.

## Proposed fix

1. Match hunks as sequences of whole lines. Split the working text into lines, tracking CRLF once. Compare each search line byte-exactly against an entire file line, and count whole-line-sequence occurrences to detect Ambiguous.
2. Parse `@@ <text>` into an anchor. When present, it must equal a whole (trimmed) line, and the search starts only after it (V4A semantics), which also lets it disambiguate.
3. Refuse a hunk with no context or '-' lines unless the file is empty or the hunk is an explicit end-of-file append (`*** End of File`). The refusal says 'add a context line so the insertion point is unambiguous'.
4. In apply_to, carry one working copy per path across sections, or reject a repeated path at parse time with 'put all hunks for <path> under one *** Update File header'.
5. In the registry, check every precondition (version or absence) for every change before the first commit. If a commit still fails partway, either roll the landed files back from the snapshot, or return a result that names them in structured_json so the agent emits write events and syntax checks for them.
6. Take pre_edit_clean_ for every path from apply_patch::paths_in_patch before execute.
7. Optionally run hunk bodies through strip_line_numbers, as replace_in_file already does.
8. Add engine tests for mid-line, prefix, anchored add-only, anchor disambiguation, repeated-path and mid-commit failure.
graft is untouched.

## Evidence (file:line)

- src/tools/apply_patch.hpp:8-12: its contract reads 'Exact context/preimage matching only'
- src/tools/apply_patch.hpp:149-151: hunk bodies are newline-joined with no trailing newline; 210-218: locate_exact uses hay.find(needle); 549: locate_exact(working, search)
- src/tools/apply_patch.hpp:334: a line starting '@@ ' opens a hunk and the anchor text is discarded; 513-531: an empty search means 'insert at end of file'
- src/tools/registry.cpp:1788: the tool description says matching is 'byte-exact … never fuzzy'; src/loop/approval.cpp:687-688: apply_patch updates stay ungated as 'exact match or refuse'
- src/tools/apply_patch.hpp:489-497: each Update op re-reads the original file (no working copy carried across sections for the same path)
- src/tools/registry.cpp:1842: the snapshot keeps the first copy of each path; 1884-1886: expected_version comes from the ledger, else sha(snapshot); src/tools/memory_file.cpp:169: commit erases the ledger
- src/tools/registry.cpp:1908-1912: returns write_failure partway through the loop, after earlier files already committed
- src/loop/agent.cpp:2353: write events are recorded only when result.ok(); files_touched is built from write events
- src/loop/agent.cpp:2269-2276: pre_edit_clean_ is keyed on the `path` param; 1058-1059: a missing entry defaults to was_clean=true, producing '[syntax] …: FAILED'
- Repro (g++ on the real header, ap_test.cpp): file `    max = 1` with hunk -x = 1/+x = 2 gives Applied and 'max = 2'; file `account = 0` with -count = 0/+count = 5 gives Applied and 'account = 5'
- Repro (ap_test2.cpp): '@@ class A:' with '+    X = 1' appends the line at EOF inside class B; '@@ class B:' with context repeated in A and B gives Ambiguous
- Repro (dup_test.cpp): two '*** Update File: a.py' sections: commit 1 ok, commit 2 CONFLICT; retrying duplicates 'import sys', and 'return 3' never lands

## How the finder suggested verifying it

apply_patch.hpp uses only the standard library and compiles with g++ here. Turn ap_test.cpp, ap_test2.cpp and dup_test.cpp into tests that assert NoMatch, Ambiguous or the correct site, and that same-path sections apply cumulatively or are rejected at parse time. The registry and agent parts need macOS tests: the precondition check before the first commit, and landed files appearing in files_touched.

Back to the [index](README.md).
