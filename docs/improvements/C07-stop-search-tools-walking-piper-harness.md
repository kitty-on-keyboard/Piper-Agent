# C07: Stop search tools walking .piper/ harness state and aborting silently on one oversized line

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| UNVERIFIED | piper | bug | high | S |

> **Not yet verified.** Open every cited line below and confirm the problem still holds before you implement. Drop or reshape the task if it does not.

Merged from these finder reports:

- Stop workspace search tools walking .piper/ harness state, and stop search aborting silently on one long line

## Problem

Every orchestrated slice writes harness state inside the workspace, under <cwd>/.piper/slices/<id>/. That includes task.json, prompt.md, result.json and git.diff. It also includes two files written live during the run: events.jsonl and live.jsonl. events.jsonl has one JSON line per tool_result carrying the full observation, often over 8 KB for a numbered read_file. live.jsonl holds the model's streamed thinking. `.piper` is not in the search skip list and sorts before src/, so it is walked first. This causes three problems.
(a) `search` aborts the whole walk at the first hit whose full line would exceed max_result_bytes (8192). After the model reads a file over about 5 KB, searching for anything in it returns '(no matches)' or a few junk lines. This happens already in the first slice.
(b) locate_symbol ranks stale `+def foo(` / `-def foo(` lines from earlier slices' git.diff equal to the live definition and breaks the tie by walk order, so the stale lines come first. It also prints whole multi-KB event lines with no byte limit.
(c) search never reports that it was truncated, although find_files does.
It gets worse with every slice as .piper accumulates.

## Why it matters

Priority 2, on the user's only workflow. The model is told a symbol does not exist, or is pointed at stale definitions and at its own earlier thinking. It then re-creates code, edits the wrong thing, or burns turns re-searching. One locate_symbol call can push tens of KB of junk into a 96k-token window and force a compaction, which the code's own notes measure at 22x TTFT. It starts in the first slice of every orchestrated workspace and gets worse with each slice.

## Proposed fix

(1) Add '.piper' to skip_during_descent. It is harness state, the same class as .lmp_spool and .lmp_tmp. Passing it explicitly as `subdir` still searches it. load_skill reads .piper/skills by explicit path, so skills are unaffected. (2) In `search`, emit each hit as a bounded window around the match: at most about 200 chars, matching edit_diagnostics' kHunkMaxLineChars, with '… (+N chars)'. On reaching the 200-match or byte budget, stop and append '(truncated at N matches — narrow with subdir)'. A single oversized line must never end the walk. (3) In locate_symbol, clip candidate text the same way and bound total output to max_result_bytes. (4) Add registry tests with a workspace containing .piper/slices/x/events.jsonl (an over-8 KB line holding the needle) and a src file. search must return the src hit, and locate_symbol output must stay within max_result_bytes.

## Evidence (file:line)

- src/tools/ignore_dirs.hpp:20-35: the skip list's comment says it covers 'our own run artifacts', and it lists .lmp_spool and .lmp_tmp but not .piper
- src/tools/registry.cpp:1249-1253: `if (matches >= 200 || output.size() + hit_size > ctx_.max_result_bytes) return false;`. hit_size includes the full line, so one oversized line ends the whole walk
- src/tools/registry.cpp:1271: '(no matches)' with no truncation note (find_files at 1328-1339 does print '(truncated -- narrow the pattern or pass subdir)')
- src/surface/session.cpp:188: wctx.max_result_bytes = 8192
- src/tools/registry.cpp:1384-1385, 1404-1410, 1435: locate_symbol walks '.', appends the full line for each candidate (up to 400), and does not bound its output size
- src/surface/sidecar.cpp:1934-1941: events.jsonl is opened at <result_dir>/events.jsonl and written during the run; 1989: LiveJournal(result_dir/'live.jsonl') holds thinking and answer tokens
- src/loop/agent.cpp:2464-2470: the tool_result event carries the full result summary, i.e. the whole read_file text, as one JSON line
- scripts/piper_worker.py:637-642: the default packet dir is <cwd>/.piper/slices/<id>/; src/surface/worker.cpp:1031 writes git.diff there
- Repro (Python port of the walk and caps, scratchpad search_port.py): src/server.py (5.7 KB, defines handle_request) plus .piper/slices/slice-001/events.jsonl holding that file's read_file result as an 8,444-byte line; search('handle_request') returns '(no matches)'
- Repro (g++ on the real symbol_index.hpp, scratchpad sym_test.cpp): '.piper/slices/…/git.diff:+def handle_request(' and '-def handle_request(' score 3 and sort ahead of 'src/server.py:42:def handle_request('. The locate_symbol port emitted 8,539 bytes, 8,444 of them a single event line

## How the finder suggested verifying it

Both bugs are reproduced in the scratchpad: a Python port of the walk and caps (search_port.py) and g++ on the real symbol_index.hpp (sym_test.cpp). Rerun both after the change. The in-repo registry tests need the macOS build. For a field check, grep an orchestrated workspace's .piper/slices/*/events.jsonl for tool_result lines over 8 KB, and for search calls that returned '(no matches)' for identifiers that exist in src.

Back to the [index](README.md).
