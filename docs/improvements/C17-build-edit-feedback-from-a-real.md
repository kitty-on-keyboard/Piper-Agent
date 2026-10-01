# C17: Build edit feedback from a real line alignment: NoMatch diagnostics that name the divergent line, receipts that list every changed region

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| UNVERIFIED | piper | success-rate | medium | M |

> **Not yet verified.** Open every cited line below and confirm the problem still holds before you implement. Drop or reshape the task if it does not.

Merged from these finder reports:

- Fix replace_in_file NoMatch diagnostics: similarity scores over 100% send the model to the wrong site
- Build edit receipts from a real line diff so multi-hunk patches and overwrites report what actually changed

## Problem

The edit tools describe both failures and successes with heuristics instead of aligning lines.
(1) Failure: replace_in_file NoMatch is the most common edit failure, and its diagnostic is wrong in two ways.
- The overlap score mixes units: it counts want-token positions, duplicates included, but divides by the window's raw token count. Scores therefore go above 100%, and windows cut short at EOF score higher. In a file of similar functions, it ranked the tail of the wrong function first at 'overlap 125%' and offered the imports at 100%, while the real site was not listed at all.
- The separate 41-line ground-truth dump is keyed on the first textual occurrence of old_text's first line. For common lines such as `return None`, that is an unrelated region.
- Neither part says which line differs. Each miss costs about 2 KB.
(2) Success: applied_hunk reports a single prefix/suffix span.
- Two one-line hunks 100 lines apart come back as a 106-line rewrite. The first hunk's new content is elided, while the receipt says 'this is the whole change, so you do not need to read the file back'.
- write_file over an existing file returns only 'wrote N bytes'. A model that wrote '# ... rest unchanged' over a 500-line file is never told it removed 440 lines.
registry.cpp already includes pcc/diff.hpp, a real line diff, but never calls it.

## Why it matters

Priority 2. NoMatch is the most frequent edit failure. A diagnostic that names the divergent line turns it into a one-turn fix, instead of repeated misses or an edit redirected to the wrong function, and it shrinks each miss from about 2 KB to about 0.4 KB. Correct multi-hunk receipts stop needless re-reads and attempts to fix rewrites that never happened. An overwrite receipt that shows '-440/+60' lets the model catch a truncating rewrite itself, before the cloud has to.

## Proposed fix

Replace both heuristics with a line alignment. This changes feedback only; graft is untouched.
(1) NoMatch diagnostics:
- Tokenize each line the way graft does.
- Find file positions where old_text lines match token-exactly, extend them in order, and pick the window with the most matched lines. On a tie, report the competing sites.
- Name the first divergence, e.g. 'old_text lines 1-3 match <path> lines 60-62; old_text line 4 `…` differs from line 63 `…`'.
- Show only that window (old_text's height plus 2 context lines), and drop the separate first-occurrence 41-line dump.
- If any Jaccard ranking is kept, compute it with a set or min-count multiset intersection over full-height windows, so no score can exceed 100%.
(2) Receipts:
- Render from the existing pcc::unified_diff, used as-is. Exposing its hunk list would change the measured PCC engine and would need a re-score first.
- Show each changed region in the existing line-number-plus-text form with post-image numbering, bounded per region and in total by the kHunkMax* caps.
- Use this for apply_patch, per file.
- Also use it for write_file and commit_think_block when they replace existing content, so an overwrite reports e.g. '-440/+60 lines'.
- bytes_changed becomes the sum over regions.
replace_in_file receipts stay as they are, because graft changes exactly one span.

## Evidence (file:line)

- src/tools/edit_diagnostics.hpp:192: `inter` is a popcount over want positions, duplicates included. 244-246: uni = want_size + b_size - inter, so the score can exceed 1.0. 182: windows at EOF are shorter, which inflates their score
- src/tools/registry.cpp:1719: the ground-truth dump is keyed on f.bytes.find(first line), i.e. the first occurrence. 1728-1729: it spans start-5 to +40 lines. 1745: format_nearest is appended
- tests/tools/test_registry.cpp:1317-1333: only a 3-function fixture is asserted
- src/tools/edit_diagnostics.hpp:403-418: applied_hunk computes one prefix/suffix span. 482: the receipt says 'this is the whole change, so you do not need to read the file back'
- src/tools/registry.cpp:1786-1795: apply_patch says 'Prefer this for multi-hunk edits'. 1925: format_applied runs once per file. 1522-1527: write_file success says only 'wrote N bytes', even though the pre-image is read at 1487-1496
- src/tools/registry.cpp:22 includes src/pcc/diff.hpp; unified_diff is only called from src/pcc/cas.cpp:223
- Repro (g++ on the real graft and edit_diagnostics, scratchpad nomatch_test.cpp): with 12 near-identical handlers the candidates were 'line 96, overlap 125%', 'line 56, 103%' and 'line 1, 100%' (the imports). The true site, line 60, was missing, and the payload was 1,948 bytes. An old_text starting with 'return None' dumped lines 1-41
- Repro (hunk_test.cpp): hunks at lines 5 and 110 gave 'edited a.py at line 5 (-106/+106 lines)', with the new line hidden inside '… (172 more changed lines)'. diff_test.cpp: pcc::unified_diff on the same pair gave two correct one-line hunks in 224 bytes

## How the finder suggested verifying it

edit_diagnostics.hpp, graft_engine.hpp and src/pcc/diff.cpp compile with g++ here. Extend nomatch_test.cpp to assert that the reported site is handler_7 at line 60, that the divergent line is named, and that no score exceeds 1.0. Extend the hunk and diff tests to assert that a two-hunk receipt lists both regions with correct post-image line numbers within the byte cap, and that an overwrite receipt reports removed and added line counts. Keep the existing replace_nomatch tests passing on macOS.

Back to the [index](README.md).
