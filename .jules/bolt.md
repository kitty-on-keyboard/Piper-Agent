## 2024-05-24 - Layout Thrashing in requestAnimationFrame Render Loops
**Learning:** In `webview.ts`, reading `feed.scrollHeight` immediately after mutating the DOM inside `drain(budget)` (and inside `add()`'s `queueOp`) forces a synchronous layout calculation in every animation frame. This causes layout thrashing and high CPU usage during text streaming.
**Action:** Always follow the Read-Modify-Write pattern in render loops. Read layout dimensions (like `scrollHeight`, `clientHeight`, `scrollTop`) *before* applying DOM mutations. For auto-scrolling to the bottom, setting `scrollTop = 1e9` avoids the need to read `scrollHeight` again after mutating the DOM.

## 2026-03-30 - O(1) Fast Path for Sibling Checkpoint Discovery
**Learning:** `findSiblingMtp` previously scanned all sibling directories in a model folder and read/parsed `config.json` for every directory before evaluating guessed sibling names. In directories with many model checkpoints, this caused unnecessary filesystem I/O and JSON parsing for unrelated models.
**Action:** Always check targeted guesses/known naming conventions first via direct path existence checks before falling back to full directory scanning and filtering.

## 2026-04-01 - Fast-Path State Tracking for Number Literal Parsing
**Learning:** `StreamParser` performed repeated linear scans (`buf_.find('.')` and `buf_.find_first_of("eE")`) over the accumulated buffer on every byte in `push_byte()`, `allowed_bytes()`, and `state_signature()`. For streaming/token probing on float/exponent number literals, this turned byte accumulation and constraint checking into an $O(N^2)$ operation per token.
**Action:** Maintain boolean state flags (`num_has_dot_`, `num_has_exp_`) during incremental token accumulation to allow $O(1)$ grammar and constraint validation.

## 2026-03-31 - Redundant IIFE Closure Allocations in Webview Render Loop
**Learning:** In `webview.ts`, `renderMd` used a legacy IIFE pattern `((ev) => () => applyMd(ctx, ev))(e)` inside `for (const e of events)`. In ES6, `const` loop variables create per-iteration bindings natively, so the IIFE allocated two closure function instances instead of one on every non-text markdown event during live streaming.
**Action:** Avoid IIFE wrappers in `for (const x of ...)` loops; pass `() => fn(x)` directly to queue callbacks.

## 2026-04-05 - O(1) Indexing Fast Path for Flat JSON Arrays in parsephony
**Learning:** In `parsephony`, `Value::operator[](size_t i)` walked the DOM tree using `next_sibling` for each element index $i$, leading to $O(N^2)$ time complexity when looping over array elements by index. Because flat arrays (where all elements are scalars occupying 1 node on the tape) satisfy `n.off - (idx_ + 1) == n.len`, element $i$ can be retrieved in $O(1)$ time at tape index `idx_ + 1 + i`.
**Action:** Check `n.off - start == n.len` before walking siblings in tape-based JSON array indexing to achieve $O(1)$ random access for flat arrays.

## 2026-04-10 - Single-Pass Corpus Token Frequency Map for Static Analysis
**Learning:** In `scripts/run_ratchets.py`, `gate_dead_code` previously compiled a regex for each declared symbol and searched the entire file corpus $M$ times ($O(M \times N_{files} \times L_{file})$), taking ~57 seconds. By scanning all corpus files in a single pass to build a `collections.Counter` of identifier tokens (`\b[A-Za-z_][A-Za-z0-9_]*\b`), symbol frequencies can be queried in $O(1)$ time per symbol (~0.3 seconds total, ~180x speedup).
**Action:** Build corpus-wide token frequency maps in a single pass when checking multiple identifier references instead of running repeated full-corpus regex scans per symbol.

## 2026-04-15 - Slice-Based Fast Path for Raw Text Parameters in parsephony ToolCallGuard
**Learning:** `ToolCallGuard::feed` previously processed input byte-by-byte via `push_byte(c)` and `value_append(c)` during raw text parameter parsing (`Ph::ValueText`). For multi-line text parameter values, character-at-a-time string appending caused excessive function call overhead and frequent `std::string` reallocations.
**Action:** In `ToolCallGuard::feed(std::string_view bytes)`, scan for contiguous non-newline, non-control byte ranges in `Ph::ValueText` and bulk-append `std::string_view` slices via `value_append(slice)`.

## 2026-04-18 - Short-Circuiting Workspace Relative Path Resolution for Absolute Path Diagnostics Queries
**Learning:** `code_intel.ts`'s `diagnostics(path)` previously called `relPath(uri, cache)` (which queries VS Code workspace API `getWorkspaceFolder` and `asRelativePath`) for every URI in `vscode.languages.getDiagnostics()` before checking path filters. When querying diagnostics for an absolute target path, evaluating relative paths for non-matching URIs is redundant because relative paths never match absolute target paths.
**Action:** In `diagnostics(path)`, check `isAbsPath(path) && uri.fsPath !== path` to short-circuit and skip `relPath()` calls for non-matching URIs across the workspace.

## 2026-05-18 - Batching Partial Terminator Buffer Flushes and Reserving Capacity in ToolCallGuard
**Learning:** Appending characters individually to parameter accumulators (`value_append(c)`) and making separate `value_append` calls for partial parameter terminators (`kTerm.substr(0, term_pos_)`) caused frequent `std::string` reallocations and Copy-On-Write overhead in `ToolCallGuard`. Combining the partial match buffer with incoming text bytes into single stack-buffered `string_view` appends, alongside pre-reserving `value_` capacity, reduced string manipulation overhead and improved parameter streaming throughput by ~15%.
**Action:** Pre-allocate string capacity when initializing shared accumulators and batch partial delimiter flushes with content slices into single `value_append(std::string_view)` calls.

## 2026-04-20 - Document-Level Array Index Offset Caching in parsephony
**Learning:** `Value::operator[](size_t i)` walked non-flat array DOM subtrees using `next_sibling` for each element index $i$. When indexing temporary `Value` instances (such as `doc.root()[i]`) or performing random/non-sequential access, per-`Value` iteration state was lost, causing $O(N^2)$ time complexity.
**Action:** Store a fixed-capacity LRU array offset cache on `Document` to populate and cache element node indices for non-flat arrays on first lookup, ensuring $O(1)$ random access across all `Value` instances and access patterns.

## 2026-05-20 - Fast-Path String Searching in Fenced Block Parsing
**Learning:** `extract_fenced_blocks` stepped character-by-character (`++i` and `++j`) through response texts and block bodies to check `line_start` and fence conditions on every byte. For large code blocks, this turned line scanning into an expensive byte-by-byte loop.
**Action:** Use `text.find("```", i)` to jump directly to candidate opening backticks and `text.find('\n', j)` to skip block body lines directly to the next line start.

## 2026-05-25 - Avoid spreading querySelectorAll in Webview Hot Paths
**Learning:** Constructing arrays from `querySelectorAll` results using the spread operator (`[...feed.querySelectorAll('.msg')]`) requires allocating a full list in memory and evaluating an $O(N)$ query over the entire subtree. In frequently-called loops such as message stream management or finding the last `.msg.assistant`, this creates unnecessary DOM lookups and array allocations.
**Action:** Replace `querySelectorAll` array spreading with direct DOM traversals (e.g., using `while (feed.firstChild && feed.firstChild !== live) { feed.firstChild.remove(); }` or walking backwards via `live.previousElementSibling`) to achieve $O(1)$ and $O(K)$ performance without allocations.

## 2026-05-26 - Zero-Allocation Line Numbering for Tool File Observations
**Learning:** `number_lines()` in `src/tools/text_view.hpp` previously called `std::to_string(line)` on every line during file reads (`read_file`, `read_many`, `read_slice`). For multi-thousand line files, this triggered thousands of heap allocations and string constructions per read. Using `std::to_chars` with a stack buffer (`char num_buf[32]`) formats integer line numbers with zero allocations.
**Action:** Use `std::to_chars` into local stack buffers for loop-level string formatting of integer line numbers instead of `std::to_string`.
