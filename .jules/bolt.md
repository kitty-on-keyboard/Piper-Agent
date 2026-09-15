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

## 2024-03-24 - One precise optimization
**Learning:** Avoid shotgun-style micro-optimizations across multiple files that reduce readability (like manually unrolling arrays or rewriting DOM queries to more verbose forms) unless they solve a measurable bottleneck.
**Action:** When asked for ONE small optimization, pick the single highest-impact and lowest-risk change, such as avoiding quadratic allocations during high-frequency parsing.
