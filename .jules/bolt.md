## 2024-05-24 - Layout Thrashing in requestAnimationFrame Render Loops
**Learning:** In `webview.ts`, reading `feed.scrollHeight` immediately after mutating the DOM inside `drain(budget)` (and inside `add()`'s `queueOp`) forces a synchronous layout calculation in every animation frame. This causes layout thrashing and high CPU usage during text streaming.
**Action:** Always follow the Read-Modify-Write pattern in render loops. Read layout dimensions (like `scrollHeight`, `clientHeight`, `scrollTop`) *before* applying DOM mutations. For auto-scrolling to the bottom, setting `scrollTop = 1e9` avoids the need to read `scrollHeight` again after mutating the DOM.

## 2026-03-30 - O(1) Fast Path for Sibling Checkpoint Discovery
**Learning:** `findSiblingMtp` previously scanned all sibling directories in a model folder and read/parsed `config.json` for every directory before evaluating guessed sibling names. In directories with many model checkpoints, this caused unnecessary filesystem I/O and JSON parsing for unrelated models.
**Action:** Always check targeted guesses/known naming conventions first via direct path existence checks before falling back to full directory scanning and filtering.

## 2026-03-31 - Redundant IIFE Closure Allocations in Webview Render Loop
**Learning:** In `webview.ts`, `renderMd` used a legacy IIFE pattern `((ev) => () => applyMd(ctx, ev))(e)` inside `for (const e of events)`. In ES6, `const` loop variables create per-iteration bindings natively, so the IIFE allocated two closure function instances instead of one on every non-text markdown event during live streaming.
**Action:** Avoid IIFE wrappers in `for (const x of ...)` loops; pass `() => fn(x)` directly to queue callbacks.
