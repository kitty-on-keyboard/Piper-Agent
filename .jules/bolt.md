## 2026-09-07 - Single-Pass Question Parsing in Webview UI
**Learning:** Parsing question text with multiple regex filters and `slice()` operations resulted in redundant string trimmings (4x per line) and extra array allocations for every streamed question.
**Action:** Consolidate array scans into a single index loop, caching trimmed lines and avoiding array slice/filter allocations.

## 2026-09-08 - [Cache config.json parses during checkpoint classification]
**Learning:** `classifyCheckpoint`, `isUsableMtp`, and `findSiblingMtp` repeatedly call `parseConfig`, causing synchronous file I/O and JSON parsing for `config.json` across parent directory scans.
**Action:** Use an in-memory `Map<string, ConfigJson | null>` cache for `parseConfig` results in `extension/src/checkpoint.ts` to eliminate duplicate disk reads and parsing.
