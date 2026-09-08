## 2026-09-08 - [Cache config.json parses during checkpoint classification]
**Learning:** `classifyCheckpoint`, `isUsableMtp`, and `findSiblingMtp` repeatedly call `parseConfig`, causing synchronous file I/O and JSON parsing for `config.json` across parent directory scans.
**Action:** Use an in-memory `Map<string, ConfigJson | null>` cache for `parseConfig` results in `extension/src/checkpoint.ts` to eliminate duplicate disk reads and parsing.
