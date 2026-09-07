## 2026-09-07 - Single-Pass Question Parsing in Webview UI
**Learning:** Parsing question text with multiple regex filters and `slice()` operations resulted in redundant string trimmings (4x per line) and extra array allocations for every streamed question.
**Action:** Consolidate array scans into a single index loop, caching trimmed lines and avoiding array slice/filter allocations.
