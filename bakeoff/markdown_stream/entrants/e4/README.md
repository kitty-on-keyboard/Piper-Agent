# MarkdownStream

`MarkdownStream` is an incremental Markdown renderer state machine designed specifically for streaming chat UIs, where text arrives a few characters at a time. It ensures rendering is visually stable, avoiding flickering code blocks or swallowing user input.

## Supported Subset
- Fenced code blocks with language tags (using \`\`\`)
- Inline code (using arbitrary number of \` matching pairs)
- ATX headings (1 to 6 `#` characters followed by a space)
- Unordered list items (`-` or `*`) and ordered list items (`1.` etc.) with nesting
- Paragraph breaks (via empty lines)

## Design Guarantees

### Holdback Bound
To prevent UI hangs while waiting for ambiguous characters to resolve (such as ```` ` ```` or list markers), `MarkdownStream` utilizes a **Bounded Holdback** technique.
**The maximum holdback bound is strictly 64 bytes.**
If the internal buffer reaches 64 bytes without definitively matching a markdown element, it will eagerly force-flush characters as standard `Text`, preventing unbounded delays.

### Split Invariance
A strict requirement of `MarkdownStream` is that feeding input in a single chunk (`feed(chunk)`) produces the exact same event stream as feeding that input split at any arbitrary byte boundary.
This is achieved by implementing a byte-by-byte state machine under the hood. The `feed` method appends chunks to a small internal holdback buffer, and the core processing loop strictly transitions states based on unambiguous marker criteria, buffering only what is absolutely necessary up to the 64-byte limit. Thus, chunk boundaries never influence syntactic resolution. Exhaustive testing splits various inputs at every single possible byte offset to formally verify this split-invariance property.

### No Mocks, No Externals
This library relies entirely on the C++20 Standard Library (`<string>`, `<vector>`, `<cstdint>`, etc.) and links nothing else. It natively compiles and guarantees determinism across Linux x86-64 and macOS/arm64 targets. Tests are powered by GoogleTest.
