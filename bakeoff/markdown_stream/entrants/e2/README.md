# MarkdownStream

An incremental markdown renderer state machine that guarantees deterministic event output regardless of how the input text stream is chunked.

## Features Supported
- Fenced code blocks with language tags (`` ``` `` or `~~~`)
- Inline code (`\``)
- ATX Headings (`#`)
- Unordered lists (`-`, `*`, `+`)
- Ordered lists (digits followed by `.` or `)`)
- Paragraphs
- Raw text
- No swallowed elements (open blocks are correctly terminated at the end of the stream)

## Performance Bounds
- **Maximum Holdback:** The implementation maintains a strict bound on withheld bytes. The state machine never holds back more than **10 bytes** when waiting for a possible closing fence marker, or **64 bytes** when waiting for an info string.

## Split Invariance
Split invariance is the core principle of this implementation: for any valid markdown text, `stream.feed(doc)` will yield identically matching stream events to calling `stream.feed()` over splits of that text, at *every possible chunk boundary*.

Split invariance is guaranteed by:
1. Buffering exactly up to the required lookahead logic points (potential fences).
2. Deferring ambiguous inline tokens (`\``) when encountered at the chunk boundary, resolving them upon the next token or `finish()`.
3. Applying deterministic newline matching logic.
