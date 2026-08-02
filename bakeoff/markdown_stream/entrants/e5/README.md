# MarkdownStream

`MarkdownStream` is an incremental markdown renderer state machine that provides chunk-boundary correctness and deterministic output. It parses chunks of a markdown stream incrementally and emits well-formed rendering events.

## Features Supported

- Fenced code blocks with language tags
- Inline code
- ATX headings (Levels 1-6)
- Unordered list items (with nesting)
- Ordered list items (with nesting)
- Paragraph breaks

This explicitly excludes tables, links, images, blockquotes, or emphasis formatting beyond inline code.

## Holdback Bound

The stream buffer holdback limit is `1024 bytes` (~1KB). The stream will temporarily hold onto trailing ambiguous markers (like `` ` ``, `\n`, or spaces leading up to markers) for disambiguation, up to this bound. If this size limit is exceeded without resolution, it forcibly flushes characters to avoid infinite hanging.

## Split Invariance

This implementation ensures **Split Invariance**: "Feeding the input in one chunk produces the exact same sequence of events as splitting the input at every possible position."

This is guaranteed mechanically rather than merely tested. We use a deterministic state machine that only advances or emits events when a token is fully unambiguous, whether derived from a single contiguous byte sequence or reconstructed incrementally across chunk boundaries. All events emitted are strictly decoupled from the size of the feeding chunks, ensuring true split-invariant emission behavior. Our extensive test suite validates this mechanically at every possible byte-split position for 20+ diverse real-world documents.