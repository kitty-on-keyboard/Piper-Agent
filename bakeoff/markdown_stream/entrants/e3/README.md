# MarkdownStream

An incremental markdown renderer state machine. It takes chunks of text and emits parsed events, ensuring that code blocks and other formatting elements do not incorrectly parse until disambiguated.

## Holdback Bound
The maximum number of bytes withheld across chunk boundaries is small and strictly bounded (e.g., maximum 64 bytes for language tags or short markers like backticks and lists). It will never hold back an entire stream waiting for an unbounded marker to complete.

## Supported Subset
- Text (plain prose)
- Fenced Code Blocks (```) with language tag
- Inline Code (`code`)
- ATX Headings (1-6 levels)
- Ordered (1.) and Unordered (-, *) Lists with nesting inferred by 2 spaces per depth
- Paragraph Breaks (\n\n)

## Guarantee of Split Invariance
Split invariance is guaranteed by maintaining all incoming text in a `holdback_` buffer. The internal parsing loop only consumes from `holdback_` when a marker has definitively matched or definitively failed to match. Ambiguous states (e.g. trailing backticks, trailing spaces) simply break the parsing loop and leave the characters in the buffer. The events emitted do not depend on the chunk sizes because the parser operates on the continuously appended `holdback_` buffer identical to a single concatenated string.
