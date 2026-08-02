# MarkdownStream cook-off (Jules round 2, Brief E)

An incremental markdown/code-fence state machine: bytes in, render events out, no HTML and no
DOM. Brief text in [docs/JULES_BRIEFS_ROUND2.md](../../docs/JULES_BRIEFS_ROUND2.md).

Repo: `cat-collector-king/MarkdownStream`, one PR per entrant.

## Standing, 2026-08-01 — COMPLETE, 5 of 5, plus an amalgamation that is the only clean row

```
            split  strict  skel  code  lose    hold   fin  utf8  reset  pend    us/KB
e1  (PR 1)     0      29     6     0     0      64     0     0      0     2      44.4
e2  (PR 2)    14      29     4     1     3    8192     1     3      0     0      39.2
e3  (PR 3)     1      30     1     0     0    8192     0     0      0     1      40.6
e4  (PR 4)     0      29     1     0     0      63     0     0      0     0      48.3
e5  (PR 5)     0      29     2     0     0    1024     0     0      0     0      44.4
amalgam        0      29     0     0     0      63     0     0      0     0      49.1
```

The handoff predicted the discriminating column would not be one the brief names. It was not.
Split invariance — the property the brief says to build around — separates only e2 and e3. The
column that sorts the field is **`hold`**, and it sorts it into "streams" and "hangs".

### `hold`: two entrants buffer the entire stream, and both say in writing that they do not

Fed 8 KB of nothing but backticks, one byte at a time, **e2 and e3 emit no event at all until
the stream ends.** In the webview that is a model producing 8 KB of output and a user watching
a blank bubble. The brief called this out in as many words — "Holding the whole stream waiting
for a fence that never comes is a hang, not caution" — and asked for a stated constant plus a
test that it is never exceeded. Both wrote the constant down. Neither test caught this.

| entrant | README claims | measured, all-backticks |
|---|---|---|
| e2 | "never holds back more than **10 bytes** … or **64 bytes** when waiting for an info string" | **8192** |
| e3 | "maximum **64 bytes** … It will **never hold back an entire stream** waiting for an unbounded marker to complete" | **8192** |
| e5 | (no constant stated) | 1024 |
| e1 | (no constant stated) | 64 |
| e4 | "Bounded Holdback" (no constant stated) | 63 |

e3's sentence is the exact behaviour it fails to have. Both entrants tested the bound on
well-formed input, where nothing is ever withheld for long, so the tests passed. The failing
input is the one the brief names and neither entrant tried.

The other adversarial inputs discriminate much less: on plain prose every entrant is 0, on
`**` repeated every entrant is 1-2, and on an open fence followed by 8 KB that never closes it
every entrant is bounded (3, except e4/amalgam's 63). It is specifically the ambiguous-marker
run that separates them.

### The rest of the field

- **e1** — the only entrant that drops *content structure*. Three ordered list items become
  one; two unordered items become one; a dedent back to level 0 stops being a list item at
  all; and in `` `code` ``/fence/`` `code` `` its inline-code span stays open **across** the
  fenced block (`IO; CO[]; CC; IC;`). Text bytes survive, so nothing is lost — but a list
  renders as one item and the rest as prose. `pend=2` besides.
- **e2** — last on every axis that matters: `split=14`, `utf8=3`, and the only entrant that
  **loses bytes**. `` `` `` at end of stream yields `InlineCodeOpen InlineCodeClose` and no
  text; ``` ``` ``` yields `CodeBlockOpen CodeBlockClose` and no text. The brief says
  `finish()` "flushes held-back bytes as text"; e2 invents structure and drops them.
- **e3** — the *only* entrant that gets a three-level nested list right, and the only one to
  fail split invariance on a document of nothing but blank lines. Undone by `hold`.
- **e4** — cleanest row in the field. One defect, one line. See below.
- **e5** — e4's list defect plus a 1024-byte holdback.

`fin` is **zero for four of the five**, and e2's 1 is its byte-drop above rather than a
swallowed fence. The handoff flagged unterminated-fence swallowing as a likely differentiator
and it is not one: all five close an open fence at `finish()`, and all five still emit the
text preceding it. That question is settled, negatively.

### The amalgamation is two characters

`amalgam/` is **e4 with `spaces < 3` changed to `spaces < 32`** at `markdown_stream.cpp:46`
(and the identical line at :283).

e4 caps its leading-space scan at three, which is CommonMark's rule for how far a *top-level*
block may be indented. Applied to a nested list item it is the wrong rule: `    - deeper` at
four spaces falls past every marker test and lands in the parent item's text. e4 already keeps
a `m_list_indents` stack and already calls `close_lists(indent)`, so the machinery for
arbitrary depth was there — only the scan was too short to reach it. e5 has the same defect.

The change is narrow. Fenced content at four spaces is still inviolate, and deeply indented
prose is still prose:

```
                            e4                          amalgam
- a/  - b/    - c/- d   LOpen<0> LOpen<1>"b@    - c@"    LOpen<0> LOpen<1> LOpen<2> LOpen<0>
"text\n        prose"   Text (unchanged)                 Text (unchanged)
"```\n    - x\n```"     CBText (unchanged)               CBText (unchanged)
```

**Adopted**, ported to TypeScript — see the adoption note below.

## The harness

`scoreboard.cpp` is neutral: written before any entrant was read, graded against a brute-force
non-incremental reference, using only the public API the brief pinned. `score.sh` compiles it
once per entrant. Columns are documented at the top of `scoreboard.cpp`.

Two columns exist because the brief does not pin the semantics and grading either dialect as
the only correct one would have been wrong:

- **inline-code dialect.** Entrants are graded against *both* a lookahead reference (a backtick
  pairs with the next one on the line) and an eager one (each backtick run toggles), and are
  credited for consistently implementing either. Lookahead is what a whole-document parser
  does and is what the first version of this reference did alone — but deciding whether a
  backtick has a partner requires scanning to end of line, which is the unbounded lookahead
  brief item 3 forbids. Grading only against lookahead marks an entrant down for obeying the
  brief.
- **`skel` list nesting.** Some entrants emit flat `Open/Close` per item, others nest
  `Open,Open,Close,Close` for a sublist. The brief specifies neither and both render. `skel`
  compares the sequence of open *depths* plus a balance flag, so losing an item still fires
  and choosing a nesting model does not.

`strict` is reported but never a failure: it counts documents where the raw, unmerged event
stream differs across split points. Every entrant scores ~29 and the clean base 26, so nobody
holds the stricter property and it separates nobody.

### Four harness bugs, all found by the falsifiers or by a suspicious sweep

Per the standing rule, the harness was suspected before the subjects, and four times that was
correct:

1. **`hold` measured the wrong thing.** The first version tracked bytes fed minus payload bytes
   emitted, and reported **8192 for a correct implementation** — on an all-backticks stream
   every input byte legitimately *is* a marker, consumed and never echoed, so payload lag grows
   without bound while events stream out fine. Counting *events* instead is immune to how many
   input bytes are markers.
2. **The reference paired inline backticks by lookahead only** (above).
3. **The reference hard-coded three backticks**, so a ```` ```` ````-fence's info tag read as
   `` ` `` and the fence never closed. Fences open on a run and close on a run at least as long.
4. **The reference invented structure at EOF.** A bare ``` ``` ``` at end of stream opened a
   code block; four of five entrants correctly flush it as text, because without the newline
   the info tag was never known to be complete. The reference was wrong and the field was
   right — this one was caught only because four rows failed identically.

## Falsifiers — the harness is shown red before it is believed

`falsifiers/` holds a clean base plus three deliberately broken derivatives, all built by
`score.sh` on every run. Each falsifier is the base plus exactly ONE planted defect, so the
delta against the base row is unambiguous.

| falsifier | planted defect | fires |
|---|---|---|
| `clean_base` | none | **nothing** — the columns are satisfiable |
| `broken_holdback` | holdback cap removed | `hold=8192` — **and nothing else** |
| `broken_swallow` | `finish()` does not close an open fence | `fin=2`, `skel=3` |
| `broken_split` | `#` resolved at the chunk boundary instead of waiting one byte | `split=5`, `utf8=1` |

`broken_holdback` is the important row: one column, seven others clean, which is what says the
board discriminates rather than alarms. `broken_swallow` and `broken_split` each fire a second
column, but consistently — an unclosed fence necessarily changes the skeleton, and the UTF-8
corpus contains a heading, so a heading defect shows there too.

`clean_base` is not an entry and was not scored as one. It exists so that "every column can be
zero" is a demonstrated fact rather than an assumption.

## Adoption note — this cannot drop in as-is, and it is not a coverage regression

Two things the entrants could not know, both checked before recommending anything.

**The webview renders no markdown today.** `lmp/token` carries `{channel, text}`, and
[webview.ts:439](../../extension/src/webview.ts:439) inserts it as a raw text node into a
`white-space: pre-wrap` div. Brief E deliberately scopes out tables, links, images,
blockquotes and emphasis, and the handoff flagged adopting it as a possible coverage
regression. It is not: none of those render today either. Markdown currently reaches the user
as literal characters. Anything here is strictly more than what ships.

**But the render target is TypeScript and every entrant is C++.** The two options were a new
render-event notification plus a webview rewrite, or a port into the webview. The port won:
the events are a view concern the sidecar has no use for, and a render-event notification
would freeze one rendering dialect into the wire protocol and have to be versioned forever.

### What was adopted, and how it stays honest

`amalgam/` is ported into `extension/src/webview.ts` as `markdownStreamSource()` — the view
script is injected as text, so the parser ships as a string rather than a module. Two
deliberate deviations from the C++, both because JS strings are UTF-16 where the original
counted bytes:

- the holdback flush never splits a surrogate pair;
- an info tag that runs past the holdback bound opens the block instead of draining one byte
  per feed, which in a webview is a visibly frozen bubble.

One addition beyond the brief: `listOpen` carries `ordered` and `start`, so `1. 2. 3.`
renders as `<ol>` rather than bullets. Brief E's `Event` has only `level`, so every entrant
loses the distinction — fine for an event-stream exercise, wrong for a chat UI.

**`scripts/verify-markdown-stream.js` evals the shipped source string** and diffs its event
trace against the C++ amalgam over `corpus.txt` (37 documents, whole-feed and byte-at-a-time,
`dump.cpp` on the other side). It verifies what ships, not a copy of it. Currently 340 trace
lines, identical:

```bash
cd extension && npm run compile && node scripts/verify-markdown-stream.js ../bakeoff/markdown_stream/corpus.txt
```

`scripts/preview-webview.js` renders the real view script in a browser with the sidecar
faked, which is how the DOM side was checked — the trace test says nothing about what the
page looks like.

Nothing here is wired into the gate. `scoreboard.cpp` is standalone and built only by
`score.sh`.
