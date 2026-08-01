# Handoff: judging the MarkdownStream cook-off (Brief E)

Written 2026-08-01, end of the session that landed speculative decoding. This is the next
session's starting point.

---

## READ THIS FIRST: there may be nothing to judge yet

**As of 2026-08-01 22:20, `cat-collector-king/MarkdownStream` has ZERO PRs.** The repo
exists, `main` is near-empty, and no entrant has landed. Briefs C and D from the same round
both filled to 5/5 over about 75 minutes; E was launched alongside them and has produced
nothing so far.

So the first command of the next session is:

```bash
gh pr list -R cat-collector-king/MarkdownStream --state all --limit 10
```

- **5 entrants:** proceed to "How to judge it" below.
- **1-4 entrants:** the harness is still worth building — it is the long pole, and it can be
  written and falsified before the field is complete. Do not adopt from a partial field
  without saying so; see the Brief D note below for what that looked like.
- **0 entrants:** the cook-off did not run. Say so plainly rather than inventing work, and
  ask whether to relaunch it. Brief E's text is in
  [JULES_BRIEFS_ROUND2.md](JULES_BRIEFS_ROUND2.md), ready to paste.

`gh` needs the second account for these repos, and the ambient credential helper serves the
wrong token — `gh api` works while `git fetch` fails with "Repository not found". Clone with
`gh repo clone`, then fetch branches with the helper list RESET first:

```bash
git -c credential.helper= -c credential.helper='!gh auth git-credential' fetch origin 'refs/heads/*:refs/remotes/origin/*'
```

---

## What MarkdownStream is, and where it lands

An incremental markdown/code-fence state machine: bytes in, render events out, no HTML and
no DOM. The problem it solves is visible and specific — a fenced code block opens with
```` ``` ```` and the closing fence has not arrived, so the UI either shows raw backticks,
opens a code block that swallows the rest of the conversation, or flickers between the two
on every token.

It lands in **the webview**, and it is independent of everything speculative decoding
touched. It became worth building the moment token streaming landed (`7cb3139`): before
that, `Observer::on_token` fired once per TURN with the whole decoded block, so there was no
partial markdown to get wrong. There is now — 51 notifications per turn on the real sidecar.

The consuming code is the extension's webview. Check what it does with partial text today
before adopting anything; the streaming commit already had to fix it creating a new bubble
and a new reasoning disclosure per notification.

---

## How to judge it

**Do not read five implementations and form an opinion.** Build a neutral scoreboard,
compile every entrant against it, and score. That is what separated the winner from the pack
in round 1, and in round 2 it caught defects in all ten entrants that their own tests
missed. Two working examples, both landed this session:

- `bakeoff/prefix_ledger/` — reference-implementation grading, falsifiers, `score.sh`
- `bakeoff/spec_verifier/` — statistical grading, two regimes reported separately

Copy `score.sh` from either; it takes a directory of entrant trees and prints one row each.

### The rules that earned their place

1. **Write the scoreboard BEFORE reading any entrant.** Both round-2 boards were, and both
   found things a code read would not have.

2. **Falsify the harness before believing it.** Every entrant passing is not evidence until
   the board has been shown red. Plant deliberate defects in `falsifiers/` and build them on
   every run. The best falsifier fires exactly ONE column and leaves the rest clean — that
   is what says the board discriminates rather than alarms.

3. **Suspect the harness before the subjects.** This has now been the right call four times
   across three sessions. In round 2 it went the other way once and was still worth
   checking: five SpecVerifier entrants produced byte-identical output, which looked like a
   broken harness and turned out to be five independent implementations picking splitmix64
   with the same canonical constants, because the brief pinned the RNG state to one uint64.

4. **The discriminating column is usually not the one the brief names.** Brief D asked for
   fingerprint sensitivity and every entrant passed; the column that mattered was
   *constructed* collisions, which the first version of the board did not test. Brief C
   named the bad-drafter histogram as the decisive test; the falsifier proved the
   good-drafter one is.

5. **An amalgamation is a legitimate outcome, and often the right one.** Round 1 shipped two
   (`SuffixProposer`, `moetrace`). Round 2's Brief D shipped a third, which beat all five
   entrants by removing the constraint that caused their shared defect. Do not assume the
   job is picking a winner.

### For Brief E specifically

The brief names **split invariance** as the property to build around: feeding an input in
one chunk and split at every possible byte offset must produce the identical event stream.
That is the right headline, and it is mechanically checkable — take ~20 documents, compare
the one-chunk stream against every single-byte split point, assert equality. An entrant
either has it or does not.

Two columns the brief does NOT ask for, and which are where the interesting differences will
be:

- **Holdback bound under adversarial input.** The brief asks for a stated constant and a
  test that it is never exceeded. Feed it a stream of nothing but backticks, then a
  half-open fence a megabyte long. An implementation that buffers to find a fence that never
  arrives is a hang, not caution, and it will pass every well-formed test.
- **What `finish()` does to an unterminated fence.** The brief says it must close. Check
  that the preceding text is still emitted, in order, and not dropped on the floor — that is
  the failure that loses a user's entire answer.

And one thing to check before adopting, which no entrant can know: **the webview's existing
rendering path**. Brief E deliberately scopes out tables, links, images, blockquotes and
emphasis. If the webview currently renders those, adopting this state machine is a
regression in coverage even if every column is green. Decide that deliberately.

---

## State of everything else, 2026-08-01

Landed and green on `perf/streaming-and-parallel-dispatch` (22/22 gate, 6/6 ratchets):

- **Speculative decoding**, complete and measured. OFF by default (`LMP_SPECULATIVE=1`).
  See [HANDOFF_SPECULATIVE.md](HANDOFF_SPECULATIVE.md) — sections 6 and 7 are the current
  state and what is left.
- **Brief C (SpecVerifier)** — 5/5 scored, entrant e3 adopted into `src/model/`.
- **Brief D (PrefixLedger)** — 5/5 scored, amalgamation adopted into `KvCacheLedger`.
- **Brief E (MarkdownStream)** — the subject of this document. Nothing landed.

The single highest-value follow-up on the speculative side, if Brief E turns out to be
empty: **sweep `SuffixProposer`'s `min_support` / `min_match_len` / `draft_cost_ratio` on
real agent traces.** Speculation is trigger-limited, not acceptance-limited — 28 blocks
against 1269 fallbacks, but 84% of drafted tokens accepted. Those thresholds were tuned on
synthetic data and are the whole lever. It needs no new machinery.

## Gotchas that still apply

- **Never two MLX processes.** One model is 19 GB on a 48 GB host. `ctest --preset realmodel`
  pins jobs=1; `scripts/drive.py` must run alone. This is not theoretical — doing it took
  the machine down.
- **Adding a test edits `gate_manifest.txt` twice**, count AND name, by design.
- **A traced run's throughput is meaningless.** `LMP_MOE_TRACE` forces a mid-graph sync: 52
  tok/s traced against 85 untraced.
- **`timeout` does not exist on this macOS.** Use the tool's own deadline flags.
