# Jules round 2 — three tasks, five entrants each

Run each brief as its own repo with 5 entrants. Five is what made round 1 work: the winning
draft proposer was only identifiable because four others did the same thing worse, and the
one wrong `moetrace` was only visible because five agreed against it. Fifteen separate tasks
would give fifteen unverifiable single points.

All three are pure C++20, no GPU, no platform syscalls, fully testable on Linux — the
constraint that separated round 1 (all adoptable) from the two rounds before it (all
stubbed or Linux-only).

---

## SHARED PREAMBLE — paste at the top of each brief

> **Environment reality.** You are building on Linux x86-64. The consuming project runs on
> macOS / Apple Silicon (arm64). Therefore:
>
> - **No GPU code of any kind.** No CUDA, no Metal, no MLX, no ROCm.
> - **No Linux-only syscalls.** No `memfd_create`, `eventfd`, `futex`, `io_uring`. Must
>   compile and pass tests on both Linux and macOS/arm64. Use `<atomic>`, `<mutex>`,
>   `<thread>`.
> - **No external dependencies.** C++20 standard library only. CMake may fetch GoogleTest
>   and nothing else.
> - **No stubs, no placeholders, no mocks.** Do not return a fixed value with a comment
>   saying a real implementation would go here. If something cannot be implemented, say so
>   in the PR description and leave it unimplemented rather than implementing a lie.
> - **Tests must be capable of failing.** For at least one test, show in the PR description
>   that you ran it against a deliberately broken implementation and watched it go red.
> - **Determinism.** Same inputs, same outputs. No `std::random_device`, no unseeded RNG, no
>   dependence on unordered-container iteration order in any output.
> - **Report honest numbers.** If you miss a stated performance budget, print the number you
>   actually hit. Do not quietly relax the target.

---

# BRIEF C — `SpecVerifier`: the acceptance rule for speculative decoding

## What this is for

A language model normally produces one token at a time, each costing a full pass over the
model. **Speculative decoding** makes something cheap guess the next k tokens, then has the
real model check all k in a single pass. Tokens the model agrees with are free.

The subtle part — and the whole of this task — is *deciding which guesses to keep*. Do it
naively (keep a guess if it is the model's top choice) and you silently change what the
model produces: the output is no longer a sample from the model's real distribution, it is
biased toward confident tokens. The correct procedure keeps the output distribution
**exactly identical** to ordinary one-at-a-time sampling. It is a form of rejection
sampling, and it is easy to get subtly wrong in a way no casual test catches.

You are implementing that procedure. There is no model here — you are given probability
distributions as arrays of floats.

## The exact interface

`include/spec_verifier.hpp`, namespace `spec`. Adopted as written; do not rename.

```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace spec {

using TokenId = std::int32_t;

struct Result {
    // Tokens to commit, in order. Always contains at least one token: even when every
    // draft token is rejected, the verification pass yields one fresh token, which is
    // where speculative decoding's guaranteed floor comes from.
    std::vector<TokenId> accepted;
    // How many DRAFT tokens survived. accepted.size() == accepted_drafts + 1 normally.
    std::size_t accepted_drafts = 0;
    // True when the final token came from the corrected distribution after a rejection,
    // rather than from the model's own next-position row.
    bool ended_on_rejection = false;
};

class SpecVerifier {
  public:
    // `seed` fixes the RNG. Two verifiers with the same seed and inputs agree exactly.
    explicit SpecVerifier(std::uint64_t seed);

    // `draft` holds k proposed tokens.
    // `draft_probs[i]` is the DRAFTER's probability for draft[i] (one scalar per position).
    // `target_rows[i]` is the TARGET model's full probability row for position i --
    //   target_rows.size() == draft.size() + 1, because verification also produces the
    //   distribution for the position after the last draft token.
    // Every row is a proper distribution: non-negative, sums to ~1.
    [[nodiscard]] Result verify(std::span<const TokenId> draft,
                                std::span<const float> draft_probs,
                                std::span<const std::span<const float>> target_rows);

  private:
    std::uint64_t rng_;
};

} // namespace spec
```

## The procedure (implement exactly this)

Walk the draft left to right. For draft token `t` at position `i`, with drafter probability
`q` and target probability `p = target_rows[i][t]`:

- Draw `u` uniform in [0,1). **Accept** `t` if `u < min(1, p/q)`. Move to position `i+1`.
- Otherwise **reject**: stop, and sample the replacement token from the *residual*
  distribution `norm(max(0, target_rows[i] - drafter_row_at_i))`. Since you are given only
  the scalar `q` rather than the drafter's full row, use the standard reduction: subtract
  `q` from the rejected token's mass only, clamp negatives to zero, renormalise, sample.
  Append that token and finish, with `ended_on_rejection = true`.
- If every draft token is accepted, sample one extra token from `target_rows[k]` and append
  it.

Guard the degenerate cases: `q == 0`, rows that are all zero after the residual subtraction,
and `draft` empty (then just sample from `target_rows[0]`).

## Tests you must write

- **Distribution preservation — the one that matters.** Fix a target distribution. Run tens
  of thousands of verifications with a deliberately BAD drafter (e.g. one that always
  proposes the least likely token) and histogram the committed first tokens. The histogram
  must converge to the target distribution. Then repeat with a GOOD drafter and assert the
  same convergence. **If acceptance is implemented naively this test fails and nothing else
  does — it is the reason this task exists.** State the statistical tolerance you used and
  why.
- **The floor.** `accepted` is never empty, for any input, including an all-rejected draft.
- **Perfect drafter.** When the drafter's proposals are the target's argmax and `q` equals
  `p`, every token is accepted and `accepted.size() == k+1`.
- **Determinism.** Same seed and inputs, identical result, across runs.
- **Degenerate inputs.** Empty draft, `q == 0`, single-token vocabulary, a row that is all
  mass on one token.
- **Falsification.** Show one test going red against a broken implementation.

## Deliverables

Header, source, tests, `CMakeLists.txt`, and a README stating the acceptance rule in your
own words, the statistical test design, and the measured convergence error.

---

# BRIEF D — `PrefixLedger`: a token-sequence ledger with rollback

## What this is for

An LLM caches its internal state for the tokens it has already processed. When the next
request shares a prefix with the last one, that cached work can be reused instead of
recomputed — the difference between a fast follow-up and a slow one.

Two operations make that safe, and the second is the hard one:

- Given a new token sequence, how long a prefix of it matches what is already cached?
- **Undo.** Discard the last N tokens and return to an earlier state exactly. Speculative
  decoding needs this: it processes k guessed tokens, then keeps only the ones that survive
  and must roll the rest back as if they never happened.

You are building the bookkeeping for both. No model, no cache — just an exact, fast record
of a token sequence supporting append, prefix comparison, and truncation.

## The exact interface

`include/prefix_ledger.hpp`, namespace `kv`.

```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace kv {

using TokenId = std::int32_t;

struct ReuseDecision {
    // Length of the common prefix between the ledger and the candidate sequence.
    std::size_t reusable = 0;
    // True when the candidate DIVERGES from the ledger before the ledger ends -- the
    // caller must discard cached state past `reusable`. False when the candidate simply
    // extends the ledger (or equals it), which needs no invalidation.
    bool divergent = false;
};

class PrefixLedger {
  public:
    void append(TokenId id);
    void append(std::span<const TokenId> ids);

    // Must not modify the ledger. Called on every request.
    [[nodiscard]] ReuseDecision plan_reuse(std::span<const TokenId> candidate) const;

    // Drop the last `n` tokens. Truncating more than is held clears it. THE ROLLBACK.
    void truncate_last(std::size_t n);
    // Keep exactly the first `n`.
    void truncate_to(std::size_t n);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::span<const TokenId> tokens() const noexcept;
    void clear() noexcept;

    // Cheap change-detector so a caller can tell whether cached state is still valid
    // without re-comparing token by token. MUST change on any mutation that changes the
    // contents, and MUST be equal for two ledgers holding the same tokens.
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;
  private:
    std::vector<TokenId> ids_;
};

} // namespace kv
```

## What "good" means

- `plan_reuse` runs on every request against sequences of tens of thousands of tokens.
  Budget: **under 20 microseconds at 100,000 tokens.** Benchmark it; report p50 and p99.
- `truncate_last` must be O(n) in the number removed at worst, and must leave the ledger
  byte-identical to one that had those tokens never appended. Prove that in a test.
- `fingerprint()` must satisfy both directions: equal contents give equal fingerprints, and
  any content change changes it. A rolling hash that survives truncation cheaply is the
  interesting design question — say what you chose and why.

## Tests you must write

- **Rollback equivalence, by construction.** Build ledger A by appending 1000 tokens then
  truncating the last 300. Build ledger B by appending only the first 700. A and B must be
  equal in `tokens()`, `size()`, and `fingerprint()`. Randomise the lengths and repeat.
- **Divergence detection.** Candidate equal to the ledger, a strict prefix of it, a strict
  extension of it, one differing at position 0, and one differing at the last position.
  Assert `reusable` and `divergent` for each — including that a pure extension is NOT
  divergent while a strict prefix IS.
- **Fingerprint sensitivity.** Changing any single token anywhere changes the fingerprint.
  Test at the front, middle, and end.
- **Interleaving.** Random sequences of append/truncate operations against a
  `std::vector<TokenId>` reference implementation; the two must never disagree.
- **Falsification.** Show one test going red against a broken implementation.

## Deliverables

Header, source, tests, benchmark, `CMakeLists.txt`, README with the fingerprint design and
the measured p50/p99.

---

# BRIEF E — `MarkdownStream`: an incremental markdown renderer state machine

## What this is for

A chat UI receives assistant text a few characters at a time and must render it live. Naive
approaches break in a specific, very visible way: a fenced code block opens with ```` ``` ````
and the closing fence has not arrived yet, so the UI either shows raw backticks, or opens a
code block that swallows the rest of the conversation, or flickers between the two on every
keystroke.

You are building the state machine that makes incremental rendering correct. Bytes in,
render events out. **No HTML, no DOM, no UI** — just the event stream, so the same logic can
drive any front end.

## The exact interface

`include/markdown_stream.hpp`, namespace `md`.

```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace md {

enum class EventKind : std::uint8_t {
    Text,           // plain prose to append
    CodeBlockOpen,  // `info` carries the language tag, possibly empty
    CodeBlockText,  // raw code to append, never interpreted
    CodeBlockClose,
    InlineCodeOpen,
    InlineCodeClose,
    HeadingOpen,    // `level` is 1..6
    HeadingClose,
    ListItemOpen,   // `level` is the nesting depth, starting at 0
    ListItemClose,
    ParagraphBreak,
};

struct Event {
    EventKind kind = EventKind::Text;
    std::string text;   // Text / CodeBlockText payload
    std::string info;   // language tag for CodeBlockOpen
    int level = 0;      // heading level or list depth
};

class MarkdownStream {
  public:
    // Feed the next chunk. Returns only events that are now CERTAIN. A chunk ending in
    // "``" emits nothing for those bytes: they may become a fence or may be literal.
    [[nodiscard]] std::vector<Event> feed(std::string_view chunk);

    // End of stream. Flushes held-back bytes as text and closes anything still open, so an
    // unterminated code block does not swallow the output.
    [[nodiscard]] std::vector<Event> finish();

    void reset();

    // True when bytes are being withheld pending disambiguation. The UI can show a caret.
    [[nodiscard]] bool pending() const noexcept;
};

} // namespace md
```

## What "good" means

The whole task is **chunk-boundary correctness**. These must hold:

1. **Split invariance.** For any input, feeding it in ONE chunk and feeding it split at
   EVERY possible position must produce the identical event stream once concatenated. This
   is the property to build around, and the property to test hardest.
2. **Never emit a partial marker as text.** A trailing `` ` ``, `**`, or `#` that might
   still become a marker is withheld until resolved.
3. **Bounded holdback.** Never withhold more than a small constant number of bytes. Holding
   the whole stream waiting for a fence that never comes is a hang, not caution.
4. **Code blocks are inviolate.** Inside a fence, nothing is interpreted — no headings, no
   lists, no inline code. Only the closing fence ends it.
5. **`finish()` always closes.** An unterminated fence, heading, or list must be closed.

Support: fenced code blocks with language tags, inline code, ATX headings, unordered and
ordered list items with nesting, and paragraph breaks. Do NOT attempt tables, links,
images, blockquotes, or emphasis beyond inline code — scope is deliberate.

## Tests you must write

- **Split invariance, exhaustively.** Take ~20 representative documents. For each, compare
  the one-chunk event stream against the stream from splitting at every single byte offset.
  All must match. This is the headline test.
- **The adversarial fence.** Inputs ending mid-fence (`` ` ``, ``` `` ```, ```` ``` ````),
  a fence inside a code block, backticks inside a code block, a language tag split across
  chunks.
- **No swallowing.** An unterminated code block at `finish()` still yields all preceding
  text as events.
- **Holdback bound.** Assert the withheld byte count never exceeds your stated constant,
  across all test inputs.
- **Falsification.** Show one test going red against a broken implementation.

## Deliverables

Header, source, tests, `CMakeLists.txt`, README stating the holdback bound, the supported
subset, and how split invariance is guaranteed rather than merely tested.
