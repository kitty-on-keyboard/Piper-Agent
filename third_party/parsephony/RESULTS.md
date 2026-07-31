# parsephony — results

Measured 2026-07-29, M-class Apple Silicon, AppleClang 21, `-O3`.
Cookoff entries were built and run from their own branches the same day; their
vs-nlohmann ratios come from their own benchmarks, which makes "speedup over
nlohmann" the one yardstick every implementation shares.

Reproduce: `cmake -S . -B build && cmake --build build -j8 && ctest --test-dir build`
then `./build/bench`, `./build/bench_stream`, `./build/bench_qwen`.

---

## 1. Correctness

| | JSONTestSuite | streaming chunk-safety | mask↔parser agreement |
|---|---|---|---|
| **parsephony** | **283/283 mandatory** (95 y + 188 n), 0 structural mismatches vs nlohmann | **every split point of every test doc** | **exhaustive, both grammars** |
| #6 typed-schema | 318/318 by its own accounting | n/a | n/a |
| #7 bake-cache | 244/244 (subset) | n/a | n/a |
| #9 round2-a | differential fixtures only | hand-picked splits | n/a |
| #11 round2-b | 14 gtest cases | n/a | n/a |
| others | fixture round-trips only | n/a | n/a |

Test totals: `smoke` 82 · `stream_test` 5,631 · `toolcall_test` 38,934 · `oracle` PASS.
The 6 implementation-defined `i_` divergences are the same 6 entry #6 reported
(huge exponents saturating to infinity; RFC 8259 BOM rejection where nlohmann is lenient).

## 2. JSON throughput

Speedup over nlohmann, small tool-call payloads:

| implementation | vs nlohmann | caveat |
|---|---|---|
| **parsephony** | **4.4× – 11.1×** | full escape/surrogate/UTF-8 validation |
| #5 schema-codegen | ~3.5–4.1× | **never decoded escapes** |
| #8 perfect-hash | ~3.3–3.5× | **escapes skipped entirely** |
| #1 arena-zerocopy | ~2.4–3.2× | |
| #3 feat-impl | ~3.0× | |
| #6 typed-schema | ~2.9× | |
| #10 round2-d | ~1.8× (text path) | |
| #2, #7 | — | benchmarks didn't build (simdjson × AppleClang 21) |
| #4 | — | didn't compile |

Both entries that cleared 3× got there by skipping work. parsephony beats them
while matching the strictest entry's correctness.

**Honest bound:** simdjson DOM remains 1.2–3.6× faster on raw parse-only, and
On-Demand ~7× (it never validates unread values). Closing that needs NEON
structural scanning. At ~150 ns per tool call against a ~10 ms/token generation
budget, that work buys LM-Pipe nothing — and neither simdjson mode can run
token-incrementally or emit a decoding mask, which is the actual job.

## 3. Streaming

| | work to first field | vs the round-2 entries |
|---|---|---|
| **parsephony** | **2.5×** less than wait-then-parse (name known at byte 85/146) | — |
| #9 round2-a | 2.3× | beaten |
| #10 round2-d | 1.75× | beaten |

## 4. Constrained decoding — real Qwen 3.6 vocabulary

Vocabulary loaded from the model the user actually runs
(`Qwen3.6-35B-A3B-MLX-4bit/tokenizer.json`): **248,077 byte-level BPE tokens**,
33 special, exported via `scripts/export_vocab.py`.

Two grammars, one mask engine (`TokenMaskT`, templated over the automaton):

### JSON grammar
```
cold mask, document start      1.07 ms   (880 / 248,044 tokens legal)
cold mask, string interior     0.12 ms   (245,576 legal)
amortized per sampling step    8,099 ns  (34 grammar states)
steady state (warm cache)        7.2 ns
```

### Qwen XML tool-call grammar (`ToolCallGuard`)
```
1,000 constrained generations   1000/1000 valid, extracted, tool always registered
tokens emitted                  53,165  (53.2 per call)
amortized per sampling step     4,845 ns  (389 grammar states)
steady state (warm cache)         17.7 ns
```

Legal-token counts by position — the constraint visibly doing work:

| position | legal tokens (of 248,044) |
|---|---|
| at start (framing) | **2** |
| tool name after `get_` | **3** |
| parameter-name position | **12** |
| inside free-text value | 248,044 |
| inside a number value | **14** |

**17.7 ns against a ~10 ms token.** The guarantee costs ~0.0002% of generation.

### What the guard enforces

Verified against the model's own `chat_template.jinja` — Qwen 3.6 emits **XML**,
not JSON:

```
<tool_call>
<function=get_weather>
<parameter=city>
Denver
</parameter>
</function>
</tool_call>
```

Under the mask the model **cannot**: call an unregistered tool, misspell or
repeat a parameter, close `</function>` before every required parameter is
present, emit a malformed typed value, or run past `</tool_call>`. Typed values
nest the JSON PDA as a sub-automaton, so `tojson`-encoded numbers, booleans,
objects and arrays are validated by the same engine. The guard doubles as the
extraction parser — `tool_name()` and `params()` are populated on completion, so
LM-Pipe needs no second pass.

## 5. Idea provenance

| entry | idea | fate |
|---|---|---|
| #1 | zero-copy string views | ✅ core of the tape |
| #2 | flat tape, lazy decode | ✅ + the skip pointers it lacked |
| #3 | strict UTF-8 validation | ✅ table-driven, default on |
| #4 | parse-into-struct typed sink | ⏸ `Value::at()` covers the access pattern |
| #5/#8 | schema key specialization | ⏸ their edge was skipped work, not the hash |
| #6 | JSONTestSuite + differential oracle | ✅ adopted as the acceptance gate |
| #7 | bake cache | ❌ can't pay for itself on 150-byte payloads |
| #9 | resumable chunk-safe SAX | ✅ + exhaustive split-point testing |
| #10 | token-ID fast path | 🔁 generalized into `TokenMaskT` on the real vocab (its version allocated a `std::vector` per lookup against a 22-token toy) |
| #11 | grammar/parser duality | ✅ **made real** — `allowed_bytes()` + token masks, the part it only named |

## 6. How the mask stays cheap

Naively, masking 248k tokens per sampling step is milliseconds. Three structural
observations collapse it to nanoseconds:

1. **Configurations recur.** Masks are cached by `state_signature()`; an entire
   tool-call grammar visits 389 distinct configurations, a JSON document 34.
2. **String interiors are nearly unconstrained.** Tokens are pre-classified once
   at construction, so string states start from a precomputed base mask and
   simulate only the ~few thousand tokens containing quotes, backslashes or
   control bytes.
3. **Everywhere else is narrow.** Tokens are bucketed by first byte (CSR), so a
   state accepting two bytes touches those two buckets, not the vocabulary.

Correctness of the optimization is not assumed — `toolcall_test` checks, for
every prefix of canonical calls, that `allowed_bytes()` accepts a byte **iff**
the automaton does (38,934 assertions), and `bench_qwen` verifies across 53,165
emitted tokens that the mask never blocks a legal continuation and never permits
an illegal one.
