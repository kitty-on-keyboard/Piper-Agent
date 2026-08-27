# parsephone

A JSON parser and **constrained-decoding engine** for local LLM agent loops,
built for Piper Agent and Qwen 3.6.

Synthesized from an 11-way agent cookoff (`parsephony`,
PRs #1–#11) — every entry's best idea kept, every entry's flaw fixed, then
pushed well past what any of them attempted. See [DESIGN.md](DESIGN.md) for
provenance and [RESULTS.md](RESULTS.md) for measurements.

## What it does

**Parses JSON** — a flat tape of 12-byte nodes with skip pointers, lazy scalar
decode, zero-copy strings. 4.4–11.1× faster than nlohmann on tool-call payloads
while passing 283/283 mandatory JSONTestSuite cases.

**Streams** — a resumable byte-level PDA. Chunk boundaries may fall anywhere,
including mid-escape and between surrogate-pair halves. Fields surface as they
arrive rather than after the document closes.

**Constrains generation** — the same automaton that validates output also
answers *"which of the 248,077 Qwen tokens may come next?"*, so the model is
structurally unable to emit an invalid tool call.

```
1,000 constrained generations   1000/1000 valid, extracted, tool always registered
steady-state mask cost              17.7 ns per sampling step
one Qwen 3.6 token                  ~10,000,000 ns
```

## Qwen 3.6 speaks XML, not JSON

Verified against the model's own `chat_template.jinja`:

```
<tool_call>
<function=get_weather>
<parameter=city>
Denver
</parameter>
</function>
</tool_call>
```

`ToolCallGuard` implements that grammar, with the JSON PDA nested inside it for
`tojson`-encoded typed values. Given a tool registry it enforces — byte by byte,
at decode time — that the model cannot call an unregistered tool, misspell or
repeat a parameter, close `</function>` before every required parameter is
present, emit a malformed typed value, or run past `</tool_call>`.

It is also the extractor: on completion, `tool_name()` and `params()` hold the
parsed call. No second pass.

## Quick start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

Benchmarks: `./build/bench` (JSON head-to-head), `./build/bench_stream` (TTFF),
`./build/bench_qwen` (real-vocab constrained decoding — needs the vocab export
below).

```bash
python3 scripts/export_vocab.py \
  ~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit/tokenizer.json \
  fixtures/qwen36_vocab.bin
```

## Using it

Parse and pull fields out:

```cpp
#include "parsephony/parsephony.hpp"

parsephony::Document doc;
if (parsephony::parse(payload, doc) == parsephony::Error::Ok) {
    auto name = doc.root().at("tool_calls.0.function.name").get_string();
}
```

Constrain generation and extract in one pass:

```cpp
#include "parsephony/toolcall.hpp"
#include "parsephony/mask.hpp"

std::vector<parsephony::ToolSpec> tools = {
    {"get_weather", {{"city", parsephony::ParamType::Text, /*required=*/true},
                     {"days", parsephony::ParamType::Number, false}}},
};

parsephony::Vocab vocab;
vocab.load("fixtures/qwen36_vocab.bin");

parsephony::ToolCallGuard guard(tools);
parsephony::TokenMaskT<parsephony::ToolCallGuard> masker(vocab);

while (!guard.complete()) {
    const auto& mask = masker.compute(guard);       // ~18 ns warm
    for (size_t i = 0; i < vocab.size(); ++i)
        if (!parsephony::TokenMaskT<parsephony::ToolCallGuard>::test(mask, i))
            logits[i] = -1e9f;                      // illegal tokens unsampleable
    int tok = sample(logits);
    guard.feed(vocab.tokens[tok]);
}
// guard.tool_name() and guard.params() are now populated.
```

See [INTEGRATION.md](INTEGRATION.md) for wiring this into LM-Pipe's decode loop.

## Layout

| path | what |
|---|---|
| `include/parsephony/parsephony.hpp` | tape, `Document`, `Value`, one-shot `Parser` |
| `include/parsephony/stream.hpp` | resumable JSON PDA, SAX `Handler`, `ByteSet` |
| `include/parsephony/toolcall.hpp` | Qwen XML tool-call grammar + schema enforcement |
| `include/parsephony/mask.hpp` | `TokenMaskT`, `Vocab` — token masks over any automaton |
| `include/parsephony/swar.hpp` | 8-bytes-per-iteration string scanning |
| `scripts/export_vocab.py` | HF `tokenizer.json` → flat binary vocab |

## Status

`smoke` 82 checks · `stream_test` 5,631 · `toolcall_test` 38,934 · `oracle` PASS.
All green as of 2026-07-29.

One gap left open deliberately: simdjson's DOM parser is still 1.2–3.6× faster
on raw parse-only. Closing it needs NEON structural scanning, which buys nothing
against a 10 ms/token budget — and neither simdjson mode can run
token-incrementally or emit a decoding mask.
