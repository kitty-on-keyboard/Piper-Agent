# mlx_qwen_tokenizer

C++20 BPE tokenizer for Qwen 3.6, targeting MLX on Apple Silicon. Built to replace
`tokenizers-cpp` (the C++ → C → Rust FFI wrapper over HF `tokenizers`) in LM_Pipe.

Merged from six independent implementations of the same specification, keeping the parts that
survived review and fixing the defects each of them shipped.

## Build and test

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

Nothing to download, nothing to place by hand — the vocabulary is committed under
`testdata/qwen36/`. Dependencies (PCRE2, nlohmann/json, utf8proc, GoogleTest) are fetched by
CMake.

MLX is **off** by default, so the library, CLI, tests and benchmark all build on a machine with no
MLX installed. Enable the `mlx::core::array` bridge with `-DMLX_QWEN_TOKENIZER_ENABLE_MLX=ON`.

## Versus the incumbent

All five workloads LM_Pipe performs, measured back-to-back against `tokenizers-cpp` on the same
machine (macOS arm64, AppleClang `-O2`), the **same 19.06 MB production Qwen 3.6 tokenizer.json**,
and the **same corpus bytes** — `bench/bench_tokenizer.cpp` and `bench/bench_incumbent.cpp`
generate identical inputs so every row is directly comparable:

| Workload | Incumbent | This library | Ratio |
|---|---|---|---|
| LOAD — **first ever** for a model | 211 ms | **348 ms** | **0.6x — slower** |
| LOAD — every load after | 211 ms | 5.9 ms | **36x** |
| ENCODE 256 KB bulk | 5.52 MB/s | 68.4 MB/s | **12.4x** |
| DECODE per-token (generation hot path) | 170 ns/tok, **corrupts** | 8 ns/tok, correct | **21x** |
| DECODE batch ×20000 | 53 ns/tok | 4 ns/tok | **13x** |
| VOCAB single-char sweep ×248077 | 77.8 ms | 0.035 ms | **2200x** |
| ENCODE constant string ×10000 | 4388 ns/call | 505 ns/call | **8.7x** |

Medians of 3–4 interleaved runs on an idle machine. Encode throughput is the most load-sensitive
row: under compile contention it drops to ~41 MB/s, still ~8x the incumbent.

**The first load of any given model is ~1.6x slower than the incumbent, not faster.** That run
parses the JSON *and* bakes the binary cache; every subsequent load reads the cache in ~6 ms. The
trade is deliberate — one slower start per model install buys 36x on every start thereafter — but
anyone benchmarking a fresh checkout against a fresh model will measure 348 ms and should see that
number here rather than conclude the table is wrong.

### What this does *not* buy

Against LM_Pipe's measured runtime — ~57 s model load, ~46 ms/token forward pass, ~23 s of decode
per turn — these savings are: load 205 ms of ~57 s (0.4%), decode 162 ns against 46 ms/token
(0.0003%), encode a few ms per turn. The vocab sweep sits behind a flag that is currently off.

**The tokenizer is not a bottleneck in this consumer and replacing it will not make the agent
measurably faster.** The case for the swap is correctness — the split-codepoint corruption below,
and the encoder being right on inputs where the incumbent pipeline was not — plus owning the
format. The ratios above are headroom, not user-visible latency.

"Warm" load reads the binary vocab cache (below); "first-ever" is the one-time JSON parse that
builds it. Reproduce both columns with the two benchmark binaries — no number above is from a
corpus either implementation generated for itself.

## Correctness guarantees

**Drop-in equivalence with the incumbent on real consumer prompts.** `bench/compare_incumbent_equivalence.cpp`
assembles LM_Pipe-shaped prompts using its ChatML template verbatim — `<|im_start|>` control
tokens, the KV-baked `<tools>` schema block, `<tool_call>` XML, `<tool_response>` turns,
multi-turn transcripts carrying source code, diagnostics, and unicode — and asserts id-for-id
equality against `tokenizers-cpp` itself. **54 prompts, 173,154 tokens, largest 79,357 tokens:
zero mismatches**, plus exact round-trip on every one. Probe-shaped agreement is not the same as
agreement on the bytes the consumer actually sends; this is the test that says drop-in.

Build it with `-DTOKENIZERS_CPP_DIR=<checkout> -DTOKENIZERS_CPP_LIB_DIR=<built libs>`; without
those the default build has no dependency on tokenizers-cpp.

**HF parity on encode.** Byte-identical token ids to HuggingFace `tokenizers` 0.22.2 across
15,045 generated cases — prose, code, ChatML, CJK, emoji, combining marks, decomposed input,
digit runs, long words — on **both** the committed test vocabulary and the 19 MB production
Qwen 3.6 vocabulary, via both load paths (JSON and cache). The pre-tokenizer Split regex and the
NFC normalizer are read from `tokenizer.json` rather than assumed: the 248k-vocab Qwen 3.6
pattern keeps combining marks attached (`[\p{L}\p{M}]+`), and NFC composes decomposed input
(`n` + U+0303 → `ñ`, one token) — both diverge from what a hardcoded pipeline produces, and both
were live encode bugs here until checked against the reference.

**Streaming decode is exact.** Feeding ids one at a time through `StreamingDecoder` and
concatenating the results is byte-for-byte identical to batch decode. This is the property the
incumbent's per-token path lacks: it lossily converts each token independently, so any codepoint
whose UTF-8 bytes split across two tokens becomes U+FFFD — `🎉` decodes as `���`
(`scripts/repro_split_codepoint_corruption.py` reproduces this against the real vocab; 944 of the
248k tokens are byte fragments that are not standalone UTF-8, so it is reachable, not
theoretical). The decoder holds an incomplete-but-valid sequence across calls, emits it the
moment it completes, and resyncs immediately on bytes already proven invalid rather than stalling.
Pinned by `StreamingDecodeEqualsBatchDecode` and the split-sequence tests.

**Malformed input round-trips.** Invalid UTF-8 is tokenized byte-level and decodes back to the
identical bytes (the incumbent replaces it with U+FFFD). Damage from a bad byte is confined to
that byte — it does not re-segment the rest of the document.

## Binary vocab cache

After the first JSON parse, every table the runtime needs (token text blobs, decoded-bytes blob,
merge hash table, sorted index) is written out as flat PODs and read back with no per-entry work.

**The cache never goes in the model directory.** It lands in this library's own per-user cache —
`$XDG_CACHE_HOME/mlx_qwen_tokenizer`, else `~/Library/Caches/mlx_qwen_tokenizer` on macOS, else
`~/.cache/mlx_qwen_tokenizer` — named by a hash of the tokenizer.json's absolute path so two
models cannot collide. The directory holding the model is opened read-only: it belongs to whoever
installed the model (LM Studio re-verifies its model directories, the volume may be read-only, and
a sidecar there would vanish on re-download).

Override with `LoadOptions::cache_dir`, disable with `LoadOptions::use_cache = false`, and locate
it with `Vocab::cache_path_for(path, cache_dir)`. Validity is source size + mtime plus a payload
checksum; a stale or corrupt cache silently falls back to the JSON and is rewritten. Write failure
is silent — the cache is an optimization, never a requirement.

```cpp
mlx_qwen_tokenizer::LoadOptions opts;
opts.cache_dir = "/var/cache/myapp/tokenizers";   // optional; defaults to the per-user cache
tokenizer.load(path_to_tokenizer_json, opts);
```

## API notes for consumers

- **Generation loop:** construct one `StreamingDecoder` per stream and call `decode(id)` per
  token. Batch `Tokenizer::decode(ids)` is a straight concatenation of per-token bytes.
- **`Vocab::single_char_table(out)`** fills `out[id]` with the single byte token `id` decodes to
  (or `'\0'`) in one pass — replaces the quarter-million per-id decode calls the JSON-RPC grammar
  sampler would otherwise make.
- **`Vocab::id_to_bytes_view(id)`** is the raw decoded bytes of any token, borrowed, no copy.
  Anything currently discovered by calling decode in a loop over the vocab should read this table
  instead.

## Performance provenance

Encode scaling is linear (~2.0x per input doubling; the benchmark prints the ratio so a
super-linear regression is visible). The pathological cases are covered: a 64 KB unbroken word
encodes via a heap-based merge in milliseconds where a rescan merge took 4.3 s quadratically
(`VeryLongWordDoesNotBlowUp`), and PCRE2 runs with `PCRE2_SUPPORT_JIT` actually enabled — the
`pcre2_jit_compile` call is silently a no-op without it, which is worth checking in any PCRE2
project claiming JIT.

Every optimization here was differentially verified byte-identical before landing, and every
regression test was run against the buggy code it pins to prove it fails.

## Layout

```
include/mlx_qwen_tokenizer/   public headers
src/                          vocab, trie, normalizer, pretokenizer, bpe, streaming decoder, mlx bridge
cli/                          qwen_tok
bench/                        bench_tokenizer (this library), bench_incumbent (tokenizers-cpp)
tests/                        gtest suite + oracle fixtures
scripts/                      generate_qwen_oracle.py, repro_split_codepoint_corruption.py
testdata/qwen36/              committed Qwen 3.6 vocabulary
docs/                         consumer integration findings (LM_Pipe)
```
