# Third-party notices

Piper Agent's own code is Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).
This file credits software that is vendored, fetched at build time, or linked from the
macOS SDK.

## Vendored (in `third_party/`)

| Component | License | Where the full text is |
|---|---|---|
| simdjson 4.6.1 | Apache-2.0 | [third_party/simdjson/LICENSE](third_party/simdjson/LICENSE) |
| PCRE2 10.43 | BSD 3-Clause | [third_party/pcre2/LICENCE](third_party/pcre2/LICENCE) |
| utf8proc | MIT + Unicode data terms | [third_party/utf8proc/LICENSE.md](third_party/utf8proc/LICENSE.md) |
| nlohmann/json 3.11.3 | MIT | [third_party/nlohmann/LICENSE.MIT](third_party/nlohmann/LICENSE.MIT) |
| robin_hood hashing | MIT | Copyright (c) 2018-2021 Martin Ankerl; header in `third_party/frankentok/robin_hood.h` |
| frankentok | Apache-2.0 (this project) | Qwen tokenizer; original to this tree |
| parsephony | Apache-2.0 (this project) | JSON PDA + tool-call guard; original to this tree |

PCRE2's optional GNU readline binding is GPL and is **not** enabled in this build.

## MLX (fetched at build when `LMP_WITH_MLX=ON`)

[MLX](https://github.com/ml-explore/mlx) v0.32.0 is downloaded by CMake FetchContent and
linked statically. A shipped sidecar contains MLX (and `mlx.metallib`). MLX is MIT:

```
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING WITHOUT LIMITATION THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

MLX itself fetches, at configure time: `{fmt}` (MIT), nlohmann/json (MIT, a second copy),
and Apple metal-cpp (Apple sample-code terms). Those copies live in the CMake build
directory, not in this git tree.

## System libraries (macOS SDK, not redistributed as source)

Accelerate, Metal, Foundation, QuartzCore, ImageIO, CoreGraphics, CoreFoundation,
SQLite3, zlib.

## Model weights

This repository does **not** include Qwen (or any other) model weights. You download a
checkpoint separately and point `lmPipe.modelDir` / `LMP_QWEN_DIR` at it. Comply with
**that checkpoint's** license; typical Qwen3 community checkpoints are Apache-2.0, but
some Qwen3.8 variants use a different community license.

## Test / bakeoff-only

GoogleTest (BSD-3-Clause) and extra nlohmann/json copies may be FetchContent'd by bakeoff
entrants. They are not part of the sidecar.
