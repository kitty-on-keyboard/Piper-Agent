# Contributing

Piper Agent is Apple Silicon only. There is no portable fallback.

## Build

```bash
cmake --preset dev && cmake --build --preset dev -j8
ctest --preset gate
```

The gate (`ctest -L gate`) must stay green. It uses no model, no network, and no
workspace. Real-model tests are labelled `realmodel` and must not run in parallel.

`dev` compiles [MLX](https://github.com/ml-explore/mlx) from source (tag `v0.32.0`). That
is slow the first time. CI builds with `-DLMP_WITH_MLX=OFF`.

## Model weights

Do not commit checkpoints, `.safetensors`, or tokenizer dumps from a live model.
Point tests at a local directory with `LMP_QWEN_DIR`.

## Scope

One in-process Qwen3 model via MLX. No second inference server, no subagents, no other
model families (they refuse at load). The agent does not run git commands; it edits the
workspace and leaves staging/commits to you.

## Extension

```bash
cmake --build --preset dev --target lmp_sidecar -j8
cd extension && npm install && npm run compile && node scripts/verify-question-options.js && node scripts/verify-inline-code-order.js
```

`npm run install-local` packages a `.vsix` and installs it into every VS Code-family
editor it finds. Override with `LMP_EDITOR_CLI`.
