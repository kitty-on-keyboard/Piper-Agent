# Warm LSP prove

Machine: this Mac. Command: `python3 scripts/prove_warm_lsp.py` with `LMP_WARM_LSP=1` and `LMP_GODOT_BIN`, `GODOER_GODOT_BIN`, and `GODOT_BIN` unset. N=10. Cold is `lsp_restart` then `lsp_diagnostics` on a fresh process. Warm is `lsp_diagnostics` on the process left running, with a new trailing comment each call so the server must publish again. The timer wraps only `lsp_diagnostics`. Percentile index is `round((p/100)*(n-1))` (Python 3 half-even). RSS is the sandbox-exec process plus its descendants, in KB, from `ps`.

Godot binary reported by `lsp_status` on that run: `/Users/dev/godot-godoer/bin/godot.macos.editor.arm64` (`4.6.3.stable.custom_build.429fb6e2c`). Stock `/Applications/Godot.app` is not the default.

Servers: Apple clangd 21.0.0, basedpyright 1.40.1, Godoer headless `--editor --lsp-port`.

## Latency and RSS

| Language | Cold P50 ms | Cold P95 ms | Warm P50 ms | Warm P95 ms | RSS after first warm KB | RSS after 10 warm edits KB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C++ | 28.8 | 30.5 | 59.8 | 66.9 | 24160 | 24688 |
| Python | 280.3 | 298.5 | 259.2 | 261.1 | 204192 | 206080 |
| GDScript | 1089.8 | 1102.9 | 6.5 | 9.6 | 883360 | 883360 |

C++ cold ms: 30.4, 30.5, 29.0, 28.2, 28.8, 29.2, 28.8, 29.3, 28.6, 28.5.

C++ warm ms: 59.8, 55.7, 63.3, 62.7, 65.0, 59.2, 57.8, 58.0, 66.9, 66.4.

Python cold ms: 294.3, 278.7, 279.9, 298.5, 280.3, 281.0, 291.4, 282.5, 279.0, 279.7.

Python warm ms: 259.2, 260.5, 260.4, 255.9, 260.8, 258.5, 258.8, 261.1, 259.9, 257.6.

GDScript cold ms: 1101.5, 1081.2, 1096.8, 1077.3, 1089.8, 1097.3, 1093.7, 1081.9, 1102.9, 1072.8.

GDScript warm ms: 6.7, 2.5, 9.6, 5.4, 7.6, 6.5, 5.4, 9.4, 5.7, 7.4.

Python RSS includes the Node child (about 185 MB of the ~199 MB). C++ RSS is clangd. GDScript RSS is the headless editor and did not grow across the 10 edits.

Warm P50 does not beat cold on C++ (59.8 ms vs 28.8 ms). Python warm P50 is 21 ms under cold (259.2 vs 280.3), which is a small gap on a ~280 ms call. GDScript warm P50 is 6.5 ms against a 1089.8 ms cold spawn.

An identical buffer does not get a second `publishDiagnostics` from clangd. The server returns the previous diagnostics for an unchanged text instead of waiting out the 20s kill timeout. The table above is real edits, not that cache.

## Seatbelt

Profile from `seatbelt_profile` on this repo, applied with `sandbox-exec`:

- `touch /private/tmp/warm-lsp-escape-probe` exited 1: `Operation not permitted`.
- `ls /Users/dev` exited 1: `Operation not permitted`.
- `ls testdata/warm_lsp` exited 0.

`python3 scripts/test_warm_lsp.py` passed on this machine after those checks, including flag-off (no child server), a planted Python diagnostic, a planted C++ diagnostic, a planted GDScript diagnostic from the Godoer fork, and a Swift tool error that contains `parked`.

## Swift

Parked. `/usr/bin/sourcekit-lsp` under this profile exits with `SOURCEKITD FATAL ERROR: Service is invalid` and publishes no diagnostics. Adding `SDKROOT` and the existing DerivedData / SwiftPM cache allows did not change that. The profile was not widened.

## Agent task

Same planted function in two copies, Qwen3.8-27B-MLX-4bit, `reasoning_effort` event `requested=xhigh supported=1 instructed=1` on both runs. One sidecar at a time. `LMP_DRAFT_DIR` pointed at `Qwen3.8-27B-MTP-4bit` for two attempts of the flag-off slice. Both died in decode with `MTLCompilerService` unavailable (`Unable to build metal library from source`) after `decode_begin speculative=1 mtp=1`, 0 tokens. The runs below left `LMP_DRAFT_DIR` unset.

Flag off (`ab-off`, no `trust_mcp`, `LMP_WARM_LSP` unset): status ok, 6 turns, wall_seconds 98.207, check exit 0. Journal tools: `plan`, `read_file`, `replace_in_file`, `finish`. No `lsp_diagnostics`. Edit: `return "no" if x >= 0 else x` became `return 1 if x >= 0 else x`.

Flag on (`ab-on`, `--trust-mcp warm-lsp`, `LMP_WARM_LSP=1`): status ok, 8 turns, wall_seconds 260.404, check exit 0. `lsp_diagnostics` ran before the edit and reported `Type "int | Literal['no']" is not assignable to return type "int"` at line 2. The edit was `return -1 if x >= 0 else x`. A second `lsp_diagnostics` returned no diagnostics. The journal's only write is that one `replace_in_file`. `files_touched` also lists the pre-existing dirty worktree; those files were not in the tool log.

The flag-on agent used the diagnostic and was slower (260.404 s, 8 turns) than the flag-off agent (98.207 s, 6 turns).

## Dogfood

The MCP server, fixtures, and prove script were written in this session by the parent. Piper was dispatched only for the agent comparison, after the prove hang on an unchanged clangd buffer was fixed. Phase 4 (a real Piper/Godoer edit under the flag) was not run: the KEEP-seed bar was not met.

## Verdict

KEEP-narrow.

The pass bar for KEEP-seed was warm P50 clearly beating cold on C++ and Python. That did not happen. C++ warm is slower than a cold clangd on this one-line fixture. Python's warm edge is 21 ms.

The server stays, flag default off. GDScript on the Godoer fork is the latency result that is large (1089.8 ms cold, 6.5 ms warm) and it costs about 863 MB RSS. C++ and Python diagnostics are correct under the same Seatbelt profile; do not describe them as a warm-latency win. Swift stays parked. Stock Godot.app is not a fallback.
