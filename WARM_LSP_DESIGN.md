# Warm LSP design

Flag: `LMP_WARM_LSP`. Default off. Unset or any value other than `1` registers the MCP tools and returns a tool error. No language server is spawned.

Branch: `warm-lsp-mcp`. This does not change decode, MTP, or MoE.

## Seam

Piper already hosts MCP servers from `.mcp.json` when a packet names them with `trust_mcp`. `warm-lsp` is that server: `scripts/warm_lsp_mcp.py`. It speaks newline MCP to `McpHost` and Content-Length LSP to the children. ToolCallGuard sees `lsp_status`, `lsp_diagnostics`, and `lsp_restart` through the existing schema path. There is no second MCP stack.

`spawn_env` drops parent `LMP_*`. `build_start_message` copies `LMP_WARM_LSP=1` onto the warm-lsp child only when the parent value is `1`, and copies `LMP_GODOT_BIN`, `GODOER_GODOT_BIN`, and `GODOT_BIN` when they are set. With none of those set, GDScript still launches the Godoer fork.

## Godot binary

Default editor, in order:

1. `LMP_GODOT_BIN`
2. `GODOER_GODOT_BIN`
3. `GODOT_BIN`
4. `/Users/dev/godot-godoer/bin/godot.macos.editor.arm64`

That file is `4.6.3.stable.custom_build`. `/Applications/Godot.app` is not a fallback. The LSP is TCP (`--headless --editor --lsp-port`), which this fork serves without a GUI. `HOME` for that process is `.piper/lsp-home` inside the workspace, because the editor refuses to start when its config directory is not writable.

Seatbelt allows file-read of the Godoer checkout (`…/godot-godoer`, the parent of `bin/`) so the editor binary can be mapped. That allow is not a read of the rest of `$HOME`.

## Seatbelt

Profile is built by `seatbelt_profile` and applied with `sandbox-exec` to each language server. The MCP process itself stays outside Seatbelt, same as every other trusted server.

Measured on this Mac:

- `deny file-read* (subpath $HOME)` then `allow file-read* (subpath <workspace>)` lets `/bin/cat` read a workspace file.
- The same profile makes `ls $HOME/Desktop/Models` fail with `Operation not permitted`.
- `deny file-read-data` is not overridden by a later `allow file-read*`. The home deny has to be `file-read*`.
- Ancestor `file-read-metadata (literal …)` nodes are required. Without them clangd reports `Failed to resolve path` because the workspace lives under `$HOME`.
- With those literals, `clangd --check` on `int x = "hi"` reported `cannot initialize a variable of type 'int'` and `All checks completed, 1 errors`.
- `touch /private/tmp/…` under the profile is denied. Workspace listing is allowed. `ls $HOME` is denied. Covered by `scripts/test_warm_lsp.py`.

Writes stay denied outside the workspace. Named SwiftPM and DerivedData directories are not opened. That is why Swift is parked.

## Languages

| Language | Server | State |
| --- | --- | --- |
| C++ | `/usr/bin/clangd` | Wired. Fixture `testdata/warm_lsp/cpp`. |
| Python | basedpyright in `.piper/lsp-py`, launched with `/opt/homebrew/bin/python3` | Wired. The uv Python under `~/.local` is not used: that path is inside the home deny. |
| GDScript | Godoer fork, headless LSP | Wired. Fixture `testdata/warm_lsp/gdscript`. |
| Swift | `/usr/bin/sourcekit-lsp` | Parked. Under this profile sourcekitd exits with `SOURCEKITD FATAL ERROR: Service is invalid` and publishes nothing. `SDKROOT` plus the existing SwiftPM cache paths did not change that. |

## Not killed early

A scoped profile can exec clangd, basedpyright, and the Godoer editor and still deny home listings and writes outside the workspace. Swift does not justify widening that profile.
