# commit_think_block — handoff for Grok (Cursor)

You are implementing this for Sean's C++ Piper sidecar (`LM_Pipe_2`). Be conservative: default **off**, exact-byte copies only, never auto-write from think. Wrong writes are worse than wasted decode tokens.

## 0. Non-negotiables

- Repo: `/Users/dev/Desktop/seans_projects_local/LM_Pipe_2`
- Branch: `fix/greenfield-write-caps` (HEAD ≈ `2a568b8`; tree is dirty — **do not clean/stash/reset**)
- Feature flag default **off**: env `LMP_COMMIT_THINK=0` (or unset = off). `=1` enables tool declaration + harvest.
- **Never** auto-commit on `</think>`. Model must call the tool explicitly after think.
- **Never** invent, merge, or "best-guess" fence content. Unclosed / ambiguous fences → drop, no block.
- Same sandbox + read-before-write / versioning rules as `write_file` (see `src/tools/registry.cpp` write_file). Do not bypass them.
- Tool **illegal inside** `TurnPhase::Think`. Legal only in Text / same window as other tools after `</think>`.
- Hardware: M5 Pro 48GB; do not load two big models while testing.
- Related but separate: KV/compaction shadow-swap is `docs/KV_COMPACTION_SHADOW_SWAP_HANDOFF.md` — do not expand this handoff into that work.

## 1. Product goal

Qwen often drafts a full code fence inside `<think>`, then retypes it into `write_file` — wasted **decode** tokens (KV already holds the think text).

**Fix:** after think closes, sidecar harvests closed fences into turn state. Model may call `commit_think_block` with a tiny arg list; sidecar copies **exact** stored bytes through the normal write path.

Success = correct file bytes + far fewer tool-body tokens on a smoke where the model commits from think. Failure mode to avoid = wrong path / partial fence / silent write.

## 2. Design (do not freestyle)

### 2.1 Harvest (on `</think>` / think→text transition)

When think closes and `LMP_COMMIT_THINK=1`:

1. Decode think token ids → text (same decode path the UI uses).
2. Extract **closed** markdown fences: ` ```lang?\n ... \n``` `
3. For each block store: `block_id` (stable int from 0), optional `language`, `content` bytes, `sha256`, `char_len`, source token span if cheap.
4. Caps: max **3** blocks/turn; max size per block (pick something sane, e.g. 200k chars — document it); drop overflow with a clear event/log line.
5. Unclosed fence, nested weirdness, empty body → **skip that fence**, do not repair.

Store on the turn / agent object that tools can read for **this turn only** (cleared next turn unless you have a strong reason not to).

### 2.2 Tool: `commit_think_block`

Suggested schema:

| param | required | meaning |
|---|---|---|
| `path` | yes | workspace-relative path (same as write_file) |
| `block_id` | yes | index from harvest |
| `expect_sha256` | no but encouraged | if present, must match or ToolError |

Behavior:

1. Look up block; missing → ToolError, disk unchanged.
2. If `expect_sha256` set and mismatch → ToolError.
3. Apply **identical** gates as `write_file` (sandbox, read-before-write / content version, approvals).
4. Write **exact** `content` bytes (no trailing cleanup that changes sha).
5. Return summary including path, block_id, sha256, bytes written.

Optional later (not v1): `append` vs overwrite — default overwrite matching write_file semantics.

### 2.3 Grammar / mode

- Declare tool only when flag on and mode allows writes (same filter as write_file).
- Prompt nudge (short): if a full file was drafted in think, prefer `commit_think_block` over pasting into `write_file`. Still allow write_file.
- Mask: tool name must be openable only post-think (existing phase rules should already block tools in Think — verify with a test).

### 2.4 Observability

Emit an event on harvest: `think_blocks` count, sizes, shas (not full content).  
Emit on commit success/fail. Needed for L3 token accounting.

## 3. Files to touch (expected)

- `src/tools/registry.cpp` — declare + handler (reuse write primitives; do not fork a second write stack)
- Turn/agent state — harvest storage (likely `src/loop/` near grammar advance / turn end)
- `src/model/grammar.*` only if phase/tool legality needs an explicit assert
- Prompt/persona string if there is a central conventions block
- Tests under `tests/tools/`, `tests/loop/`, maybe `tests/model/` for fence parser

Prefer a small pure `extract_fenced_blocks(string_view) → vector<Block>` in a testable .cpp/.hpp with **no** GPU.

## 4. Milestones + tests (must pass in order)

### M0 — Fence parser unit tests (no model)

Cases: one fence; lang tag; multiple; unclosed dropped; empty dropped; content with internal backticks if you support it (or explicitly unsupported); exact sha of known fixture.

### M1 — Tool seam (no GPU)

Fake turn state with 2 blocks → commit writes exact bytes.  
Bad block_id / bad sha / sandbox / missing prior read (if required) → ToolError, disk unchanged.

### M2 — Grammar / phase

Tool illegal in Think; legal after think-close in a harness that drives TurnGrammar.  
No empty-mask / abandoned-storm from declaring the tool.

### M3 — Realmodel smoke (flag on, one short prompt)

Prompt that induces a small fenced file in think then a commit call.  
Assert: file bytes sha == block sha; tool call text does **not** contain the full file body.  
Record decode token count vs a write_file baseline if easy.

### M4 — Regression

Flag **off**: existing tool/loop tests + a quick agent smoke unchanged.  
Do **not** run full Aider 49 as a gate for v1.

## 5. Non-goals (v1)

- Auto-write on think close
- Mid-think tool execution
- Diff/patch from think
- Multi-file magic from one fence
- LoRA / Swift studio (unrelated)

## 6. Report back

1. Flag name + default  
2. Parser + tool API  
3. Test list green (M0–M2 required; M3 if weights available via `LMP_QWEN_DIR`)  
4. What you deliberately did not do  

## 7. Why this lines up

Sean measured wasted decode when Qwen retypes think-code into tools. He chose harvest-after-think + explicit tool over mid-think tools or prompt-only. Safety over cleverness.
