# Defaults ON + Advanced toggles — handoff for Grok (Cursor)

Flip **commit-from-think** and **KV shadow-compact** to **default on**, expose both in the sidebar settings drawer under an **Advanced** section, and keep **env overrides** for bakeoff / headless.

Do **not** change compaction quality, shadow-swap algorithm, or commit_think_block semantics. This is wiring + defaults only.

## 0. Non-negotiables

- Repo: `/Users/dev/Desktop/seans_projects_local/LM_Pipe_2`
- Branch: `fix/greenfield-write-caps` (dirty tree — **do not clean/stash/reset**)
- Flags today are **env-only and default OFF**:
  - `LMP_COMMIT_THINK=1` → `WorkspaceContext::commit_think` (`src/surface/session.cpp` ~150)
  - `LMP_SHADOW_COMPACT=1` → `AgentConfig::shadow_compact` (`src/surface/sidecar.cpp` ~740)
- Struct defaults today: `commit_think = false` (`src/tools/registry.hpp`), `shadow_compact = false` (`src/loop/agent.hpp`)
- Product decision (Sean 2026-09-06): **default ON** for normal IDE use; still easy to turn off for fair benches
- Related: `docs/COMMIT_THINK_BLOCK_HANDOFF.md`, `docs/KV_COMPACTION_SHADOW_SWAP_HANDOFF.md` — do not re-litigate those designs
- After protocol schema edits, regenerate with `scripts/gen_protocol.py` (do not hand-edit `protocol.generated.ts` / `protocol_generated.hpp`)

## 1. Goal

| Knob | IDE default | Sidebar | Env override (bakeoff) |
|------|-------------|---------|------------------------|
| Commit closed think fences (`commit_think_block`) | **on** | Advanced toggle | `LMP_COMMIT_THINK=0\|1` |
| Shadow-swap after compaction / collapse | **on** | Advanced toggle | `LMP_SHADOW_COMPACT=0\|1` |

Precedence (must be documented in code comments once):

1. If env var is set to exactly `0` or `1` → that wins (headless / bakeoff)
2. Else if start-message / `RunSettings` field is present → that wins
3. Else C++ struct default (**true** after this change)

Unset env + no field → on. That matches “defaults on” without breaking old clients that omit the new fields (once struct defaults flip).

## 2. Protocol

Edit `protocol/schema.json` → `RunSettings` add two bools (snake_case on the wire, same as peers):

```json
{
  "name": "commit_think",
  "type": "bool",
  "_comment": "When true, harvest closed markdown fences after </think> and declare commit_think_block. Default true for IDE; set LMP_COMMIT_THINK=0 for bakeoffs that must not see the tool."
},
{
  "name": "shadow_compact",
  "type": "bool",
  "_comment": "When true, warm a post-compaction/collapse KV off the live path and swap when identity checks pass. Default true; LMP_SHADOW_COMPACT=0 forces classic full re-prefill for A/B."
}
```

Then run the generator (`python3 scripts/gen_protocol.py` or whatever the repo’s documented regen step is — follow existing README / script header). Confirm:

- `extension/src/protocol.generated.ts` gains `commit_think` / `shadow_compact` on `RunSettings`
- `src/surface/protocol_generated.hpp` matches

## 3. Sidecar / session

### 3.1 `shadow_compact`

In `src/surface/sidecar.cpp` where `config` is filled from the start message (near `context_budget_tokens` / current `LMP_SHADOW_COMPACT` block ~740):

- Default `config.shadow_compact = true` via `AgentConfig` default in `src/loop/agent.hpp`
- If message has `shadow_compact`, set from `bool_field`
- Then apply env override: if `LMP_SHADOW_COMPACT` is exactly `0` → false; exactly `1` → true; otherwise leave

Remove the old “env alone can turn on” path that ignored settings.

### 3.2 `commit_think`

Today only `session.cpp` reads env when building `WorkspaceContext`. That is too late / too narrow for IDE settings.

- Default `WorkspaceContext::commit_think = true` in `src/tools/registry.hpp`
- Thread the start-message / session config bool into registry construction:
  - Prefer: store `commit_think` on the same config path as other run settings, then when `session.cpp` builds `wctx`, set `wctx.commit_think` from that config (then env override)
  - Or: read `commit_think` from the start message in the same place other RunSettings are applied, stash on session, apply in `session.cpp`
- Env override identical to shadow: `LMP_COMMIT_THINK=0|1` wins when set

Update comments that say “default off” / “only when LMP_COMMIT_THINK=1” in `registry.hpp`, `registry.cpp` (`commit_think_block` registration), and any mode_brief / docs strings if they claim off-by-default.

### 3.3 Headless scripts

`scripts/drive.py` / `scripts/agent_eval.py`: optional flags are nice but **not required** if env works. Document in script `--help` or a one-line comment that bakeoff fairness uses:

```bash
LMP_COMMIT_THINK=0 LMP_SHADOW_COMPACT=0 ...
```

Do not flip bakeoff harness defaults without an explicit ask — leave agent_eval as-is unless it already sets these envs.

## 4. Extension settings + Advanced drawer

Mirror `lmPipe.speculativeDecoding` / auto-approve toggles.

### 4.1 `extension/package.json` `contributes.configuration.properties`

```json
"lmPipe.commitThink": {
  "type": "boolean",
  "default": true,
  "description": "Harvest closed fences from thinking and expose commit_think_block so the model can commit exact bytes by block id. Advanced sidebar toggle; LMP_COMMIT_THINK=0 overrides for benches."
},
"lmPipe.shadowCompact": {
  "type": "boolean",
  "default": true,
  "description": "After context compaction or collapse, warm KV off-path and swap instead of full re-prefill. Advanced sidebar toggle; LMP_SHADOW_COMPACT=0 overrides for benches."
}
```

### 4.2 `settingsFromConfig()` in `extension/src/extension.ts`

Add to the returned `RunSettings`:

```ts
commit_think: cfg.get<boolean>("commitThink", true),
shadow_compact: cfg.get<boolean>("shadowCompact", true),
```

### 4.3 Sidebar drawer UI (`extension/src/webview.ts`)

There is **no** Advanced section today. Add one at the **bottom** of `#drawer` (after system prompt / before view build, or after view build — either is fine; prefer just above `View build` so the build stamp stays last):

```html
<div class="set">
  <label>Advanced</label>
  <div class="toggle"><span>Commit from thinking</span><div class="sw" id="swCommitThink"></div></div>
  <div class="toggle"><span>Shadow KV after compact</span><div class="sw" id="swShadowCompact"></div></div>
  <div class="warnbox" id="advWarn"></div>
</div>
```

Wire like `swExec` / `swWrite`:

- `sw('swCommitThink', 'commitThink');`
- `sw('swShadowCompact', 'shadowCompact');`
- In the settings paint path, toggle `.on` from `settings.commitThink` / `settings.shadowCompact` (whatever key the host posts — match how `autoApproveExec` is mirrored into the drawer payload in `sidebar.ts`)

### 4.4 Host → drawer settings payload (`extension/src/sidebar.ts`)

Wherever the drawer `settings` object is built / posted (includes `autoApproveExec`, `speculativeDecoding`, etc.), include:

- `commitThink` / `shadowCompact` booleans from config

Allow-list any `put` / `msg.key` handler the same way other boolean keys are (do **not** open a generic “write any key” path — follow the existing allowlist pattern around speculativeDecoding).

No reload-model side effect needed (unlike speculative decoding). Next `lmp/start` / new run picks them up. If a run is already in flight, same as other run settings: take effect on the **next** start (call that out in the Advanced warnbox one-liner if helpful: “Applies on the next run.”).

## 5. Acceptance

1. Clean build of sidecar + extension; `gen_protocol` regenerated and committed with the schema change
2. Fresh IDE (or reload window): both Advanced toggles show **on** with empty/default settings
3. Start a short run with toggles on → event / tool list shows `commit_think_block` available when thinking produces a closed fence path; after a forced compact (small `--context-budget` via drive.py or a long chat), journal/events show shadow path when applicable (reuse ≫ 0 or existing shadow events — do not invent new metrics)
4. Flip both toggles **off**, new run → no `commit_think_block` in tools; compact behaves like classic re-prefill (reuse ~0) when you force compact
5. With toggles **on** in settings but `LMP_COMMIT_THINK=0 LMP_SHADOW_COMPACT=0` in the sidecar process env → both features **off** (prove env wins)
6. With toggles **off** but env `=1` → both **on**
7. Package + install extension into Antigravity the same way as 2.0.1 if that is still the install path; bump patch version if you normally do on packaging

## 6. Out of scope

- Mid-turn schema withhold when harvest empty
- Prompt nudge against post-commit `write_file`
- Fixing `test_spec_cache`
- Changing 75%→35% compact policy
- Bakeoff scorer changes
- Renaming tools or events

## 7. Done when

Sean can leave both Advanced toggles alone and get commit-think + shadow-compact on every IDE run; can turn either off from the drawer; can force both off with env for Piper vs Cline fairness without editing settings.json.
