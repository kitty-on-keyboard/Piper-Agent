# piper init, without touching Godoer briefs

Gemini. Piper CLI only. Do not stash or reset the dirty tree. Do not edit
Godoer. Do not invent a default webhook host. Do not write `AGENTS.md`,
`GEMINI.md`, or `CLAUDE.md`. Godoer owns those. A second init must not wipe
or duplicate them.

## Why

Cursor and Gemini load the Godoer briefs in the folder they open. They will
not open `docs/AGENT_WAKE.md` on their own, and they often skip `--help`.
Piper still has to tell the parent the contract. It does that in two places
it owns, never in Godoer's files.

## What to add

1. `piper worker run` prints the parent contract to stderr before the sidecar
   starts, every time, attached or not. One short paragraph. The launching
   agent sees it because it started the command. This is the reliable path.
   `piper init` is optional.

2. `piper init` and `piper init .` write only:
   - `PIPER.md` — the contract from `docs/AGENT_WAKE.md`
   - `.cursor/rules/piper-parent.mdc` — the same short rule, for Cursor

   Do not create or edit `AGENTS.md`, `GEMINI.md`, `CLAUDE.md`, `.mcp.json`,
   or anything under `game/`.

   Print one line: `piper init: PIPER.md and .cursor/rules/piper-parent.mdc.
   Godoer briefs left alone. Stay attached, or pass --orch-webhook.`

3. The stderr paragraph on `piper worker run`:

```
piper: you are the parent. Stay attached and read the exit and result.json.
Do not detach unless you pass --orch-webhook. No default URL. Events:
ask (write answer.json), done, stalled (not success), died (do not relaunch).
See PIPER.md if present.
```

## Where

`scripts/piper_worker.py` only.

- Add an `init` subcommand next to `worker` in `main`. No sidecar.
- Print the parent paragraph in `main` after the webhook check and before
  the sidecar exec (the block that already prints the one-line `agent path:`
  line). Print even when `PIPER.md` is missing. Stderr, not stdout.
- `PIPER.md` body is the text of `docs/AGENT_WAKE.md`.
- `.cursor/rules/piper-parent.mdc` is a Cursor rule with the same short
  paragraph. Create `.cursor/rules/` if needed. Do not touch `.cursor/mcp.json`.

## Done when

- `piper init` in a godoer project does not change `AGENTS.md`, `GEMINI.md`,
  or `CLAUDE.md`.
- Init twice does not duplicate `PIPER.md` or the cursor rule.
- `piper worker run` prints the paragraph before the sidecar starts, even
  when `PIPER.md` is missing.
- Detached launch with no webhook still exits before the sidecar starts.
