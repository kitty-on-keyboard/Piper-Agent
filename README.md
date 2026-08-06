# LM_Pipe v2

A local coding agent for Apple Silicon: a VS Code extension plus one native sidecar that
loads a **single** Qwen3 model in-process via MLX and drives a tool-using ReAct loop
against the user's workspace. **Product scope:** Mac-local Qwen/MLX, one model loaded,
no subagents, no second inference server.

**Status: all nine phases landed.** The model runs in-process on MLX and completes a
grammar-accepted turn at 22.9 tok/s. See [docs/PHASES.md](docs/PHASES.md) for every
measurement and the places the spec was overruled. Remaining gaps include T2 containers;
model-free speculative decode exists but stays **opt-in** (`LMP_SPECULATIVE=1`).

The build order in the spec is not advisory. Phases 0–2 come before any model code,
because v1 built the loop first and spent the next month discovering what it could not
measure.

## Quick start

```bash
cmake --preset dev && cmake --build --preset dev -j8 && ctest --preset gate
```

The gate must be green to merge. Nothing lands red.

## The gate

`ctest -L gate` runs with no model, no network and no workspace. Its contents are pinned
in [tests/gate/gate_manifest.txt](tests/gate/gate_manifest.txt) by **count and by name,
recorded separately** — adding a test fails the count, renaming one fails the names, and
neither can be repaired by editing the other half. That redundancy is the whole point: v1
selected its gate with `ctest -E realmodel` for months, `-E` excludes by name, the pattern
matched nothing, and every "48/48" it printed was a count of a set nobody had checked.

Real-model tests carry the `realmodel` label, are excluded from the gate, and **must never
run in parallel** — concurrent large model loads exhaust unified memory and take the
machine down. `ctest --preset realmodel` pins `jobs: 1` for that reason.

## Gates

```bash
./scripts/run_ratchets.py --root .              # the gates
./scripts/run_ratchets.py --root . --self-test  # prove each one can go red
```

Four gates, configured in [scripts/ratchets.json](scripts/ratchets.json): layer include
direction, dead code, protocol drift (§4.4), tool honesty (§6.3).

Every one checks a **property** — the dependency graph points one way, a declared symbol
is used, both sides of the protocol come from one schema, a tool description names tools
that exist. A gate here has to be a thing that is true or false of the code, whose false
case a reviewer would call a defect on its own.

There was a fifth, prose correctives (§9.2), which required every corrective arm in the
loop to contain a mechanism rather than a composed sentence. It went with the corrective
machinery itself on 2026-08-05 — the loop no longer steers the model at all, so there are
no arms to inspect. If steering ever returns, the gate must return with it.

There was a sixth that counted lines — 800 per file, 80 per function, nesting 3. It was
removed on 2026-08-02 and **is not coming back in any form**. What it produced was code
shaped to satisfy it: an API that swallowed its own error message to buy back a line,
files split on length instead of responsibility, and a sidecar sitting at exactly 800/800.
Good design is enforced by review; a budget only moves the mess around. If a file is
badly organised, say so about its organisation.

Two of its three limits also **never ran**. The function scanner only began a body at
brace depth 0, and `namespace lmp::x {` puts every file in `src/` at depth 1 from its
first line — so it saw zero functions in 107 of 149 files, and its self-test went green
against a probe with no namespace, a shape nothing in the repo has. Only the file cap was
ever live. Worth stating because it inverts the obvious defence of a size limit: nothing
ever tripped the function cap because the function cap was not looking.

Exemptions are the other half of a gate's honesty. `dead_code_exempt` is **empty** — the
sixteen protocol mirrors it used to list are one `generated` rule now, and the two that
remained had both outlived their own stated expiry conditions without anyone noticing.
A by-symbol exemption list is written when a gate is inconvenient and never read again.

A dormant gate does not report green. It reports `DORMANT`, and it **fails** the moment it
finds subjects — so the day a tool registry lands, the tool-honesty gate stops being
dormant by breaking the build, and someone has to implement it deliberately.

`--self-test` plants a violation for each live gate in a throwaway copy of the tree and
requires the gate to fire. A gate nobody has seen fail is a gate nobody has falsified, and
its greens mean nothing.

## The bake-off corpora

[bakeoff/](bakeoff/) is ported byte-for-byte from v1 and is the most valuable thing in the
repo. It arrives now, in phase 0, because an answer key written after the thing it grades
is not an answer key.

| | |
|---|---|
| `blast_radius/` | 187 cases + 42 held out, 10 published labelling rules, 11 blind entrants |
| `log_triage/` | 25 real source trees (971,544 bytes of committed build output) + 7 held out, **answer key written by the compilers themselves** |

Two things to read before quoting any number from it:

- [bakeoff/blast_radius/KEY_CORRECTIONS.md](bakeoff/blast_radius/KEY_CORRECTIONS.md) —
  three labels corrected in v2's key, with the argument; plus the finding that v1's own
  results table predates a fix the same document describes.
- `bakeoff/*/README.md` — verbatim v1, deliberately not edited, and therefore historical
  record rather than current measurement.

```bash
cmake --preset bakeoff && cmake --build --preset bakeoff -j8   # all 28 scoreboards
./build-bakeoff/bakeoff/blast_radius_score_e12
```

## Layout

```
src/platform/   L0  arenas, event log, SPSC channel, clock, fs
src/model/      L1  tokenizer, vocab, KV cache, sampler, grammar mask, MLX backends
src/tools/      L2  registry, schemas, structured results, sandbox, capability classifier
src/context/    L3  event store, tiering, compaction, prompt assembly
src/loop/       L4  the loop: classifier, repeat cache, HITL gate, steering, operator check
src/surface/    L5  JSON-RPC protocol, extension, webview UI, settings
```

Each layer depends only on those above it. The include direction is enforced by
`scripts/run_ratchets.py`, and the link direction by CMake — both, because a header-only
violation links fine and a link-only violation compiles fine.

## Hard constraints

Apple Silicon only. MLX in-process, **one model**, no inference server, **no subagents**.
Qwen3 only; other families are refused at load, not adapted. Speculative decode is
implemented and stays opt-in (`LMP_SPECULATIVE=1`). C++20, `-Wall -Wextra -Wpedantic
-Werror`, assertions on in every configuration. **The agent never touches git** — it
edits the workspace; staging, committing and branching are the user's.
