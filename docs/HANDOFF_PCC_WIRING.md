# Handoff: make PCC real — wire the durable context store into the agent

Paste this whole file as the opening prompt of a fresh session in
`/Users/dev/Desktop/seans_projects_local/LM_Pipe_2`.

## The one-sentence version

`src/pcc/` is a complete, tested, bi-temporal context store with budgeted retrieval — and
the agent cannot read a single byte of it. Wire it in, and the agent's effective context
stops being its window and starts being everything it has ever seen in this workspace.

## Read this first: the measured status, not the intended one

Do not trust the design docs about what is live. On 2026-08-03 I measured this:

```
$ sqlite3 ~/Desktop/Agent_testing/ResMon/.lmp-context.db "select kind,count(*) from item group by kind;"
(no rows)
$ grep -c '"kind":"compaction"' ~/Library/Logs/LM_Pipe/events.jsonl
0
```

That workspace has had several **full 80-turn runs** through it. The database has its
schema and **zero rows**. Two independent gaps produce that:

**Gap 1 — nothing reads it.** No tool in `src/tools/` or `src/loop/` calls
`pcc::recall()`. `context_recall` / `context_rehydrate` exist only inside the separate
`pcc_mcp_server` binary, which is **not staged into the VSIX** and needs a hand-written
`lmPipe.mcpServers` entry with an absolute `--db` path. Out of the box the store is
write-only.

**Gap 2 — the writes barely fire.** `src/surface/context_journal.cpp` attaches the store
as `ContextStore`'s **compaction sink only** (`ctx.set_compaction_sink(...)`). Nothing is
persisted until the prompt crosses `Agent::kCompactAtPercent` (75% of
`context_budget_tokens`). Long runs burn turns that each add little, so on real work the
threshold is never crossed and the store stays empty. The one component whose whole job is
"remember what got trimmed" has never been given anything to remember.

Fixing only Gap 1 gives you a recall tool that searches an empty database. Do both.

## Why this is worth doing properly

Sean's framing when PCC was built: *"make something that enables the model to have much
greater context capability."* The design answer, from `docs/PCC.md`, is that **an agent's
effective context is the size of what it can get back, not the size of its window** —
which is why `recall()` takes a *token budget* and returns what fits plus
`pcc://item/{id}` URIs for the rest.

Concretely, on this codebase's own observed failures: runs re-read the same files because
they forgot what was in them, re-derived facts they had already established, and lost
everything the moment a window closed. A working recall path is the difference between an
agent with a 96k window and an agent with a 96k *working set* over an unbounded history.

## The existing API — use it, do not reinvent it

`src/pcc/store.hpp`:

```cpp
namespace kind {
  kTurn = "turn";        // one exchange, verbatim        (append-only)
  kSpan = "span";        // a compacted range's summary   (append-only)
  kFact = "fact";        // a durable claim               (supersedable by `key`)
  kArtifact = "artifact";// a named revision of content   (CAS-backed via `hash`)
}

struct Record { session, kind, key, title, body, hash, first_event, last_event, valid_from };
struct Item   { id, session, kind, key, title, body, hash, valid_from, valid_to, ... };

class Store {
  std::int64_t append(Record);                    // append-only kinds
  // + supersede-by-key writers, current(), stats(), artifact_content()
};
```

`src/pcc/recall.hpp`:

```cpp
struct RecallRequest {
  std::string query;
  std::string session;          // EMPTY searches every session
  std::size_t token_budget = 1500;
  AsOf as_of;                   // bi-temporal; defaults to now
  int candidates = 60;
};
struct Recall { std::string text; std::vector<RecallEntry> entries;
                std::size_t tokens_used, included, pointers_only; };

Recall recall(const Store&, const RecallRequest&, const TokenCounter& = estimate_tokens);
Recall rehydrate(const Store&, std::uint64_t first_event, std::uint64_t last_event,
                 std::size_t token_budget, std::string_view session = {},
                 const TokenCounter& = estimate_tokens);
```

`Recall::text` is **already formatted to hand to a model**. The tool body is mostly
argument marshalling.

## Task 1 — native `context_recall` and `context_rehydrate`

Register them as **native tools backed by the session's own `pcc::Store`**. Not MCP.

Why native rather than shipping the MCP server:
- No configuration. `docs/` records the standing rule that "works when configured" does
  not count for this product — the model is meant to be in the box.
- No process hop, no JSON round-trip per call.
- **It gets the real tokenizer.** `TokenCounter` defaults to `bytes/4`, an estimate the
  header is careful to name as one at every call site. In-process you can pass
  `[&tok](std::string_view s){ return tok.encode_content(std::string(s)).size(); }`.
  The budget is the entire point of `recall()`; an out-of-process server cannot make it
  honest. **This alone justifies the native path** — say so in the comment.

Mechanics:
- `Registry::declare()` is **private**. `declare_remote()` is public but is explicitly for
  out-of-process tools and marks them accordingly — do **not** smuggle these through it.
  The intended route, per the note already in `src/tools/registry.hpp:134-142`, is a member
  function **defined in its own translation unit**, e.g. `src/tools/context_tools.cpp`
  declaring `Registry::declare_context_tools(pcc::Store&, TokenCounter)`. Add the
  declaration to the class, keep the definition out of `registry.cpp`.
- Wire it from `src/surface/sidecar.cpp` in `start_mission`, right after the journal opens
  (~line 591) — the store must exist first. Guard on a null journal: a run whose store
  failed to open must still work, with the tools simply absent. That is the same
  degrade-don't-fail contract `ContextJournal::open()` already documents.
- Neither tool mutates the workspace or executes commands: `mutates_workspace = false`,
  `executes_commands = false`, `irreversible = false`. They must be readable at **T1** like
  `search` and `locate_symbol` — do not route them through `tier_for()`.
- Reuse the tool descriptions from `src/pcc/mcp_server.cpp:88-126` almost verbatim. They
  are well written and already tell the model *when* to reach for each.

## Task 2 — write turns as they happen

Today: compaction sink only. Change to append each turn to the store **as it is recorded**,
and keep compaction as the step that additionally writes the `kSpan` summary.

Design points to get right:
- `ContextStore::add_turn()` (`src/context/context.hpp:206`) is the natural hook, but
  `ContextStore` is L3 and must keep knowing nothing about databases — that layering is
  deliberate and `context_journal.hpp` explains why the adapter lives at L5. Add a second
  sink (a `TurnSink`) alongside `CompactionSink`, or drive it from the loop. **Do not**
  `#include` pcc from `src/context/`.
- `turn_body()` in `context_journal.cpp:12-30` is the serializer; reuse it unchanged.
- Set `first_event` / `last_event` from the `TurnRecord` — `rehydrate()` keys on that range
  and it is what makes the "events 40-91" pointer in a compacted summary resolvable.
- Appending on every turn means a row per turn per run. That is the intended volume for an
  append-only kind, but check `Store::stats()` growth on a long run and confirm the FTS5
  index keeps up.

## Task 3 — decide the session partitioning, deliberately

`ContextJournal::open(workspace, id, ctx)` is called with `id` = the **JSON-RPC request id
of `lmp/start`**, i.e. the run id (`sidecar.cpp:592`). A follow-up via `lmp/message` reuses
`session.ctx` and does not re-open the journal, so a conversation keeps one session id —
but **every new mission is a new partition**.

So: `RecallRequest::session` empty searches *all* history in that workspace; set to the
current session it searches only this conversation. This is the single most important
product decision in the task and it is currently unmade.

**My recommendation:** default `context_recall` to searching **all sessions** (leave
`session` empty), and expose an optional `this_session_only` parameter. The whole value
proposition is "what did I learn about this repo last week", and recency fusion already
biases toward the current run. Verify that claim on real data before committing to it —
`recall()` fuses ranks, so an old-but-relevant item should surface without drowning the
present.

## Task 4 — reconcile with the existing `remember` tool

`Registry::remember_fact()` (`registry.cpp:846`) appends to `.lmp-memory.md` — 16 KiB of
undated one-line bullets, no search, no time. PCC's `kFact` kind is the strictly better
home: supersedable by key, bi-temporal, searchable.

Do **not** silently break the existing tool or orphan the existing file. Either back
`remember` with `kFact` while keeping the markdown as a human-readable mirror, or leave it
alone this pass and note it. Say which you did and why.

## Verification — required, and there is a trap

**Run BOTH preset builds.** The `dev` preset alone is not enough for this component:

```bash
cmake --preset dev  -B /tmp/nomlx -DLMP_MLX_PYTHON=/usr/bin/false
cmake --build /tmp/nomlx -j8 && ctest --test-dir /tmp/nomlx -L gate

cmake --preset asan -B /tmp/asan  -DLMP_MLX_PYTHON=/usr/bin/false
cmake --build /tmp/asan -j8 && ctest --test-dir /tmp/asan -L gate
```

`-DLMP_MLX_PYTHON=/usr/bin/false` is mandatory — CI has no MLX and compiles the `#else`
half of `src/model/mlx_backend.cpp` that a local build never touches. The asan preset
picks up a *different* MLX than dev if you let it probe, producing a confusing unrelated
`mx::device_info` error.

**The ASan trap, on this exact component:** a previous pass had
`store.current(...)->body` binding a `const auto&` into a temporary `optional`. `const
auto&` on a *subobject* does not extend the temporary's lifetime. The plain build read the
freed memory and compared equal anyway — every test passed. **Only ASan caught it.** If you
write `const auto& x = store.current(k)->field;` you have probably just written that bug.

`tests/gate/gate_manifest.txt` pins the test count and is the authority — update it if you
add a gate test, or `test_gate_manifest` fails on the count.

**Prove it end to end, not just in unit tests.** The failure mode this task exists to fix
is precisely "the unit tests pass and the feature does nothing in a real run." Required
evidence:

1. Run a real mission in `~/Desktop/Agent_testing/` through the editor.
2. `sqlite3 <workspace>/.lmp-context.db "select kind,count(*) from item group by kind;"` —
   must show `turn` rows climbing during the run, without waiting for compaction.
3. Start a **second, later** mission in the same workspace and confirm the agent can
   `context_recall` something that only exists in the first mission's history.

Point 3 is the whole feature. If it does not work, nothing else counts.

## Ground rules

- **Do not re-litigate the no-embedding decision.** BM25 over FTS5, RRF fusion at k=60, by
  design: a real embedder needs a model this component deliberately does not link, and one
  19 GB model on a 48 GB host means there cannot be a second MLX process. `recall()` fuses
  *rank lists* precisely so an embedder drops in later as a third list. `docs/PCC.md` and
  `docs/BAKEOFF_PCC.md` carry the reasoning.
- Match the surrounding comment style: comments here explain *why*, and cite the measured
  failure that motivated the code. Several of the sharpest bugs in this repo were found
  because a comment recorded what had already gone wrong. Keep that up — and if you find
  the design docs disagree with the code, the code is the truth and the doc is a bug.
- `git log` is the record; the commit messages carry reasoning worth reading before
  changing anything they touch.
