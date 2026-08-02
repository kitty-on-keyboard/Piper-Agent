# PCC — the Persistent Context Core

`src/pcc/`. A durable, searchable, bi-temporal store for everything an agent learns, with
an MCP server in front of it. Built 2026-08-02 out of `docs/BAKEOFF_PCC.md`.

## The problem it solves

LM_Pipe already has a good working-context manager. `src/context/context.hpp` runs four
tiers and compacts on trim, and the design comment is right that context is where agent
runs fail silently. What it does *not* have is anywhere to put what it drops.

`ContextStore::compact_oldest()` summarizes the oldest turns to one anchor line each and
then erases them:

```cpp
spans_.push_back(std::move(span));
recent_.erase(recent_.begin(), recent_.begin() + drop);
```

The summary it leaves behind even names the range it came from — `events 40-91` — which
reads like a provenance pointer and, before this component, pointed at nothing. The full
observation text was gone. Twenty turns later, a run that needed a detail the anchor line
had truncated could not get it back at any price.

The other half of the problem is cross-session memory, which was a 16 KiB markdown dotfile
of undated one-line bullets (`.lmp-memory.md`, `src/tools/memory_file.cpp`). No search, no
structure, and — the part that actually bites — no time: a note written last week that is
now false is indistinguishable from one written this morning that is not.

**An agent's effective context is not the size of its window. It is the size of what it
can get back.** That is what this component is for.

## Shape

```
sqlite.hpp     RAII over SQLite: Db, Stmt, Transaction (nestable, via SAVEPOINT)
diff.hpp       line-oriented unified diff
cas.hpp        content-addressed artifacts, delta chains, bounded
store.hpp      the bi-temporal item store, FTS5-indexed
recall.hpp     rank fusion and budgeted packing
mcp_server.cpp the store as an MCP server, on src/mcp
```

L1, beside `src/model` and independent of it. It links `sqlite3` and `z`, both of which
ship in the macOS SDK, so it adds no install step for a contributor or for CI.

## Bi-temporal, and why it earns its complexity

Every row carries two timelines:

- **valid time** (`valid_from`, `valid_to`) — when the fact was true of the world;
- **system time** (`system_time`) — when we recorded it.

Writing a key again does not overwrite. It closes the open row at the instant the new one
becomes valid and inserts the successor, so:

- retrieval defaults to now and returns the **current** truth, which is what stops a stale
  note from outranking what the agent can observe today;
- `context_history` answers "what did I believe, and when did I stop" — the question you
  need when a run went wrong an hour ago;
- system time makes a *replay* honest: a fact recorded at 15:00 about a state valid from
  13:00 is invisible to a query that asks what was known at 14:00.

Intervals are half-open, so no instant exists at which a key appears to hold two values.
There is no delete, only `forget`, which closes a fact without replacing it.

## Retrieval: BM25, and no fake embedding

Retrieval is BM25 over SQLite's FTS5, with the title weighted 4× the body.

There is deliberately **no embedding**, and no placeholder for one. Six of the eleven
cook-off entrants that implemented "semantic search" did it over character sums, hash
bytes, `[0.1] * 384`, or `np.random.seed(hash(text))` — retrieval that returns arbitrary
rows with total confidence and passes every test anyone wrote for it. BM25 is weaker than
a good embedder on paraphrase and stronger on the identifiers, paths, symbol names and
error strings that agent memory is mostly made of, and it can be measured rather than
trusted.

A real embedder is a genuine improvement and is not free — it needs a model this component
does not link, and `never-run-two-mlx-processes` rules out a second one alongside the main
model. `recall()` fuses **rank lists rather than scores** (reciprocal rank fusion, k=60)
precisely so that embedder can arrive later as a third list with nothing else changing.
Fusing ranks also sidesteps the scale problem that made one entrant's freshness factor
rank fresh memories strictly worse — see the bug named in `docs/BAKEOFF_PCC.md`.

Ties in the fused score are common rather than exotic (two lists over one candidate set
tie whenever their orders are reverses), and freshness breaks them.

## The budget is the argument

A search that returns twenty rows has not helped an agent; it has moved the problem. So
`recall()` takes a **token budget** and returns text that fits it:

- entries are packed best-first, and an entry too large to fit is skipped while packing
  **continues** — one 40 KB tool result must not evict the eight small facts behind it;
- everything withheld is listed as a `pcc://item/{id}` URI, so the caller can fetch what it
  needs in a second call;
- the pointer list is capped at a quarter of the budget, because thirty URIs and no content
  is worse than two facts and a truncated list;
- `tokens_used` is counted from the final string, not accumulated, so the number reported
  is the number a caller would measure.

Token counting is injectable. The default is a bytes/4 **estimate** and is named as one at
every call site; the sidecar passes the real tokenizer.

## Artifacts

SHA-256 content addressing, so identical content is stored once however often it arrives.
A revision may be stored as a delta against its predecessor, compressed with the
predecessor as a zlib **preset dictionary** — binary-clean and one pass, where the
cook-off's `diff-match-patch` approach was text-only and corrupted binary artifacts in two
entrants.

Delta chains are **bounded at `kMaxChainDepth` (8)**. At the bound the next revision is
stored whole, re-basing the chain. Reconstruction is therefore always at most 9 inflates,
and a damaged blob loses at most 8 revisions rather than a file's entire history. No
cook-off entrant bounded this; all nine that implemented deltas recursed without limit.

Every reconstruction is re-hashed against the hash that was asked for before it is
returned, so a corrupt chain reports corruption instead of handing back plausible bytes.

Measured by `tests/pcc/test_pcc_cas.cpp` on a synthetic source file, printed on every run:

```
9 revisions of a 22212-byte file: 199260 logical, 3481 stored (57.2x),
8 deltas cost 1504 bytes
```

The honest limit: zlib's dictionary window is 32 KiB, so for artifacts larger than that
only the tail of the base seeds the dictionary and the ratio decays toward plain deflate.

## Compaction stops being destructive

`ContextStore` gained one hook:

```cpp
using CompactionSink =
    std::function<void(const std::vector<TurnRecord>& dropped, std::size_t span_index)>;
void set_compaction_sink(CompactionSink sink);
```

It is called with the turns compaction is about to erase, **before** it erases them. A
sink rather than a store reference for two reasons: `render()` stays pure and diffable,
which is the property the whole prompt-purity argument rests on, and `src/context` keeps
knowing nothing about databases, so a `ContextStore` in a unit test still needs no fixture.
Unset, behaviour is exactly what it was.

With it wired, a trim keeps the summary in the prompt and the full text one query away,
reachable two ways: by the event range the summary printed (`context_rehydrate`), or by
searching for what you remember (`context_recall`). `tests/pcc/test_pcc_context.cpp` is
that scenario end to end.

## The MCP server

`pcc_mcp_server`, built on `src/mcp` — our own C++20 MCP server, not a third-party SDK.

```
context_recall          search everything, packed to a token budget
context_rehydrate       an event range back into full turns
context_remember        store a fact under a key; writing again supersedes
context_forget          close a fact without replacing it
context_history         every revision of a key, with its validity window
context_put_artifact    store a revision; dedup and delta are automatic
context_artifact_diff   unified diff between two revisions
```

Resources: `pcc://item/{id}` (resolves to CAS content for artifacts, body otherwise) and
`pcc://stats`.

Running it from Cursor or Antigravity against the same database the sidecar uses means a
conclusion reached in one is available in the other a second later. Add to an MCP client
config:

```json
{ "command": "/path/to/build/src/pcc/pcc_mcp_server", "args": ["--db", "/path/to/pcc.db"] }
```

## Verification

`ctest -L gate` — `test_pcc_cas`, `test_pcc_store`, `test_pcc_recall`, `test_pcc_context`.
Each file carries a `check_framework_can_still_fail_here` case, so its greens count
(S2.1.2). Both CI configurations were run locally per `reproduce-lm-pipe-ci-locally`:
38/38 under `dev` and under `asan`, both with MLX forced off.

The ASan run earned its keep immediately — it caught a dangling reference in the *test*
code (`store.current(...)->body` binds into a temporary optional; binding `const auto&` to
a subobject does not extend the temporary's lifetime) that the plain build read as freed
memory and compared equal anyway.

## Wiring

`src/surface/context_journal.hpp` is the adapter, and it lives at L5 because it is the one
place that can see both an L3 `ContextStore` and an L1 `pcc::Store` while the dependency
still points downward. The sidecar opens one per session at
`<workspace>/.lmp-context.db` — a dotfile beside `.lmp-memory.md`, so nothing has to
create a directory.

It **degrades rather than fails**: if the database cannot be opened, `open()` reports to
stderr and returns null, and the run proceeds exactly as it did before this existed. A
throw inside the sink is swallowed for the same reason. Journalling makes a run better;
wedging the agent because a disk is full would be a worse trade.

`Session::journal` is declared **before** `Session::ctx` on purpose: the sink captures a
raw pointer into the journal's store, so the context must be destroyed first.

## Not done yet

- **No embedder**, by choice — see above.
- **Nothing reads the store back into the prompt automatically.** The agent can reach it
  through the MCP tools, and `rehydrate()` exists and is tested, but no policy in
  `src/loop` decides on its own to pull a compacted span back. Deciding *when* an agent
  should spend tokens re-reading its own past is a real design question and guessing at it
  is how you get a loop that thrashes.
- **Auto-linking on insert** (kNN the index at write time, weight edges `1/(1+distance)`)
  is a good idea from the cook-off that needs a real embedder first.
- **No cross-session `remember` migration.** `.lmp-memory.md` still exists and still works;
  nothing reads it into the store.
