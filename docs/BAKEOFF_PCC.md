# Cook-off: the agent context database (`pcc-test-1`, `context-db`)

Judged 2026-08-02. Twelve entrants in a `pcc-test-1` cook-off, plus fourteen
from a `context-db` round on 2026-07-31, read for their architecture only.

**Verdict: nothing adoptable as code, three ideas worth taking, and one whole-category
failure that is the most useful thing in the corpus.** The result is `src/pcc/`.

Read this before re-reviewing either repo.

## What the brief asked for

The `pcc-test-1` brief pinned three pillars, and pinned them tightly enough that all
twelve converged on the same file layout:

1. an artifact compression layer — SHA-256 CAS, zstd, `diff-match-patch` deltas;
2. a hybrid memory store — LanceDB vectors plus a SQLite entity-relationship graph, with
   a background decay/compaction thread;
3. a native MCP interface — `store_memory`, `search_knowledge`, `create_artifact`,
   `get_artifact_diff`, `prune_context`, and the resource URIs `memory://entities/{id}`
   and `artifact://snapshots/{hash}`, over stdio and SSE.

Convergence that complete means the deviations carry the information, not the agreements.

## The category failure: invented embeddings

Eleven of the twelve implemented semantic search. **Six of those eleven invented the
embedding**, counted from the branches:

| entrant | "embedding" |
|---|---|
| `entrant-10` | `[0.1] * 384` — a constant, so every vector is identical |
| `entrant-2` | SHA-256 digest bytes scaled to [-1, 1] |
| `entrant-3` | `np.random.seed(abs(hash(text)))` then `np.random.rand(128)` |
| `entrant-7` | `np.random.seed(hash(text))` then `np.random.randn(128)` |
| `add-mcp-agentic-db` | "deterministic mock vector based on text length and ascii values" |
| `feat/cpp20-hybrid-memory` | sum of character codes into 128 buckets, normalised |

Each of these is a hash, not an embedding: it is deterministic, so the same text retrieves
itself, and *unrelated to meaning*, so everything else comes back in arbitrary order. The
failure is invisible from the outside — the tests pass, the tool returns rows, the rows
look like rows.

Two entrants then built on top of it. `entrant-3` runs an O(N²) all-pairs
`np.allclose(vec1, vec2)` over the table and calls it "semantic clustering deduplication";
against random vectors it can only ever match byte-identical text. `entrant-6`'s
compaction has the same shape and is honest enough to say so in a comment
(`deduplicate by exact text match`).

The five that used a real model (`sentence-transformers`, `all-MiniLM-L6-v2`) are
`entrant-1`, `entrant-6`, `entrant-8`, `entrant-9` and `pcc-database-agent`. That is the
line worth drawing through the corpus: the brief said "semantic", and half the field
delivered something that has the shape of semantic search and none of the substance.

**This is why `src/pcc` ships with no embedding at all rather than a placeholder one.**
BM25 over FTS5 is weaker than a good embedder on paraphrase and stronger on identifiers,
paths and error strings — and, decisively, it is a retrieval quality you can *measure*
rather than one you have to trust.

## The ideas worth taking

**1. The parent artifact as a compression dictionary (`entrant-8`).** The only genuinely
good implementation idea in the corpus, and the only entrant not to reach for
`diff-match-patch`:

```python
zdict = zstd.ZstdCompressionDict(base_content)
compressed = zstd.ZstdCompressor(dict_data=zdict).compress(content)
```

Binary-safe, one pass, no patch format to parse. The `diff-match-patch` alternative that
the other seven chose is text-only: `entrant-6` catches the `UnicodeDecodeError` and
silently falls back to a whole copy, and `entrant-3` decodes with `errors='replace'` and
**corrupts the artifact**. Taken, with zlib's `deflateSetDictionary` in place of zstd, so
the store adds no dependency that is not already in the macOS SDK.

**2. Bi-temporal records (`context-db`, all fourteen).** Every record carries `valid_from`
/ `valid_to` — when the fact was true — separately from `system_time` — when it was
recorded. It is a 1990s data-warehousing idea and it is the single most valuable thing in
either repo for an agent, because an agent's notes go stale silently. Taken, and it is the
spine of `src/pcc/store.hpp`.

The `context-db` *implementations* are unusable here: `io_uring`, `eBPF`/libbpf, `memfd`,
persistent-memory tiering, and CRDT sync are all Linux-specific, which is the trap
recorded in `docs/BAKEOFF_MCP.md` and hit again. The model is portable; none of the code
is.

**3. Auto-linking on insert (`feat/cpp20-hybrid-memory`).** On store, kNN the existing
index and write graph edges weighted `1 / (1 + distance)`, so the knowledge graph builds
itself instead of waiting for the agent to declare relations. Genuinely good, and **not
taken yet** — it needs a real embedder to mean anything, and against that entrant's own
character-sum vectors it links documents by length. Recorded here so it is not lost.

## What nobody did

- **Nobody bounded the delta chain.** Nine branches carry a parent/base delta path, and
  every one chains a revision onto whatever base it is handed and reconstructs by
  recursion. A file edited forty times becomes a forty-link chain; reading the newest
  revision costs forty decompressions, and one damaged link loses every revision after it.
  `src/pcc` bounds it at `kMaxChainDepth` and re-bases, which is standard packfile
  practice.
- **Nobody verified a reconstruction.** No entrant re-hashes the reassembled content
  against the hash that was asked for, so a corrupt chain returns plausible bytes.
- **Nobody made compaction reversible.** Every entrant treats `prune_context` as delete.
  `entrant-8` deletes any memory whose decay score falls below 0.2; `entrant-10` deletes
  anything older than 30 days. For an agent this is backwards — the whole reason to have
  a store is that the *prompt* has to forget and the *store* must not.
- **The background decay/compaction threads are mostly ceremony.** Read directly:
  `feat/cpp20-hybrid-memory` sleeps ten seconds in a loop around a comment reading
  `// Placeholder for decay logic`; `entrant-10`'s compaction body is `pass`; `entrant-6`
  says in its own docstring that it "is a stub"; `entrant-1`'s `compact_memory` runs
  `VACUUM` and nothing else. Four confirmed by reading; the remainder were not audited
  one by one, so no count is claimed here.
- **Nobody budgeted retrieval.** Every `search_knowledge` returns a fixed `limit=5` rows.
  The scarce resource for an agent is prompt tokens, and a row count is not a token count.

## One bug worth naming

`entrant-8` blends recency into ranking like this:

```python
r['final_score'] = r['_distance'] * r['decay_score']
return sorted(results, key=lambda x: x['final_score'])
```

`_distance` is a distance (lower is better) and `decay_score` is freshness (higher is
better, 1.0 when new, floored at 0.1). Multiplying and sorting ascending means a **fresh**
memory scores *higher* — i.e. worse — than a stale one at the same distance. The intended
ranking is exactly inverted, and nothing about the output would look wrong.

`src/pcc/recall.cpp` fuses *ranks* rather than scores for this reason: reciprocal rank
fusion only compares positions, so there is no scale to get backwards.
`recall_prefers_the_fresher_of_two_equally_relevant_facts` in `tests/pcc/test_pcc_recall.cpp`
is the check that would have caught it.

## Provenance

Entrants were read from the branches directly, not from the PR descriptions, which
overstate consistently — `feat/cpp20-hybrid-memory`'s body describes a hybrid store whose
`create_artifact` and `prune_context` handlers both return the string
`"Not implemented"`. That matches the lesson already recorded for `mlx-redo`: never take a
Jules PR body at face value, open the file.

```bash
gh pr list --state all
```
