# Parallelism plan, and what the cook-offs are worth

Written 2026-08-01, after reviewing all 10 landed entrants of `cat-collector-king/agent-cookoff`
and all 10 of `cat-collector-king/mcp-transport`.

This started from one question: **can we put more SPSC queues in the pipeline to kill
bottlenecks?** The answer is no for the decode chain and yes for exactly one other place.
The rest of this document says which ideas survive, which do not, and in what order to take
them.

---

## The finding that governs everything else

Autoregressive decode is `forward_N -> token_N -> forward_{N+1}`. That is a hard data
dependency, so stages cannot overlap and extra queues only add latency and synchronisation.

The agent cook-off is an accidental controlled experiment on this. Ten agents were told to
build a multi-stage SPSC pipeline; it degraded in every one:

- **E7** wired four stages and never closed the loop. It generates exactly one token.
- **E10** makes the model thread block on a two-ring round-trip per token, strictly worse
  than calling `sample()` inline, plus a 1 ms sleep capping it at 1000 tok/s.
- **E6, E8** call `sample_async(...)` and `.get()` on the next line -- a thread spawned per
  token purely to wait on it.
- **E3** gave up and ran the stages sequentially in `main`, with a comment conceding the
  queues are decorative.

Our own ceiling is already documented and is not a topology problem:
[`HANDOFF_PERF.md` -- "Why async_eval is not available"](HANDOFF_PERF.md). `mx::async_eval`
is worth ~20% and is structurally foreclosed because our sampler is host-side, which
mask-first constrained decode requires. No queue arrangement touches that.

**Corollary: parallelism must come from somewhere other than the decode chain.** Everything
below respects that.

---

## Workstream 1 -- Stream tokens to the surface

**This is a product gap, not a perf item. Do it first.**

`AgentObserver::on_token` (`src/loop/agent.hpp:132`) is named per-token but fires **once per
turn**, after `backend_.generate()` returns, with the fully decoded `reasoning` and
`assistant_text` blocks (`src/loop/agent.cpp:314`). The real per-token hook,
`GrammarSink::on_token` (`src/loop/agent.cpp:14`), only walks the grammar and emits nothing.
We measure and report TTFT, then show the user nothing until the turn ends.

All 10 agent-cook-off entrants stream. It is the one thing they uniformly have that we do
not, and a local agent that goes silent for a whole turn feels broken regardless of its
tok/s.

**DONE** (2026-08-01). `src/loop/token_stream.cpp`, tests in `tests/loop/test_token_stream.cpp`.
Verified against the real sidecar: **51 token notifications for one turn, mean 4 characters
each, first at 3.63 s** — where the old path emitted one message per channel per turn.

Two things this plan got wrong before the work started, both corrected by contact with the
code:

1. **UTF-8 was already solved, not "has to be written".** `QwenTokenizer::Stream` already
   wrapped frankentok's `StreamingDecoder` with exact split-codepoint handling, verified
   against this checkpoint. It existed and was used by exactly one test and nothing else.
   The real correction is subtler: **"does the concatenation match" does not test the
   hazard.** Byte-level BPE splits a character into fragments that add back up, so a naive
   per-token emitter passes that check. What breaks is the *individual* message — a
   fragment is not valid UTF-8 alone and cannot be JSON-encoded. Measured here: every token
   of a pure multi-byte string is such a fragment. The test asserts per-chunk validity.
2. **Thread placement.** Decode and emit must not run on the generation thread — not for
   CPU cost (microseconds against a 12 ms step) but because `write_line()` does
   `fwrite` + `fflush` to a pipe whose reader is the extension, so a slow reader would
   throttle generation. One SPSC hop after sampling. Grammar advance stays on the
   generation thread because it produces the next step's mask.

This is the *only* stage decomposition in the whole cook-off that is sound, and it is sound
precisely because it is downstream-only and never feeds back.

Two things it broke that had to be fixed with it: the sidecar's `write_line` needed a mutex
(the streamer thread and the run thread both write stdout, and the payload and its newline
are two calls), and the webview created a **new** assistant bubble and a **new** reasoning
disclosure per notification — invisible while the notification fired once per turn, and one
element per token afterwards.

---

## Workstream 2 -- Parallel dispatch of independent tool calls

**The real parallelism win, and the one the original question was reaching for.**

`src/loop/agent.cpp:353` executes a turn's batched calls **strictly in order**. The grammar
allows up to `kMaxCallsPerTurn = 4`, and the model batches them *because they are
independent* -- that is the entire reason the feature exists (`src/model/grammar.hpp:95`).
Four file reads take four times the wall time they need to.

### Why not a queue per tool

We have five tools: `read_file`, `read_slice`, `write_file`, `shell`, `plan`. SPSC means one
consumer, so a queue per tool serialises **same-tool** calls and parallelises
**different-tool** calls. Our canonical batch is four `read_file` calls -- the case
`agent.cpp:342` calls out by name. A per-tool queue would serialise exactly that and leave
the `write_file`, `shell`, and `plan` threads idle. It is backwards for this workload, it
dedicates threads to the three tools that must stay serial, and threads should scale with
concurrency demand rather than catalogue size.

### The design

Per-**call** concurrency through one bounded pool, not per-tool:

- Admit a call to the parallel path only if its `blast_radius` capability flags say
  read-only. The classifier already exists -- `risk_score()` in `src/loop/agent.cpp:40`
  reads `writes_outside_workspace`, `destroys_data`, `escalates_privileges` and the rest.
  Anything with a write, destroy, or privilege bit stays on the serial path, as do
  `Unparseable` and `PartiallyParsed` hints.
- `write_file` stays serial for write ordering and the irreversibility gate. `shell` stays
  serial behind HITL. `plan` is in-memory and instant; parallelising it is pointless.
- Results must be collected back in **call order** so history records, the deliverable
  ledger and the UI rows are unchanged. Parallelism is an execution detail; the transcript
  must be identical to the serial one.

### The primitive: NOT the MPMC ring, and this plan was wrong to specify one

**DONE** (2026-08-01). `src/loop/parallel_calls.cpp`, tests in
`tests/loop/test_parallel_calls.cpp` (gate). Confirmed on the real model: a turn batching
`calc.py` and `notes.txt` ran both reads under one generation.

This plan said to port `mcp-transport` PR #9 -- a faithful **Vyukov bounded MPMC queue**,
whose orderings I did check against the canonical algorithm and which is correct, including
the `pos + mask + 1` republish on dequeue. **It is not needed and was not used.**

The cap is four calls per turn, fixed by the grammar and known before dispatch starts. A
queue is the right structure for many items and unknown parallelism; for a bounded set of
four already in hand it adds a shared consumer to coordinate and buys nothing. Four threads,
joined in index order, is the whole implementation. The repo would have rejected the ring
anyway -- it has a `dead_code` ratchet, and an unused MPMC queue is exactly what that
ratchet is for.

Worth keeping the distinction: "you need MPMC the moment you want more than one consumer"
is true, and it still does not follow that you need a *queue*. The MPMC ring remains the
right answer if the toolbox ever grows a shared work pool with unbounded arrivals.

### What made it safe

`dispatch_call` is deeply stateful -- approver, verification ledger, deliverable ledger,
checklist, event log. None of it is safe from four threads. The split that made this work:
eligible calls run **only** `Registry::execute` concurrently, and every gate and every
ledger write stays on the agent thread, after, in index order. Eligibility is stated as the
properties that make the other branches unreachable (`!mutates_workspace`,
`!executes_commands`, `!irreversible`, not `plan`, registered) rather than as a list of tool
names, so a tool added later is excluded until someone declares it harmless.

Checked before relying on it: the read path has no mutable statics, no `chdir`, no shared
buffers -- `Registry::execute` is a map lookup and the handlers read `ctx_` fields fixed at
construction.

---

## Workstream 3 -- Small steals, individually cheap

| Idea | Source | Note |
|---|---|---|
| ~~Cache the opposite index in the SPSC hot path~~ | agent-cook-off E3, the only one | **DONE.** `spsc_channel.hpp` carries `head_cached_`/`tail_cached_`, with the stale-lower-bound argument written out at :48 and :66. Verified 2026-08-02 |
| De-duplicate the repetition-penalty history | E6, E9 | See below -- correctness, and it moves the baseline |
| Hold back a streamed tail only when it is short enough to still be a tag prefix | E7 | `len - last_open < 12`. Relevant once Workstream 1 streams text |
| Cursor plus one `substr` instead of substr-per-iteration | E8 | Same |
| Raw storage plus placement-new instead of `vector<T>` | E5 | Drops default-construct-every-slot; only worth it if `T` gets expensive |

**On the repetition penalty.** `src/model/sampler.cpp:50` iterates `recent` and applies the
penalty **per occurrence**, so a token appearing three times in the 64-token window gets
`l/p/p/p`. HuggingFace's reference gathers and scatters by index, applying it once per
unique id. E6 and E9 both de-duplicate; we do not. This is a genuine divergence from the
reference — **but it changes generated tokens and would move the pinned baseline.** Do not
fold it in silently. Land it deliberately, with a re-pin. (The baseline this section
originally cited, corpus 3/6 and holdout 1/4, is long superseded; read
`evals/agent/pins.json` rather than any number written here.) Still true 2026-08-02.

---

## Workstream 4 -- Speculative decoding

The only technique that breaks one-token-per-forward, and the one nobody in either cook-off
attempted. A small draft model proposes k tokens; the target verifies all k in one forward
pass.

Why it fits us specifically:

- It is **not** blocked by the host-side sampler that forecloses `async_eval`. Verification
  reads k logit rows at once; the grammar walks the accepted prefix afterwards.
- Our constrained decoding actively *helps* it. Inside `<tool_call>` the legal set is often
  near-singular, so draft proposals get accepted at a high rate exactly where agent output
  is most boilerplate-heavy.
- Memory is fine and this is **not** the two-process hazard: a 0.6B draft at bf16 is ~1.2 GB
  in-process against a 19 GB resident checkpoint on a 48 GB host. The standing rule is never
  to run two MLX *processes*; one process holding a small draft is a different thing.

> **CLOSED — this landed later the same day, in `a6eb981`.** Speculative decoding is
> wired, gate-tested with no GPU, and measured on the real agent loop: **+4.5% to +11%
> decode on turns where it fires, ~+3.8% overall, 84% acceptance**, OFF by default
> (`LMP_SPECULATIVE=1`, or `MlxBackendConfig::speculative`). The binding constraint turned
> out to be the TRIGGER rate, not the acceptance rate. See
> [HANDOFF_SPECULATIVE.md](HANDOFF_SPECULATIVE.md) — including two measurements that
> change the design, on `draft_probs` and on the ~4% TV between batched verification and
> sequential decode.
>
> Everything below this line is the assessment as it stood on 2026-08-01, kept because how
> the four blockers were cleared is the useful part. It is history, not a to-do list.

**NOT DONE — blocked, and deliberately not half-built.** Assessed 2026-08-01. Four
blockers, three of them structural:

1. **No draft checkpoint exists on this machine.** The only local models are the 19 GB Qwen
   target and a 15 GB `gemma-4-26B-A4B-it-QAT-MLX-4bit`. A draft must share the target's
   vocabulary, and Gemma's does not — our own tokenizer refuses it by family verification
   (S5.2), which is the check that exists precisely to stop this. Obtaining one is a
   multi-GB download and is Sean's call, not mine.
2. **`forward_logits` returns the last position only.** `src/model/mlx/qwen35_moe_model.hpp`
   slices `h` to `seq_len - 1` and states the contract in a comment: *"only the final
   position's logits are ever consumed downstream."* Verification needs logits for all k
   draft positions from one pass.
3. **The KV cache cannot roll back.** The only slicing in `src/model/mlx/kv_cache.hpp` is a
   front-trim for the sliding window. Rejecting draft tokens requires truncating the
   *tail* back to the accepted length, which does not exist.
4. **`KvLedger` is append-only by design.** `src/model/kv_cache.hpp` offers `append` and
   `clear` and nothing between them; the header calls append-only KV the natural model
   (S8.3). It would need a truncate-to-N that matches the cache's.

So this is not "wire up a draft model" — it is changing an explicit architectural contract
in three files that sit on the tuned hot path, to enable something whose payoff cannot be
measured here because blocker 1 means there is nothing to measure it with. Building it
blind is the exact failure `HANDOFF_PERF.md` warns about. Left undone on purpose.

**The variant worth costing first.** Prompt-lookup (n-gram) decoding drafts by finding a
matching n-gram in the existing context and proposing its continuation. It needs **no draft
model at all**, which removes blocker 1 entirely, and it fits agent workloads unusually well
because agent output constantly repeats the prompt — file paths, tool names, code being
edited, previously-read file contents. Blockers 2–4 still apply, so the model-layer work is
the same; but it turns a "download and integrate a second model" project into a
self-contained one, and it can be measured the day it lands.

**UPDATE 2026-08-01 — the payoff is now measured, and it constrains the design.** See
[MOE_ROUTING_FINDINGS.md](MOE_ROUTING_FINDINGS.md), from 16,914 real decode steps on the
production checkpoint. Two things this plan could not have known:

- **Fixed-length drafting is a NET LOSS on this model.** Expert-bandwidth speedup is 0.678x
  at draft length 4 and 0.454x at length 8 — a stock speculative decoder makes this model
  more than twice as slow. Only short drafts or a cumulative-probability stop rule win
  (1.250x at tau=0.8). Whatever gets built here must be adaptive; a fixed `k` is a
  regression, not a speedup.
- **Do not build expert-aware drafting.** Routing is 5.15x token-determined and genuinely
  predictable, which makes steering drafts by predicted expert cost look obvious. It was
  tested three ways and does not pay; the best-scoring variant turned out to be identical to
  fixed-length-1. Section 3 of the findings has the detail.

The drafter itself is no longer hypothetical either: `SuffixProposer`, amalgamated from the
`model-free-draft-proposer` cook-off, reaches 4.17 accepted tokens per call at 0.06 wasted
against the best entrant's 4.21 at 2.66.

---

## Rejected -- do not redo these

- **More SPSC queues in the decode chain.** Hard data dependency; ten independent attempts
  all degraded. See the top of this document.
- **A queue per tool.** Backwards for a five-tool box whose canonical batch is four calls to
  one tool. See Workstream 2.
- **A faster MCP transport for our tool path.** LM_Pipe_2 contains no MCP. The five tools
  are in-process C++ calls; `read_file` is a direct filesystem read, so there is no
  transport to speed up. The sidecar says so itself: *"this sidecar speaks the private
  lmp/* namespace; it is not MCP and does not claim to be"* (`src/surface/sidecar.cpp:642`).
  Our only real IPC is sidecar to Cursor extension over newline-delimited JSON on
  stdin/stdout (`src/surface/transport.hpp`), and even with per-token streaming that is
  ~80 messages/second -- four orders of magnitude from needing shared memory.
- **The `mcp-transport` entrants as transports.** M1 and M3 use `memfd_create`, `eventfd`
  and `futex`: **Linux-only, they do not build on macOS.** Only M7 (`shm_open`) and M10
  (file-backed `mmap`) are portable. Six of the ten use no platform syscalls at all and are
  in-process rings, arenas and slabs rather than transports. The MPMC ring in Workstream 2
  is the salvage; the rest is not applicable to this project on this OS.
- **The agent cook-off model code.** All 10 ran on Linux with no MLX. Every model
  implementation is stubbed -- `hidden_size` mocked to 1024, 2 layers, mock embeddings, mock
  final norm. The best of them reaches `topk(gate_logits, ...)` and then comments that a
  real implementation would gather. We have `switch_glu.hpp` doing the real gather and
  `gated_delta.hpp` for Qwen 3.5's gated delta attention, which none of them knew existed.

---

## Where MCP *would* actually pay

Worth separating from the above, because it is a real feature and not a perf fix: if
LM_Pipe should be able to use third-party MCP servers (filesystem, GitHub, Postgres, the
rest of the ecosystem), that is a genuine capability jump. At that point the toolbox holds
remote connections with real setup cost, and the per-tool-worker pattern rejected in
Workstream 2 becomes correct -- a worker owning a live connection per server is exactly how
an MCP client pool is built.

That is the case where the transport work becomes relevant. It is a roadmap item about
capability, not latency, and it should be scoped as such.

---

## Sequencing

1. **Workstream 1 (streaming).** Product gap, unblocks the visible half of the product.
2. **Workstream 2 (parallel read-only dispatch).** Real concurrency, primitive already
   written, safety classifier already built.
3. **Workstream 3**, folded in opportunistically -- except the repetition-penalty fix, which
   waits for a deliberate baseline re-pin.
4. **Workstream 4 (speculative decoding)** only once 1 and 2 are working, per the standing
   preference for connecting cut wires over improving working ones.
