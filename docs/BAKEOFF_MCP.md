# MCP cook-off: `mcp-client-cpp` and `mcp-server-cpp`

Judged 2026-08-02. 7 client entrants, 7 server entrants, all on `cat-collector-king`.
Both repos were created the same day and were not present in the 2026-08-02 sweep
recorded in the memory note, which is why that note says no MCP client cook-off exists.
It does now.

Branch → label mapping is in `bakeoff/mcp/ENTRANTS.txt`.

## Headline

**All 14 compile and run.** This is the first Jules round where no entrant is a stub —
the Linux/no-GPU constraint that wrecked `agent-cookoff` and `mlx-redo` does not bite
here, because MCP over stdio is pure POSIX + std. The brief was well chosen.

**All 14 implement the same ~15% of the protocol**, and it is the same 15% every time:
`initialize`, `notifications/initialized`, `tools/list`, `tools/call`, plus `ping` on
the servers. Zero entrants implement resources, prompts, completion, logging,
pagination, cancellation, progress, sampling, roots, or elicitation. The file layout is
identical across all 7 in each repo, so the brief pinned the structure again.

**All 14 hardcode `protocolVersion: "2024-11-05"`** and not one performs version
negotiation. That is the oldest of the three published revisions. Verified empirically:
against a server that answers `2025-06-18`, every client carries on as if it had got
what it asked for.

So breadth does not discriminate and neither does the happy path. What discriminates is
what happens off it, and that had to be measured.

## Method

Scoreboard written before reading any entrant. Servers are directly testable over
stdin/stdout, so `bakeoff/mcp/conform.py` grades 12 spec behaviours with one process per
check, so a hang in one cannot mask another.

The board was falsified before it was believed: `bakeoff/mcp/falsifiers/` holds a
correct reference server and a deliberately broken one carrying seven planted defects.
The correct one scores 12/12, the broken one 4/12, and **each planted defect is caught
by the check meant to catch it**. Without that, a clean sweep means nothing.

## Server results

| | S1 | S2 | S3 | S4 | S5 | S6 | S7 |
|---|---|---|---|---|---|---|---|
| initialize shape | P | P | P | **F** | P | P | P |
| no reply to notification | P | P | P | P | P | P | P |
| no reply to unknown notification | P | P | P | P | **F** | P | P |
| string id preserved | P | P | P | **F** | P | P | P |
| tools/list | P | P | P | **F** | P | P | P |
| unknown method → -32601 | P | P | P | **F** | P | P | P |
| bad JSON → -32700, survives | P | P | P | **F** | P | P | P |
| unknown tool → error | P | P | P | **F** | P | P | P |
| tools/call → content[] | **F** | P | P | **F** | P | P | P |
| ping | P | P | P | **F** | P | P | P |
| rejects pre-initialize requests | **F** | **F** | **F** | **F** | **F** | **F** | P |
| stdout is JSON-RPC only | P | P | P | P | P | P | P |
| **score** | 10 | 11 | 11 | **3** | 10 | 11 | 12 |
| **deterministic over 12 runs** | **no** | yes | yes | **no** | yes | yes | yes |
| **exits at stdin EOF** | yes | yes | yes | yes | yes | **no** | yes |

Three findings the conformance score alone would have hidden:

**S4 (3/12) drops requests nondeterministically.** The same 4-request input yields 0, 1
or 2 responses across runs, never 4. Two independent causes, both in
`readerThreadFunc`: it spawns a **detached `std::jthread` per request** and then
`main()` returns at EOF while those workers are mid-write; and it tests `POLLHUP`
*before* draining `POLLIN`, so data that arrives in the same `revents` as the hangup is
discarded. A detached `jthread` is also a contradiction in terms — the type exists to
join.

**S1 is racy too**, less brutally: 12 runs of a 3-response script gave 3,3,3,3,2,2,3,2,3,2,3,3.
Same family of defect. Its one conformance failure is a symptom of the race, not a
missing feature — called by hand with correct arguments its `tools/call` is fine.

**S6 never exits.** It answers all three requests correctly and then hangs at stdin EOF
forever. It scores 11/12 only because the harness kills it at 6s. A client that spawned
it would leak the process and block in `waitpid`.

**S5** answers `notifications/cancelled` with a `-32601` error. Replying to a
notification at all is a protocol violation; it checks method-not-found before checking
whether an `id` is present.

**Only S7 refuses to serve `tools/list` before `initialize`.** The other six will answer
anything at any time.

**Winner: S7** — 12/12, deterministic across 12 runs, exits cleanly, and the only
entrant that enforces the lifecycle gate.

## Client results

Clients cannot be graded by a stdin harness, so they were graded by reading plus a wire
trace: the two entrants that accept a server command on `argv` were run against a
correct MCP server that logs the raw bytes it receives.

| | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|---|---|---|---|---|---|---|---|
| newline framing (spec) | **no** | yes | **no** | yes | yes | yes | **no** |
| incremental read buffer | yes | yes | yes | yes | yes | yes | yes |
| partial/EINTR-safe writes | **no** | yes | yes | **no** | yes | **no** | yes |
| stderr safe | yes* | yes | yes* | yes | yes | **no** | yes* |
| request timeout | yes | yes | yes | yes | yes | **no** | yes |
| fails pending requests on teardown | no | no | no | no | no | no | **yes** |

\* safe by inheriting the parent's stderr rather than piping it.

**C1, C3 and C7 emit LSP `Content-Length:` framing, not MCP framing.** MCP stdio is
newline-delimited JSON, full stop. C7's code even comments the line `// For standard
MCP, write Content-Length header`. The wire trace makes the consequence concrete — every
single message C1 sends produces a parse error at the server:

```
'Content-Length: 164\r\n'          <- not JSON
  ^^ PARSE ERROR
'\r\n'
'{"id":1,"jsonrpc":"2.0","method":"initialize",...}\n'
```

It appears to work only because a lenient line-based server skips the junk line and
parses the next one. Against a server that treats a parse error as fatal, these three
never get past `initialize`. This is the classic shape of a bug that passes against its
own mock and fails against every real server — and all three ship a matching mock.

**C6 deadlocks against a server that logs.** It pipes the child's stderr and never reads
it. Given a server that writes 150 KB to stderr at startup — entirely normal for the
npx-based reference servers — C6 hung until killed, having got 2 lines out. C1 survives
the same test only because it never pipes stderr at all. The lesson for our own build:
**inherit stderr, or pipe it and actively drain it; never pipe it and ignore it.**

**C4 sets `O_NONBLOCK` on the child's stdout** and then reads it in a loop, so it spins
rather than blocks. Its line buffering is otherwise the most careful of the seven.

**Winner: C5** — correct framing, a `Process` RAII type cleanly separated from the
protocol layer, and a single `poll()` covering stdout and stderr together, which is the
only structure in the seven that is both correct and not a busy-wait. **C2** is the
runner-up on robustness: proper partial-write and `EINTR` handling.

## What we took

- **C5** — `Process` as an RAII type separate from protocol logic; one `poll()` over
  stdout+stderr; the read buffer as a member so partial lines survive across reads.
- **C2** — the write loop that handles partial writes and retries on `EINTR`.
- **C7** — failing every pending promise with a meaningful exception at teardown,
  rather than letting `~promise` deliver a bare `broken_promise`.
- **C6** — declaring the reader thread last so member destruction order stops it first.
  (The one genuinely thoughtful line in that entrant.)
- **S7** — the lifecycle gate: nothing but `initialize` and `ping` is served until
  `notifications/initialized` has arrived.
- **S1/S2/S3/S5/S6** — the shape of the tool registry: `std::function` dispatch keyed by
  name, with the JSON Schema carried next to the handler.

## What we rejected

- `Content-Length` framing (C1, C3, C7).
- Piping stderr without draining it (C6).
- A thread per request, detached (S4).
- Honouring `POLLHUP` before draining `POLLIN` (S4).
- Returning from `main()` with workers in flight (S1, S4).
- Hardcoding `2024-11-05` with no negotiation (all 14).
- Serving requests before `initialize` (six of seven servers).
- `O_NONBLOCK` on a stream you then poll in a spin loop (C4).

## Bottom line

Nothing here is adoptable as-is: the best client speaks a protocol variant no real
server uses or is one of three that do, and the best server implements an eighth of the
spec against an 18-month-old revision. But unlike `mlx-redo` and `mcp-transport`, this
round is **not a write-off** — the code is real, it runs, and the two winners (C5, S7)
are sound designs to build on. The entrants are frozen under `bakeoff/mcp/entrants/`
and the conformance harness is now pointed at our own implementation, where it is a
regression gate rather than a scoreboard.
