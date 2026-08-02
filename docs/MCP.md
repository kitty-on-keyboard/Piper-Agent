# `lmp_mcp` — Model Context Protocol in C++20

A client and a server for MCP, written against the spec rather than against a mock.
Lives in `src/mcp/`, links nothing from LM_Pipe, and is liftable into its own repo.

Built after judging 14 Jules entrants (`docs/BAKEOFF_MCP.md`). Where a decision here
looks over-specified, it is usually because an entrant got it wrong in a way that only
showed up against real software.

## Status

**Implemented.** JSON-RPC 2.0 core with correct error codes and string/number id
preservation; the full lifecycle including version negotiation; stdio transport; an
in-process transport; tools, resources, resource templates, prompts, completion and
logging on the server; all of those plus roots on the client; ping, cancellation,
progress, pagination; `structuredContent` and `outputSchema`.

**Not implemented, deliberately.** Streamable HTTP transport and the OAuth
authorization spec that goes with it. `Transport` is an interface and neither `Client`
nor `Server` contains any I/O, so this is a new file rather than a rewrite — but it is
not written yet, and nothing here should be read as claiming HTTP works. Sampling and
elicitation are declined cleanly (`-32601`) rather than silently ignored, so a server
that asks gets an answer instead of a hang.

Protocol revision `2025-06-18`, negotiating down to `2025-03-26` and `2024-11-05`.

## It works against real software

Not against our own mock. Both directions, on 2026-08-02:

**Our client → official servers** (`@modelcontextprotocol/server-everything` 2026.7.4,
`server-filesystem` 0.2.0, `server-memory` 0.6.3). Negotiated `2025-06-18`, enumerated
13 tools / 7 resources / 4 prompts from `server-everything`, received live progress
notifications during `trigger-long-running-operation`, and round-tripped real work
through `server-filesystem` — read a file, wrote a file, verified the bytes on disk.

**Official client → our server.** The `@modelcontextprotocol/sdk` 1.30.0 TypeScript
client drives `mcp_demo_server` through 18 assertions covering the handshake,
instructions, tool schemas, `structuredContent`, tool failure vs protocol error,
progress, resources, prompts, `completion/complete`, `ping`, `logging/setLevel` and
error codes. **18 passed, 0 failed.** This is the direction that matters most: our own
conformance harness can only find bugs we thought to look for.

**Conformance board.** `bakeoff/mcp/conform.py` scores 12 spec behaviours. Our server
takes **12/12**; the best entrant took 12/12 and the worst 3/12. The board is falsified
in both directions first — `bakeoff/mcp/falsifiers/` holds a correct reference server
that must score 12/12 and a broken one carrying seven planted defects that must be
caught individually.

It earned its keep immediately: it caught a real bug in *this* implementation, where the
transport treated stdin EOF as "closed" and dropped every reply the worker pool had not
yet written. That scored 5/12 before the fix.

## The transport question

> "I see you are using stdio but you told me that's horrendously slow. Can we use SPSC
> queues instead or something?"

Measured, not asserted (`bakeoff/mcp/bench_transport.cpp`, M-series, 20,000 requests):

| | p50 | p99 | throughput |
|---|---|---|---|
| stdio, 32 B, serial | 19.5 µs | 33.5 µs | 48k req/s |
| stdio, 32 B, 8 in flight | 77 µs | 142 µs | **99k req/s** |
| stdio, 32 B, 32 in flight | 276 µs | 347 µs | **115k req/s** |
| stdio, 16 KB, serial | 224 µs | 250 µs | 4.5k req/s |
| in-process, 32 B, serial | 15.7 µs | 27 µs | 57k req/s |
| in-process, 16 KB, serial | **24 µs** | 37 µs | **41k req/s** |

Three things follow.

**stdio is not the bottleneck.** A round trip is ~20 µs. A real MCP tool call reads a
file, queries a database or makes a network request — 100 µs to 10 ms. The transport is
somewhere between 0.2% and 20% of the cost, and usually nearer the low end. Replacing it
optimises the part that is not being spent.

**An SPSC ring cannot replace stdio for real servers, and the reason is not
performance.** An MCP server is a separate process, usually TypeScript or Python,
launched through `npx` or `uvx`. A lock-free queue is only reachable by a peer compiled
against that same queue — so the moment we invent a transport, every server that speaks
it is a server we wrote. That trades away the entire ecosystem MCP exists to give us, to
buy back 20 µs. The spec defines exactly two transports for this reason, and neither is
shared memory.

**So the ring goes where both ends are genuinely ours.** `InProcessTransport` is a
linked pair of endpoints handing `nlohmann::json` values across a queue — no pipe, no
`dump()`, no `parse()`. LM_Pipe's own tools can be exposed through the same `Server` API
as everything else and reached without leaving the process. Identical protocol,
identical handlers, identical tests; the agent above cannot tell which it got. On small
payloads it is ~1.2× faster, which is unremarkable. On 16 KB payloads it is **9×**
faster, because that is where serialisation actually dominates.

**Where the real throughput win is: concurrency, not bandwidth.** Serial stdio does 48k
req/s; the same pipe with 32 requests outstanding does 115k — 2.4×, bigger than
anything the transport swap buys. That is why requests are concurrent by construction
here: `send_async` returns a future, the reader thread completes whichever reply lands
first, and the server answers on a worker pool. Every cook-off client could have done
this and none did.

If in-process ever needs to be faster, the honest next step is replacing that queue's
mutex with a real lock-free SPSC ring — but the measurement above says to spend the
effort elsewhere first.

## Design notes

**Framing.** MCP stdio is newline-delimited JSON. There is no length header, and three
of seven cook-off clients wrote LSP-style `Content-Length:` preambles anyway — one under
a comment reading "For standard MCP". `LineFramer` is a class, not a `getline()` loop,
because a `read()` boundary lands wherever the kernel puts it; half a message now and
half in 40 ms is the normal case. `encode_line` asserts no interior newline survives.

**Ids are a variant, not a `uint64`.** Six of seven clients keyed their correlation map
on `uint64`, which works right up until a peer sends `"abc-1"` and every string id
collapses to 0. A server has no choice: it must echo back exactly what it was sent.

**stderr: inherit, or drain — never pipe and ignore.** Entrant C6 piped the child's
stderr and never read it; against a server writing 150 KB of startup logs it deadlocked
until killed, because the child blocks in `write(2)` once the 64 KB pipe buffer fills.
Default is `kInherit`; `kCapture` is only honoured by a transport that polls it.

**`POLLIN` before `POLLHUP`.** A peer that writes its last message and exits delivers
both in the same `revents`. Checking the hangup first throws that message away — entrant
S4 does exactly this and drops replies nondeterministically.

**A joined worker pool, not a thread per request.** S4 spawned a detached `jthread` per
request and returned from `main()` with workers mid-write, producing 0, 1 or 2 responses
to the same four-request input. Here requests run on a fixed pool that is drained and
joined before `run()` returns. Notifications are handled inline on the reader thread, so
`notifications/cancelled` stays responsive while every worker is busy — which is the
whole reason cancellation is not just another queued item.

**The lifecycle gate.** Nothing but `initialize` and `ping` is served until
`notifications/initialized` arrives. Six of seven entrants would answer `tools/list`
from a peer that had never handshaken.

**Tool failure is a result, not a protocol error.** `isError: true` travels as a normal
result, because the model needs to see that the file was not found in order to try
another path. A JSON-RPC error means the call could not be made at all.

## Stability

- 3 test binaries, 23 cases, clean under **ASan+UBSan** and **ThreadSanitizer**
  (0 warnings, including 6 threads sharing one pipe).
- 20,000-request soak: p99 33 µs, 4.8 MB peak RSS, **0 leaked child processes**.
- 20 sequential spawn/teardown cycles leave nothing behind.
- Every test file ends with a case proving its own checks can go red
  (`EXPECT_FAILING_CHECKS`), per the repo's rule.

## Using it

```cpp
// Client: talk to any MCP server.
Client client(Client::Info{"my-agent", "1.0"});
Subprocess::Options opts;
opts.program = "npx";
opts.args = {"-y", "@modelcontextprotocol/server-filesystem", "/tmp"};
client.connect_stdio(std::move(opts));

const ServerInfo info = client.initialize();
for (const Tool& t : client.list_tools()) { /* ... */ }
const ToolResult r = client.call_tool("read_text_file", {{"path", "/tmp/x"}});
```

```cpp
// Server: expose tools over stdio.
Server server(Server::Info{"my-server", "1.0"});
Tool t;
t.name = "greet";
t.input_schema = {{"type","object"},{"properties",{{"who",{{"type","string"}}}}}};
server.add_tool(std::move(t), [](const nlohmann::json& args, RequestContext&) {
    return ToolResult::text("hello " + args.value("who", "world"));
});
StdioTransport transport;
server.run(transport);
```

Two binaries ship with it:

- `mcp_probe` — point it at any MCP server and it reports what the server offers, calls
  a tool, reads a resource. This is how the interop above was run.
- `mcp_demo_server` — a server exercising progress, cancellation, structured content,
  resources, prompts and completion, for pointing real clients at.

```bash
./build/src/mcp/mcp_probe --call echo --args '{"message":"hi"}' -- npx -y @modelcontextprotocol/server-everything
```

## Next

1. **Wire it into the agent.** The client exists but nothing in `src/loop` calls it yet;
   external MCP tools should appear in the `src/tools` registry alongside native ones.
   This is the step that makes it useful rather than correct.
2. **Streamable HTTP transport** plus the authorization spec — the one real gap against
   the published spec.
3. **URI-template matching** for resource templates. Today the first registered template
   handler is offered the URI and may decline; RFC 6570 matching is not implemented.
4. **Sampling and elicitation**, if the agent ever wants servers to be able to ask it
   for model output.
