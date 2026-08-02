# Handoff: everything remaining

Paste this whole file as the opening prompt of a fresh session in
`/Users/dev/Desktop/seans_projects_local/LM_Pipe_2` (branch `main`).

Written 2026-08-02, after landing `src/mcp` (PR #6) and re-pinning the stale grammar test
(PR #7). Everything below was verified against the tree that day, not copied from the
older plans — several items those plans list are now done, and this document says so
rather than leaving you to find out.

---

## Read this first

The goal is unchanged: **a fast, local, competitive 2026 coding agent Sean can drive
against real work.** Single model, single process, Apple Silicon, Qwen3 via MLX
in-process.

**Closed, do not re-open.** Performance (six passes, `HANDOFF_PERF.md`). Speculative
decoding (wired, off by default, `HANDOFF_SPECULATIVE.md`). Token streaming. Cross-session
memory (`remember`). The miniature tokenizer fixture. Editor-API edits through
`WorkspaceEdit`. The refusal ledger. MLX itself is not worth replacing
(`MOE_ROUTING_FINDINGS.md`, and the memory note behind it).

**Where things actually stand, measured 2026-08-02:**

```
ctest -L gate            32/32,  ~15 s      (both preset configs, and CI)
ctest -L realmodel -j1    5/5,   ~29 s      (local only -- 3 tests, see R1)
agent_eval pins          corpus 5/6 solved, 2 completed, 2 verified
                         holdout 2/4 solved, 2 completed, 3 verified
MCP conformance          12/12; official TS SDK client drives our server 18/18
extension                installed into Antigravity IDE and Cursor, sidecar verified live
```

**Updated 2026-08-02 (later session).** **R1 and V2 are done**, and H1 is done except for
the `PHASES.md` mutation figure, which belongs to V1. The gate went 29 → 32: `test_grammar`
and `test_token_stream_gate` (R1) and `test_pin_attribution` (V2). `realmodel` is still
5 tests but only **3** of them are now CI-invisible — the other two are the same sources
registered a second time against the real checkpoint.

Two findings from doing it, both recorded in full below:

- Putting the tokenizer path under sanitizers for the first time exposed **real undefined
  behaviour in vendored utf8proc**, on the production NFC path, reproduced on the real
  checkpoint and not just the fixture. Fixed in `52df37d`.
- **`cmake --preset asan` does not work on this machine without `-DLMP_MLX_PYTHON`.** The
  standing rule above said it did. Corrected.

---

## Standing rules

Non-negotiable, and every one of them exists because it was violated once:

- `ctest -L gate` green, **under a preset, and with `LMP_MLX_PYTHON` pointed at the venv**.
  The default probe uses `python3`, which on this machine is 3.9 — MLX stopped shipping
  cp39 wheels after 0.29.3, so the build silently pins to an MLX two minor versions behind
  and fails in `src/model/mlx_backend.cpp` with `no member named 'device_info'`. That is a
  configuration artifact, not a broken tree.
  **The presets do not set it, and a preset alone is not enough** — this file used to say
  "use `cmake --preset dev` or `--preset asan`", and `--preset asan` on a fresh build dir
  fails exactly this way. `build/` only works because an earlier session's
  `-DLMP_MLX_PYTHON` is still in its cache. Corrected 2026-08-02, after it cost a build
  cycle. Configure with:

  ```
  cmake --preset asan -DLMP_MLX_PYTHON=$HOME/.venvs/lmp-mlx/bin/python
  ```
- `./scripts/run_ratchets.py` 6/6, with `--self-test` still able to redden each one.
- A new test bumps **both** the count and the names in `tests/gate/gate_manifest.txt`, in
  the same commit. Two edits, deliberately.
- `python3 scripts/agent_eval.py run` before and after any loop change. Pins are floors: a
  drop fails, an improvement asks to be re-pinned deliberately.
- **One MLX process at a time.** One checkpoint is 19 GB on a 48 GB host. `ctest -L
  realmodel` must be `-j1`, and nothing else may hold a model while it runs.
- Reproduce CI locally before pushing: both presets with `-DLMP_MLX_PYTHON=/usr/bin/false`.
  CI has no MLX and compiles the `#else` half a local build never touches.

---

## Ranked

| # | Item | Size | Risk | Why here |
|---|---|---|---|---|
| ~~**R1**~~ | ~~`realmodel` tests never run in CI~~ | S | — | **DONE.** Invisible set 5 → 3; falsifier run in the CI config |
| ~~**M1**~~ | ~~Wire MCP into the agent~~ | M | — | **DONE.** Verified against the official reference servers |
| **C1** | Declared contract is not checked against the mission | M | **high** | Solved 5, completed 2 — the gap is here |
| **V1** | Mutation harness is absent; its number is unreproducible | M | medium | A measurement with no instrument |
| ~~**V2**~~ | ~~`pins.json` stores only aggregates~~ | S | — | **DONE.** Attribution lands on the next deliberate re-pin |
| **M2** | MCP: streamable HTTP + authorization | L | low | The one real spec gap |
| **D1** | No git write | M | low | It can read its diff and not commit it |
| **C2** | No resume | L | medium | A crashed run is lost |
| **D2** | `locate_symbol` is ranked grep, not an index | M | low | clangd already runs in this repo |
| **C3** | Repetition penalty diverges from the reference | S | medium | Moves the baseline; must be a deliberate re-pin |
| **C4** | Nothing retries | S | low | Read the classification before wiring |
| **M3** | MCP: URI templates, sampling, elicitation | M | low | Completeness, no user waiting on it |
| **P1** | Workstream 3 small steals | S | low | Individually cheap |
| **D3** | T2 containers refuse | XL | medium | **Do not start without asking Sean** |
| **H1** | Stale docs | XS | low | Cheap, and they mislead the next session |

---

## R1 — `realmodel` tests never run in CI — **DONE 2026-08-02**

> **Landed** in `e8818e0`, taking option 2 below. Each of the two tokenizer-only sources is
> now registered **twice** off one file: a `gate` variant compiled with
> `LMP_MINI_VOCAB_JSON` that runs on the generated miniature vocabulary and therefore runs
> on CI, and the original `realmodel` variant against the real checkpoint. Duplicating
> rather than moving costs one extra binary and keeps the real-vocab coverage that
> motivated the label in the first place. Gate 29 → 31 here (32 after V2).
>
> **Falsifier, run in the configuration CI actually builds** (`-DLMP_MLX_PYTHON=/usr/bin/false`,
> no MLX at all): re-applying `4300a3c`'s semantics reddens `-L gate`, 1 failed of 31.
> The same mutation left the gate green before. Also confirmed under `dev` and `asan`.
>
> **What it caught on the way in.** `gate-asan` selects `gate`, so sanitizers had never
> once run over the tokenizer path. The first run aborted on real undefined behaviour in
> vendored utf8proc — `utf8proc.c:391` forms `NULL+4` on the documented two-pass measuring
> call that `frankentok/src/normalizer.cpp:37` makes on every NFC normalization. Benign on
> any real ABI, fatal under `-fno-sanitize-recover=undefined`. **Reproduced on the real
> checkpoint before fixing**, so it is a production path and not a fixture artifact. Fixed
> in `52df37d`, marked as a local modification to re-apply on upgrade.
>
> **Residual, deliberately left.** Three tests still never run on CI —
> `test_realmodel`, `test_kv_reuse_realmodel`, `test_spec_cache` — because they genuinely
> need weights on a GPU. The invisible set is smaller, not empty, and `realmodel_count` in
> the manifest still pins it at 5. Note the corollary the utf8proc find makes concrete:
> **nothing in that set has ever run under a sanitizer.** If any of the three is ever
> reduced to something a vocabulary can drive, it should move the same way.

**What was wrong.** `.github/workflows/gate.yml` runs `ctest --preset gate` and
`gate-asan`. Neither selects the `realmodel` label, and nothing else does. Five tests —
`test_grammar_realmodel`, `test_realmodel`, `test_kv_reuse_realmodel`, `test_spec_cache`,
`test_token_stream` — run only when a human remembers.

**This is not hypothetical.** On 2026-07-31 commit `4300a3c` changed `grammar.cpp` (+23/−12)
and did not touch its test. Batching deliberately changed what closing a tool call means:
`</tool_call>` now returns `Advance::Ok` and moves to `TurnPhase::Text` so a second call
may follow, where it used to return `Accepted` and finish the turn. The test asserted the
old contract in four places and **stayed red for two days.** Nobody saw it because the
label hides it. Fixed in PR #7 — but the mechanism that hid it is untouched.

Worse, a red grammar test *reads* like broken tool calling. It cost real time on
2026-08-02 to establish that the agent was fine and only the test was stale.

**Design.** CI runners have no Apple Silicon GPU and no 19 GB checkpoint, so these cannot
simply be added to the CI job. Two candidates, and the second is better:

1. A scheduled local run — a `launchd` job or a `make check-real` someone runs. Weak: it
   restores the same "when a human remembers" property that failed.
2. **Split the label.** Most of what these tests assert needs a *vocabulary*, not a GPU —
   `test_grammar_realmodel` loads only `tokenizer.json`, and the mini-vocab fixture
   (`tests/fixtures/make_mini_vocab.py`) already exists and is generated at build time.
   Move everything that needs only a tokenizer onto the fixture and into `gate`; leave the
   `realmodel` label for what genuinely needs weights on a GPU.

Prefer 2. It shrinks the invisible set instead of scheduling a reminder about it.

**Tests that must exist.** Whatever moves into `gate` must be in the manifest, with the
count bumped. `test_gate_manifest` already proves the selector selects.

**Falsifier.** Re-apply the `4300a3c` semantics to `grammar.cpp` (make the close return
`Accepted` and set `Done`) and confirm **CI** goes red — not a local run. If CI stays
green the split did not cover the case that actually rotted.

---

## M1 — Wire MCP into the agent — **DONE 2026-08-02**

> **Landed** in `95bc956`. `src/tools/mcp_host.{hpp,cpp}` connects each configured server
> at run start and registers its tools into the same `Registry` the native tools live in,
> namespaced `mcp__<server>__<tool>`, with the `inputSchema` converted to a
> `parsephony::ToolSpec` so `ToolCallGuard` constrains remote calls exactly as it does
> native ones. Settings ride in `RunSettings.mcp_servers`; the wire parsing lives in
> `src/surface/mcp_settings.{hpp,cpp}`.
>
> **Sean's decision on containment (asked, per this section's own instruction): untrusted
> by default, per-server opt-in.** An untrusted server's tools are all `irreversible`, so
> every call raises a card. The server's own `readOnlyHint`/`destructiveHint` annotations
> deliberately do not participate — the MCP spec says they must not drive a security
> decision, and they are claims made by the thing being contained. `trusted` must be a
> literal boolean `true`; it is tested against `"true"`, `1`, `null`, `"yes"`, `0` and
> absent (S7.5).
>
> **The generator gained an array type.** `RunSettings.allowed_commands` documents the old
> workaround — newline-separated, which works only because a shell command cannot carry a
> raw newline. A list of objects has no such separator, so `repeated` is now a schema field
> and both sides generate from it.
>
> **Verified against software we did not write**, which is the part that matters:
> `mcp_host_probe` drove the official `server-filesystem` (14 tools) and
> `server-everything` (13), none rejected. The filesystem server has its own `read_file`,
> which registered as `mcp__probe__read_file` and did **not** shadow the native one — the
> exact collision the naming exists for, against a real server that has it. See
> `bakeoff/mcp/interop/README.md`.
>
> **ASan found a real bug in it** once the tests reached `gate-asan`: the handlers captured
> a raw `mcp::Client*` the host owned, so `close()` was a use-after-free on the next call.
> Connections are `shared_ptr` now, correct by construction rather than by destruction
> order.
>
> **Still open here.** Exposing OUR tools to other MCP clients (the `Server` half, and
> `InProcessTransport`) is deliberately not built — Sean chose client-first. That is the
> obvious next MCP item, ahead of M2.

## M1 — the original plan

**What is wrong.** `src/mcp` is complete, interop-verified in both directions, and
**nothing calls it.** `src/loop` has no MCP client; `src/tools/registry.cpp` has no MCP
tools. Today the library is correct and useless.

**Where.** `src/tools/registry.{hpp,cpp}`, `src/loop/agent.cpp`, and the settings path
from the extension (`protocol/schema.json` → `RunSettings`).

**Design.**

- A server list in `RunSettings` — command, args, env — so the extension can carry it and
  the sidecar does not read a config file of its own.
- At run start, connect each configured server, `initialize`, `list_tools`, and register
  each remote tool into the existing `Registry` with a namespaced name (`mcp__<server>__<tool>`)
  so a remote `read_file` cannot shadow the native one.
- The tool's `inputSchema` becomes the `ToolDecl` schema, which means
  `parsephony::ToolCallGuard` constrains generation of remote calls **exactly as it does
  native ones** — this is the part that makes MCP tools first-class here rather than
  bolted on, and it is why the schema must be carried through rather than flattened.
- A failed or slow server must degrade to "that tool is absent", never stall the turn.
  `Client` already has per-request timeouts; use them and log to the event log.
- Approval and sandbox: **a remote tool is not covered by Seatbelt.** It runs in the MCP
  server's process, outside our jail. Decide deliberately whether remote tools are
  `irreversible` by default and require a card. Getting this wrong reintroduces the exact
  hole `PLAN_GAP_CLOSURE.md` finding 2 describes, one layer out. **Ask Sean before
  choosing the default.**
- `InProcessTransport` exists for the mirror image — exposing *our* tools to other MCP
  clients over the same `Server` API without a pipe. Not required for M1; do not build it
  until someone wants it.

**Tests.** Gate-level, against `mcp_demo_server` (already built): a registry populated
from a live server, a namespaced call round-tripping, a server that dies mid-run leaving
the agent able to finish, and a remote tool whose schema constrains generation.

**Falsifier.** Point it at `npx -y @modelcontextprotocol/server-filesystem /tmp` and have
the agent solve a task that requires a remote tool. Then kill the server mid-run and
require the run to end cleanly rather than hang.

---

## C1 — The declared contract is not checked against the mission

**What is wrong.** Completion is evidential: a recorded deliverable plus a falsifiable
passing verification of **the contract the model declared**. Nothing checks that the
declared contract covers the mission. `PLAN_GAP_CLOSURE.md` finding 3 records
`rename_across_files` finishing `completed=yes verified=yes solved=no` — the model
declared `pytest -q`, made it pass, and the harness agreed, while the ground truth also
required no residual `calc_total`.

The current pins say the same thing from the other side: **corpus solved 5, completed 2.**
The agent is solving tasks it cannot prove it solved. That gap is this item.

**Where.** `src/loop/verification.cpp`, `src/loop/turn.cpp`, and whatever declares the
contract in `src/loop/agent.cpp`.

**Design.** Deliberately unresolved — this is the interesting design question in the loop
and it should not be answered by the first idea. Three shapes, in increasing ambition:

1. **Contract coverage as a checklist item.** The mission's own acceptance criteria are
   extracted at plan time and each must map to a verification. Cheap; only as good as the
   extraction.
2. **Adversarial verification.** After the model declares its contract, a second pass asks
   what the contract would miss. Costs a turn; catches the `rename_across_files` shape.
3. **Ground truth is never the model's to choose.** The harness derives the contract from
   the mission and the model may only add to it. Strongest and most invasive.

Do not start until you have re-read finding 3 and can state which shape you are building
and why the other two lose.

**Falsifier.** `rename_across_files` must end `completed=no` *or* `solved=yes` — the
disagreement cell must empty. And the whole suite must not regress: this is exactly the
change that can make the agent conservative and drop `solved`.

---

## V1 — The mutation harness is absent

**What is wrong.** `docs/PHASES.md` reports "mutation testing 3/8 killed, 5 survivors" and
names the survivors. **The harness is not in this repo** — `find . -name '*mutat*'` returns
nothing. That figure cannot be re-measured, which is precisely what `ctest -E realmodel`
was condemned for on the previous page of the same document.

`PHASES.md` already says: "Either the harness gets committed or the figure gets struck; it
should not stay as a measurement with no instrument."

**Design.** Commit it. It is described in `PHASES.md` in enough detail to rebuild:
copy the tree, plant one mutation, build, run the gate, require red. Its own history is
worth preserving in the commit message — its first run reported 6/8 because three separate
defects made every copy fail for reasons unrelated to any mutation, so a **null mutant runs
first** and an unmutated copy must build and pass before any kill is believed.

**Then re-measure.** The survivors list is stale in at least one place: `grammar.cpp:105`
was blocked on the missing vocabulary fixture, and that fixture now exists.

**Falsifier.** The null mutant must pass and a hand-planted mutation must be killed, in
the same run.

---

## V2 — `pins.json` stores only aggregates — **DONE 2026-08-02**

> **Landed** in `a22dd1f`. `summarize()` carries a per-task map beside the counts and
> `compare()` names what changed; the aggregate stays the gate, so this adds attribution
> and not strictness. It also closes a case aggregate-only pins could never show: one task
> regressing while another starts passing leaves `solved` identical and the suite quietly
> different — reported, deliberately not gated.
>
> **`evals/agent/pins.json` was deliberately NOT rewritten.** `--pin` overwrites the floors
> with whatever a single temperature-0.6 run produced, so re-pinning is an act someone
> takes on purpose, not a side effect of landing the harness. Until then the pin carries no
> per-task record and `compare()` says so rather than reporting a diff it cannot compute.
> **The next deliberate `--pin` is what makes drops attributable.**
>
> **The falsifier is now automated.** "Change one task's outcome by hand and confirm the
> diff names it" is a one-off, and a full run costs ~30 minutes and a 19 GB load, so by
> hand means once and never again. It is `agent_eval.py self-test` — four scenarios, no
> model, no sidecar — and runs in the gate as `test_pin_attribution`. Shown red by stubbing
> out the regression detection (4 failures), green when restored.

**What was wrong.** `evals/agent/pins.json` records `solved: 5` and not *which* five.
`PLAN_GAP_CLOSURE.md` finding 2 hit this directly: "Whether this was a regression could not
be established: `pins.json` stores only aggregates, so there is no record of which 3 of 6
corpus tasks passed on 2026-07-31."

**Design.** Record the per-task outcome alongside the aggregate. Keep the aggregate as the
gate — per-task results at temperature 0.6 are too noisy to be floors — but store the
detail so a drop can be attributed. Small change to `scripts/agent_eval.py`.

**Falsifier.** Re-pin, change one task's outcome by hand, and confirm the diff names the
task rather than only the count.

---

## M2 — MCP: streamable HTTP and authorization

**What is wrong.** `docs/MCP.md` states it plainly: stdio only. `Transport` is an
interface and neither `Client` nor `Server` contains any I/O, so this is a new file rather
than a rewrite — but it is not written, and a growing share of real MCP servers are remote.

**Design.** `HttpTransport` implementing `Transport`: POST/GET to a single endpoint, SSE
for server→client, `Mcp-Session-Id` and `MCP-Protocol-Version` headers. Then the OAuth
2.1 authorization spec on top. The authorization half is the larger and more dangerous
piece; **do not store tokens without asking Sean where.**

**Falsifier.** The same interop discipline as stdio: drive a real remote MCP server, and
have the official SDK client drive ours over HTTP. `bakeoff/mcp/interop/` is the pattern.
A conformance pass against our own harness is not sufficient evidence and was not for
stdio either.

---

## D1 — No git write

`git_diff`, `git_log` and `git_status` are registered; nothing commits or branches. The
agent can read its own diff and cannot record it.

**Design.** `git_commit` and `git_branch`, both `irreversible` in `ToolDecl` so they raise
an approval card. Never `git push` — that is outward-facing and belongs to the human.
Refuse to commit when the working tree contains changes the run did not make.

**Falsifier.** A run that commits its own work, and a run whose commit is denied and which
then finishes cleanly rather than retrying forever (the `RefusalLedger` path).

---

## C2 — No resume

A crashed run is lost. `ReplayBackend` exists but only for tests.

**Design.** The event log (`src/platform/event_log.{hpp,cpp}`) already records the run. A
resume reads it back to reconstruct `ContextStore` and the turn index. The hard part is not
the log, it is deciding what is safe to replay: tool calls with side effects must not be
re-executed. Probably: replay context, re-run nothing, resume at the next turn boundary.

**Falsifier.** Kill the sidecar mid-run, resume, and require the same deliverable without
re-executing any side-effecting tool. Prove the second part by counting executions in the
event log, not by inspection.

---

## D2 — `locate_symbol` is ranked grep

`src/tools/symbol_index.hpp` shells out to `grep -rnE` and ranks the hits. Its own comment
calls it "the cheap middle". This repo already runs clangd, and `CMAKE_EXPORT_COMPILE_COMMANDS`
is on, so `compile_commands.json` exists.

**Design.** A clangd-backed index behind the same `locate_symbol` interface, falling back
to the current grep when there is no `compile_commands.json` — which is most workspaces the
agent will be pointed at. **Measure before believing it is better:** ranked grep is
surprisingly good on the tasks in the eval suite, and this must be justified by a pin
movement, not by being architecturally nicer.

---

## C3 — Repetition penalty diverges from the reference

`apply_repetition_penalty` in `src/model/sampler.cpp` loops `for (TokenId id : recent)`
and applies the penalty **per occurrence**, so a token appearing three times in the
64-token window gets `l/p/p/p`. HuggingFace gathers and scatters by index, applying it
once per unique id. Still true as of 2026-08-02; two cook-off entrants de-duplicated and
we do not.

This is a real divergence, **and it changes generated tokens.** Landing it silently moves
the pinned baseline and every comparison across it becomes meaningless. Land it
deliberately, in its own commit, with a re-pin and both numbers stated.

---

## C4 — Nothing retries

`ToolResult.retryable` is now read — `src/loop/turn.cpp:185` gates `BreakRepeat` on it — but
nothing actually retries.

**Read the classification before wiring a consumer.** `shell` marks every non-zero exit
`Transient` + retryable, and a failing build is neither. Auto-retrying it re-runs a
deterministic failure and burns a turn. Fix the labels first or the feature is negative
value.

---

## M3 — MCP completeness

Three small gaps in `src/mcp`, none with a user waiting:

- **URI-template matching.** `Server::handle_resources_read` offers the first registered
  template handler the URI and lets it decline. RFC 6570 matching is not implemented.
- **Sampling and elicitation.** Declined cleanly with `-32601` today, which is correct
  behaviour but not support. Sampling means a server can ask *our* model for a completion —
  interesting here specifically, because we have the model in-process.
- **Server-side response routing.** `Server::on_message` (`server.cpp:268`) drops replies
  to requests the *server* initiated toward the client — `roots/list`, sampling — because
  nothing waits on them yet. Sampling cannot land until it does.

---

## P1 — Workstream 3 small steals

From `PLAN_PARALLELISM.md`, individually cheap and independently landable — but **check
each against the tree first, because that table has decayed.** Its headline item, caching
the opposite index in the SPSC hot path, is **already done**: `spsc_channel.hpp` carries
`head_cached_`/`tail_cached_` with the stale-lower-bound argument written out. Verified
2026-08-02.

What is still open there: the streamed-tail holdback (`len - last_open < 12`) and
cursor-plus-`substr` instead of substr-per-iteration, both of which only became relevant
once streaming landed. None of these move the eval pins; do not let them displace R1, M1
or C1.

---

## D3 — T2 containers

`src/tools/container.cpp` refuses unconditionally, by design, and says so in its refusal
text. Unattended runs are unavailable as a result (S7.2). This is XL and it changes the
security posture.

**Do not start without asking Sean.**

---

## H1 — Stale documents — **DONE except the mutation figure**

Cheap, and they actively mislead. Each claim was checked against the tree before editing,
not taken from this list:

- ~~`HANDOFF_AGENT.md`'s ranked table lists the tokenizer fixture and editor-API edits as
  open; both are closed.~~ **Done.** Both rows struck. Two more were stale and are
  corrected: `ToolResult.retryable` is no longer "read by nobody" (`src/loop/turn.cpp:185`
  gates `BreakRepeat` on it, and nothing retries — which is C4, not this), and "20 tests in
  ~7 s" is now 32 in ~15 s. Its `max_new_tokens` and `rename_across_files` notes are still
  live and were left alone.
- ~~`PLAN_PARALLELISM.md` Workstream 4 says speculative decoding is "NOT DONE — blocked".~~
  **Done.** Marked closed with the measured result, pointing at `HANDOFF_SPECULATIVE.md`;
  the four blockers are kept below the line and labelled history. Workstream 3's first row
  was stale the same way — the SPSC opposite-index cache is **already implemented**
  (`spsc_channel.hpp:48`, `:66`) — and its repetition-penalty note cited a baseline of
  corpus 3/6, holdout 1/4, which is two re-pins out of date. Both fixed.
- `PHASES.md`'s mutation figure — **still open**, and it belongs to V1: the honest fix is
  to restore the instrument, not to strike the number. Left for whoever does V1.

---

## How to run things

```bash
cmake --preset dev -B build && cmake --build build -j8
ctest --test-dir build -L gate                    # 32/32
ctest --test-dir build -L realmodel -j1           # 5/5, ONE at a time
./scripts/run_ratchets.py                         # 6/6
python3 scripts/agent_eval.py self-test           # 4 scenarios, no model
python3 scripts/agent_eval.py run                 # ~30 min, loads the model
```

The sanitizer build needs the interpreter passed explicitly — see Standing rules:

```bash
cmake --preset asan -DLMP_MLX_PYTHON=$HOME/.venvs/lmp-mlx/bin/python && cmake --build --preset asan -j8 && ctest --preset gate-asan
```

```bash
cd extension && npm run install-local             # Antigravity IDE + Cursor
```

```bash
python3 bakeoff/mcp/conform.py build/src/mcp/mcp_demo_server                     # 12/12
cd bakeoff/mcp/interop && node drive_our_server.mjs ../../../build/src/mcp/mcp_demo_server
```

Reproduce CI before pushing:

```bash
cmake --preset dev -B /tmp/nomlx -DLMP_MLX_PYTHON=/usr/bin/false && cmake --build /tmp/nomlx -j8 && ctest --test-dir /tmp/nomlx -L gate
```

---

## Working agreements

- Measure before believing, including documents in this repo and including this one.
- A green result counts only once the check has been shown capable of going red. Every
  test file here ends with a case that proves its own checks can fail; keep that.
- A survivor, a stale pin or an unreproducible number is a **finding**, not an
  embarrassment. Record it where it will be read, as `PHASES.md` does for the mutation
  figure.
- Do not re-open performance, speculative decoding, or replacing MLX. Each is closed with
  numbers.
- Ask before: T2 containers, storing OAuth tokens, the sandbox default for remote MCP
  tools, and anything that downloads multiple GB.
