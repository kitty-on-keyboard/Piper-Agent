# C04: Make dispatch output a compact, complete card: telemetry to a log file, failure evidence on the card, UNVERIFIED without a check

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| VERIFIED (reproduced) | piper | cloud-tokens | high | S |

> **Verified.** The problem and fix below are the skeptic's corrected versions. They replace the finder's original claim where the two differ.

Merged from these finder reports:

- Keep worker stderr telemetry out of the dispatch output; put failure detail on the card
- Make the review card carry the fail evidence: error, failing check tail, how the run ended, UNVERIFIED when no check ran

## Problem

Mostly accurate, with conditions and some overstatements.

(1) Telemetry flood. This is real on the cold path only. scripts/piper_worker.py:2573 runs `subprocess.Popen(cmd)` and inherits stdio. src/model/mlx_backend.cpp writes to stderr unconditionally on every generate call, and there is one generate per turn (agent.cpp:1297). That is about 11 lines (~1 KB) per turn: generate_enter, the `generate:` line, prefill_start, a chunk_begin/chunk_end pair per 2048-token chunk, prefill_done, decode_begin printed twice (1164/1172), mem decode_begin, decode_first_token and decode_end.
- Warm path: when `piper worker serve` is alive, `lmp_sidecar --worker` forwards to the daemon (socket_reader.cpp forward_to_daemon). The breadcrumbs then go to the daemon's stderr, not the dispatch output. Only the banner, one ask notice and the card reach the cloud.
- Claude Code specifically: its Bash tool runs commands with stdin=/dev/null and stdout as a regular file (checked in this container). piper_worker.py:2535-2541 therefore classifies dispatch as detached. With no wake URL it exits 3. With `.piper/orch_webhook` present it double-forks and returns exit 0 in 0.09 s with no card and no result.json. So in Claude Code the flood appears only with LMP_DAEMONIZE=0, or once that separate bug is fixed. In orchestrators with a pty or pipe for stdio it appears by default.
- Size: my repro of the real formats through the real dispatch code gave 355 lines / 29.5 KB for 30 turns, with the card in the last 11 lines. Claude Code 2.1.x keeps the first 30,000 characters inline (TaskOutput reads from offset 0) and saves the rest to a file. Past about 30 turns the card leaves the inline result. That always happens for 60-turn trust_mcp slices.

(2) Card. The claims hold:
- It has no `error`, no test.output_tail, and no turns or wall time.
- A check timeout shows as `exit=-1`. The `[timeout after Ns]` marker exists only in output_tail (worker.cpp:149-151).
- review_verdict returns PASS when status is ok and test.ran is false.

Overstatements:
- A failing check is not a false success. The card already shows `status: error` and `verdict: FAIL`. The problem is missing evidence and an extra round trip, not a wrong verdict.
- "Docs discourage result.json" is too strong. PIPER.md names result.json as a review input; "no cat/jq" is about status polling.
- `--json` really does carry the result plus the card (1929 vs 966 bytes), but no documented orchestrator flow uses `--json`.

(3) Flaws in the proposed fix:
- O_APPEND across re-dispatches of the same slice mixes runs. In my prototype, the DIED card's log tail showed the previous run's 'finished' line. The C++ side already archives and truncates events.jsonl per run for exactly this reason (archive_prior_events, worker.cpp:1177).
- Adding the raw log tail to every non-ok card mostly adds MLX `mem` lines, which is noise for error and stalled results. The raw tail is only useful for DIED.
- Routing stderr to a file moves the ask question off the live stream. The card has to bring it back.

## Why it matters

Priority 3, on every slice: about 1 KB of telemetry per model turn (thousands of input tokens per dispatch) shrinks to a card of about 1 KB. Putting the failing assertion on the card also saves the extra result.json read or check rerun on every non-PASS slice. Priority 1: no PASS without a check, the tool-output cap can no longer truncate the card away, and the model's own report never appears alone next to a failing check.

## Fix to ship

Python only, all in scripts/piper_worker.py plus tests and docs.

(1) Telemetry to a per-slice log, in main()'s C++ branch around line 2573:
- Compute log = dirname(result_path)/worker.stderr.log.
- Rotate an existing log to worker.stderr.prev.log, so each run starts fresh. This mirrors archive_prior_events for events.jsonl. Do not rely on O_APPEND across runs.
- Open it with `os.open(O_WRONLY|O_CREAT|O_TRUNC|O_APPEND, 0o644)`, pass the fd as `subprocess.Popen(cmd, stderr=fd)`, then close it in the parent. The child writes the fd directly, so breadcrumbs survive a SIGKILL.
- Leave stdout alone, so `--jsonl` is unchanged.
- Add an escape hatch, `LMP_WORKER_STDERR=inherit`, for humans running `piper run` in a terminal.
- Optionally point the agent_eval fallback path at the same file instead of DEVNULL.
- Optional: skip the 4-line banner for `dispatch`.

(2) format_review_card:
- Always add `run: turns=N wall=Ns`.
- On anything other than PASS, add `error: <result.error>`.
- When test.ran is true and exit_code is not 0, add a `check:` block with the last ~10 non-empty lines of test.output_tail. The C++ timeout marker is already the last line of that tail, so no special parsing is needed; a structured timed_out field belongs to C10.
- On anything other than PASS, add the last 3 or fewer worker.stderr.log lines that start with `piper:`. This brings back the ask question, a daemon disconnect, or a failed webhook post.
- On DIED (no result.json), add the log path and its raw last ~8 lines (crash breadcrumbs).
- Keep the card to about 25 lines.
- Leave out ended/question/scope; those come from their own candidates.

(3) review_verdict: return UNVERIFIED when status is ok and exit is 0 but test.ran is false.
- Update self_test case 19 and prove_deterministic_harness step 5.
- Add fixtures: failing check, timeout tail, no check, stalled, and died with a log that has a prior-run log present.
- Assert the error, check and verdict lines, and that the dispatch output is under 1.5 KB with the telemetry in worker.stderr.log.

(4) Docs: SKILL.md §4 and PIPER.md step 5 should say UNVERIFIED means the model finished but nothing was checked, so review the diff or re-dispatch with --check. Mention the log path.

(5) Drop the `--json` restructure.

Land this together with, or after, the separate fix that makes `dispatch` always attached under Claude Code's stdin=/dev/null and regular-file stdout. Until then, Claude Code orchestrators never reach this code path without LMP_DAEMONIZE=0.

## Evidence (file:line)

- scripts/piper_worker.py:2573: `proc = subprocess.Popen(cmd)` inherits stderr; 2544: the banner prints on every run
- src/model/mlx_backend.cpp:583-589: log_mlx_mem; per-turn call sites at 952-1245 (decode_begin twice at 1164/1172); speculative path at 616-799
- src/surface/sidecar.cpp:2192-2203 and src/surface/worker.cpp:1626-1632: 'Waiting for answer.json' heartbeats every 5 seconds
- scripts/piper_worker.py:980-1018: format_review_card cuts the message to 160 chars and prints the test line as just exit+cmd, with no error, output_tail, turns or wall time
- scripts/piper_worker.py:959-968: review_verdict gives PASS when status is ok and exit is 0, even if test.ran is false
- scripts/piper_worker.py:1050-1056, 1095-1101: --json embeds the full result plus the card (2205 bytes against a 950-byte result.json in the repro)
- src/surface/worker.cpp:149-151: a check timeout only appends '[timeout after Ns]' to the output, so the card shows exit=-1
- SKILL.md:105-108 tells the cloud to read `error` and to target `test.output_tail`, but neither is on the card; PIPER.md:94 says 'no cat/jq'; SKILL.md:130 lists 'trusting a model done with no --check' as an anti-pattern
- Repros (scratchpad r4, r1): a fake sidecar emitting the mlx_backend line formats made the dispatch output 345 lines / 28527 bytes; a packet emitted with --check printed 'test: not run … verdict: PASS'; for a synthetic failing-check result the card showed the model's 'All done' next to 'test: exit=1', and the AssertionError was nowhere on it

## Skeptic verdict

real=True, reproduced=True, keep=True, root-cause fix=True, already handled=False, violates constraint=False

Evidence I checked myself:
- All cited lines match the code: piper_worker.py:959-968, 980-1018, 1050-1056, 1095-1101, 2544, 2573; mlx_backend.cpp:583-589, 952-1245; sidecar.cpp:2196-2202; worker.cpp:149-151 and 1626-1632; SKILL.md:105-108 and 130; PIPER.md:94.
- Repro (scratchpad/c04): a fake worker emitting the exact mlx_backend formats, run through the real `dispatch` with LMP_DAEMONIZE=0, produced 355 lines / 29,552 bytes. With my prototype patch applied to a scratch copy, the same run produced 16 lines / 718 bytes, and the 340 telemetry lines were in worker.stderr.log. The patch gave correct cards for a failing check (AssertionError lines shown), a timeout (the `[timeout after 60.000000s]` marker appears as the last tail line with no parsing), a run with no check (UNVERIFIED), a stall (error: agent did not complete (max_turns)), and a death (log tail shown).
- The death test is what exposed the O_APPEND run-mixing flaw.
- Claude Code truncation: I read the installed binary (2.1.286). Bash output over 30,000 characters keeps the head inline and saves the rest to a file.

Already handled? Partly, and only by accident:
- The keep-warm daemon path keeps the breadcrumbs out of the dispatch output.
- agent_eval.py:486-488 already sends sidecar stderr to DEVNULL or to LMP_SIDECAR_STDERR. The C++ launch path in piper_worker.py never adopted that policy, so the fix matches an existing convention.

Not a band-aid:
- The breadcrumbs exist for crash forensics ("Last line on a SIGKILL still has to be on disk"), so cutting them back would be wrong.
- The fix separates the diagnostics channel from the orchestrator channel at the launch parent, which is the layer that owns the cloud-facing output. Durability is kept because the child writes to the file descriptor directly.
- No constraint is touched: the change is Python only, makes no C++ changes, and touches neither git nor the measured engines.

Against it:
- Part (1) has no effect for users who always keep the daemon warm.
- In Claude Code, part (1) only matters once dispatch actually runs attached.
- Part (4), the `--json` dedupe, is churn with no documented consumer.
- The ended/question/scope fields depend on C02, C05, C03, C09 and C10 and should not be bundled here.

The card additions and UNVERIFIED apply to every slice, whatever the mode, and are cheap. The existing tests that assert PASS for runs with no check (self_test case 19, prove_deterministic_harness step 5) need updating.

Related bugs I found that are outside this candidate:
- Severe, and it gates C04 in Claude Code: `piper dispatch` is misdetected as detached in Claude Code. It exits 3, or exits 0 immediately with no card when a wake file exists. A false "exit 0 = completed" is a priority-1 problem.
- Minor: a SIGKILLed worker makes dispatch exit 247 (proc.wait() returns -9). That is outside the documented 0-3 exit codes.
- The repo's self-test has one failure on Linux before any change (the 'failed to parse event log' stderr case).

Back to the [index](README.md).
