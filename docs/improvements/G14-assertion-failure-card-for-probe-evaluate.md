# G14: Assertion-failure card for probe/evaluate/improve: join the failing check to its probe line, the trace's delivered signals and the handler code

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| UNVERIFIED | both | feature | medium | M |

> **Not yet verified.** Open every cited line below and confirm the problem still holds before you implement. Drop or reshape the task if it does not.

Merged from these finder reports:

- ideas-godoer#4 Assertion-failure card

## Problem

Behavioral failures with no script error are the most common red result in the evaluate and improve loop, and distill cannot help with them, because the failure is a synthetic USER_ERROR with no file. The agent gets the assertion text plus 'the reading is in the receipt', and then has to open run-N.json, the trace and the scripts itself. improve() hands the author adapter only the raw criteria.

The facts that decide most cases are already recorded deterministically:
- whether the signal was emitted;
- how many connections it reached;
- the numeric miss from near().

## Why it matters

This covers the highest-volume red result in the evaluate/improve loop, and saves the cloud agent the receipt, trace and script crawl on every failed attempt. In improve(), the card gives the author a cited hypothesis instead of raw criteria. Linking behavior to code is the hard part, so the decisive facts are precomputed and the verdicts are a closed set.

## Proposed fix

1. Hooks.
   - A G13 collector with the predicate 'a ProbeResult with failed checks, or a summaries count mismatch'.
   - In evaluate, after the criteria loop, when any outcome is 'fail'.
   - In improve.py:101, read the ready card for the attempt's packet_id into feedback-{n}.json as 'diagnosis'. If it is not ready, record the packet_id instead, so nothing waits.

2. Collector (deterministic). facts = {failing descriptions, near() actual/want/tol parsed, expected vs ran counts, replay agreement}. Evidence:
   (a) probe source lines whose expect/near description literal matches, ±8 lines, from the frozen evaluator copy;
   (b) from the trace, delivered signals and player actions in the window before the last check, flagging emissions on asserted nodes with delivered == 0;
   (c) for each such signal, the body of the connected handler (connections from verify/tscn.py, or `.connect(` hits);
   (d) in improve, a trace.Comparison against the previous attempt.

3. Report schema, a Piper job kind (G01/G05):
   - verdict: enum {never_emitted, emitted_not_delivered, delivered_wrong_effect, value_off, deadline, fixture_broken, probe_contract}
   - cited: [{ref, line}] across probe and game
   - observed: {actual, want, tol}
   - fix: {ref, line, change}

4. Deterministic short-circuit: if an asserted node's signal has delivered == 0 and a [connection] names a missing method, emit status=deterministic, verdict=emitted_not_delivered, with no model call.

## Evidence (file:line)

- godoer:godoer/probe.py:467-478 failing checks become one USER_ERROR '<n> of <m> probe assertion(s) failed: ...' with raw='' and no file
- godoer:godoer/evaluate.py:250-271 on failure, _check_outcome returns the name plus 'The evaluate run was traced; the reading is in the receipt' and the probe's prints
- godoer:godoer/improve.py:101-104 feedback-{n}.json = {project, manifest, criteria, instructions}, with no diagnosis
- godoer:godoer/trace.py:17-20 of 4,961 emissions only 6 reached a connection, so the delivered timeline is small enough to cite

## How the finder suggested verifying it

Here: unit-test the collector on a ProbeResult built from synthetic GODOER_CHECK lines plus a hand-written .trace (trace.read_trace format). Cover probe-line lookup by description, near() parsing, and the zero-delivery short-circuit. Check that a passing probe makes no packet, and that improve feedback gains 'diagnosis' or 'packet_id' without blocking.

Mac: use GodoerBench tasks with known-failing starting states as a labeled corpus, and score verdict and citation accuracy against the reference solution diffs.

Back to the [index](README.md).
