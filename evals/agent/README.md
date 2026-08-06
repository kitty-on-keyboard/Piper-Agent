# Agent eval suite

End-to-end fixtures for `scripts/agent_eval.py`. Ground truth is an **immutable
operator checker** run after the agent finishes — never a question asked of the model.

## Layout

```
evals/agent/
  pins.json                 # corpus/holdout floors only (private is never pinned)
  tasks/<name>/
    task.json               # mission, split, protect, optional grader/harness
    check.sh                # optional external grader (sibling of workspace/)
    workspace/              # copied into a throwaway cwd for the agent
```

`workspace/` is the only tree the agent may edit. `task.json` and `check.sh` stay outside
that copy. The harness captures their bytes before start, sends the composed command as
`verify_contract`, and fails the run if metadata/grader/protected fixtures are tampered.

## Splits

| split | Role |
|-------|------|
| `corpus` | Tuned-against set. Pin floors apply. |
| `holdout` | Reported beside corpus. **Not established as harder** today (historically 4/4 vs 5/6). Diagnostic only until rebuilt. |
| `private` | Never tuned against. Never written into pin floors. Long-horizon / security fixtures land here. |

## How to run

Model-free policy + fixture checks (CI-safe):

```bash
python3 scripts/agent_eval.py self-test
python3 scripts/agent_eval.py list
```

Deterministic smoke (temperature 0, default seed 0) after loop changes:

```bash
python3 scripts/agent_eval.py run --smoke
python3 scripts/agent_eval.py run --smoke --split corpus
```

Local quality / latency before a deliberate re-pin (three seeds, temperature 0.6):

```bash
python3 scripts/agent_eval.py run --seed 7,13,42
python3 scripts/agent_eval.py run --seed 7,13,42 --split private
```

Long-context / compaction fixture (skipped by default):

```bash
python3 scripts/agent_eval.py run --include-heavy --only long_context
```

Re-pin only after a deliberate, like-for-like quality run:

```bash
python3 scripts/agent_eval.py run --seed 7,13,42 --pin
```

Private outcomes may be printed as `_private_observed` but are never pin floors.

## Scoring

- **solved** — immutable checker exit 0 and integrity intact
- **completed** — run ended with the model's own completion signal
- **verified** — a passing `verify_contract` reading was observed on the wire
- **intact** — `protect` paths unchanged (and external grader bytes unchanged)

Report solve rate and integrity/security failures first; turns/tokens are secondary.
