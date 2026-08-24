# Handoff: run the arms

Paste this whole file as the opening prompt of a fresh session in
`/Users/dev/Desktop/seans_projects_local/LM_Pipe_2`.

Written 2026-08-20. The protocol, the denominator and the grader are **built and
validated**. Nothing is left to design. What remains is to run three arms and publish the
number.

Read [SWEBENCH_BARE_METAL.md](SWEBENCH_BARE_METAL.md) first — it is the full record,
including the nine defects that had to be fixed to make the denominator honest, and the
reasoning you should not relitigate.

---

## The job

**Run LM_Pipe, Cline and mini-swe-agent over the same 197 SWE-bench instances, on this
machine, same weights, and publish solve rate AND wall clock AND turns — whichever way it
comes out.**

The whole point is that the tasks are Princeton's, not ours. Our own six-task bakeoff
(LM_Pipe 6/6 in 141.4 s vs Cline 5/6 in 186.1 s) invites "you wrote the benchmark you
win", and this is the answer to that.

**A losing number honestly reported is worth more than the six-task win.** Do not tune
anything to improve it. If an arm loses, publish it.

---

## What is already settled — do not redo or relitigate

### The denominator: 197

`runs/inclusion/included_instances.txt` — **197 instances**, frozen before any arm ran.
`runs/inclusion/included_relaxed.txt` — 212, a documented secondary. **Headline uses the
strict 197.**

Per repo: django 93, sympy 77, pytest 13, seaborn 4, pylint 3, xarray 3, flask 3, requests
1, sphinx 0. It is largely a django-and-sympy benchmark and the writeup says so.

103 instances are excluded, all for reasons visible before any agent ran: 52 have no
hostable environment (astropy/matplotlib/scikit-learn pin numpy 1.17.3/1.19.2, no arm64
wheels), 16 are sphinx (old Jinja2/sphinxcontrib pins that no longer resolve), and the
rest failed SWE-bench's own precondition on this machine. Full ledger in
`runs/inclusion/inclusion.json`.

### The agent gets the project's environment

Settled by an A/B on 10 stride-sampled instances, both arms graded:

| | resolved | turns | wall clock |
|---|---|---|---|
| with environment | 5/10 | 240 | **21.7 min** |
| blind | 4/10 | 247 | 33.5 min |

The solve rate is **not** the finding — one instance flipped and runs are not reproducible
at fixed seed. The finding is **1.54x wall clock at identical turn counts**: a blind agent
cannot tell when it is done, so `django__django-12470` spread its patch over six files in
57 turns instead of one file in 29. Keep the environment. It is also the condition the
published anchor runs in.

### Why not a container

Asked and answered, in Sean's words: "container is not going to work for an agent whos
entire sell is bare metal." The container never runs the agent — but the *anchor* runs
inside one, so a blind agent scored against it measures the missing environment. Everything
here runs on the host, and grading imports `swebench.harness.grading` with no Docker.

---

## Run it

Work dir is `~/Desktop/seans_projects_local/swebench_work` (call it `$W`). Its venv is the
interpreter for every script below: `$W/venv/bin/python`.

**Hold this configuration constant across all three arms.** The pilot used
`--wall-clock 900` and it bit once (`sympy__sympy-16106` ended at 855 s), so the real run
raises it to 1200. Set it once; never per-arm.

```bash
W=/Users/dev/Desktop/seans_projects_local/swebench_work
V=$W/venv/bin/python
IDS=$(paste -sd, $W/runs/inclusion/included_instances.txt)
```

### Arm 1 — LM_Pipe (~7 hours)

```bash
$V scripts/swebench_run.py run --instances "$IDS" --out $W/runs/arm_lmpipe \
   --max-iterations 60 --wall-clock 1200 > $W/runs/arm_lmpipe.log 2>&1
```

Then grade it:

```bash
$V scripts/swebench_grade.py --predictions $W/runs/arm_lmpipe/predictions.json \
   --out $W/runs/grade_lmpipe --only-included $W/runs/inclusion/included_instances.txt
```

~30 s per instance, so about 1.5 hours. Grading needs no MLX, so it can run while nothing
else does — but **never while another arm is generating**, because CPU contention would
distort that arm's wall clock, and wall clock is half the result.

### Arm 2 and 3 — Cline and mini-swe-agent

Both need an OpenAI-compatible server on the **same checkpoint**, which is a second MLX
process, so LM_Pipe must be finished and gone first. From BAKEOFF_HARNESS:

```bash
~/.lmstudio/extensions/backends/vendor/_amphibian/app-mlx-generate-mac26-arm64@33/bin/python \
  -m mlx_lm server --model /Users/dev/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit \
  --port 8080
```

- **Cline** 3.0.55, provider id **`openai-compatible`** (`openai-native` posts to
  `/v1/responses`, which mlx_lm answers 404). Use `--data-dir` to keep state out of
  `~/.cline`. `scripts/xharness.py` already drives it for our own tasks; it needs a
  SWE-bench arm adding — reuse `swebench_run.py`'s checkout and patch extraction rather
  than writing new ones, or the numbers stop being comparable.
- **mini-swe-agent** 2.4.6 is installed in `$W/venv`. It has an `environment_class: local`
  (`minisweagent/environments/local.py`) so it runs host-side; its `swebench.yaml` config
  defaults to docker and `/testbed`, so point `cwd` at our checkout and set the class.

Both arms must get the **same venv on PATH** that LM_Pipe gets, or the comparison is
between an agent with an interpreter and two without.

---

## Traps that have already cost hours

- **Never two MLX processes.** One checkpoint is 15–19 GB resident on a 48 GB host and
  doing it has taken this machine down. `swebench_run.py` refuses to start if it sees one;
  the mlx_lm server has no such guard, so check `pgrep -f 'mlx_lm|lmp_sidecar'` yourself.
- **Do not pipe a long background run through `tail`.** It buffers and you see nothing for
  hours. Redirect to a file; `rows.json` is also written incrementally after every
  instance, so progress is always readable.
- **An exclusion that looks plausible is more dangerous than one that looks broken.** A
  100% failure rate gets investigated; a 60% one gets believed. Three separate parser bugs
  hid behind believable-looking ledgers. If a result looks reasonable, check *why* before
  accepting it.
- **`--only-included` on the grader.** Without it you will grade instances that are not in
  the set and the denominator moves.
- Runs are **not reproducible at fixed seed** (temperature 0.6). A single differing
  instance is never evidence. The pins are `corpus 18/18, holdout 12/12` over seeds
  7/13/42 — a floor set at the ceiling, so it will occasionally report a false FAIL. Re-run
  the named task before believing it, and **never lower the pin after a red**.

---

## Known-broken: both entries closed 2026-08-24

**Cancellation** (entry 6) was never the agent's. The harness sent `lmp/cancel` from
inside its `for raw in proc.stdout` loop, so the schedule was only consulted when a line
arrived — and nothing arrives while a tool executes. The sidecar's own path (StdinReader
sets the token on its reader thread; `pump_output` polls it at 200 ms) was correct all
along, and the extension's cancel never had the bug. Fixed with a watchdog thread plus a
`cancel_when_file` trigger; `cancel_long_shell` is now a real falsifier (900 s sleep vs
every timeout that could impersonate a cancel, and a ≤30 s prompt-kill assertion);
self-test scenario 14 pins the harness defect itself. Verified end-to-end on the real
sidecar: cancelled in 16 s with the child provably mid-flight. Full record in
`docs/AGENT_DEFECTS_SWEBENCH.md` entry 6.

**Fix-by-deletion** (entry 4): measured over all 197 arm-1 patches — 1 pure deletion in
178 non-empty (0.6%), zero broken tests from it, and two deletion-heavy patches actually
*resolved* their instances. Not a pattern; closed with no prompt change.

**Also fixed for the re-run:** a run that ends `awaiting_user` in an unattended harness is
now answered once via `lmp/message` ("no operator; decide yourself") and continued, instead
of shutting down against an empty patch — arm 1 lost `django__django-16910` exactly that
way. Turns are summed across yields; rows carry `unattended_replies`.

## Cleaning up

```bash
./scripts/swebench_cleanup.py          # dry run: every path, its size, why it exists
./scripts/swebench_cleanup.py --yes    # remove it
```

About 6 GB outside the repo (mirrors, per-version venvs, the harness venv). `runs/` is held
back unless `--include-runs` — **that directory is the measurements**. Stray per-instance
checkouts land in `$TMPDIR` and are ~250 MB each; `--only tmp --yes` clears them and it is
worth doing between arms.

---

## Uncommitted state

Everything from this work is uncommitted, alongside pre-existing changes from earlier
sessions. New this session:

```
scripts/swebench_run.py        generate patches (leak-free checkout, env, patch extraction)
scripts/swebench_probe_env.py  build one venv per (repo, version)
scripts/swebench_inclusion.py  the pre-registered inclusion rule
scripts/swebench_grade.py      patches -> resolved/unresolved, no Docker
scripts/swebench_cleanup.py    account for and remove everything outside the repo
docs/SWEBENCH_BARE_METAL.md    the protocol and every measurement
docs/AGENT_DEFECTS_SWEBENCH.md six LM_Pipe defects, three fixed this session
```

C++ changes: `.git/info/exclude` for LM_Pipe's workspace artifacts
(`src/surface/context_journal.*`), an `lmp/notice` protocol notification for survivable
degradations (`protocol/schema.json` + regenerated both sides + sidecar + extension), and a
post-compaction recall nudge (`src/context/context.cpp`). 49/49 gate green, six new tests.

`evals/agent/pins.json` is a three-seed quality pin measured this session.

**Sean has not been asked whether to commit any of it.** Ask before committing or pushing.
