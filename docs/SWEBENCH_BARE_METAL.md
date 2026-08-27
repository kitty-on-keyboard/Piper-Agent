# SWE-bench on bare metal: the protocol, and what it costs

Written 2026-08-19, continuing [BAKEOFF_HARNESS.md](BAKEOFF_HARNESS.md). This records a **protocol and its feasibility**,
not a score. No arm has been run. Every number below is about the machine, not the agent.

## Why not SWE-bench as published

The container is worth being precise about, because the obvious objection lands on the
wrong step. SWE-bench's container **never runs the agent**: the agent produces a `git
diff`, and the container only applies that patch and runs the project's test suite.
`sb-cli` moves even that off this machine. Containers alone do not disqualify us.

The real problems are two, and neither is about grading.

**SWE-bench measures the model, not the harness.** The claim from BAKEOFF_HARNESS is
same-weights, same-machine: 6/6 against Cline's 5/6, and 1.32x–1.74x on wall clock. Both
harnesses *solved* the same tasks — the entire separation was speed. A pass-rate benchmark
scores Qwen3.6-35B and returns a near-null on the axis where we differ.

**The published anchor is not in our condition.** `mini-swe-agent` at 18.8% runs *inside*
the container, where the repository's dependencies are installed and it can run the tests,
reproduce the bug, and check its own fix. An agent on a bare checkout is blind. Scoring
that against 18.8% measures the missing environment, not the loop.

## What we run instead

Keep SWE-bench's **instances** — they are Princeton's, which is the whole point, since our
own six tasks invite "you wrote the benchmark you win" — and drop its container.

| | |
|---|---|
| **tasks** | SWE-bench Lite, 300 instances, unmodified |
| **environment** | one `uv` venv per (repo, version) on the host, built from SWE-bench's own spec table |
| **grading** | the official `FAIL_TO_PASS` / `PASS_TO_PASS` lists, parsed by `swebench.harness.grading` |
| **arms** | LM_Pipe, Cline, `mini-swe-agent` (its `local` environment class), same weights, same machine, sequential |
| **reported** | solve rate **and** wall clock **and** turns |

Nothing leaves the machine and nothing runs in a container. The agent can run the project's
tests, which closes the handicap rather than apologising for it.

### The pre-registered inclusion rule

An instance is in the set **iff, at `base_commit` in a fresh venv, every `PASS_TO_PASS`
test passes and every `FAIL_TO_PASS` test fails.**

Mechanical, checkable, and settled *before* any arm runs. It excludes instances this
machine cannot host, never instances an arm found hard. The size of the excluded set is
published with the result. Choosing the subset any other way re-opens the objection this
whole exercise exists to close.

## What the machine can actually host

`scripts/swebench_probe_env.py`, one venv per (repo, version), SWE-bench's own python
version / pip packages / install command:

| | pairs built | instances |
|---|---|---|
| pure Python — django, sympy, pytest, sphinx, requests, pylint, flask, seaborn | 51/51 | **243/243** |
| xarray (compiles, but modern numpy suffices) | 1/1 | 5/5 |
| **astropy, matplotlib, scikit-learn** | **0/12** | **0/52** |
| **total** | **52/64** | **248/300 (83%)** |

Pure-Python environments build in **2–5 seconds each**. The 52 failures are one cause:
those three projects pin an era-appropriate numpy (1.17.3, 1.19.2) that has no arm64
wheels and does not compile on a modern toolchain. Nothing about them is agent-related, and
xarray shows the split is about the *pin*, not about compiling as such.

**83% is the ceiling on the denominator, not the denominator.** The inclusion rule below
has not been run yet, and it can only shrink this.

### Two deviations, both recorded by the run rather than remembered

**Python 3.6 is unavailable.** 77 of the 300 instances name it — django 3.0–3.2 (56),
scikit-learn (19), astropy (2). `uv` distributes 3.8+, and 3.6 does not build cleanly on
Apple silicon. Those pairs are built on the nearest available interpreter and carry
`python_deviation` in the results. Django 3.0 running its own suite under 3.8 is a real
difference; the inclusion rule above is what decides whether it matters, per instance.

The full spread, measured rather than assumed:

```
3.6    77   django 56, scikit-learn 19, astropy 2      unavailable -> deviation
3.8    20   django 19, matplotlib 1                    available
3.9   165   sympy 77, django 30, pytest 17, sphinx 16  available
3.10    5   xarray 5                                   available
3.11   33   matplotlib 22, django 9, flask 2           available
```

**One install needed a newer-pip fallback.** pylint 2.15's build backend predates PEP 660,
so a modern editable install refuses where the benchmark's container — with an old pip
that still did legacy editable installs — never noticed. The probe tries the spec command
verbatim, then `editable_mode=compat`, then non-editable, and records which one worked in
`install_fallback`. An unrecorded fallback is a silent protocol change.

## The leak that had to be closed first

A plain clone of a mirror carries the repository's **entire default branch**, including the
commit that fixed the issue. On `astropy__astropy-12907`:

```
git log --all --oneline | grep -i separab
2f1bfd254d Backport PR #12907: Correctly calculate the separability of a nested compound model
```

`git show` then hands over the reference patch verbatim. Any score measured against a
repository in that state is measuring `grep`.

`checkout()` now makes `base_commit` the only reachable ref — one branch, no remote, no
tags — and `assert_history_truncated()` refuses the instance if anything still points past
it. Verified on astropy, django and sympy: one ref, zero tags, and the newest reachable
commit *is* the base commit. Ancestors survive, which is exactly what a developer sitting
on that commit would have.

Objects for later commits remain in the hardlinked pack and could in principle be reached
by enumerating them with `git cat-file --batch-all-objects`. No benchmark of this shape
closes that; it is written down here rather than left for someone else to find.

## The context store is part of the agent, and it is half-live here

`src/pcc` is not an optional extra — a benchmark run with a dead store measures a
crippled LM_Pipe. It opens unconditionally, so there is no setting to get wrong, but a
failure to open is logged as a *survivable degradation* (`context_journal opened=0`) and
the run continues silently without durable context. "It should be on" is not evidence, so
every instance now records four independent signals.

Measured on the first two instances (django-14999, sympy-16503):

| | django-14999 | sympy-16503 |
|---|---|---|
| turns | 8 | 26 |
| store bytes | 576,808 | 1,849,888 |
| `context_recall` / `context_rehydrate` calls | **0** | **0** |
| `recall_scope` at mission start | items 1, sessions 1, **advertised 0** | items 1, sessions 1, **advertised 0** |

**The write side is live and busy.** 0.5–1.8 MB journalled per instance.

**The read side is structurally unreachable in this configuration.** The sidecar advertises
recall only when an *earlier* session left something — `sessions > 1 && items > 0`. A fresh
checkout per instance means `sessions == 1` every time, so recall is never advertised and
the model never reaches for it. That is a property of how the benchmark provisions
workspaces, not a defect in `src/pcc`, whose measured value has always been cross-session.

Two consequences worth deciding deliberately rather than by default:

- The **primary run keeps a fresh store per instance.** It is the standard, comparable,
  instance-independent configuration, and it is leak-safe.
- A persistent per-repo store would exercise the read side, but reintroduces exactly the
  leak closed above: instances of one repo sit at different commits, so an earlier
  instance's snapshots can contain a later instance's future. It is only safe **ordered by
  base-commit date ascending**, where the store can never hold anything newer than the
  current base. Run that way, on django's 114 instances, it measures something nobody
  publishes: what cross-task memory is worth on SWE-bench. It is a *secondary* result
  because it breaks instance independence, and it must be labelled as such.

### A metric that nearly lied

The first version of this evidence sized the store with `db_bytes` alone and reported
4096 bytes on both instances — a bare SQLite header page, which reads as "the store is
empty". SQLite in WAL mode leaves the main database untouched until a checkpoint; the
1.8 MB was in `.lmp-context.db-wal`. `store_bytes` is now db + wal.

`remember_fact` is likewise **not** a proxy for writes: the turn sink persists every turn
from inside the context store without any tool call, so counting the tool would report
zero on a run that wrote continuously.

### And it was going into the patches

`.lmp-context.db` is written to the **workspace root**, so `git add -A` staged a
multi-megabyte SQLite blob into every generated patch. Now excluded by pathspec rather
than deleted, so the evidence above survives extraction. Verified absent from both smoke
patches.

## A checker bug that would have thrown away good instances

The first full inclusion run excluded django instances at an implausible rate with
`f2p_not_failing`. It was wrong, and the way it was wrong is worth keeping.

`django__django-11099` printed `AssertionError: ValidationError not raised` **twice** —
its bug reproducing exactly as the benchmark says it should — and the checker still called
its `FAIL_TO_PASS` tests "not failing". A test that fails inside `with self.subTest(...)`
is reported under a **bare method name**, so the parser yields

```
FAILED  'test_ascii_validator'
```

while `FAIL_TO_PASS` asks for

```
'test_ascii_validator (auth_tests.test_validators.UsernameValidatorsTests)'
```

**SWE-bench's own grading never meets this.** It asks whether F2P tests *pass* after a
patch, and a passing test always prints its fully qualified name. Only the
**precondition** — are these tests failing *before* any patch — has to read the name of a
failing subtest, and that is a check their harness does not run at evaluation time. Reusing
their parser did not protect against it, because the parser is fine; the mismatch is in what
we asked of it.

`TestNameResolver` now falls back to the bare name, and **refuses when it is ambiguous**:
`test_help_text` exists in four different classes in one django module, so a bare key is
trusted only when the bare name is unique among the wanted tests *and* appears once in the
output. Excluding an instance costs one data point; attributing one class's failure to
another class's identically named method corrupts the result and still looks like a score.

The general lesson, which this project keeps relearning: **an exclusion that looks
plausible is more dangerous than one that looks broken.** A 100% exclusion rate gets
investigated; a 60% one gets believed.

### The same bug again, one layer down

The corrected run still excluded django heavily, now as `p2p_not_passing`. Recording
*which* status each failing test actually had is what broke it open: **224 ABSENT against
1 ERROR**. Almost nothing was failing. Almost everything was missing.

Under `--verbosity 2`, a django test that HAS a docstring prints over two lines — the
identity on one, the docstring and the verdict on the next:

```
test_squashed_name_with_start_migration_name (migrations.test_commands.SquashMigrationsTests)
--squashed-name specifies the new migration's name. ... ok
```

`parse_log_django` keys on the line carrying the verdict, so the result is filed under the
**docstring** and the real name never appears. `PASS_TO_PASS` asks for the real name, finds
nothing, and the instance is excluded for tests that "did not run" when they ran and passed.

That it is a deviation and not how the benchmark works is checkable: SWE-bench's own
`PASS_TO_PASS` for `django__django-11039` holds exactly **4** docstring-shaped entries;
our run produced **61**. Their lists were built with this same parser, so in their
environment those tests printed their names.

`django_docstring_aliases` recovers the name from the preceding line. Aliases are
**additive and never override** — those 4 genuine docstring keys are in the published list
and must keep resolving, so the parser's own output always wins on an exact key.

| `django__django-11039` | P2P absent | verdict |
|---|---|---|
| before | 59 / 88 | OUT |
| after | **0 / 88** | **IN** |

Two bugs, same family: **reusing someone's parser does not protect you from asking it a
question it was never asked.** SWE-bench's grading only ever reads names of tests that
*passed*, after a patch. The precondition has to read the names of tests that *failed*, and
of every test that ran — and both of those paths were unexercised.

### And one genuine exclusion, for a reason worth knowing

`django__django-11099` is *still* excluded after the fix, on one test:
`test_help_text (auth_tests.test_validators.UserAttributeSimilarityValidatorTest)` is in
its `FAIL_TO_PASS` list, but the instance's `test_patch` never touches `test_help_text` and
the change is about a username regex. It passes here. SWE-bench's F2P list captured a
failure that was environment-dependent in their container and does not reproduce in ours.

That is the inclusion rule doing its job. The instance's stated precondition does not hold
on this machine, so no arm is scored on it, and the reason is in the ledger.

## The full catalogue of what had to be fixed

Every one of these excluded real instances while looking like a legitimate verdict. They
are listed because the pattern matters more than any single entry: **each was found only
by refusing to believe a plausible exclusion rate**, and each was invisible in the
summary — the ledger said "tests did not fail", which is what a correct run of a
well-behaved instance also says.

| # | what | how it presented | instances at risk |
|---|---|---|---|
| 1 | subTest failures are reported under a **bare method name** | `f2p_not_failing` on an instance whose log literally says `AssertionError: ValidationError not raised` | scattered |
| 2 | docstring'd django tests print **name and verdict on different lines** | `p2p_not_passing`, 224 ABSENT vs 1 real failure | most of django |
| 3 | two tests **concatenated onto one line** when a verdict never arrives | both tests vanish; parser invents a name like `test_a (mod.A) ... test_b (mod.B)` | rare |
| 4 | django's **test requirements never installed** — the spec says `requirements.txt` and the probe treated that as a no-op | 30 P2P tests SKIPPED, which is not passing | ~9 |
| 5 | deleting **all** git refs broke `setuptools_scm` | pytest installs as `0.1.dev10157`, its own `tox.ini` refuses: *"requires pytest-2.0"* | all 17 pytest |
| 6 | `tox` never installed, though the published `test_cmd` is `tox --current-env` | `seen=0`, empty log | all 16 sphinx |
| 7 | hardcoded `pip install -e .` instead of the spec's `pip install -e .[test]` | conftest dies on `import docutils` | sphinx |
| 8 | spec **`pre_install`** steps ignored (the seds pinning Jinja2, markupsafe, sphinxcontrib) | subtly wrong environment, tests fail as if the code were broken | sphinx |
| 9 | setuptools ≥81 **removed `pkg_resources`** | `No module named 'pkg_resources'` at import; no test runs | old sphinx |

Fixes 1–3 are parsing, 4–9 are environment. Two of them — 5 and 7 — were self-inflicted:
the leak fix in the previous section deleted tags that `setuptools_scm` needs, and a
hardcoded install command quietly built a different environment from the one the benchmark
specifies.

**Fix 5 is the one to remember.** Closing the history leak and keeping the build working
looked like a trade-off and was not: a tag that is an **ancestor** of `base_commit` leaks
nothing, because every commit reachable from it is already reachable from `base_commit`.
`checkout()` now deletes only refs pointing at or past the future. Re-verified afterwards
on `astropy__astropy-12907` — the fix commit `2f1bfd254d` and the PR #12907 merge remain
unreachable, while 6 genuine *ancestor* commits mentioning separability stay visible,
exactly as they would for a developer sitting on that commit.

### What is still excluded, honestly

**sphinx (16 instances) cannot be hosted.** After tox, `.[test]`, `pre_install` and the
setuptools pin, its tests finally collect and run — and then error with
`sphinx.errors.ExtensionError` from Jinja2/sphinxcontrib version drift that the era's
pins no longer resolve on a modern toolchain. This was stopped deliberately rather than
chased further; the inclusion rule exists to exclude what the machine cannot host, and
saying so is the honest outcome.

**astropy, matplotlib, scikit-learn (52 instances)** pin numpy 1.17.3 / 1.19.2, which has
no arm64 wheels and does not build. Same category, decided earlier.

## What exists

| | |
|---|---|
| `scripts/swebench_run.py` | provisions a leak-free checkout, drives the agent, extracts the patch |
| `scripts/swebench_probe_env.py` | builds one venv per (repo, version); feasibility |
| `scripts/swebench_cleanup.py` | accounts for every byte written outside the repo, and removes it |
| `scripts/agent_eval.py` | `drive_sidecar()` extracted, so both evaluators share one wire conversation |

`drive_sidecar` is the important one. A second evaluator that reimplemented the protocol
would drift, and a SWE-bench number would stop being a statement about the same agent the
pins measure. It was extracted and then validated by a full 6/6 corpus run through it.

## Cleaning up

Everything lives in **`$LMP_SWEBENCH_WORK`** (default `~/swebench_work`) — mirrors, venvs,
the parquet, and the measurements. Nothing is inside the repo except source.

```bash
./scripts/swebench_cleanup.py
```

Dry run by default; it prints every path with its size and why it exists, and separates
what it may delete from what it merely touched (`~/.cache/uv` and the borrowed Homebrew
interpreter are shared with the rest of the machine and are never removed). `--yes`
deletes. `runs/` is held back unless `--include-runs` is passed, because that directory is
the measurements.

Deletion is safe: mirrors re-clone, venvs rebuild in seconds, the parquet is a 1 MB
download.

## The denominator

The inclusion rule, run over all 248 hostable instances:

| | instances | of Lite |
|---|---|---|
| SWE-bench Lite | 300 | |
| no hostable environment | 52 | |
| checked | 248 | |
| **INCLUDED — strict** (every F2P fails, every P2P passes) | **197** | **65%** |
| **INCLUDED — relaxed** (at least one F2P fails, every P2P passes) | **212** | **71%** |

| repo | checked | strict | relaxed |
|---|---|---|---|
| django/django | 114 | 93 | 108 |
| sympy/sympy | 77 | **77** | **77** |
| pytest-dev/pytest | 17 | 13 | 13 |
| sphinx-doc/sphinx | 16 | 0 | 0 |
| psf/requests | 6 | 1 | 1 |
| pylint-dev/pylint | 6 | 3 | 3 |
| pydata/xarray | 5 | 3 | 3 |
| mwaskom/seaborn | 4 | 4 | 4 |
| pallets/flask | 3 | 3 | 3 |

sympy is 77/77. django is 82% strict and 95% relaxed. Between them they are 185 of the
197 included instances, which is worth stating plainly: **this is largely a django and
sympy benchmark**, and any per-repo effect will be dominated by those two.

Lists: `runs/inclusion/included_instances.txt` (197) and
`runs/inclusion/included_relaxed.txt` (212). Both are fixed now, before any arm runs.

### Why there are two, and which is the headline

The rule at the top of this document is stricter than SWE-bench's own resolve condition
needs. An instance resolves when every F2P passes *after* the patch; a test already
passing *before* it will still pass after, so it never blocks resolution — and it grants
no free point either, because a do-nothing agent is still caught by the F2P tests that do
fail at base.

Of the F2P tests that pass at base, **20 of 22 are not mentioned anywhere in the
instance's own `test_patch`**. They are pre-existing tests that happened to fail in
Princeton's container, not tests that prove the fix.

This is a rule changed after seeing data, which is exactly what pre-registration forbids,
so it is handled the only way that stays honest: **no arm has run**, so the change cannot
be informed by any score, and **both sets are published**. The headline stays on the
strict 197; the relaxed 212 is reported beside it with its own N. Anyone who distrusts the
reasoning can use the strict list and ignore the argument entirely.

## Does the environment earn its keep? A/B, graded

The protocol's central assumption is that giving the agent the project's own interpreter
matters. That was untested, and the first single instance argued the other way — with an
environment `django__django-14999` took 21 turns and 152 s against 10 turns and 60 s blind,
both correct. So it was measured properly: ten instances chosen by a fixed stride over the
included list (a rule set before any result was seen), run twice, graded both times.

| | resolved | turns | wall clock |
|---|---|---|---|
| **with the project's environment** | **5/10** | 240 | **21.7 min** |
| blind (`--no-environment`) | 4/10 | 247 | 33.5 min |

**The solve rate is not the finding.** Exactly one instance flipped — `mwaskom__seaborn-3190`,
resolved with an environment and unfixed without — and runs at temperature 0.6 are not
reproducible even at a fixed seed. One flip out of ten is not evidence of a capability
difference and should not be reported as one.

**The wall clock is the finding: 1.54x, at identical turn counts.** Same work, far less of
it wasted. It concentrates in the instances where a blind agent cannot tell whether it has
finished:

```
django__django-12470    195 s / 29 turns   ->   460 s / 57 turns blind, 6 files touched
sympy__sympy-16106      244 s / 37 turns   ->   855 s / 36 turns blind (near the 900 s cap)
```

`django__django-12470` is the shape of it: unable to verify, the blind run kept editing and
spread its patch across **six files** instead of one. Turn counts being equal in total while
wall clock differs by half means the blind agent is not taking *more* steps, it is taking
*longer, less decisive* ones.

**Decision: keep the environment.** It is the fairer condition — it is the one the published
anchor runs in — it is materially faster on the axis this project actually competes on, and
its effect on the solve rate is at worst neutral.

### What the pilot says about the full run

Ten instances took 21.7 minutes of generation. Extrapolated to the 197-instance included
set, LM_Pipe's arm is roughly **7 hours**, plus about 1.5 hours of grading at ~30 s an
instance. That is an overnight run, not the multi-day commitment the earlier handoff
assumed. Three arms is a couple of days of machine time.

The 5/10 is not a headline. It is ten instances from an easier-than-Verified set under an
inclusion rule of our own construction, and it is reported here only to show that
generation and grading connect end to end.

## Status, and what is not done

- **Done.** Protocol fixed; history leak closed and re-verified; environments built for
  248 of 300; grading imported from `swebench.harness` with no Docker; **the inclusion
  rule has been run end to end** and the denominator is settled at 197 strict / 212
  relaxed.
- **Not done.** No arm has been run. There is no score in this document, by design.
- **Not done.** The agent is not yet given its instance's venv, so it still cannot run
  the project's tests while solving. The environments now exist — wiring them into
  `swebench_run.py` is what closes the handicap this protocol was built to close.

The denominator came in smaller than 300 and the reasons are all written down, which was
the point. What it cost to get there is nine distinct defects, every one of which
presented as a plausible verdict rather than as a failure.

## A pin correction, made in passing

The corpus pin was stale (`rename_across_files` now solves). Re-pinning exposed that a
**single-seed pin at temperature 0.6 is a coin flip**: one run put holdout at 3/4 on an
`email_regex_edges` stall that three seeds do not reproduce. `summarize()` already said so
in a comment — *"at temperature 0.6 an individual task is far too noisy to be a floor"* —
and already supported `name@seed=N` keys.

`evals/agent/pins.json` is now a three-seed quality pin: **corpus 18/18, holdout 12/12**
over seeds 7/13/42, with the run settings recorded (the previous pin predated that field
and compared against a "legacy 0.6/seed-7 convention"). Private, never a floor, observed
13/15.
