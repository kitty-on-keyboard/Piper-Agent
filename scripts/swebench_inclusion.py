#!/usr/bin/env python3
"""Run SWE-bench's own precondition on this machine, and fix the benchmark's denominator.

THE RULE, decided before any arm runs and never revisited afterwards:

    An instance is INCLUDED iff, at base_commit in its venv, every PASS_TO_PASS test
    passes and every FAIL_TO_PASS test fails.

WHY IT IS NEEDED. SWE-bench grades by applying the agent's patch, applying the test patch,
and running the tests; an instance resolves when every F2P passes and every P2P still
passes. That is only meaningful if the tests behave as the benchmark claims BEFORE any
agent touches them. On Princeton's Linux container that holds by construction. We moved
the environment -- macOS arm64, and 56 instances on Python 3.8 where the spec says 3.6 --
so it has to be re-established. Two ways it breaks, distorting in opposite directions:

    P2P fails at base    grading requires every P2P to pass, so the instance scores
                         unresolved for EVERY arm whatever patch it writes. Pure noise,
                         and it penalises whichever agent came closest.

    F2P passes at base   the bug does not reproduce here, so an agent that does NOTHING
                         satisfies the resolve condition. A free point for every arm.

Neither says anything about an agent. This strips both out.

THIS IS SWE-BENCH'S CRITERION, NOT OURS. It is the validation they run when constructing
instances. We re-execute it because we moved the ground underneath it.

NOTHING HERE IS REIMPLEMENTED. The test command, the test-file restoration and the
directives come from the instance's own `eval_script`; the output is parsed by the
project's own `log_parser` out of swebench.harness. What this script contributes is the
adaptation from container to host, which is listed in `adapt_eval_script` and is the only
place a deviation can hide.

PRE-REGISTRATION. The output is a fixed list plus an exclusion ledger, written once and
used unchanged by every arm. An instance can only be excluded for a reason visible before
any agent runs, so no score can influence membership. Choosing the subset any other way
re-opens the "you wrote the benchmark you win" objection in a new costume.

Usage:
  swebench_inclusion.py --limit 5 --out runs/inclusion
  swebench_inclusion.py --repo django/django --out runs/inclusion
  swebench_inclusion.py --out runs/inclusion            all hostable instances
"""

import argparse
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

SPEC_TABLE = {
    "django/django": "SPECS_DJANGO", "sympy/sympy": "SPECS_SYMPY",
    "pytest-dev/pytest": "SPECS_PYTEST", "sphinx-doc/sphinx": "SPECS_SPHINX",
    "psf/requests": "SPECS_REQUESTS", "pylint-dev/pylint": "SPECS_PYLINT",
    "pallets/flask": "SPECS_FLASK", "mwaskom/seaborn": "SPECS_SEABORN",
    "matplotlib/matplotlib": "SPECS_MATPLOTLIB",
    "scikit-learn/scikit-learn": "SPECS_SKLEARN",
    "astropy/astropy": "SPECS_ASTROPY", "pydata/xarray": "SPECS_XARRAY",
}

WORK = os.environ.get(
    "LMP_SWEBENCH_WORK", os.path.expanduser("~/swebench_work")
)
DATASET_V5 = os.path.join(WORK, "data", "swebench_lite_test_v5.parquet")
ENV_PROBE = os.path.join(WORK, "runs", "env_probe.json")
ENVS = os.path.join(WORK, "envs")
SPEC_FILE = os.path.join(WORK, "data", "swebench_python_specs.py")

START_MARKER = ">>>>> Start Test Output"
END_MARKER = ">>>>> End Test Output"


def adapt_eval_script(script, workspace, env_dir):
    """Container -> host. Every deviation from the published script lives here.

    Four substitutions, and nothing else is touched:

      /testbed                     -> this instance's checkout
      conda activate testbed       -> neutralised; the venv is on PATH instead
      source .../conda activate    -> neutralised, same reason
      git config --global ...      -> neutralised. It appends a safe.directory entry to
                                      the USER'S ~/.gitconfig, once per instance; 248 of
                                      those is a benchmark editing the machine it runs on.
                                      We own the checkout, so it was never needed.

    The test command, the test-file restoration and the directives are left exactly as
    published.
    """
    out = []
    for line in script.split("\n"):
        stripped = line.strip()
        if stripped.startswith("source /opt/miniconda3/bin/activate"):
            out.append(": # conda neutralised; venv is on PATH")
            continue
        if stripped.startswith("conda activate"):
            out.append(": # conda neutralised; venv is on PATH")
            continue
        if stripped.startswith("git config --global"):
            out.append(": # refused: would write to the user's ~/.gitconfig")
            continue
        out.append(line.replace("/testbed", workspace))
    return "\n".join(out)


def extract_test_output(stdout):
    """Only what lies between the markers; the parsers assume nothing else is present."""
    if START_MARKER not in stdout:
        return None
    body = stdout.split(START_MARKER, 1)[1]
    if END_MARKER in body:
        body = body.split(END_MARKER, 1)[0]
    return body


def hostable_keys():
    """(repo, version) pairs whose environment actually built. See swebench_probe_env.py."""
    if not os.path.exists(ENV_PROBE):
        sys.exit(f"no env probe at {ENV_PROBE}; run swebench_probe_env.py first")
    with open(ENV_PROBE) as fh:
        return {r["key"]: r for r in json.load(fh) if r.get("ok")}


def spec_for(repo, version):
    """The benchmark's own install recipe for this (repo, version), or None."""
    ns = runpy.run_path(SPEC_FILE)
    table = SPEC_TABLE.get(repo)
    return ns[table].get(version) if table else None


def install_instance(workspace, env_dir, timeout=1800, spec=None):
    """Point the venv at THIS checkout before the eval script runs.

    The environment probe installed each package from a temporary tree and then deleted
    it, so a venv's editable install dangles and a non-editable one is a stale copy of the
    environment-setup commit. Either way the tests would run against the wrong code, which
    is the kind of error that produces a plausible number rather than an obvious failure.
    """
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = env_dir
    env["PATH"] = os.path.join(env_dir, "bin") + os.pathsep + env["PATH"]
    env.pop("PYTHONHOME", None)
    # THE SPEC'S COMMAND FIRST, not a hardcoded one. sphinx's is
    # `python -m pip install -e .[test]`, and installing plain `-e .` drops the test
    # extra -- its conftest then dies on `import docutils` and every sphinx instance is
    # excluded for running no tests. A hardcoded install command quietly builds a
    # different environment from the one the benchmark specifies.
    base = (spec or {}).get("install", "python -m pip install -e .")
    attempts = [
        base,
        base + " --config-settings editable_mode=compat",
        base.replace(" -e ", " "),
        "python -m pip install -e .",
    ]
    if (spec or {}).get("pre_install"):
        for step in spec["pre_install"]:
            step = re.sub(r"\bsed -i (?!')", "sed -i '' ", step)
            subprocess.run(step, cwd=workspace, shell=True, env=env,
                           capture_output=True, text=True, timeout=600)
    for command in attempts:
        res = subprocess.run(command, cwd=workspace, shell=True, env=env,
                             capture_output=True, text=True, timeout=timeout)
        if res.returncode == 0:
            return True, command, ""
    return False, attempts[-1], (res.stderr or res.stdout)[-800:]


# A test identity that has been spliced onto the END of another test's line.
DJANGO_RUN_ON = re.compile(r"(?<=\.\.\. )(?=[A-Za-z_]\w* \([\w.]+\) \.\.\. )")


def normalise_django_log(log):
    """Split test results that django ran together onto one line.

    Observed on django__django-11905:

        test_isnull_non_boolean_value (lookup.tests.LookupTests) ... test_iterator (lookup.tests.LookupTests) ... ok

    unittest writes the identity, runs the test, then writes the verdict. Anything the test
    itself emits on the same stream lands between the two, and a verdict that never arrives
    leaves the next test's identity appended to the previous line. `parse_log_django` then
    takes everything before ` ... ok` as one name and BOTH tests disappear.

    Splitting on an identity that follows a ` ... ` restores one result per line. It is
    rare -- one line in that log, none in another -- so this recovers few instances; it is
    here because a parser silently inventing a test called
    "test_a (mod.A) ... test_b (mod.B)" is the kind of thing that quietly moves a score.
    """
    return DJANGO_RUN_ON.sub("\n", log)


DJANGO_TEST_ID = re.compile(r"^[\w.]+ \([\w.]+\)$")
DJANGO_VERDICTS = (
    (" ... ok", "PASSED"), (" ... OK", "PASSED"), (" ... FAIL", "FAILED"),
    (" ... ERROR", "ERROR"), (" ... skipped", "SKIPPED"),
)


def django_docstring_aliases(log):
    """Recover the test names django hides behind docstrings.

    THE BUG. With `--verbosity 2`, a test method that HAS a docstring is printed over two
    lines -- the identity on one, the docstring and the verdict on the next:

        test_squashed_name_with_start_migration_name (migrations.test_commands.SquashMigrationsTests)
        --squashed-name specifies the new migration's name. ... ok

    `parse_log_django` keys on the line that carries the verdict, so it files that result
    under the DOCSTRING and the real test name never appears. PASS_TO_PASS asks for the
    real name, finds nothing, and the instance is excluded for having tests that "did not
    run" -- when they ran and passed.

    Measured on django__django-11039: 59 of 88 P2P tests absent before this, 0 after, and
    the instance flips from excluded to included.

    HOW WE KNOW THIS IS A DEVIATION AND NOT HOW THE BENCHMARK WORKS. SWE-bench's own
    PASS_TO_PASS list for that instance contains exactly 4 docstring-shaped entries; our
    run produced 61. Their lists were built with this same parser, so in their environment
    those tests printed their names and here they print their docstrings. Whatever causes
    that -- it is not worth chasing -- the log still holds both halves, one line apart.

    ALIASES ARE ADDITIVE, NEVER OVERRIDES. Those 4 genuine docstring keys are in the
    published list and must keep resolving, so the parser's own output always wins on an
    exact key and this only fills what it left empty.
    """
    aliases = {}
    lines = [line.strip() for line in log.split("\n")]
    for index, line in enumerate(lines[:-1]):
        if not DJANGO_TEST_ID.match(line):
            continue
        following = lines[index + 1]
        for suffix, status in DJANGO_VERDICTS:
            if suffix in following:
                aliases[line] = status
                break
    return aliases


class TestNameResolver:
    """Matches a wanted test name against parsed statuses, tolerating subTest reporting.

    THE BUG THIS EXISTS FOR. A test that fails inside `with self.subTest(...)` is reported
    by django under a BARE method name, so the parser yields

        FAILED  'test_ascii_validator'

    while FAIL_TO_PASS asks for

        'test_ascii_validator (auth_tests.test_validators.UsernameValidatorsTests)'

    An exact-match check reads that as "the test did not fail" and excludes a perfectly
    good instance. Measured on django__django-11099: the run printed
    "AssertionError: ValidationError not raised" twice and the checker still called it
    f2p_not_failing.

    SWE-bench's own grading never meets this. It asks whether F2P tests PASS after a patch,
    and a passing test always prints its fully qualified name. Only the PRECONDITION -- are
    these tests failing BEFORE any patch -- has to read the name of a failing subtest, and
    that is a check their harness does not run at evaluation time.

    AMBIGUITY IS REFUSED, NOT GUESSED. `test_help_text` exists in four different classes in
    one django test module. A bare key is only trusted when the bare name is unique among
    the wanted tests AND appears once in the parsed output; otherwise the name is recorded
    as ambiguous and resolves to None, which excludes the instance. Excluding an instance
    costs one data point. Attributing one class's failure to another's identically named
    method corrupts the result and looks like a score.
    """

    def __init__(self, statuses, wanted):
        self.statuses = statuses
        self.bare_used = 0
        self.ambiguous = []
        self._wanted_bare = {}
        for name in wanted:
            self._wanted_bare.setdefault(bare_name(name), []).append(name)

    def status(self, wanted):
        if wanted in self.statuses:
            return self.statuses[wanted]
        bare = bare_name(wanted)
        if bare not in self.statuses:
            return None
        if len(self._wanted_bare.get(bare, [])) != 1:
            if bare not in self.ambiguous:
                self.ambiguous.append(bare)
            return None
        self.bare_used += 1
        return self.statuses[bare]


def bare_name(name):
    """`test_x (a.b.C)` -> `test_x`; a name with no qualifier is already bare."""
    return name.split(" ", 1)[0]


def check_instance(row, env_dir, args, checkout):
    """Returns the precondition verdict for one instance."""
    from swebench.harness.utils import make_test_spec
    import swebench.harness.log_parsers as parsers

    instance = {k: row[k] for k in row.index}
    spec = make_test_spec(instance)
    f2p, p2p = list(spec.FAIL_TO_PASS), list(spec.PASS_TO_PASS)

    verdict = {
        "instance_id": row["instance_id"], "repo": row["repo"],
        "version": row["version"], "env_key": os.path.basename(env_dir),
        "n_f2p": len(f2p), "n_p2p": len(p2p),
    }
    workspace = tempfile.mkdtemp(prefix=f"swe_incl_{row['instance_id']}_")
    started = time.time()
    try:
        checkout({"repo": row["repo"], "base_commit": row["base_commit"]}, workspace)

        spec = spec_for(row["repo"], row["version"])
        installed, command, error = install_instance(workspace, env_dir,
                                                     args.install_timeout, spec)
        verdict["install_cmd"] = command
        if not installed:
            verdict.update(included=False, reason="install_failed", error=error)
            return verdict

        script = adapt_eval_script(row["eval_script"], workspace, env_dir)
        script_path = os.path.join(workspace, ".swebench_eval.sh")
        with open(script_path, "w") as fh:
            fh.write(script)

        env = os.environ.copy()
        env["VIRTUAL_ENV"] = env_dir
        env["PATH"] = os.path.join(env_dir, "bin") + os.pathsep + env["PATH"]
        env.pop("PYTHONHOME", None)
        try:
            # ONE STREAM, as the container does. The script runs under `set -x`, so the
            # `>>>>> Start Test Output` markers are emitted by the xtrace on STDERR while
            # the test runners write to whichever stream they please -- django uses
            # stderr, pytest uses stdout. Capturing them separately loses the interleaving
            # and the markers land in a different string from the output they delimit,
            # which reads as "the tests never ran" on a run that ran 120 of them.
            res = subprocess.run(["/bin/bash", script_path], cwd=workspace, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, timeout=args.test_timeout)
        except subprocess.TimeoutExpired:
            verdict.update(included=False, reason="test_timeout")
            return verdict

        body = extract_test_output(res.stdout)
        if body is None:
            verdict.update(included=False, reason="no_test_output",
                           error=(res.stderr or res.stdout)[-800:])
            return verdict

        parser = getattr(parsers, row["log_parser"], None)
        if parser is None:
            verdict.update(included=False, reason="no_parser",
                           error=f"unknown log_parser {row['log_parser']}")
            return verdict
        statuses = parser(body, spec)
        if row["log_parser"] == "parse_log_django":
            body = normalise_django_log(body)
            statuses = parser(body, spec)
            recovered = django_docstring_aliases(body)
            merged = dict(recovered)
            merged.update(statuses)  # the parser's own keys always win
            verdict["docstring_aliases"] = len(recovered)
            statuses = merged
        verdict["tests_seen"] = len(statuses)
        resolve = TestNameResolver(statuses, f2p + p2p)
        verdict["bare_name_matches"] = resolve.bare_used
        verdict["ambiguous_names"] = resolve.ambiguous[:8]

        # THE RULE. At base commit, with no model patch applied: the bug's tests must
        # fail, and everything else must pass. A test that did not run at all counts
        # against the precondition -- silence is not a pass.
        f2p_bad = [t for t in f2p if resolve.status(t) not in ("FAILED", "ERROR")]
        p2p_bad = [t for t in p2p if resolve.status(t) != "PASSED"]
        # PASSED and absent are both "not failing", and they mean opposite things: the
        # first is a precondition that genuinely does not hold on this machine, the second
        # is a test that never ran, which is a harness fault dressed as a verdict. The
        # subTest bug looked exactly like the former until the statuses were printed.
        verdict["f2p_not_failing"] = [
            {"test": t, "status": resolve.status(t) or "ABSENT"} for t in f2p_bad[:8]
        ]
        verdict["p2p_not_passing"] = [
            {"test": t, "status": resolve.status(t) or "ABSENT"} for t in p2p_bad[:8]
        ]
        verdict["f2p_absent"] = sum(1 for t in f2p_bad if resolve.status(t) is None)
        verdict["p2p_absent"] = sum(1 for t in p2p_bad if resolve.status(t) is None)
        verdict["n_f2p_not_failing"] = len(f2p_bad)
        verdict["n_p2p_not_passing"] = len(p2p_bad)

        # TWO VERDICTS, BOTH PUBLISHED. The rule at the top of this file -- every F2P test
        # fails -- turned out to be stricter than SWE-bench's own resolve condition needs,
        # and it excluded instances for a reason that cannot affect resolvability.
        #
        # Measured: of the F2P tests that PASS at base, 20 of 22 are not mentioned anywhere
        # in the instance's own test_patch. They are pre-existing tests that happened to
        # fail in Princeton's container, not tests that prove the fix. An instance resolves
        # when every F2P passes AFTER the patch; a test already passing before will still
        # pass after, so it never blocks resolution and never grants a free point either --
        # a do-nothing agent is still caught by the F2P tests that DO fail at base.
        #
        # So the relaxed rule is: all P2P pass, and AT LEAST ONE F2P fails.
        #
        # This is a rule changed after seeing data, which pre-registration exists to
        # forbid, so it is handled the only way that stays honest: NO ARM HAS RUN, the
        # change cannot be informed by any score, and BOTH sets are written out. The
        # headline stays on the strict set; the relaxed set is reported beside it with its
        # own N. Anyone who distrusts the reasoning can use the strict list.
        verdict["included_strict"] = not f2p_bad and not p2p_bad
        verdict["included_relaxed"] = (not p2p_bad) and len(f2p_bad) < len(f2p)

        if f2p_bad and p2p_bad:
            reason = "f2p_not_failing+p2p_not_passing"
        elif f2p_bad:
            reason = "f2p_not_failing"
        elif p2p_bad:
            reason = "p2p_not_passing"
        else:
            reason = "ok"
        verdict.update(included=(reason == "ok"), reason=reason)
        return verdict
    except Exception as exc:
        verdict.update(included=False, reason="error",
                       error=f"{type(exc).__name__}: {exc}")
        return verdict
    finally:
        verdict["seconds"] = round(time.time() - started, 1)
        shutil.rmtree(workspace, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo")
    ap.add_argument("--instances")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--test-timeout", type=int, default=1800)
    ap.add_argument("--install-timeout", type=int, default=1800)
    args = ap.parse_args()

    import pandas as pd
    import swebench_run as sr  # for checkout(), so the tree matches what the agent gets

    df = pd.read_parquet(DATASET_V5)
    envs = hostable_keys()

    rows = []
    for _, row in df.iterrows():
        key = f"{row['repo'].replace('/', '__')}@{row['version']}"
        if key not in envs:
            rows.append({"instance_id": row["instance_id"], "repo": row["repo"],
                         "version": row["version"], "env_key": key, "included": False,
                         "reason": "no_environment"})
            continue
        rows.append(row)

    selected = []
    for item in rows:
        if isinstance(item, dict):
            continue
        if args.repo and item["repo"] != args.repo:
            continue
        if args.instances and item["instance_id"] not in args.instances.split(","):
            continue
        selected.append(item)
    if args.limit:
        selected = selected[: args.limit]

    no_env = [r for r in rows if isinstance(r, dict)]
    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "inclusion.json")
    done = {}
    if os.path.exists(results_path):
        with open(results_path) as fh:
            done = {r["instance_id"]: r for r in json.load(fh)}

    print(f"  {len(selected)} instance(s) to check "
          f"({len(no_env)} already excluded: no environment)")
    verdicts = list(no_env)
    for index, row in enumerate(selected, 1):
        iid = row["instance_id"]
        if iid in done and done[iid].get("reason") not in (None, "error"):
            verdicts.append(done[iid])
            print(f"[{index}/{len(selected)}] {iid:<34} cached {done[iid]['reason']}")
            continue
        env_dir = os.path.join(ENVS, f"{row['repo'].replace('/', '__')}@{row['version']}")
        print(f"[{index}/{len(selected)}] {iid:<34} ...", end="", flush=True)
        verdict = check_instance(row, env_dir, args, sr.checkout)
        verdicts.append(verdict)
        print(f" {'IN' if verdict['included'] else 'OUT'} "
              f"({verdict['reason']}, {verdict.get('seconds')}s)", flush=True)
        merged = {v["instance_id"]: v for v in verdicts}
        merged.update({k: v for k, v in done.items() if k not in merged})
        with open(results_path, "w") as fh:
            json.dump(list(merged.values()), fh, indent=2)

    merged = {v["instance_id"]: v for v in verdicts}
    merged.update({k: v for k, v in done.items() if k not in merged})
    final = list(merged.values())
    with open(results_path, "w") as fh:
        json.dump(final, fh, indent=2)

    included = [v for v in final if v.get("included")]
    with open(os.path.join(args.out, "included_instances.txt"), "w") as fh:
        fh.write("\n".join(sorted(v["instance_id"] for v in included)) + "\n")

    import collections
    ledger = collections.Counter(v.get("reason") for v in final if not v.get("included"))
    print(f"\n  INCLUDED {len(included)}/{len(final)}")
    print("  exclusion ledger:")
    for reason, count in ledger.most_common():
        print(f"    {reason:<32} {count}")
    print(f"\n  -> {os.path.join(args.out, 'included_instances.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
