#!/usr/bin/env python3
"""Can SWE-bench Lite's environments be built on bare metal, on THIS machine?

WHY THIS EXISTS. The plan is to run every arm on the host with no container, which only
works if the repositories' own dependencies install on macOS arm64. That is an empirical
question about 64 (repo, version) pairs and nobody's README answers it. This script
answers it by trying, and writes down what happened -- including the failures, which are
the interesting half.

WHAT IT MEASURES, AND WHAT IT DOES NOT. This builds the environment and checks the
package imports. It does NOT run the benchmark's tests; that is stage two, and it is a
different question (`swebench_probe_tests.py`). Feasibility first, because if the
environment cannot be built the test question never arises.

THE SPECS ARE SWE-BENCH'S, NOT OURS. Python version, pip packages, install command and
test command are read from the constants table published with the benchmark (fetched to
data/swebench_python_specs.py). Inventing our own install recipe would make any later
number a statement about our packaging skill rather than about the agent.

THE DEVIATION THIS WILL RECORD. 77 of the 300 instances name Python 3.6, which `uv` does
not distribute and which does not build cleanly on Apple silicon: django 3.0-3.2 (56),
scikit-learn (19) and astropy (2). Where the specified interpreter is unavailable the
probe retries on the nearest available one and records `python_deviation`, so the
disclosure is produced by the run rather than remembered afterwards. A pair that only
passes under a deviation is not silently equivalent to one that passes as specified --
django 3.0 running its own test suite under 3.8 is a real difference, and stage two is
what decides whether it matters.

The full spread, measured rather than assumed:

    3.6   77   django 56, scikit-learn 19, astropy 2      unavailable -> deviation
    3.8   20   django 19, matplotlib 1                    available
    3.9  165   sympy 77, django 30, pytest 17, sphinx 16, ...   available
    3.10   5   xarray 5                                   available
    3.11  33   matplotlib 22, django 9, flask 2           available

Usage:
  swebench_probe_env.py --pure-only          the 8 repos needing no compiler
  swebench_probe_env.py --repo sympy/sympy   one project
  swebench_probe_env.py                      all 64 pairs
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

WORK = os.environ.get(
    "LMP_SWEBENCH_WORK", os.path.expanduser("~/swebench_work")
)
DATASET = os.path.join(WORK, "data", "swebench_lite_test.parquet")
SPEC_FILE = os.path.join(WORK, "data", "swebench_python_specs.py")
ENVS = os.path.join(WORK, "envs")
MIRRORS = os.path.join(WORK, "repos")
RESULTS = os.path.join(WORK, "runs", "env_probe.json")

SPEC_TABLE = {
    "django/django": "SPECS_DJANGO", "sympy/sympy": "SPECS_SYMPY",
    "pytest-dev/pytest": "SPECS_PYTEST", "sphinx-doc/sphinx": "SPECS_SPHINX",
    "psf/requests": "SPECS_REQUESTS", "pylint-dev/pylint": "SPECS_PYLINT",
    "pallets/flask": "SPECS_FLASK", "mwaskom/seaborn": "SPECS_SEABORN",
    "matplotlib/matplotlib": "SPECS_MATPLOTLIB",
    "scikit-learn/scikit-learn": "SPECS_SKLEARN",
    "astropy/astropy": "SPECS_ASTROPY", "pydata/xarray": "SPECS_XARRAY",
}

# Projects whose install builds C/C++/Fortran extensions against an era-appropriate
# numpy. These are the ones expected to fight arm64, so they are ordered last and can be
# skipped entirely with --pure-only: a failure here says nothing about the 81% that do
# not need a compiler, and running them first would just delay the useful answer.
COMPILED = {"matplotlib/matplotlib", "scikit-learn/scikit-learn",
            "astropy/astropy", "pydata/xarray"}

# What `uv python` can actually supply here. Anything else is a deviation, not a failure.
AVAILABLE_PYTHONS = ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13"]

# The import that proves the install did something, per project. `pip install` exiting 0
# is not evidence: an editable install of a broken tree exits 0 and imports nothing.
IMPORT_CHECK = {
    "django/django": "django", "sympy/sympy": "sympy", "pytest-dev/pytest": "pytest",
    "sphinx-doc/sphinx": "sphinx", "psf/requests": "requests",
    "pylint-dev/pylint": "pylint", "pallets/flask": "flask",
    "mwaskom/seaborn": "seaborn", "matplotlib/matplotlib": "matplotlib",
    "scikit-learn/scikit-learn": "sklearn", "astropy/astropy": "astropy",
    "pydata/xarray": "xarray",
}


# Where each project keeps the requirements its TEST SUITE needs, as opposed to the ones
# its package needs to import. SWE-bench's spec says `packages: requirements.txt` for
# django, sympy, pylint and flask, and the first version of this probe treated that as a
# no-op -- so django installed nothing beyond its own package, and every test needing
# jinja2, Pillow or docutils SKIPPED. A skipped test is not a passing test, so those
# instances failed the inclusion rule for a reason that was ours, not theirs.
REQUIREMENT_FILES = (
    "tests/requirements/py3.txt",     # django
    "requirements_test.txt",          # pylint
    "requirements_test_min.txt",      # pylint
    "requirements/tests.txt",         # flask
    "requirements/dev.txt",           # flask
    "requirements-dev.txt",
    "requirements.txt",
)


def install_test_requirements(py, src, timeout=1800):
    """Best effort, package by package. Returns (file_used, installed, failed).

    Whole-file installs are attempted first and usually fail on at least one line: these
    lists were written for Linux and pin things like `pylibmc` (needs libmemcached) or
    `pywatchman` (needs watchman). One unbuildable package must not cost the other twenty,
    so a failed batch falls back to installing each requirement on its own and recording
    what did not make it. What is missing is then a known quantity rather than a mystery
    skip in a test log.
    """
    for relative in REQUIREMENT_FILES:
        path = os.path.join(src, relative)
        if not os.path.exists(path):
            continue
        res = run(["uv", "pip", "install", "--python", py, "-r", path], timeout=timeout)
        if res.returncode == 0:
            return relative, ["<all>"], []
        installed, failed = [], []
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if not line or line.startswith("-"):
                    continue
                one = run(["uv", "pip", "install", "--python", py, line], timeout=timeout)
                (installed if one.returncode == 0 else failed).append(line)
        return relative, installed, failed
    return None, [], []


def venv_env_for(env_dir):
    """Environment with the venv first on PATH, so `python`/`pip` mean the venv's."""
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = env_dir
    env["PATH"] = os.path.join(env_dir, "bin") + os.pathsep + env["PATH"]
    env.pop("PYTHONHOME", None)
    return env


def load_specs():
    if not os.path.exists(SPEC_FILE):
        sys.exit(f"no spec table at {SPEC_FILE}")
    return runpy.run_path(SPEC_FILE)


def resolve_python(spec_python):
    """The interpreter to actually use, and whether that is a deviation."""
    if spec_python in AVAILABLE_PYTHONS:
        return spec_python, None
    wanted = tuple(int(p) for p in spec_python.split("."))
    for candidate in AVAILABLE_PYTHONS:
        if tuple(int(p) for p in candidate.split(".")) >= wanted:
            return candidate, f"spec wants {spec_python}, unavailable here; used {candidate}"
    return AVAILABLE_PYTHONS[-1], f"spec wants {spec_python}; used {AVAILABLE_PYTHONS[-1]}"


def run(cmd, cwd=None, timeout=1800, env=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, env=env,
                          shell=isinstance(cmd, str))


def checkout_at(repo, commit, dest):
    mirror = os.path.join(MIRRORS, repo.replace("/", "__") + ".git")
    if not os.path.isdir(mirror):
        raise RuntimeError(f"no mirror for {repo}; run swebench_run.py first")
    run(["git", "clone", "--local", "--no-checkout", mirror, dest], timeout=600)
    res = run(["git", "checkout", "--detach", commit], cwd=dest, timeout=600)
    if res.returncode != 0:
        raise RuntimeError(f"checkout {commit[:10]} failed: {res.stderr[-300:]}")
    return dest


def build_env(repo, version, spec, env_commit, args):
    """One virtualenv per (repo, version), which is the granularity SWE-bench itself uses.

    114 django instances span 7 versions, so building per instance would do the same work
    16 times over.
    """
    key = f"{repo.replace('/', '__')}@{version}"
    env_dir = os.path.join(ENVS, key)
    python, deviation = resolve_python(spec["python"])
    row = {"repo": repo, "version": version, "key": key,
           "spec_python": spec["python"], "python_used": python,
           "python_deviation": deviation, "compiled": repo in COMPILED}

    started = time.time()
    if os.path.isdir(env_dir):
        shutil.rmtree(env_dir, ignore_errors=True)
    # --seed puts pip/setuptools in the venv so the benchmark's own install command
    # can be run VERBATIM rather than rewritten into uv's syntax. Rewriting it is how
    # a probe ends up measuring the rewrite.
    res = run(["uv", "venv", "--seed", "--python", python, env_dir], timeout=600)
    if res.returncode != 0:
        row.update(ok=False, stage="venv", error=res.stderr[-500:],
                   seconds=round(time.time() - started, 1))
        return row

    py = os.path.join(env_dir, "bin", "python")

    # pkg_resources, which setuptools REMOVED in v81. Every project here predates that:
    # sphinx 3.5 does `from pkg_resources import iter_entry_points` at import time, so a
    # modern seeded venv gives "No module named 'pkg_resources'" and not one test runs.
    # SWE-bench's containers are conda environments pinned to the era's setuptools and
    # never meet this. Pinning below 81 restores the module without touching anything the
    # benchmark specifies.
    res = run(["uv", "pip", "install", "--python", py, "setuptools<81", "wheel"],
              timeout=600)
    row["setuptools_pinned"] = res.returncode == 0
    # No "@" in the path: pip resolves a local project as a file:// URL and
    # percent-encodes it, so swe_env_pylint@2.15_x becomes ...pylint%402.15_x and the
    # install fails on a directory that plainly exists.
    src = tempfile.mkdtemp(prefix=f"swe_env_{key.replace('@', '-')}_")
    try:
        checkout_at(repo, env_commit, src)

        # pip_packages are pinned by the benchmark; `packages` is a loose list unless it
        # names a file, in which case the repo's own requirements are the source of truth
        # and pip -e . will pull what it needs.
        wanted = list(spec.get("pip_packages") or [])
        loose = spec.get("packages", "")
        if loose and loose not in ("requirements.txt", "environment.yml"):
            wanted += loose.split()
        if wanted:
            res = run(["uv", "pip", "install", "--python", py, *wanted], timeout=1800)
            if res.returncode != 0:
                row.update(ok=False, stage="deps", error=res.stderr[-800:],
                           deps=wanted, seconds=round(time.time() - started, 1))
                return row
        row["deps"] = wanted

        # PRE-INSTALL, which the benchmark runs before anything else. For old sphinx these
        # are the sed lines that pin Jinja2<3.0, markupsafe<=2.0.1 and the sphinxcontrib-*
        # packages to versions that still work with it. Skipping them does not fail
        # loudly; it installs a subtly wrong environment whose tests then fail for
        # reasons that look like the agent's problem.
        #
        # `sed -i` differs between GNU and BSD: GNU takes `-i` with no argument, BSD
        # requires a suffix. `-i ''` is the BSD spelling of the same edit.
        for step in spec.get("pre_install") or []:
            step = re.sub(r"\bsed -i (?!')", "sed -i '' ", step)
            res = run(step, cwd=src, timeout=600, env=venv_env_for(env_dir))
            row.setdefault("pre_install_failed", [])
            if res.returncode != 0:
                row["pre_install_failed"].append(step[:80])

        # The published test_cmd is the contract, and sphinx's is
        # `tox --current-env -epy39 -v --`. The spec table lists no packages for sphinx,
        # because the benchmark's own image installs tox in its Dockerfile rather than
        # through this table. Without it the command is simply not found, the log is
        # empty, and all 16 sphinx instances are excluded for "no tests ran".
        # tox-current-env is what makes --current-env work.
        if spec.get("test_cmd", "").strip().startswith("tox"):
            res = run(["uv", "pip", "install", "--python", py, "tox", "tox-current-env"],
                      timeout=args.install_timeout)
            row["tox_installed"] = res.returncode == 0
            if res.returncode != 0:
                row["tox_error"] = res.stderr[-400:]

        if spec.get("packages") == "requirements.txt":
            used, got, missed = install_test_requirements(py, src, args.install_timeout)
            row["requirements_file"] = used
            row["requirements_installed"] = len(got)
            row["requirements_failed"] = missed

        install = spec.get("install", "python -m pip install -e .")
        venv_env = venv_env_for(env_dir)
        # The spec's command is tried verbatim first. Where it fails for a reason that is
        # about THIS machine's toolchain rather than about the project, a fallback is
        # tried and RECORDED -- pylint 2.15 has a build backend older than PEP 660, which
        # the benchmark's container never noticed because its pip still did legacy
        # editable installs. A fallback that is not recorded is a silent protocol change.
        attempts = [
            (install, None),
            (install + " --config-settings editable_mode=compat", "pep660_compat"),
            (install.replace(" -e ", " "), "non_editable"),
        ]
        res = None
        for command, fallback in attempts:
            res = run(command, cwd=src, timeout=args.install_timeout, env=venv_env)
            if res.returncode == 0:
                row["install_cmd"] = command
                row["install_fallback"] = fallback
                break
        if res is None or res.returncode != 0:
            row.update(ok=False, stage="install", install_cmd=install,
                       error=(res.stderr or res.stdout)[-1200:] if res else "no attempt",
                       seconds=round(time.time() - started, 1))
            return row

        module = IMPORT_CHECK[repo]
        res = run([py, "-c", f"import {module}; print({module}.__file__)"],
                  cwd=src, timeout=300)
        if res.returncode != 0:
            row.update(ok=False, stage="import", error=res.stderr[-800:],
                       seconds=round(time.time() - started, 1))
            return row
        row["import_ok"] = res.stdout.strip()[-120:]
    except Exception as exc:
        row.update(ok=False, stage="setup", error=f"{type(exc).__name__}: {exc}",
                   seconds=round(time.time() - started, 1))
        return row
    finally:
        shutil.rmtree(src, ignore_errors=True)

    row.update(ok=True, stage="done", seconds=round(time.time() - started, 1))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pure-only", action="store_true",
                    help="skip the four projects that build native extensions")
    ap.add_argument("--repo", help="restrict to one repo")
    ap.add_argument("--install-timeout", type=int, default=2400)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even pairs already recorded as ok")
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args()

    import pandas as pd
    ns = load_specs()
    df = pd.read_parquet(DATASET)

    pairs = {}
    for r in df.itertuples():
        pairs.setdefault((r.repo, r.version),
                         {"n": 0, "env_commit": r.environment_setup_commit})
        pairs[(r.repo, r.version)]["n"] += 1

    selected = [
        (repo, ver, info) for (repo, ver), info in pairs.items()
        if (not args.repo or repo == args.repo)
        and not (args.pure_only and repo in COMPILED)
    ]
    # Cheap and likely first, native-extension builds last: an early answer on the pure
    # projects is worth more than a strictly ordered one.
    selected.sort(key=lambda t: (t[0] in COMPILED, t[0], t[1]))

    os.makedirs(ENVS, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = {}
    if os.path.exists(args.out):
        with open(args.out) as fh:
            done = {r["key"]: r for r in json.load(fh)}

    print(f"  {len(selected)} (repo, version) pair(s), "
          f"{sum(i['n'] for _, _, i in selected)} instance(s)")
    rows = []
    for index, (repo, ver, info) in enumerate(selected, 1):
        key = f"{repo.replace('/', '__')}@{ver}"
        if not args.force and key in done and done[key].get("ok"):
            rows.append(done[key])
            print(f"[{index}/{len(selected)}] {key:<38} cached ok")
            continue
        spec = ns[SPEC_TABLE[repo]].get(ver)
        if spec is None:
            rows.append({"repo": repo, "version": ver, "key": key, "ok": False,
                         "stage": "spec", "error": "no spec for this version"})
            continue
        print(f"[{index}/{len(selected)}] {key:<38} py={spec['python']} "
              f"n={info['n']} ...", end="", flush=True)
        row = build_env(repo, ver, spec, info["env_commit"], args)
        row["instances"] = info["n"]
        rows.append(row)
        print(f" {'OK' if row['ok'] else 'FAIL@' + row['stage']} ({row['seconds']}s)"
              + (f"  [{row['python_deviation']}]" if row.get("python_deviation") else ""),
              flush=True)
        incremental = dict(done)
        incremental.update({r["key"]: r for r in rows})
        with open(args.out, "w") as fh:
            json.dump(list(incremental.values()), fh, indent=2)

    # Merge, never overwrite: a --repo run must not erase the pairs it did not select.
    # Writing `rows` alone once discarded 51 good results and left three behind.
    merged = dict(done)
    merged.update({r["key"]: r for r in rows})
    with open(args.out, "w") as fh:
        json.dump(list(merged.values()), fh, indent=2)

    ok = [r for r in rows if r.get("ok")]
    covered = sum(r.get("instances", 0) for r in ok)
    deviated = sum(r.get("instances", 0) for r in ok if r.get("python_deviation"))
    total = sum(i["n"] for _, _, i in selected)
    print(f"\n  {len(ok)}/{len(rows)} pair(s) built")
    print(f"  {covered}/{total} instance(s) covered "
          f"({round(100 * covered / max(total, 1))}%), "
          f"{deviated} of them under a python deviation")
    for row in rows:
        if not row.get("ok"):
            print(f"    FAIL {row['key']:<38} @{row.get('stage')}: "
                  f"{(row.get('error') or '')[:110]}")
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
