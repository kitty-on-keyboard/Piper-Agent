#!/usr/bin/env python3
"""Evaluation harness (spec S11.3 - S11.5).

  ./scripts/eval.py score        bake-off scores, corpus AND held-out, both pinned
  ./scripts/eval.py mutants      mutation testing -- a survivor is a finding about the SUITE
  ./scripts/eval.py reliability  N-run ledger; one number is not a result

Three rules this file exists to enforce:

  * A held-out set is scored ONCE and both numbers are pinned, and the holdout must
    stay the harder of the two (S11.3). If the holdout ever scores better than the
    tuned-against corpus, the "holdout" has leaked.
  * A surviving mutation is a finding about the test suite, not about the mutant
    (S11.4).
  * Reliability is a RATE. v1 measured 462 s, then 275 s and 258 s on the IDENTICAL
    binary; three runs minimum before believing any timing signal (S11.5).
"""

import argparse
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

# Pinned scores. Measured in this repo on 2026-07-30 with the ported engines.
# `holdout` is scored once and never tuned against.
PINS = {
    "blast_radius": {
        "corpus_weighted_misses": 0,
        "corpus_exact": 179,
        "corpus_total": 179,
        "holdout_weighted_misses": 15,
        "holdout_exact": 34,
        "holdout_total": 42,
    },
    "log_triage": {"corpus_weighted": 34, "corpus_exact": 71, "corpus_total": 75},
}

MUTATIONS = [
    (r"\breturn true;", "return false;"),
    (r"\breturn false;", "return true;"),
    (r" == ", " != "),
    (r" && ", " || "),
    (r" < ", " <= "),
    (r"\+\+", "--"),
]


def copy_tree_for_probe(root, dest):
    """Copies the repo, excluding build output BY DIRECTORY NAME.

    Two harness defects live here, both found by the null mutant:
      * shutil.ignore_patterns("build*") also matches
        third_party/simdjson/include/simdjson/builder.h, so every copy failed to
        compile for an unrelated reason and the score came back a perfect 6/6.
      * excluding node_modules dropped the extension typecheck test, which changed
        what `-L gate` selects and failed the pinned manifest -- again for a reason
        unrelated to any mutation."""
    def ignore(directory, names):
        drop = set()
        for name in names:
            if not os.path.isdir(os.path.join(directory, name)):
                continue
            if name in {".git", "out", "__pycache__"}:
                drop.add(name)
            elif name == "build" or name.startswith("build-"):
                drop.add(name)
        return drop

    # symlinks=True: npm's node_modules/.bin entries are symlinks whose targets are
    # resolved RELATIVE to the link. Dereferencing them turns each into a real file
    # whose `require('../typescript/bin/tsc')` no longer resolves, and the extension
    # typecheck fails in the copy for a reason unrelated to any mutation. Third thing
    # the null mutant caught.
    shutil.copytree(root, dest, ignore=ignore, symlinks=True)


def run(cmd, cwd=None, timeout=900):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, check=False)


def parse_scoreboard(text):
    out = {}
    for key, pattern in [("weighted", r"weighted misses\s+(\d+)"),
                         ("exact", r"exact cases\s+(\d+)\s*/\s*(\d+)")]:
        m = re.search(pattern, text)
        if not m:
            continue
        out[key] = int(m.group(1))
        if key == "exact":
            out["total"] = int(m.group(2))
    return out


def cmd_score(root, args):
    # LMP_BUILD_DIR lets a probe copy (whose build dir is not named "build") point at
    # its own. Another thing the null mutant found.
    build = os.environ.get("LMP_BUILD_DIR") or os.path.join(root, "build")
    br = os.path.join(build, "bakeoff", "blast_radius_score_e00_merged")
    if not os.path.exists(br):
        print("build the scoreboards first: cmake --build --preset dev")
        return 1
    corpus = parse_scoreboard(run([br]).stdout)
    holdout = parse_scoreboard(
        run([br, os.path.join(root, "bakeoff/blast_radius/holdout.jsonl")]).stdout)

    pins = PINS["blast_radius"]
    print("blast_radius (engine: src/security/blast_radius.hpp)")
    print(f"  corpus  wmiss={corpus.get('weighted')} exact={corpus.get('exact')}"
          f"/{corpus.get('total')}   pinned {pins['corpus_weighted_misses']}"
          f" / {pins['corpus_exact']}")
    print(f"  holdout wmiss={holdout.get('weighted')} exact={holdout.get('exact')}"
          f"/{holdout.get('total')}   pinned {pins['holdout_weighted_misses']}"
          f" / {pins['holdout_exact']}")

    failures = []
    if corpus.get("weighted") != pins["corpus_weighted_misses"]:
        failures.append("corpus weighted misses moved")
    if holdout.get("weighted") != pins["holdout_weighted_misses"]:
        failures.append("holdout weighted misses moved")

    # The holdout must stay HARDER per point. The corpus was iterated against with the
    # key open; the holdout never was, so a holdout that scores better means it leaked.
    corpus_rate = corpus.get("exact", 0) / max(corpus.get("total", 1), 1)
    holdout_rate = holdout.get("exact", 0) / max(holdout.get("total", 1), 1)
    print(f"  exact rate: corpus {corpus_rate:.3f} vs holdout {holdout_rate:.3f}")
    if holdout_rate >= corpus_rate:
        failures.append("the holdout is no longer harder than the tuned-against corpus "
                        "-- it has leaked and is no longer evidence")

    for f in failures:
        print(f"FAIL: {f}")
    return 1 if failures else 0


def cmd_mutants(root, args):
    """Applies one mutation at a time to a COPY of the tree and runs the gate.

    A mutant the gate still passes is a hole in the suite: some line changed meaning
    and no assertion varied. That is the finding -- not that the mutant was clever."""
    targets = []
    for base in ("src/platform", "src/model", "src/tools", "src/context", "src/loop",
                 "src/surface"):
        for dirpath, _, files in os.walk(os.path.join(root, base)):
            if "/mlx" in dirpath:
                continue  # vendored numerics: graded by the real-model tests, not here
            targets.extend(os.path.join(dirpath, f) for f in files
                           if f.endswith((".cpp", ".hpp")))

    rng = random.Random(args.seed)
    planned = []
    while len(planned) < args.count and targets:
        path = rng.choice(targets)
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        candidates = [(i, p, r) for i, l in enumerate(lines)
                      for p, r in MUTATIONS
                      if re.search(p, l) and not l.strip().startswith("//")]
        if candidates:
            planned.append((path, rng.choice(candidates)))

    # THE NULL MUTANT. Before believing any kill, prove an UNMUTATED copy builds and
    # passes. Without this the score silently measures the harness: the first run of
    # this tool reported 6/6 killed while every copy was failing to compile for an
    # unrelated reason. A green here is what makes the kills below mean anything.
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, "repo")
        copy_tree_for_probe(root, work)
        cfg = run(["cmake", "-S", work, "-B", os.path.join(work, "b"),
                   "-DCMAKE_BUILD_TYPE=RelWithDebInfo"])
        build = run(["cmake", "--build", os.path.join(work, "b"), "-j8"])
        test = run(["ctest", "--test-dir", os.path.join(work, "b"), "-L", "gate"])
        if cfg.returncode != 0 or build.returncode != 0 or test.returncode != 0:
            print("ABORT: the NULL mutant (no mutation at all) does not build and pass "
                  "in a fresh copy, so every 'kill' below would be measuring this "
                  "harness rather than the suite.")
            tail = (build.stderr or build.stdout or cfg.stderr or test.stdout)[-1200:]
            print(tail)
            return 1
    print("null mutant: builds and passes -- kills below are real\n")

    killed, survived = 0, []
    for path, (lineno, pattern, repl) in planned:
        with tempfile.TemporaryDirectory() as tmp:
            work = os.path.join(tmp, "repo")
            copy_tree_for_probe(root, work)
            rel = os.path.relpath(path, root)
            target = os.path.join(work, rel)
            with open(target, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            original = lines[lineno]
            lines[lineno] = re.sub(pattern, repl, original, count=1)
            with open(target, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")

            cfg = run(["cmake", "-S", work, "-B", os.path.join(work, "b"),
                       "-DCMAKE_BUILD_TYPE=RelWithDebInfo"])
            build = run(["cmake", "--build", os.path.join(work, "b"), "-j8"])
            if cfg.returncode != 0 or build.returncode != 0:
                killed += 1  # a mutant that will not compile is killed by the compiler
                print(f"KILLED (compile) {rel}:{lineno + 1} {pattern!r}")
                continue
            test = run(["ctest", "--test-dir", os.path.join(work, "b"), "-L", "gate"])
            if test.returncode != 0:
                killed += 1
                print(f"KILLED (gate)    {rel}:{lineno + 1} {pattern!r}")
            else:
                survived.append((rel, lineno + 1, original.strip()))
                print(f"SURVIVED         {rel}:{lineno + 1} {pattern!r}")

    total = killed + len(survived)
    print(f"\nmutation score: {killed}/{total} killed")
    if survived:
        print("\nSurvivors are findings about the SUITE -- each is a line whose meaning "
              "changed with no assertion noticing:")
        for rel, line, text in survived:
            print(f"  {rel}:{line}  {text}")
    return 0


def cmd_reliability(root, args):
    """An N-run ledger, not one number (S11.5)."""
    build = os.path.join(root, "build")
    label = args.label
    runs = []
    for i in range(args.runs):
        r = run(["ctest", "--test-dir", build, "-L", label], timeout=3600)
        ok = r.returncode == 0
        seconds = 0.0
        m = re.search(r"Total Test time \(real\) =\s+([0-9.]+)", r.stdout)
        if m:
            seconds = float(m.group(1))
        runs.append((ok, seconds))
        print(f"run {i + 1}/{args.runs}: {'PASS' if ok else 'FAIL'}  {seconds:.2f}s")

    passes = sum(1 for ok, _ in runs if ok)
    times = [s for _, s in runs]
    print(f"\nreliability: {passes}/{len(runs)} = {passes / len(runs):.1%}")
    if len(times) >= 3:
        print(f"time: median {statistics.median(times):.2f}s  "
              f"min {min(times):.2f}s  max {max(times):.2f}s  "
              f"spread {max(times) - min(times):.2f}s")
    else:
        print("NOTE: fewer than 3 runs. v1 measured 462s, then 275s and 258s on the "
              "IDENTICAL binary -- do not believe a timing signal from fewer than 3.")
    return 0 if passes == len(runs) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("score")
    m = sub.add_parser("mutants")
    m.add_argument("--count", type=int, default=10)
    m.add_argument("--seed", type=int, default=1)
    r = sub.add_parser("reliability")
    r.add_argument("--runs", type=int, default=5)
    r.add_argument("--label", default="gate")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    return {"score": cmd_score, "mutants": cmd_mutants,
            "reliability": cmd_reliability}[args.cmd](root, args)


if __name__ == "__main__":
    sys.exit(main())
