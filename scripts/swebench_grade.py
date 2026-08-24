#!/usr/bin/env python3
"""Turn generated patches into a SWE-bench score, on this machine, without a container.

THE RESOLVE CONDITION, which is SWE-bench's and not ours:

    An instance is RESOLVED iff, after applying the model's patch and then the benchmark's
    test patch, every FAIL_TO_PASS test passes and every PASS_TO_PASS test still passes.

WHAT THIS SHARES WITH THE INCLUSION RUN, DELIBERATELY. Environment build, patch
application, the eval script's container-to-host adaptation, the log parsers and the
subTest / docstring name recovery all come from scripts/swebench_inclusion.py. The two
runs differ in exactly one line -- whether the model's patch is applied before the tests
are run -- and everything else being literally the same code is what makes "this instance
was red before and green after" a statement about the patch rather than about two
harnesses that drifted.

THE AGENT CANNOT CHEAT THE TESTS. The eval script's own first act is
`git checkout <base_commit> <test files>`, so any edit the agent made to a test is thrown
away before the suite runs. The model patch is applied first and the restoration happens
after it, which means a patch that "fixes" a failing test by editing it scores zero.

Usage:
  swebench_grade.py --predictions runs/pilot/predictions.json --out runs/pilot/grade
  swebench_grade.py --predictions ... --only-included runs/inclusion/included_instances.txt
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import swebench_inclusion as si  # noqa: E402
import swebench_run as sr  # noqa: E402


def apply_model_patch(workspace, patch):
    """`git apply` the agent's diff. A patch that will not apply is a real zero.

    Not `--3way`, and not a fuzzy fallback: SWE-bench applies the prediction to a clean
    checkout and so does this. Being generous here would score patches that the benchmark
    itself would reject.
    """
    if not patch.strip():
        return False, "empty patch"
    proc = subprocess.run(["git", "apply", "-v", "-"], cwd=workspace, input=patch,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout)[-400:]
    return True, ""


def grade_one(row, prediction, args):
    """Resolved / not, for one instance, with the reason when not."""
    from swebench.harness.utils import make_test_spec
    import swebench.harness.log_parsers as parsers

    instance = {k: row[k] for k in row.index}
    spec = make_test_spec(instance)
    f2p, p2p = list(spec.FAIL_TO_PASS), list(spec.PASS_TO_PASS)
    verdict = {
        "instance_id": row["instance_id"], "repo": row["repo"],
        "version": row["version"], "n_f2p": len(f2p), "n_p2p": len(p2p),
        "patch_bytes": len(prediction or ""),
    }
    env_dir = os.path.join(si.ENVS,
                           f"{row['repo'].replace('/', '__')}@{row['version']}")
    workspace = tempfile.mkdtemp(prefix=f"swe_grade_{row['instance_id']}_")
    started = time.time()
    try:
        sr.checkout({"repo": row["repo"], "base_commit": row["base_commit"]}, workspace)

        # BEFORE the environment install: an editable install writes egg-info into the
        # tree, and a patch that touches packaging metadata would then conflict with it.
        applied, why = apply_model_patch(workspace, prediction or "")
        if not applied:
            verdict.update(resolved=False, reason="patch_did_not_apply", error=why)
            return verdict

        installed, command, error = si.install_instance(
            workspace, env_dir, args.install_timeout,
            si.spec_for(row["repo"], row["version"]))
        verdict["install_cmd"] = command
        if not installed:
            verdict.update(resolved=False, reason="install_failed", error=error)
            return verdict

        script = si.adapt_eval_script(row["eval_script"], workspace, env_dir)
        path = os.path.join(workspace, ".swebench_eval.sh")
        with open(path, "w") as fh:
            fh.write(script)
        env = os.environ.copy()
        env["VIRTUAL_ENV"] = env_dir
        env["PATH"] = os.path.join(env_dir, "bin") + os.pathsep + env["PATH"]
        env.pop("PYTHONHOME", None)
        try:
            res = subprocess.run(["/bin/bash", path], cwd=workspace, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, timeout=args.test_timeout)
        except subprocess.TimeoutExpired:
            verdict.update(resolved=False, reason="test_timeout")
            return verdict

        body = si.extract_test_output(res.stdout)
        if body is None:
            verdict.update(resolved=False, reason="no_test_output")
            return verdict
        parser = getattr(parsers, row["log_parser"], None)
        if parser is None:
            verdict.update(resolved=False, reason="no_parser")
            return verdict
        if row["log_parser"] == "parse_log_django":
            body = si.normalise_django_log(body)
            statuses = parser(body, spec)
            merged = si.django_docstring_aliases(body)
            merged.update(statuses)
            statuses = merged
        else:
            statuses = parser(body, spec)

        resolve = si.TestNameResolver(statuses, f2p + p2p)
        f2p_bad = [t for t in f2p if resolve.status(t) != "PASSED"]
        p2p_bad = [t for t in p2p if resolve.status(t) != "PASSED"]
        verdict["n_f2p_failing"] = len(f2p_bad)
        verdict["n_p2p_broken"] = len(p2p_bad)
        verdict["f2p_failing"] = [
            {"test": t, "status": resolve.status(t) or "ABSENT"} for t in f2p_bad[:6]]
        verdict["p2p_broken"] = [
            {"test": t, "status": resolve.status(t) or "ABSENT"} for t in p2p_bad[:6]]
        if not f2p_bad and not p2p_bad:
            reason = "resolved"
        elif f2p_bad and p2p_bad:
            # Worth distinguishing: this patch did not fix the bug AND broke something
            # else, which is a different failure from simply missing.
            reason = "unfixed+regressed"
        elif f2p_bad:
            reason = "unfixed"
        else:
            reason = "regressed"
        verdict.update(resolved=(reason == "resolved"), reason=reason)
        return verdict
    except Exception as exc:
        verdict.update(resolved=False, reason="error",
                       error=f"{type(exc).__name__}: {exc}")
        return verdict
    finally:
        verdict["seconds"] = round(time.time() - started, 1)
        shutil.rmtree(workspace, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-included",
                    help="path to included_instances.txt; grade only those")
    ap.add_argument("--test-timeout", type=int, default=900)
    ap.add_argument("--install-timeout", type=int, default=1800)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(si.DATASET_V5)
    with open(args.predictions) as fh:
        raw = json.load(fh)
    preds = ({p["instance_id"]: p.get("model_patch", "") for p in raw}
             if isinstance(raw, list)
             else {k: v.get("model_patch", "") for k, v in raw.items()})

    included = None
    if args.only_included:
        with open(args.only_included) as fh:
            included = {line.strip() for line in fh if line.strip()}

    rows = [r for _, r in df.iterrows() if r["instance_id"] in preds
            and (included is None or r["instance_id"] in included)]
    skipped = [i for i in preds if included is not None and i not in included]

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "grade.json")
    print(f"  grading {len(rows)} instance(s)"
          + (f"; {len(skipped)} not in the included set" if skipped else ""))
    verdicts = []
    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] {row['instance_id']:<34} ...", end="", flush=True)
        v = grade_one(row, preds[row["instance_id"]], args)
        verdicts.append(v)
        print(f" {'RESOLVED' if v['resolved'] else v['reason']} ({v.get('seconds')}s)",
              flush=True)
        with open(results_path, "w") as fh:
            json.dump(verdicts, fh, indent=2)

    with open(results_path, "w") as fh:
        json.dump(verdicts, fh, indent=2)
    import collections
    resolved = [v for v in verdicts if v["resolved"]]
    ledger = collections.Counter(v["reason"] for v in verdicts if not v["resolved"])
    print(f"\n  RESOLVED {len(resolved)}/{len(verdicts)}"
          + (f"  ({100 * len(resolved) // len(verdicts)}%)" if verdicts else ""))
    for reason, count in ledger.most_common():
        print(f"    {reason:<24} {count}")
    print(f"\n  -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
