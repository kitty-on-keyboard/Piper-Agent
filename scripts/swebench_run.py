#!/usr/bin/env python3
"""Run LM_Pipe over SWE-bench instances and emit patches for SWE-bench's own grader.

WHY THIS EXISTS. Every score this project has published is first-person: our agent, our
six tasks, our checker. docs/BAKEOFF_HARNESS.md beat Cline 6/6 to 5/6 on the same weights
and the same machine, and the obvious objection is that we wrote the benchmark we win.
SWE-bench answers that objection because someone else wrote it, someone else grades it,
and there is a published anchor to calibrate against.

WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT DO. This script only *generates*
patches. It checks out a real repository at the instance's base commit, hands the agent
the issue text, and diffs the tree afterwards. It never decides whether a patch is
correct -- that is SWE-bench's job, in SWE-bench's containers, against tests this script
never reads. The split matters: the moment a local scorer exists, the objection above
comes straight back.

WHAT THE AGENT IS AND IS NOT GIVEN:

    given       problem_statement, and a checkout at base_commit
    withheld    patch (the reference fix), test_patch, FAIL_TO_PASS, PASS_TO_PASS

Withholding is not politeness, it is the whole measurement. `test_patch` names the files
that prove the fix, so leaking it converts the benchmark into a pointing exercise.

THE HANDICAP, STATED UP FRONT. The published mini-swe-agent anchor (18.8% on Verified,
Bash-only, Qwen3-Coder-30B-A3B) runs the agent INSIDE the SWE-bench container, where the
repository's dependencies are installed and its test suite actually runs. This script
runs on the host against a bare checkout with no environment, so the agent cannot execute
the project's tests, cannot reproduce the bug, and cannot check its own work by running
anything. That is a materially harder condition, and any comparison against the published
number has to say so. It is a consequence of LM_Pipe being a native macOS binary with MLX
in-process; closing it needs the T2 container work in docs/PLAN_GAP_CLOSURE.md.

Loads the model, so it is subject to the one-MLX-process-at-a-time rule: one checkpoint
is 15-19 GB resident on a 48 GB host, and running two has taken this machine down. The
guard below refuses to start rather than trusting anyone to remember.

Usage:
  swebench_run.py list --limit 5
  swebench_run.py run --instances astropy__astropy-12907 --out runs/smoke
  swebench_run.py run --limit 300 --out runs/lite_full
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
import agent_eval as ae  # noqa: E402

WORK = os.environ.get(
    "LMP_SWEBENCH_WORK", "/Users/dev/Desktop/seans_projects_local/swebench_work"
)
DATASET = os.path.join(WORK, "data", "swebench_lite_test.parquet")
MIRRORS = os.path.join(WORK, "repos")

# Files the agent may leave behind that are noise in a patch rather than a change to the
# project. Anything matched here is removed before the diff is taken, so a stray cache
# directory cannot make a patch fail to apply in SWE-bench's container.
PATCH_NOISE = ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")

MISSION = """\
Solve the following issue in the Python repository checked out in this workspace.

<issue>
{problem}
</issue>

Make the smallest source change that fixes the issue described above. Do not modify or
add tests -- the fix is judged by the project's existing test suite, which is restored
before it is run, so edits to tests are discarded.

This project's dependencies ARE installed and `python` is its own interpreter, so you can
import the package, reproduce the issue, and run its tests to check your work.
"""

# The mission above used to end "the repository's dependencies are not installed here, so
# you cannot run its test suite". That was true and it was the handicap this whole
# protocol exists to remove: the published mini-swe-agent anchor runs INSIDE a container
# where it can reproduce the bug and check its own fix, and scoring a blind agent against
# it measures the missing environment rather than the loop.

# A verification command the agent is allowed to see. It must not leak which tests prove
# the fix, so it checks only that whatever the agent wrote is syntactically valid Python.
# An EMPTY verify_contract is a known-degraded mode -- it leaves a writing run with no
# feedback loop at all, and has previously produced runs that reported completed with
# zero writes -- so the honest choice is the strongest check that reveals nothing.
SYNTAX_VERIFY = (
    "changed=$(git diff --name-only HEAD -- '*.py'; "
    "git ls-files --others --exclude-standard -- '*.py'); "
    "[ -z \"$changed\" ] && exit 0; "
    "python3 -m compileall -q $changed"
)


def load_instances():
    if not os.path.exists(DATASET):
        sys.exit(f"no dataset at {DATASET}; see docs/HANDOFF_SWEBENCH.md")
    try:
        import pandas as pd
    except ImportError:
        sys.exit(
            "pandas is needed to read the parquet. This script runs under the swebench "
            f"venv: {os.path.join(WORK, 'venv', 'bin', 'python')}"
        )
    df = pd.read_parquet(DATASET)
    return [
        {
            "instance_id": r.instance_id,
            "repo": r.repo,
            "base_commit": r.base_commit,
            "problem_statement": r.problem_statement,
            # Needed to find the (repo, version) virtualenv; see prepare_environment.
            "version": r.version,
        }
        for r in df.itertuples()
    ]


def assert_no_other_mlx():
    """One checkpoint is 15-19 GB resident. Two have taken this machine down."""
    proc = subprocess.run(
        ["pgrep", "-fl", "mlx_lm|lmp_sidecar|fmbench"],
        capture_output=True, text=True,
    )
    live = [
        line for line in proc.stdout.splitlines()
        if "pgrep" not in line and str(os.getpid()) not in line.split()[:1]
    ]
    if live:
        sys.exit(
            "another MLX process is alive; refusing to start a second one:\n  "
            + "\n  ".join(live)
        )


def mirror_for(repo):
    """One bare mirror per project, shared by every instance that uses it.

    300 instances span 12 repositories, so cloning per instance would fetch django 114
    times. The per-instance checkout below hardlinks out of this mirror, which makes it
    close to free in both time and disk.
    """
    path = os.path.join(MIRRORS, repo.replace("/", "__") + ".git")
    if not os.path.isdir(path):
        os.makedirs(MIRRORS, exist_ok=True)
        print(f"    cloning mirror {repo} (first use, this is slow) ...", flush=True)
        subprocess.run(
            ["git", "clone", "--bare", f"https://github.com/{repo}.git", path],
            check=True, capture_output=True, text=True,
        )
    return path


def checkout(instance, dest):
    """A writable tree at base_commit that cannot see its own future.

    THE LEAK THIS CLOSES. A plain clone of the mirror carries the WHOLE default branch,
    which includes the commit that actually fixed the issue. On astropy__astropy-12907,
    one command finds it:

        git log --all --oneline | grep -i separab
        2f1bfd254d Backport PR #12907: Correctly calculate the separability of a nested...

    and `git show` then hands over the reference patch verbatim. Any score measured
    against a repository in that state is measuring grep, not engineering.

    The fix is to make base_commit the only reachable ref: one branch pointing at it, no
    remote, no tags. What survives is exactly what a developer sitting on that commit
    would have -- its ancestors, and nothing later. Objects for later commits are still
    in the hardlinked pack and could in principle be reached by enumerating them with
    `git cat-file --batch-all-objects`, which is a leak no benchmark of this shape
    closes; it is recorded here rather than left for someone to find.
    """
    mirror = mirror_for(instance["repo"])
    subprocess.run(
        ["git", "clone", "--local", "--no-checkout", mirror, dest],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        # -B, not -b: the clone already carries the project's own default branch, and
        # on most of these repositories it is called `main`.
        ["git", "checkout", "-B", "main", instance["base_commit"]],
        cwd=dest, check=True, capture_output=True, text=True,
    )
    subprocess.run(["git", "remote", "remove", "origin"], cwd=dest,
                   capture_output=True, text=True)
    # DELETE ONLY WHAT POINTS FORWARD. The first version of this deleted every ref
    # including tags, which closes the leak and also breaks the build: setuptools_scm
    # derives a package version from the nearest tag, so a tagless checkout of pytest
    # installs as `0.1.dev10157+g4a2fdce62`, and pytest's own tox.ini then refuses to run
    # with "requires pytest-2.0, actual pytest-0.1.dev...". That silently wiped all 17
    # pytest instances out of the benchmark.
    #
    # A tag that is an ANCESTOR of base_commit leaks nothing: every commit reachable from
    # it is already reachable from base_commit. It is also exactly what a developer
    # sitting on that commit would have. Only refs pointing at or past the future are
    # removed.
    refs = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname)"],
        cwd=dest, capture_output=True, text=True, check=True,
    ).stdout.split()
    for ref in refs:
        if ref == "refs/heads/main":
            continue
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ref + "^{commit}",
             instance["base_commit"]],
            cwd=dest, capture_output=True, text=True,
        )
        if ancestor.returncode != 0:
            subprocess.run(["git", "update-ref", "-d", ref], cwd=dest,
                           capture_output=True, text=True)
    assert_history_truncated(dest, instance["base_commit"])
    return dest


def assert_history_truncated(dest, base_commit):
    """Refuse to run an instance whose checkout can still see past its base commit.

    A silent failure here does not break the run, it inflates the score -- which is the
    one failure mode a benchmark must never have.
    """
    refs = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname) %(objectname)"],
        cwd=dest, capture_output=True, text=True, check=True,
    ).stdout.split("\n")
    for line in [r for r in refs if r.strip()]:
        name, _, sha = line.partition(" ")
        merge_base = subprocess.run(
            # ^{commit} peels annotated tags, whose objectname is the tag object.
            ["git", "merge-base", "--is-ancestor", sha.strip() + "^{commit}",
             base_commit],
            cwd=dest, capture_output=True, text=True,
        )
        if merge_base.returncode != 0:
            raise RuntimeError(
                f"{name} points at {sha.strip()}, which is not an ancestor of "
                f"{base_commit}: the checkout can see its own fix"
            )


def extract_patch(workspace, base_commit):
    """The diff the agent produced, in the form SWE-bench applies.

    Staged rather than working-tree, so a file the agent CREATED is in the patch. A fix
    that adds a module and never mentions it in a diff scores zero for a reason that has
    nothing to do with the model.
    """
    for noise in PATCH_NOISE:
        subprocess.run(
            ["find", ".", "-name", noise, "-type", "d", "-prune", "-exec",
             "rm", "-rf", "{}", "+"],
            cwd=workspace, capture_output=True, text=True,
        )
    # The context store is EXCLUDED, not deleted: src/pcc writes `.lmp-context.db` into
    # the workspace root, so a plain `git add -A` puts a multi-megabyte SQLite blob into
    # every patch SWE-bench is asked to apply. Excluding by pathspec rather than removing
    # the file leaves it on disk for the PCC evidence below and for --keep-workspaces.
    subprocess.run(
        ["git", "add", "-A", "--",
         ".", ":(exclude).lmp-context.db", ":(exclude).lmp-context.db-wal",
         ":(exclude).lmp-context.db-shm"],
        cwd=workspace, capture_output=True, text=True,
    )
    diff = subprocess.run(
        ["git", "diff", "--cached", "--binary", base_commit],
        cwd=workspace, capture_output=True, text=True,
    )
    return diff.stdout


def prepare_environment(instance, workspace, args):
    """Give the agent the project's own interpreter, or say why it could not.

    THE HANDICAP THIS REMOVES. Without it the agent reads a repository it cannot run: no
    import, no reproduction, no way to check its own fix. The published anchor it would be
    compared against runs inside a container where all three work, so a score measured
    blind is a statement about the missing environment.

    Returns (env_dir, note). A null env_dir is NOT fatal -- the instance still runs, blind,
    and `env_note` records why, because an instance that silently solved blind while the
    rest had an interpreter would be an unmarked hole in the comparison.

    The install is per instance, not per environment. `swebench_probe_env.py` built each
    virtualenv against a temporary checkout and deleted it, so the editable install
    dangles; pointing it at THIS checkout is what makes `import` resolve to the code the
    agent is editing rather than to a stale copy of another commit.
    """
    if args.no_environment:
        return "", "disabled by --no-environment"
    key = f"{instance['repo'].replace('/', '__')}@{instance['version']}"
    env_dir = os.path.join(WORK, "envs", key)
    if not os.path.isdir(env_dir):
        return "", f"no virtualenv built for {key}"
    try:
        import swebench_inclusion as si
    except ImportError as exc:
        return "", f"cannot import the installer: {exc}"
    spec = si.spec_for(instance["repo"], instance["version"])
    try:
        installed, command, error = si.install_instance(
            workspace, env_dir, args.install_timeout, spec)
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"
    if not installed:
        return "", f"install failed: {error[:200]}"
    return env_dir, command


def context_store_evidence(workspace, harness_dir, state):
    """Proof that the durable context store was actually live for this instance.

    src/pcc is part of the agent, not an optional extra, so a benchmark run with a dead
    store would be measuring a deliberately crippled LM_Pipe. It opens unconditionally --
    there is no setting to get wrong -- but "it should be on" is not evidence, and the
    read side is the half with a history of being weak: journalled faithfully, queried on
    roughly 1% of tool calls at a 46% miss rate.

    Three independent signals, because each can fail while the others look fine:

      db_bytes        the store exists on disk and grew -- the WRITE side
      recall_scope    the sidecar's own event, what it knew it had at mission start
      reads           context_recall / context_rehydrate calls -- the READ side

    A run with a large db_bytes and zero reads is the known failure shape, and it is only
    visible if reads are counted separately from the tool total.
    """
    evidence = {"db_bytes": 0, "wal_bytes": 0, "reads": 0, "remember_calls": 0}
    for name, key in ((".lmp-context.db", "db_bytes"),
                      (".lmp-context.db-wal", "wal_bytes")):
        path = os.path.join(workspace, name)
        if os.path.exists(path):
            evidence[key] = os.path.getsize(path)

    # THE WAL IS WHERE THE DATA IS. SQLite in WAL mode leaves the main database at a bare
    # 4096-byte header page until a checkpoint, so `db_bytes` alone reads as "the store is
    # empty" on a run that journalled 1.8 MB. Measured on the first two instances: db 4096,
    # wal 572712 and 1845792. Size the store by both or misdiagnose it.
    evidence["store_bytes"] = evidence["db_bytes"] + evidence["wal_bytes"]
    evidence["journalled"] = evidence["store_bytes"] > 4096

    # Reads are TOOL calls; writes are not. The turn sink persists every turn from inside
    # the context store without a tool call, so counting `remember_fact` would say zero on
    # a run that wrote continuously. remember_fact is kept as its own number, never as a
    # proxy for whether the store was written.
    counts = state["tool_names"]
    evidence["reads"] = counts.get("context_recall", 0) + counts.get(
        "context_rehydrate", 0)
    evidence["remember_calls"] = counts.get("remember_fact", 0)

    # Two events from the sidecar's own log. `context_journal` is the one that matters
    # most: the store opening is NOT guaranteed, and a failure is logged as a survivable
    # degradation rather than raised -- the run continues with no durable context and
    # nothing at the time says so. Reading it here is what stops a benchmark from
    # silently scoring an LM_Pipe with half its memory missing.
    log_path = os.path.join(harness_dir, "events.jsonl")
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"context_journal"' not in line and '"recall_scope"' not in line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("kind") == "recall_scope":
                    evidence["recall_scope"] = {
                        "items": ev.get("items"), "sessions": ev.get("sessions"),
                        "advertised": ev.get("advertised"),
                    }
                elif ev.get("kind") == "context_journal":
                    evidence["journal_open_failed"] = ev.get("error") or "unknown"

    # No `context_journal` event is the GOOD case: the sidecar logs it only on failure.
    evidence["journal_ok"] = "journal_open_failed" not in evidence
    evidence["healthy"] = bool(evidence["journal_ok"] and evidence["journalled"])

    # Recall is advertised to the prompt only when an EARLIER session left something
    # (sessions > 1 and items > 0). A fresh workspace per instance means sessions == 1
    # every time, so the read side is never offered and can never be used. That is a
    # property of how the benchmark provisions workspaces, not a defect in src/pcc, and
    # it is recorded per instance so the distinction survives into the writeup.
    scope = evidence.get("recall_scope") or {}
    evidence["recall_advertised"] = scope.get("advertised") == "1"
    evidence["read_side_reachable"] = evidence["recall_advertised"]
    return evidence


def run_instance(instance, model_dir, args):
    workspace = tempfile.mkdtemp(prefix=f"swe_{instance['instance_id']}_")
    harness_dir = tempfile.mkdtemp(prefix=f"swe_harness_{instance['instance_id']}_")
    checkout(instance, workspace)

    meta = {
        "name": instance["instance_id"],
        "mission": MISSION.format(problem=instance["problem_statement"]),
        "mode": "agent",
        "max_iterations": args.max_iterations,
        "wall_clock_seconds": args.wall_clock,
    }
    contract = ae.TaskContract(
        check="" if args.verify == "none" else SYNTAX_VERIFY,
        task_json_path="", task_json_bytes=b"", protected=(),
    )

    env_dir, install_note = prepare_environment(instance, workspace, args)
    extra_env = None
    if env_dir:
        extra_env = {
            "VIRTUAL_ENV": env_dir,
            "PATH": os.path.join(env_dir, "bin") + os.pathsep + os.environ["PATH"],
        }

    started = time.monotonic()
    state = ae.drive_sidecar(
        meta, model_dir, workspace, harness_dir,
        ae.sampling_for(args.run_settings, args.run_settings["seeds"][0]),
        contract, args.verbose, extra_env,
    )
    pcc = context_store_evidence(workspace, harness_dir, state)
    patch = extract_patch(workspace, instance["base_commit"])

    row = {
        "instance_id": instance["instance_id"],
        "repo": instance["repo"],
        "base_commit": instance["base_commit"],
        "model_patch": patch,
        "patch_bytes": len(patch),
        "patch_files": patch.count("\ndiff --git ") + patch.startswith("diff --git "),
        "empty_patch": not patch.strip(),
        "turns": state["iterations"],
        "completed": state["completed"],
        "reason": state["reason"],
        "unattended_replies": state.get("unattended_replies", 0),
        "seconds": round(time.monotonic() - started, 1),
        "kv_reused_tokens": state["kv_reused_tokens"],
        "generated_tokens": state["generated_tokens"],
        "tool_calls": state["tool_calls"],
        "tool_names": dict(state["tool_names"]),
        "pcc": pcc,
        "env_dir": env_dir,
        "env_note": install_note,
    }
    if args.keep_workspaces:
        row["workspace"] = workspace
    else:
        shutil.rmtree(workspace, ignore_errors=True)
    shutil.rmtree(harness_dir, ignore_errors=True)
    return row


def write_outputs(rows, out_dir, model_name):
    """Both prediction shapes, because the two graders disagree about the format.

    The local swebench harness reads a list of records; sb-cli reads a dict keyed by
    instance_id. Writing one and converting later is how a run gets re-done.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rows.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    preds = [
        {"instance_id": r["instance_id"], "model_name_or_path": model_name,
         "model_patch": r["model_patch"]}
        for r in rows
    ]
    with open(os.path.join(out_dir, "predictions.json"), "w", encoding="utf-8") as fh:
        json.dump(preds, fh, indent=2)
    with open(os.path.join(out_dir, "predictions_sbcli.json"), "w", encoding="utf-8") as fh:
        json.dump({p["instance_id"]: p for p in preds}, fh, indent=2)


def cmd_list(args):
    instances = load_instances()
    for inst in instances[: args.limit]:
        head = inst["problem_statement"].split("\n")[0][:88]
        print(f"  {inst['instance_id']:<34} {inst['repo']:<24} {head}")
    print(f"\n  {len(instances)} instance(s) in the dataset")
    return 0


def cmd_run(args):
    assert_no_other_mlx()
    if not os.path.exists(ae.SIDECAR):
        sys.exit(f"no sidecar at {ae.SIDECAR}; build it first")

    instances = load_instances()
    if args.instances:
        wanted = [s.strip() for s in args.instances.split(",") if s.strip()]
        by_id = {i["instance_id"]: i for i in instances}
        missing = [w for w in wanted if w not in by_id]
        if missing:
            sys.exit(f"unknown instance_id(s): {', '.join(missing)}")
        instances = [by_id[w] for w in wanted]
    if args.limit:
        instances = instances[: args.limit]

    args.run_settings = ae.resolve_run_settings(args.seed, args.temperature, False)
    model_name = args.model_name or ("lm_pipe+" + os.path.basename(args.model))
    print(f"  {len(instances)} instance(s), model {args.model}")
    print(f"  budget max_iterations={args.max_iterations} wall={args.wall_clock}s "
          f"verify={args.verify}")

    rows = []
    for index, inst in enumerate(instances, 1):
        print(f"\n[{index}/{len(instances)}] {inst['instance_id']} ({inst['repo']})",
              flush=True)
        try:
            row = run_instance(inst, args.model, args)
        except Exception as exc:  # a broken instance must not lose the run so far
            row = {"instance_id": inst["instance_id"], "repo": inst["repo"],
                   "model_patch": "", "empty_patch": True,
                   "setup_error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        write_outputs(rows, args.out, model_name)
        pcc = row.get("pcc") or {}
        print(f"      -> patch={row.get('patch_bytes', 0)}B "
              f"files={row.get('patch_files', 0)} turns={row.get('turns')} "
              f"{row.get('reason', '')} ({row.get('seconds')}s) "
              f"| env={'y' if row.get('env_dir') else 'NO'} "
              f"| pcc store={pcc.get('store_bytes', 0)}B "
              f"reads={pcc.get('reads', 0)} "
              f"adv={'y' if pcc.get('recall_advertised') else 'n'} "
              f"{row.get('setup_error', '')}", flush=True)

    produced = sum(1 for r in rows if not r.get("empty_patch"))
    wall = sum(r.get("seconds") or 0 for r in rows)
    print(f"\n  {produced}/{len(rows)} instance(s) produced a non-empty patch "
          f"in {round(wall / 60, 1)} min")
    print(f"  predictions -> {os.path.join(args.out, 'predictions.json')}")
    print("  NOT a score. Grade with SWE-bench's harness; see docs/HANDOFF_SWEBENCH.md")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    lst = sub.add_parser("list")
    lst.add_argument("--limit", type=int, default=20)
    lst.set_defaults(func=cmd_list)

    run = sub.add_parser("run")
    run.add_argument("--instances", help="comma-separated instance_ids")
    run.add_argument("--limit", type=int)
    run.add_argument("--out", required=True)
    run.add_argument("--model", default=os.environ.get("LMP_QWEN_DIR", ae.DEFAULT_MODEL))
    run.add_argument("--model-name", help="name recorded in predictions.json")
    run.add_argument("--max-iterations", type=int, default=75)
    run.add_argument("--wall-clock", type=int, default=1200)
    run.add_argument("--verify", choices=["syntax", "none"], default="syntax")
    run.add_argument("--seed", action="append", default=[])
    run.add_argument("--temperature", type=float)
    run.add_argument("--keep-workspaces", action="store_true")
    run.add_argument("--no-environment", action="store_true",
                     help="run blind, without the project's interpreter (the old behaviour)")
    run.add_argument("--install-timeout", type=int, default=1800)
    run.add_argument("-v", "--verbose", action="store_true")
    run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
