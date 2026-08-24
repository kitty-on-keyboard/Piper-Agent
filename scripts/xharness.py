#!/usr/bin/env python3
"""Run ANY agent over LM_Pipe's eval tasks, graded by LM_Pipe's own grader.

The point is that the ground truth is not reimplemented here. `capture_task_contract`
and `verify_task_integrity` are imported from scripts/agent_eval.py, and `solved` is
computed by the same rule agent_eval uses:

    solved = check_returncode == 0 and integrity (before AND after)

So a number produced here is comparable to a number in evals/agent/pins.json. What
varies between arms is the agent command and nothing else -- same tasks, same fixture
copy, same checker, same integrity rules.

Usage:
  xharness.py --arm cline --tasks failing_test_median,handle_bad_json --out /tmp/cline.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = "/Users/dev/Desktop/seans_projects_local/LM_Pipe_2"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import agent_eval as ae  # noqa: E402


def cline_cmd(mission, workspace):
    return [
        "cline", "--yolo", "--json", "--cwd", workspace,
        "--data-dir", os.environ["CLINE_DD"], mission,
    ]


def cline_ask_cmd(mission, workspace):
    """Cline with tool approval REQUIRED.

    The counterpart to `deny_approvals: true` in a task. Running --yolo against a task
    designed around a human saying "no" removes the safety rail and then scores the
    agent for not having one, which measures nothing.
    """
    return [
        "cline", "--json", "--auto-approve", "false", "--cwd", workspace,
        "--data-dir", os.environ["CLINE_DD"], mission,
    ]


ARMS = {"cline": cline_cmd, "cline-ask": cline_ask_cmd}


def run_one(arm, task_name, cmd_builder):
    task_dir = os.path.join(ae.TASKS, task_name)
    with open(os.path.join(task_dir, "task.json")) as fh:
        disk = json.load(fh)

    loaded = {"name": task_name}
    for key in ("grader", "check"):
        if key in disk:
            loaded[key] = disk[key]
    contract = ae.capture_task_contract(loaded)
    tmp = tempfile.mkdtemp(prefix=f"xh_{arm}_{task_name}_")
    workspace = os.path.join(tmp, "workspace")
    shutil.copytree(os.path.join(task_dir, "workspace"), workspace, symlinks=True)

    integrity_before = ae.verify_task_integrity(contract, workspace)

    timeout = disk.get("wall_clock_seconds", 600)
    started = time.time()
    stdout = stderr = ""
    timed_out = False
    try:
        proc = subprocess.run(
            cmd_builder(disk["mission"], workspace), cwd=workspace,
            capture_output=True, text=True, timeout=timeout,
        )
        stdout, stderr = proc.stdout, proc.stderr
        agent_rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        agent_rc, timed_out = None, True
        stdout = (exc.stdout or b"").decode("utf-8", "replace") if exc.stdout else ""
        stderr = (exc.stderr or b"").decode("utf-8", "replace") if exc.stderr else ""
    wall = time.time() - started

    try:
        check = subprocess.run(
            ["/bin/sh", "-c", contract.check], cwd=workspace,
            capture_output=True, text=True,
            timeout=disk.get("check_timeout_seconds", 120),
        )
        check_rc = check.returncode
        check_detail = (check.stdout + check.stderr)[-2048:]
    except subprocess.TimeoutExpired as exc:
        check_rc, check_detail = None, f"checker timed out: {exc}"

    integrity_after = ae.verify_task_integrity(contract, workspace)
    integrity = {k: integrity_before[k] and integrity_after[k] for k in integrity_before}
    solved = check_rc == 0 and integrity["integrity"]

    # Turn count, where the arm reports it. Cline --json emits NDJSON events.
    turns = None
    if arm.startswith("cline"):
        turns = sum(1 for line in stdout.splitlines() if '"type":"tool_use"' in line.replace(" ", ""))

    return {
        "arm": arm, "task": task_name, "split": disk.get("split"),
        "solved": solved, "check_returncode": check_rc, "integrity": integrity["integrity"],
        "wall_seconds": round(wall, 1), "timed_out": timed_out, "agent_returncode": agent_rc,
        "turns": turns, "workspace": workspace,
        "check_detail": check_detail[-600:],
        "agent_stderr": stderr[-600:],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--tasks", required=True, help="comma-separated task names")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for name in args.tasks.split(","):
        name = name.strip()
        if not name:
            continue
        print(f"[{args.arm}] {name} ...", flush=True)
        try:
            row = run_one(args.arm, name, ARMS[args.arm])
        except Exception as exc:
            row = {"arm": args.arm, "task": name, "solved": False,
                   "setup_error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(
            f"    solved={row['solved']} wall={row.get('wall_seconds')}s "
            f"rc={row.get('agent_returncode')} timeout={row.get('timed_out')} "
            f"{row.get('setup_error','')}",
            flush=True,
        )

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    solved = sum(1 for r in rows if r["solved"])
    print(f"\n{args.arm}: {solved}/{len(rows)} solved, "
          f"{round(sum(r['wall_seconds'] for r in rows), 1)}s total")


if __name__ == "__main__":
    main()
