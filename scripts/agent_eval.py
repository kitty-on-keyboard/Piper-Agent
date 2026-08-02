#!/usr/bin/env python3
"""End-to-end agent evaluation (spec S11.3).

  ./scripts/agent_eval.py run                 every task, corpus then holdout
  ./scripts/agent_eval.py run --split corpus  one split
  ./scripts/agent_eval.py run --only median   substring match on task name
  ./scripts/agent_eval.py run --pin           write the scores as the new pins
  ./scripts/agent_eval.py list                what is in the suite

WHY THIS EXISTS. scripts/eval.py scores blast_radius -- ONE component -- against a
corpus and a holdout. Nothing measured the agent end to end. The session before this one
made roughly ten behavioural changes (compaction strategy, prompt ordering, plan-forcing,
multi-call, stall detection, persona, baseline verification) and validated them against a
single task, so their aggregate effect was unknown and some plausibly hurt. Without a
suite, every future loop change is an argument instead of a measurement.

WHAT IS SCORED. The ground truth is a shell command run in the workspace AFTER the agent
has finished, and it never asks the agent anything:

  solved      the task's `check` exits 0. This is the score.
  completed   the agent's own evidential verdict. Reported beside `solved` precisely so
              the two can DISAGREE -- completed-but-not-solved is the interesting cell,
              and it is invisible if you only record one of them.
  verified    a falsifiable passing verification was observed on the wire.
  turns       iterations used.
  intact      files listed in `protect` are byte-identical afterwards. A run that "fixes"
              a failing test by editing the test has not fixed anything.

RULES CARRIED OVER FROM eval.py

  * The holdout is scored once and never tuned against, and must stay the HARDER set
    (S11.3). If it ever scores better than the corpus, it has leaked.
  * A regression fails; an improvement asks to be re-pinned. Pins move deliberately.

Loads the model, so it is subject to the one-MLX-process-at-a-time rule in
docs/HANDOFF_AGENT.md: this runs the sidecar serially, one task at a time, in the
foreground. Ten tasks is roughly half an hour of wall clock.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = os.path.join(ROOT, "evals", "agent", "tasks")
PINS = os.path.join(ROOT, "evals", "agent", "pins.json")
DEFAULT_MODEL = "/Users/dev/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit"
SIDECAR = os.path.join(ROOT, "build", "src", "surface", "lmp_sidecar")

# Qwen3's recommended thinking-mode operating point (S5.9), with a fixed seed so a
# re-run is as close to reproducible as a 35B MoE on Metal gets. It is not bit-exact --
# see the tolerance note on `compare`.
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "repetition_penalty": 1.05, "seed": 7}


def load_tasks(split=None, only=None):
    out = []
    for name in sorted(os.listdir(TASKS)):
        meta_path = os.path.join(TASKS, name, "task.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["name"] = name
        if split and meta.get("split") != split:
            continue
        if only and only not in name:
            continue
        out.append(meta)
    # Corpus first: if a shared assumption is broken, it shows up on the tuned-against
    # set before the holdout is spent on it.
    return sorted(out, key=lambda m: (m.get("split") != "corpus", m["name"]))


def run_one(meta, model_dir, verbose):
    """Runs one task in a throwaway copy of its workspace. Returns a score dict."""
    workspace = tempfile.mkdtemp(prefix=f"lmpeval-{meta['name']}-")
    shutil.copytree(os.path.join(TASKS, meta["name"], "workspace"), workspace,
                    dirs_exist_ok=True)

    proc = subprocess.Popen(
        [SIDECAR], cwd=workspace,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1,
    )
    state = {"turns": 0, "completed": False, "verified": False, "reason": "no_run_end",
             "unfinished": 0, "approvals": 0, "denied": 0}
    lock = threading.Lock()
    ids = [10]

    def send(obj):
        with lock:
            proc.stdin.write(json.dumps(obj) + "\n")
            proc.stdin.flush()

    started = time.monotonic()
    deadline = meta.get("wall_clock_seconds", 900) + 180  # harness slack over the run's own

    for raw in proc.stdout:
        if time.monotonic() - started > deadline:
            send({"jsonrpc": "2.0", "id": "99", "method": "lmp/cancel",
                  "params": {"run_id": "1"}})
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        method, params = msg.get("method"), msg.get("params") or {}

        if method == "lmp/ready":
            send({"jsonrpc": "2.0", "id": "1", "method": "lmp/start", "params": {
                "mission": meta["mission"],
                "settings": {
                    "model_dir": model_dir, "workspace_root": workspace,
                    "mode": meta.get("mode", "agent"), "sampling": SAMPLING,
                    "max_iterations": meta.get("max_iterations", 30),
                    "wall_clock_seconds": meta.get("wall_clock_seconds", 900),
                    "sandbox_tier": 1,
                    "auto_approve_exec": True, "auto_approve_writes": True,
                    "require_approval": False, "system_prompt": "",
                    "context_budget_tokens": 96000,
                },
            }})
        elif method == "lmp/turn":
            state["turns"] += 1
            if verbose:
                detail = ""
                if params.get("tool_status") not in (None, "Ok"):
                    # The SUMMARY, not just the status. A run of identical ToolErrors is
                    # unattributable without it -- which is exactly how five consecutive
                    # replace_in_file failures looked like a line-numbering problem they
                    # turned out not to be.
                    detail = "  " + (params.get("summary") or "").split("\n")[0][:150]
                print(f"      turn {state['turns']:2d}  {params.get('tool_name') or '(text)':<16}"
                      f" {params.get('tool_status')}{detail}", flush=True)
        elif method == "lmp/approval_request":
            state["approvals"] += 1
            # A task can insist the card be denied -- that is how the destructive task is
            # scored on whether the DATA survived rather than on what the agent said.
            approved = not meta.get("deny_approvals", False)
            if not approved:
                state["denied"] += 1
            send({"jsonrpc": "2.0", "id": str(ids[0]), "method": "lmp/approve",
                  "params": {"request_id": params.get("request_id"), "approved": approved}})
            ids[0] += 1
        elif method == "lmp/verification":
            if params.get("passed") and params.get("falsifiable"):
                state["verified"] = True
        elif method == "lmp/run_end":
            state["completed"] = bool(params.get("completed"))
            state["reason"] = params.get("termination_reason")
            state["unfinished"] = params.get("unfinished_items", 0)
            send({"jsonrpc": "2.0", "id": "98", "method": "lmp/shutdown", "params": {}})

    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()

    # --- ground truth, measured after the fact and never asked of the agent ---
    check = subprocess.run(meta["check"], shell=True, cwd=workspace,
                           capture_output=True, text=True)
    solved = check.returncode == 0

    intact = True
    for rel in meta.get("protect", []):
        pristine = os.path.join(TASKS, meta["name"], "workspace", rel)
        after = os.path.join(workspace, rel)
        if not os.path.exists(after):
            intact = False
            break
        with open(pristine, "rb") as a, open(after, "rb") as b:
            if a.read() != b.read():
                intact = False
                break
    # Editing the tests is not solving the task, whatever the check says.
    if not intact:
        solved = False

    shutil.rmtree(workspace, ignore_errors=True)
    return {
        "name": meta["name"], "split": meta.get("split", "corpus"),
        "solved": solved, "intact": intact, "completed": state["completed"],
        "verified": state["verified"], "turns": state["turns"],
        "reason": state["reason"], "unfinished": state["unfinished"],
        "approvals": state["approvals"], "denied": state["denied"],
        "seconds": round(time.monotonic() - started, 1),
    }


def summarize(rows, split):
    subset = [r for r in rows if r["split"] == split]
    if not subset:
        return None
    return {
        "total": len(subset),
        "solved": sum(1 for r in subset if r["solved"]),
        "completed": sum(1 for r in subset if r["completed"]),
        "verified": sum(1 for r in subset if r["verified"]),
        "turns": sum(r["turns"] for r in subset),
    }


def report(rows):
    print()
    print(f"  {'task':<26} {'split':<8} {'solved':<7} {'compl':<6} {'verif':<6} "
          f"{'turns':<6} {'sec':<7} reason")
    print("  " + "-" * 92)
    for r in rows:
        print(f"  {r['name']:<26} {r['split']:<8} "
              f"{'YES' if r['solved'] else 'no':<7} "
              f"{'yes' if r['completed'] else '-':<6} "
              f"{'yes' if r['verified'] else '-':<6} "
              f"{r['turns']:<6} {r['seconds']:<7} {r['reason']}")
    print()
    for split in ("corpus", "holdout"):
        s = summarize(rows, split)
        if s:
            print(f"  {split:<8} solved {s['solved']}/{s['total']}   "
                  f"completed {s['completed']}   verified {s['verified']}   "
                  f"turns {s['turns']}")


def compare(rows):
    """Pins move deliberately: a regression fails, an improvement asks to be re-pinned.

    Exact equality is the wrong test here and eval.py's scorers are not a precedent for
    it -- those are deterministic functions of a fixed corpus, whereas this drives a 35B
    MoE at temperature 0.6 on Metal, where kernel scheduling and KV reuse make even a
    fixed seed only approximately reproducible. So the pin is a FLOOR on solved counts,
    not an equality. Turn counts are reported and never gated; they are the noisiest
    number here.
    """
    if not os.path.exists(PINS):
        print("\n  no pins yet -- run with --pin to establish them")
        return 0
    with open(PINS, encoding="utf-8") as fh:
        pins = json.load(fh)

    failures, improvements = [], []
    for split in ("corpus", "holdout"):
        s, p = summarize(rows, split), pins.get(split)
        if not s or not p:
            continue
        print(f"\n  {split}: solved {s['solved']}/{s['total']}   pinned "
              f"{p['solved']}/{p['total']}")
        if s["total"] != p["total"]:
            print(f"    (suite size changed {p['total']} -> {s['total']}; re-pin)")
            continue
        if s["solved"] < p["solved"]:
            failures.append(f"{split} solved {s['solved']} < pinned {p['solved']}")
        elif s["solved"] > p["solved"]:
            improvements.append(f"{split} solved {s['solved']} > pinned {p['solved']}")

    c, h = summarize(rows, "corpus"), summarize(rows, "holdout")
    if c and h and c["total"] and h["total"]:
        cr, hr = c["solved"] / c["total"], h["solved"] / h["total"]
        print(f"\n  solve rate: corpus {cr:.2f} vs holdout {hr:.2f}")
        # The corpus was iterated against with the key open; the holdout never was. A
        # holdout that scores BETTER means it leaked, not that the agent got good.
        if hr > cr:
            failures.append("the holdout now scores better than the tuned-against corpus "
                            "-- it has leaked, or the corpus has rotted")

    for line in improvements:
        print(f"  IMPROVED: {line} -- re-pin with --pin if this is real and repeatable")
    for line in failures:
        print(f"  FAIL: {line}")
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "list"])
    ap.add_argument("--split", choices=["corpus", "holdout"])
    ap.add_argument("--only")
    ap.add_argument("--model", default=os.environ.get("LMP_QWEN_DIR", DEFAULT_MODEL))
    ap.add_argument("--pin", action="store_true", help="write these scores as the pins")
    ap.add_argument("--json", help="also write the full per-task rows here")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every turn")
    args = ap.parse_args()

    tasks = load_tasks(args.split, args.only)
    if args.command == "list":
        for meta in tasks:
            print(f"  {meta['name']:<26} {meta['split']:<8} expect={meta['expect']}")
            print(f"      {meta['notes']}")
        return 0

    if not os.path.exists(SIDECAR):
        print(f"no sidecar at {SIDECAR}; build it first", file=sys.stderr)
        return 2

    print(f"  {len(tasks)} task(s), model {args.model}")
    rows = []
    for i, meta in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] {meta['name']} ({meta['split']})", flush=True)
        row = run_one(meta, args.model, args.verbose)
        rows.append(row)
        print(f"      -> solved={row['solved']} completed={row['completed']} "
              f"turns={row['turns']} {row['reason']} ({row['seconds']}s)", flush=True)

    report(rows)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)

    if args.pin:
        pins = {s: summarize(rows, s) for s in ("corpus", "holdout")
                if summarize(rows, s)}
        pins["_measured"] = time.strftime("%Y-%m-%d")
        pins["_note"] = ("Floors, not equalities: this drives a 35B MoE at temperature "
                         "0.6, where a fixed seed is only approximately reproducible. "
                         "A drop fails; an improvement asks to be re-pinned.")
        os.makedirs(os.path.dirname(PINS), exist_ok=True)
        with open(PINS, "w", encoding="utf-8") as fh:
            json.dump(pins, fh, indent=2)
            fh.write("\n")
        print(f"\n  pinned -> {os.path.relpath(PINS, ROOT)}")
        return 0
    return compare(rows)


if __name__ == "__main__":
    sys.exit(main())
