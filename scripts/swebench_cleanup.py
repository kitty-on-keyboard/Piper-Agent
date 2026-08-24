#!/usr/bin/env python3
"""Account for everything the SWE-bench work put on this machine, and remove it.

WHY THIS EXISTS. The benchmark work writes several gigabytes OUTSIDE the repository --
git mirrors, per-version virtualenvs, checkouts in $TMPDIR, a global config an installed
tool wrote without being asked. None of it is in git, so none of it shows up in `git
status`, and six months from now nobody will remember it is there. The previous session's
build tree lived in a session scratchpad and is simply gone; this is the same problem
caught earlier.

Dry-run is the default and deletion needs --yes, because the enumeration is the point:
knowing what is there is useful on its own, and a cleanup tool that deletes by default is
a cleanup tool nobody runs twice.

Nothing inside the git repository is ever a candidate. Source belongs to git, and git
already knows how to revert it.

Usage:
  swebench_cleanup.py                 what exists, and how big it is
  swebench_cleanup.py --yes           remove the removable
  swebench_cleanup.py --only tmp      one category
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get(
    "LMP_SWEBENCH_WORK", "/Users/dev/Desktop/seans_projects_local/swebench_work"
)
TMPDIR = os.environ.get("TMPDIR", "/tmp")

# Prefixes this work gives its temporary directories. `lmpeval-` is deliberately NOT
# here: agent_eval creates and removes those itself, and a run in flight is using them.
TMP_PREFIXES = ("swe_", "swe_probe_", "swe_leak_", "swe_harness_")


def du(path):
    if not os.path.exists(path):
        return None
    out = subprocess.run(["du", "-sk", path], capture_output=True, text=True).stdout
    try:
        return int(out.split()[0]) * 1024
    except (ValueError, IndexError):
        return 0


def human(size):
    if size is None:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size / 1:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def size_str(nbytes):
    if nbytes is None:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.0f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def tmp_checkouts():
    hits = []
    for prefix in TMP_PREFIXES:
        hits.extend(glob.glob(os.path.join(TMPDIR, prefix + "*")))
    return sorted(set(hits))


def removable():
    """(category, path, note) for everything this work created and may delete."""
    items = [
        ("work", os.path.join(WORK, "repos"),
         "bare git mirrors of the 12 benchmark repos; re-clonable, slow to refetch"),
        ("work", os.path.join(WORK, "envs"),
         "per-(repo, version) virtualenvs built by the installability probe"),
        ("work", os.path.join(WORK, "venv"),
         "the harness venv: pandas, mini-swe-agent, swebench"),
        ("work", os.path.join(WORK, "data"),
         "SWE-bench Lite parquet; a 1 MB download"),
        ("work", os.path.join(WORK, "runs"),
         "MEASUREMENTS -- eval logs, probe results, predictions. Keep unless finished"),
        ("tool", os.path.expanduser("~/Library/Application Support/mini-swe-agent"),
         "global config mini-swe-agent wrote on first run, without being asked"),
    ]
    items += [("tmp", p, "stray checkout from an interrupted run") for p in tmp_checkouts()]
    return items


def shared():
    """Things this work TOUCHED but must not delete: other projects share them."""
    return [
        (os.path.expanduser("~/.cache/uv"),
         "uv's download cache -- shared with every other project on this machine"),
        ("/opt/homebrew/opt/python@3.12",
         "the interpreter uv borrowed; it was already installed, and is not ours"),
        (os.path.join(ROOT, "scripts", "swebench_run.py"),
         "source, inside the repo -- git owns it, this tool never touches the repo"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually delete")
    ap.add_argument("--only", choices=["work", "tmp", "tool"],
                    help="restrict to one category")
    ap.add_argument("--keep-runs", action="store_true", default=True,
                    help="never delete the measurements (default)")
    ap.add_argument("--include-runs", action="store_true",
                    help="delete runs/ too -- this throws away measurements")
    args = ap.parse_args()

    items = [i for i in removable() if not args.only or i[0] == args.only]
    if not args.include_runs:
        items = [i for i in items if not i[1].endswith(os.sep + "runs")]

    present = [(cat, path, note, du(path)) for cat, path, note in items
               if os.path.exists(path)]
    total = sum(s or 0 for _, _, _, s in present)

    print(f"\n  SWE-bench working state  (work dir: {WORK})\n")
    if not present:
        print("    nothing to remove -- this machine is already clean\n")
    for cat, path, note, size in present:
        print(f"    [{cat:<4}] {size_str(size):>8}  {path}")
        print(f"             {note}")
    if present:
        print(f"\n    {len(present)} item(s), {size_str(total)} total")

    print("\n  Touched but NOT removable (shared with the rest of the machine):")
    for path, note in shared():
        mark = "exists" if os.path.exists(path) else "absent"
        print(f"    [{mark:<6}] {path}\n             {note}")

    if not args.include_runs and os.path.exists(os.path.join(WORK, "runs")):
        print(f"\n  Measurements kept at {os.path.join(WORK, 'runs')} "
              f"({size_str(du(os.path.join(WORK, 'runs')))}). "
              f"--include-runs deletes them too.")

    if not args.yes:
        print("\n  Dry run. Re-run with --yes to delete.\n")
        return 0

    if not present:
        return 0
    print()
    for cat, path, _note, size in present:
        try:
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
            print(f"    removed {size_str(size):>8}  {path}")
        except OSError as exc:
            print(f"    FAILED  {path}: {exc}", file=sys.stderr)
    print(f"\n  reclaimed {size_str(total)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
