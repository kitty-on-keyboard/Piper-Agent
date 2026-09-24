#!/usr/bin/env python3
"""Watcher snapshot finds a slice when the workspace root has no task.json."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piper_ui


def main():
    with tempfile.TemporaryDirectory(prefix="piper-ui-watch-") as tmp:
        slice_dir = os.path.join(tmp, ".piper", "slices", "slice-001")
        os.makedirs(slice_dir)
        task_path = os.path.join(slice_dir, "task.json")
        result_path = os.path.join(slice_dir, "result.json")
        with open(task_path, "w", encoding="utf-8") as fh:
            json.dump({
                "id": "slice-001",
                "cwd": tmp,
                "prompt": "Edit a.py only.",
                "result_path": result_path,
            }, fh)
        with open(os.path.join(slice_dir, "live.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "seq": 1, "kind": "turn", "tool": "read_file", "path": "a.py",
                "status": "ok", "summary": "read a.py",
            }) + "\n")
        with open(os.path.join(slice_dir, "orch.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "kind": "review", "id": "slice-001", "verdict": "PASS",
                "text": "verdict:  PASS",
            }) + "\n")
        status = piper_ui.collect_status(tmp)
        live = status.get("recent_live") or []
        orch = status.get("recent_orch") or []
        if not live or live[0].get("tool") != "read_file":
            print(f"FAIL live lines: {live!r}", file=sys.stderr)
            return 1
        if not orch or orch[0].get("verdict") != "PASS":
            print(f"FAIL orch lines: {orch!r}", file=sys.stderr)
            return 1
        if status.get("task", {}).get("id") != "slice-001":
            print(f"FAIL task: {status.get('task')!r}", file=sys.stderr)
            return 1
        if os.path.isfile(os.path.join(tmp, "task.json")):
            print("FAIL root task.json should not exist", file=sys.stderr)
            return 1
        print("ok")
        return 0


if __name__ == "__main__":
    sys.exit(main())
