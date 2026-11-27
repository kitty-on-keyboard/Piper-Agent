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

        # Traversal & outside-workspace active.json pointers must be ignored
        with open(os.path.join(tmp, ".piper", "active.json"), "w", encoding="utf-8") as fh:
            json.dump({
                "id": "escape",
                "task_path": "/etc/passwd",
                "result_path": "/etc/passwd",
            }, fh)
        _, active = piper_ui.resolve_watch_dir(tmp)
        if active and active.get("id") == "escape":
            print("FAIL accepted out-of-workspace active.json path", file=sys.stderr)
            return 1

        # HTTP security tests: static dir prefix traversal, /wake token + size + CORS
        import http.server
        import threading
        import urllib.error
        import urllib.request

        ui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
        broker = piper_ui.EventBroker()
        watcher = piper_ui.WorkspaceWatcher(tmp, broker)
        watcher.running = False
        handler = piper_ui.make_handler(ui_dir, tmp, broker, watcher, wake_token="secret-tok")
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            # 1. Path traversal / sibling prefix bypass
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/../piper_ui.py")
                print("FAIL static traversal succeeded", file=sys.stderr)
                return 1
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    print(f"FAIL expected 404 on static traversal, got {e.code}", file=sys.stderr)
                    return 1

            # 2. Missing wake token -> 403
            req = urllib.request.Request(f"http://127.0.0.1:{port}/wake", data=b"{}", method="POST")
            try:
                urllib.request.urlopen(req)
                print("FAIL /wake accepted missing token", file=sys.stderr)
                return 1
            except urllib.error.HTTPError as e:
                if e.code != 403:
                    print(f"FAIL expected 403 for bad token, got {e.code}", file=sys.stderr)
                    return 1

            # 3. Valid wake token -> 200
            req_ok = urllib.request.Request(
                f"http://127.0.0.1:{port}/wake?token=secret-tok",
                data=b'{"kind":"done"}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req_ok) as r:
                if r.status != 200:
                    print(f"FAIL /wake valid token got {r.status}", file=sys.stderr)
                    return 1

            # 4. Oversized Content-Length -> 413
            req_big = urllib.request.Request(
                f"http://127.0.0.1:{port}/wake?token=secret-tok",
                data=b"x",
                headers={"Content-Length": str(2 * 1024 * 1024)},
                method="POST",
            )
            try:
                urllib.request.urlopen(req_big)
                print("FAIL /wake accepted oversized Content-Length", file=sys.stderr)
                return 1
            except urllib.error.HTTPError as e:
                if e.code != 413:
                    print(f"FAIL expected 413 for oversized wake, got {e.code}", file=sys.stderr)
                    return 1
        finally:
            srv.shutdown()
            srv.server_close()

        print("ok")
        return 0


if __name__ == "__main__":
    sys.exit(main())
