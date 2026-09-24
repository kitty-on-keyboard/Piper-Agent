#!/usr/bin/env python3
"""Piper Orchestrator Visualizer Server

Zero-dependency local dashboard pairing Piper Local Worker and Cloud Orchestrator.
Streams task packets, thinking blocks, tool calls, and result diffs in real-time.

Usage:
    python3 scripts/piper_ui.py [workspace_dir] [--port 8765] [--no-open]
"""

import argparse
import http.server
import json
import mimetypes
import os
import sys
import threading
import time
import webbrowser
from urllib.parse import urlparse

# Ensure mime types are registered
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/html", ".html")


def read_appended(path, offset):
    """Return (parsed_objects, new_offset).

    Missing file -> ([], offset) and do not change offset.
    If the file shrinks below offset, start over at 0.
    Incomplete trailing JSON is left for the next read: only consume
    complete lines (split on \\n). A partial last line stays unparsed
    and the returned offset points at the start of that partial line.
    Malformed complete lines are skipped.
    """
    if not os.path.exists(path):
        return [], offset

    size = os.path.getsize(path)
    if size < offset:
        offset = 0

    if size <= offset:
        return [], offset

    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()

    lines = data.split(b"\n")

    if data.endswith(b"\n"):
        complete_lines = lines[:-1]
        has_partial = False
    else:
        complete_lines = lines[:-1]
        partial = lines[-1]
        has_partial = bool(partial)

    parsed = []
    for line in complete_lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped.decode("utf-8"))
            parsed.append(obj)
        except Exception:
            pass

    if has_partial:
        last_newline = data.rfind(b"\n")
        new_offset = offset + last_newline + 1
    else:
        new_offset = size

    return parsed, new_offset


def read_json(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def read_jsonl(path, limit=None):
    rows = []
    if not path or not os.path.isfile(path):
        return rows
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return rows
    if limit is not None and len(lines) > limit:
        lines = lines[-limit:]
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError:
            pass
    return rows


def load_active(workspace_dir):
    data = read_json(os.path.join(workspace_dir, ".piper", "active.json"))
    if not data:
        return None
    task_path = data.get("task_path")
    result_path = data.get("result_path")
    if not isinstance(task_path, str) or not isinstance(result_path, str):
        return None
    return {
        "id": str(data.get("id") or ""),
        "task_path": os.path.abspath(task_path),
        "result_path": os.path.abspath(result_path),
    }


def list_slice_records(workspace_dir):
    """Slice dirs under .piper/slices/. A missing root task.json is normal."""
    root = os.path.join(workspace_dir, ".piper", "slices")
    records = []
    if not os.path.isdir(root):
        return records
    for name in sorted(os.listdir(root)):
        directory = os.path.join(root, name)
        task_path = os.path.join(directory, "task.json")
        if not os.path.isfile(task_path):
            continue
        task = read_json(task_path) or {}
        result_path = task.get("result_path") or os.path.join(directory, "result.json")
        if not isinstance(result_path, str):
            result_path = os.path.join(directory, "result.json")
        if not os.path.isabs(result_path):
            result_path = os.path.abspath(os.path.join(directory, result_path))
        result = read_json(result_path)
        if isinstance(result, dict) and result.get("status"):
            status = str(result["status"])
        elif os.path.isfile(os.path.join(directory, "live.jsonl")):
            status = "running"
        else:
            status = "idle"
        records.append({
            "id": str(task.get("id") or name),
            "dir": os.path.abspath(directory),
            "task_path": os.path.abspath(task_path),
            "result_path": os.path.abspath(result_path),
            "status": status,
            "prompt": task.get("prompt") if isinstance(task.get("prompt"), str) else "",
        })
    return records


def read_progress(workspace_dir):
    path = os.path.join(workspace_dir, ".piper", "progress.log")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return [line.rstrip("\n") for line in fh if line.strip()]
    except OSError:
        return []


def resolve_watch_dir(workspace_dir):
    """Active slice from .piper/active.json, else the last slice dir."""
    active = load_active(workspace_dir)
    if active and active.get("result_path"):
        return os.path.dirname(active["result_path"]), active
    slices = list_slice_records(workspace_dir)
    if slices:
        last = slices[-1]
        return last["dir"], {
            "id": last["id"],
            "task_path": last["task_path"],
            "result_path": last["result_path"],
        }
    return os.path.abspath(workspace_dir), None


def collect_status(workspace_dir):
    """Snapshot the watch UI needs. Does not require a root task.json."""
    workspace_dir = os.path.abspath(workspace_dir)
    watch_dir, active = resolve_watch_dir(workspace_dir)
    task = None
    if active and os.path.isfile(active.get("task_path") or ""):
        task = read_json(active["task_path"])
    if task is None:
        task = read_json(os.path.join(watch_dir, "task.json"))
    result_path = (active or {}).get("result_path") or os.path.join(watch_dir, "result.json")
    result = read_json(result_path) if os.path.isfile(result_path) else None
    return {
        "cwd": workspace_dir,
        "active": active,
        "watch_dir": watch_dir,
        "slices": list_slice_records(workspace_dir),
        "progress": read_progress(workspace_dir),
        "task": task,
        "result": result,
        "recent_live": read_jsonl(os.path.join(watch_dir, "live.jsonl")),
        "recent_orch": read_jsonl(os.path.join(watch_dir, "orch.jsonl")),
        "recent_events": read_jsonl(os.path.join(watch_dir, "events.jsonl"), limit=200),
    }


class EventBroker:
    def __init__(self):
        self._clients = set()
        self._lock = threading.Lock()

    def register(self, client):
        with self._lock:
            self._clients.add(client)

    def unregister(self, client):
        with self._lock:
            self._clients.discard(client)

    def broadcast(self, event_type, data):
        payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
        with self._lock:
            dead_clients = set()
            for client in list(self._clients):
                try:
                    client.write(payload)
                    client.flush()
                except Exception:
                    dead_clients.add(client)
            for dc in dead_clients:
                self._clients.discard(dc)


class WorkspaceWatcher:
    """Tails the active slice. A missing workspace-root task.json is normal."""

    def __init__(self, workspace_dir, broker, log_path=None):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.broker = broker
        self.explicit_log = os.path.abspath(log_path) if log_path else None
        self.file_mtimes = {}
        self.offsets = {}
        self.watch_dir = None
        self.active_id = None
        self.running = True

    def _arm_offset(self, path):
        """Start at EOF so history comes from /api/status, not a replay."""
        if path not in self.offsets:
            try:
                self.offsets[path] = os.path.getsize(path) if os.path.isfile(path) else 0
            except OSError:
                self.offsets[path] = 0

    def start(self):
        thread = threading.Thread(target=self._watch_loop, daemon=True)
        thread.start()

    def _read_json_safe(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _watch_loop(self):
        while self.running:
            watch_dir, active = resolve_watch_dir(self.workspace_dir)
            active_id = (active or {}).get("id") or ""
            if watch_dir != self.watch_dir or active_id != self.active_id:
                self.watch_dir = watch_dir
                self.active_id = active_id
                self.file_mtimes = {}
                self.offsets = {}
                self.broker.broadcast("slice_changed", collect_status(self.workspace_dir))

            journal_names = ("live.jsonl", "orch.jsonl", "events.jsonl")
            journals = [os.path.join(watch_dir, name) for name in journal_names]
            if self.explicit_log:
                journals.append(self.explicit_log)
            event_name = {
                "live.jsonl": "live_event",
                "orch.jsonl": "orch_event",
                "events.jsonl": "log_event",
            }
            for path in journals:
                self._arm_offset(path)
                try:
                    objs, new_offset = read_appended(path, self.offsets.get(path, 0))
                except OSError:
                    continue
                self.offsets[path] = new_offset
                kind = event_name.get(os.path.basename(path), "log_event")
                for obj in objs:
                    self.broker.broadcast(kind, obj)

            contract = {}
            if active and active.get("task_path"):
                contract[active["task_path"]] = "task_updated"
            else:
                contract[os.path.join(watch_dir, "task.json")] = "task_updated"
            result_path = (active or {}).get("result_path") or os.path.join(watch_dir, "result.json")
            contract[result_path] = "result_updated"
            contract[os.path.join(watch_dir, "awaiting_user.json")] = "gate_requested"
            contract[os.path.join(watch_dir, "answer.json")] = "answer_updated"
            for filepath, event_name_s in contract.items():
                if not os.path.isfile(filepath):
                    self.file_mtimes.pop(filepath, None)
                    continue
                try:
                    mtime = os.path.getmtime(filepath)
                except OSError:
                    continue
                if self.file_mtimes.get(filepath) == mtime:
                    continue
                self.file_mtimes[filepath] = mtime
                data = self._read_json_safe(filepath)
                if data:
                    self.broker.broadcast(event_name_s, data)

            time.sleep(0.1)


def make_handler(static_dir, workspace_dir, broker, watcher):
    class VisualizerHTTPHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Suppress normal HTTP logging for clean CLI output
            pass

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                # Send initial status
                init_data = json.dumps({"cwd": workspace_dir})
                self.wfile.write(f"event: status\ndata: {init_data}\n\n".encode("utf-8"))
                self.wfile.flush()

                broker.register(self.wfile)
                try:
                    while True:
                        # Heartbeat every 15s to keep connection alive
                        time.sleep(15)
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                except Exception:
                    pass
                finally:
                    broker.unregister(self.wfile)
                return

            if path == "/api/status":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                status_data = collect_status(workspace_dir)
                watch_dir = status_data.get("watch_dir") or workspace_dir
                status_data["awaiting_user"] = read_json(os.path.join(watch_dir, "awaiting_user.json"))
                status_data["answer"] = read_json(os.path.join(watch_dir, "answer.json"))
                self.wfile.write(json.dumps(status_data).encode("utf-8"))
                return

            # Serve UI static files
            if path == "/" or path == "":
                rel_path = "index.html"
            else:
                rel_path = path.lstrip("/")

            safe_path = os.path.normpath(os.path.join(static_dir, rel_path))
            if not safe_path.startswith(static_dir) or not os.path.exists(safe_path) or os.path.isdir(safe_path):
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"404 Not Found")
                return

            mime_type, _ = mimetypes.guess_type(safe_path)
            self.send_response(200)
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self.send_header("Content-Length", str(os.path.getsize(safe_path)))
            self.end_headers()
            with open(safe_path, "rb") as fh:
                self.wfile.write(fh.read())

        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path

            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b"{}"

            if path == "/wake":
                # Webhook wake endpoint for detached runs (--orch-webhook)
                try:
                    payload = json.loads(body.decode("utf-8"))
                    broker.broadcast("wake_event", payload)
                except Exception:
                    pass
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
                return

            if path == "/api/answer":
                # Gate approval/denial — same write_answer_file path as `piper answer`
                try:
                    import piper_worker as pw
                    payload = json.loads(body.decode("utf-8"))
                    if isinstance(payload.get("text"), str):
                        body_out = pw.build_answer_payload(text=payload["text"])
                    elif "answer" in payload:
                        body_out = pw.build_answer_payload(action=payload["answer"])
                    elif "action" in payload:
                        body_out = pw.build_answer_payload(action=payload["action"])
                    else:
                        raise ValueError("answer body needs text, answer, or action")
                    answer_path = pw.write_answer_file(workspace_dir, body_out)
                    broker.broadcast("answer_updated", body_out)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", "path": answer_path}).encode("utf-8"))
                except Exception as e:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(str(e).encode("utf-8"))
                return

            self.send_response(404)
            self.end_headers()

    return VisualizerHTTPHandler


def write_wake_url_file(workspace_dir, wake_url):
    """Persist wake URL under .piper/orch_webhook — parents must not copy from this panel."""
    root = os.path.abspath(workspace_dir)
    path = os.path.join(root, ".piper", "orch_webhook")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(wake_url.strip() + "\n")
    os.replace(tmp, path)
    return path


def run_server(workspace_dir, port=8765, open_browser=True, log_path=None):
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "ui"))
    workspace_dir = os.path.abspath(workspace_dir)
    broker = EventBroker()
    watcher = WorkspaceWatcher(workspace_dir, broker, log_path=log_path)
    watcher.start()

    handler = make_handler(static_dir, workspace_dir, broker, watcher)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)

    url = f"http://127.0.0.1:{port}"
    wake_url = url + "/wake"
    wake_file = write_wake_url_file(workspace_dir, wake_url)
    print(f"┌────────────────────────────────────────────────────────┐")
    print(f"│  Piper Orchestration Visualizer                        │")
    print(f"│  Dashboard: {url:<43}│")
    print(f"│  Workspace: {workspace_dir:<43}│")
    print(f"│  Wake file: {wake_file:<43}│")
    print(f"│  (worker reads the file — do not copy the URL)         │")
    print(f"└────────────────────────────────────────────────────────┘")

    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Piper visualizer...")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Piper Orchestrator Visualizer Server")
    parser.add_argument("workspace", nargs="?", default=".", help="Workspace directory to monitor (default: current directory)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    parser.add_argument("--no-open", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--log", type=str, default=None, help="Explicit path to lmp_events.jsonl")
    args = parser.parse_args()

    run_server(
        workspace_dir=args.workspace,
        port=args.port,
        open_browser=not args.no_open,
        log_path=args.log,
    )


if __name__ == "__main__":
    main()
