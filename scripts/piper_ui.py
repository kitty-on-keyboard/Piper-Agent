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
    """Watches task.json, result.json, awaiting_user.json, answer.json, and events.jsonl."""

    def __init__(self, workspace_dir, broker, log_path=None):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.broker = broker
        self.explicit_log = log_path
        self.log_path = self._resolve_log_path(log_path)
        self.file_mtimes = {}
        self.log_offset = 0
        self.extra_offsets = {}
        self.running = True

        # Position offset to read recent events
        if os.path.exists(self.log_path):
            self.log_offset = max(0, os.path.getsize(self.log_path) - 50000)

    def _resolve_log_path(self, explicit_log=None):
        if explicit_log and os.path.exists(explicit_log):
            return os.path.abspath(explicit_log)
        candidates = [
            explicit_log,
            os.environ.get("LMP_EVENT_LOG"),
            os.path.join(self.workspace_dir, "packets", "events.jsonl"),
            os.path.join(self.workspace_dir, "events.jsonl"),
            os.path.join(self.workspace_dir, "lmp_events.jsonl"),
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return os.path.abspath(c)

        task_fp = os.path.join(self.workspace_dir, "task.json")
        if os.path.exists(task_fp):
            try:
                with open(task_fp, "r", encoding="utf-8") as f:
                    tj = json.load(f)
                    rp = tj.get("result_path")
                    if rp:
                        cand = os.path.join(os.path.dirname(rp), "events.jsonl")
                        if os.path.exists(cand):
                            return os.path.abspath(cand)
            except Exception:
                pass
        return explicit_log or os.path.join(self.workspace_dir, "packets", "events.jsonl")

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
        targets = {
            "task.json": "task_updated",
            "result.json": "result_updated",
            "awaiting_user.json": "gate_requested",
            "answer.json": "answer_updated",
        }

        while self.running:
            # 1. Watch contract files
            for filename, event_name in list(targets.items()):
                filepath = os.path.join(self.workspace_dir, filename) if not os.path.isabs(filename) else filename
                if os.path.exists(filepath):
                    try:
                        mtime = os.path.getmtime(filepath)
                        if self.file_mtimes.get(filename) != mtime:
                            self.file_mtimes[filename] = mtime
                            data = self._read_json_safe(filepath)
                            if data:
                                self.broker.broadcast(event_name, data)
                                # If task was updated, check if it specifies custom result_path
                                if filename == "task.json" and data.get("result_path"):
                                    targets[data["result_path"]] = "result_updated"
                    except Exception:
                        pass
                else:
                    if filename in self.file_mtimes:
                        self.file_mtimes.pop(filename, None)

            # Re-resolve log path if not exists
            if not os.path.exists(self.log_path):
                cand = self._resolve_log_path(self.explicit_log)
                if os.path.exists(cand):
                    self.log_path = cand
                    self.log_offset = 0

            # 2. Tail live.jsonl and orch.jsonl before the event log.
            # Live lines and log lines use different seq spaces. The page
            # stops applying log tool rows once any live line has arrived,
            # so a same-tick batch must deliver live first.
            result_dir = None
            task_fp = os.path.join(self.workspace_dir, "task.json")
            if os.path.exists(task_fp):
                try:
                    with open(task_fp, "r", encoding="utf-8") as f:
                        tj = json.load(f)
                        rp = tj.get("result_path")
                        if rp:
                            result_dir = os.path.abspath(os.path.dirname(rp))
                except Exception:
                    pass

            live_paths = set()
            orch_paths = set()
            for d in [self.workspace_dir] + ([result_dir] if result_dir else []):
                live_paths.add(os.path.join(d, "live.jsonl"))
                orch_paths.add(os.path.join(d, "orch.jsonl"))

            all_extra = live_paths | orch_paths
            for p in all_extra:
                if p not in self.extra_offsets:
                    self.extra_offsets[p] = 0

            for p in sorted(live_paths):
                if p in orch_paths:
                    continue
                offset = self.extra_offsets.get(p, 0)
                objs, new_offset = read_appended(p, offset)
                self.extra_offsets[p] = new_offset
                for obj in objs:
                    self.broker.broadcast("live_event", obj)

            for p in sorted(orch_paths):
                if p in live_paths:
                    continue
                offset = self.extra_offsets.get(p, 0)
                objs, new_offset = read_appended(p, offset)
                self.extra_offsets[p] = new_offset
                for obj in objs:
                    self.broker.broadcast("orch_event", obj)

            # 3. Tail events log (finished runs, and the fallback when no
            # live journal is being written).
            try:
                events, self.log_offset = read_appended(self.log_path, self.log_offset)
                for ev in events:
                    self.broker.broadcast("log_event", ev)
            except Exception:
                pass

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

                def read_j(fn):
                    fp = os.path.join(workspace_dir, fn)
                    if os.path.exists(fp):
                        try:
                            with open(fp, "r", encoding="utf-8") as fh:
                                return json.load(fh)
                        except Exception:
                            return None
                    return None

                recent_events = []
                log_file = watcher.log_path
                if os.path.exists(log_file):
                    try:
                        with open(log_file, "r", encoding="utf-8", errors="replace") as lf:
                            lines = lf.readlines()[-200:]
                            for l in lines:
                                l = l.strip()
                                if l:
                                    try:
                                        recent_events.append(json.loads(l))
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                task_data = read_j("task.json")
                res_data = read_j("result.json")
                if not res_data and task_data and task_data.get("result_path"):
                    rp = task_data["result_path"]
                    if os.path.exists(rp):
                        try:
                            with open(rp, "r", encoding="utf-8") as rf:
                                res_data = json.load(rf)
                        except Exception:
                            pass

                def read_jsonl(path):
                    rows = []
                    if not path or not os.path.exists(path):
                        return rows
                    try:
                        with open(path, "r", encoding="utf-8", errors="replace") as fh:
                            for raw in fh:
                                raw = raw.strip()
                                if not raw:
                                    continue
                                try:
                                    rows.append(json.loads(raw))
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    return rows

                live_dirs = [workspace_dir]
                if task_data and task_data.get("result_path"):
                    live_dirs.append(os.path.dirname(os.path.abspath(task_data["result_path"])))
                recent_live = []
                recent_orch = []
                seen_live_paths = set()
                for d in live_dirs:
                    for name, bucket in (("live.jsonl", recent_live), ("orch.jsonl", recent_orch)):
                        fp = os.path.join(d, name)
                        if fp in seen_live_paths:
                            continue
                        seen_live_paths.add(fp)
                        bucket.extend(read_jsonl(fp))

                status_data = {
                    "cwd": workspace_dir,
                    "task": task_data,
                    "result": res_data,
                    "recent_live": recent_live,
                    "recent_orch": recent_orch,
                    "awaiting_user": read_j("awaiting_user.json"),
                    "answer": read_j("answer.json"),
                    "recent_events": recent_events,
                }
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
