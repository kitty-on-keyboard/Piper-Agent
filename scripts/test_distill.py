#!/usr/bin/env python3
"""`piper distill` asks for text, waits its turn, and never loads twice.

No model is loaded here: the daemon is a fake on a Unix socket and `mlx_lm` is
a fake module on PYTHONPATH. Run: python3 scripts/test_distill.py
"""
import json
import os
import socket
import sys
import tempfile
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piper_worker as pw  # noqa: E402


INCIDENTS = {"incidents": [
    {"id": "INC-001", "kind": "SCRIPT_FATAL", "message": "Invalid call. Nonexistent function 'jump'",
     "file": "res://player.gd", "line": 12, "source_context": "player.jump()"},
]}


def _workdir():
    d = tempfile.mkdtemp(prefix="pd")
    with open(os.path.join(d, "incidents.json"), "w") as f:
        json.dump(INCIDENTS, f)
    return d


def _serve(sock_path, handler):
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(5)

    def loop():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            data = b""
            while not data.endswith(b"\n"):
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
            handler(conn, json.loads(data.decode()) if data else {})

    threading.Thread(target=loop, daemon=True).start()
    return srv


def _args(work, sock, **extra):
    return types.SimpleNamespace(
        incidents=os.path.join(work, "incidents.json"), model_dir=None,
        out=os.path.join(work, "out.json"), max_tokens=64, socket=sock,
        allow_cold=False, timeout=30, **extra)


def test_a_warm_distill_is_a_text_request():
    work = _workdir()
    sock = os.path.join(work, "w.sock")
    seen = {}

    def handler(conn, req):
        if req.get("method") == "ping":
            conn.sendall(b'{"status":"ok"}\n')
        elif req.get("method") == "run":
            seen["request"] = req
            with open(req["task"]) as f:
                seen["task"] = json.load(f)
            with open(seen["task"]["result_path"], "w") as f:
                json.dump({"message": "Root Cause: [INC-001] no jump()."}, f)
        conn.close()

    srv = _serve(sock, handler)
    try:
        args = _args(work, sock)
        args.model_dir = work  # any existing directory; the daemon owns the model
        assert pw.cmd_distill(args) == pw.EXIT_OK
    finally:
        srv.close()

    task = seen["task"]
    assert task["mode"] == "plan", task
    assert task["auto_approve_exec"] is False
    assert task["auto_approve_writes"] is False
    assert task["auto_approve_irreversible"] is False
    assert task["on_ask"] == "end"
    assert "auto_approve_all" not in seen["request"], seen["request"]
    out = json.load(open(args.out))
    assert out["status"] == "diagnosed"
    assert out["diagnoses"][0]["status"] == "ok"


def test_a_busy_daemon_is_not_an_offline_one():
    """A daemon serving someone else does not answer a ping within a second.
    Read as offline, the caller cold-loaded a second model beside it."""
    work = _workdir()
    sock = os.path.join(work, "w.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock)
    srv.listen(5)  # listening, never accepting: exactly a daemon mid-request
    try:
        assert pw._daemon_state(sock) == "busy"
        args = _args(work, sock, )
        args.allow_cold = True
        loaded = []
        real = pw._cold_generate
        pw._cold_generate = lambda *a, **k: loaded.append(1) or (["x"], "")
        try:
            assert pw.cmd_distill(args) == pw.EXIT_OK
        finally:
            pw._cold_generate = real
    finally:
        srv.close()
    assert not loaded, "a busy daemon was treated as offline and a model was cold-loaded"
    assert json.load(open(args.out))["status"] == "skipped_busy"


def test_a_refused_connect_to_a_running_daemon_is_busy():
    """Measured on a live daemon: mid-request, with its listen backlog full,
    every connect is refused -- read as offline, --allow-cold then loaded a
    second model beside the first."""
    work = _workdir()
    sock = os.path.join(work, "worker.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock)
    srv.close()  # the socket file stays; nothing accepts on it
    with open(os.path.join(work, "worker.pid"), "w") as f:
        f.write(str(os.getpid()))
    assert pw._daemon_state(sock) == "busy"
    os.remove(os.path.join(work, "worker.pid"))
    assert pw._daemon_state(sock) == "offline"


def test_a_skip_always_writes_its_status():
    """A caller that finds no file cannot tell skipped from crashed, and one
    that finds an old file reads it as this run's. The cold path, finding
    another distill holding the lock, exited 0 and wrote nothing -- so last
    week's diagnosis came back as this run's."""
    import fcntl

    work = _workdir()
    _fake_mlx(work)  # so an mlx_lm python is found and the lock is reached
    args = _args(work, os.path.join(work, "absent.sock"))
    args.allow_cold = True
    args.model_dir = work
    with open(args.out, "w") as f:
        json.dump({"status": "diagnosed", "diagnoses": [{"diagnosis": "last week"}]}, f)
    holder = open("/tmp/piper_distill.lock", "w")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
    try:
        assert pw.cmd_distill(args) == pw.EXIT_OK
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()
        os.environ.pop("PYTHONPATH", None)
    assert json.load(open(args.out))["status"] == "skipped_busy"

    offline = _args(_workdir(), os.path.join(work, "absent.sock"))
    assert pw.cmd_distill(offline) == pw.EXIT_OK
    assert json.load(open(offline.out))["status"] == "skipped_daemon_offline"


def _fake_mlx(work, *, sleep=0.0):
    pkg = os.path.join(work, "fake")
    os.makedirs(os.path.join(pkg, "mlx_lm"))
    with open(os.path.join(pkg, "mlx_lm", "__init__.py"), "w") as f:
        f.write(
            "import os, time\n"
            f"LOG = {os.path.join(work, 'loads.log')!r}\n"
            "def load(path):\n"
            "    with open(LOG, 'a') as f: f.write(str(os.getpid()) + '\\n')\n"
            "    return object(), object()\n"
            "def generate(model, tok, prompt, max_tokens, verbose):\n"
            f"    time.sleep({sleep})\n"
            "    return 'diagnosis for ' + prompt[-20:]\n"
        )
    os.environ["PYTHONPATH"] = pkg
    return os.path.join(work, "loads.log")


def test_the_model_is_loaded_once_for_every_incident():
    """One cold load per incident was 15 GB each, one after another."""
    work = _workdir()
    log = _fake_mlx(work)
    try:
        outs, err = pw._cold_generate(sys.executable, work, ["a", "b", "c"], 16, 30)
    finally:
        os.environ.pop("PYTHONPATH", None)
    assert err == "" and len(outs) == 3, (outs, err)
    assert len(open(log).read().split()) == 1


def test_a_timed_out_load_does_not_outlive_the_caller():
    """With no timeout, and a lock held only by the parent, a caller giving up
    left the model process running with the lock already released."""
    work = _workdir()
    log = _fake_mlx(work, sleep=30)
    try:
        start = time.monotonic()
        outs, err = pw._cold_generate(sys.executable, work, ["a"], 16, 1.5)
    finally:
        os.environ.pop("PYTHONPATH", None)
    assert outs is None and "timed out" in err, err
    assert time.monotonic() - start < 15
    pid = int(open(log).read().split()[0])
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    assert not alive, f"the model process {pid} survived the timeout"


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
