#!/usr/bin/env python3
"""MCP client against scripts/warm_lsp_mcp.py. No model, no sidecar."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import warm_lsp_mcp  # noqa: E402


def rpc(proc, msg):
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        err = proc.stderr.read() if proc.stderr else b""
        raise RuntimeError(f"server closed stdout: {err!r}")
    return json.loads(line)


def start(flag: str | None):
    env = os.environ.copy()
    if flag is None:
        env.pop("LMP_WARM_LSP", None)
    else:
        env["LMP_WARM_LSP"] = flag
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "warm_lsp_mcp.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(ROOT),
        env=env,
        text=True,
    )
    rpc(
        proc,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        },
    )
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
    proc.stdin.flush()
    return proc


def call(proc, name, arguments, msg_id):
    msg = rpc(
        proc,
        {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    if "error" in msg:
        raise RuntimeError(msg["error"])
    return msg["result"]


def text_of(result) -> str:
    return "\n".join(block.get("text", "") for block in result.get("content") or [])


def check(cond, message):
    if not cond:
        raise SystemExit(message)


def test_flag_off():
    proc = start(None)
    try:
        status = call(proc, "lsp_status", {}, 2)
        check(status.get("isError") is False, f"status should succeed, got {status}")
        body = json.loads(text_of(status))
        check(body["enabled"] is False, f"flag off should report enabled false: {body}")
        diag = call(proc, "lsp_diagnostics", {"path": "testdata/warm_lsp/python/bad.py"}, 3)
        check(diag.get("isError") is True, f"diagnostics must error when off: {diag}")
        check("LMP_WARM_LSP is off" in text_of(diag), text_of(diag))
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_stub_and_real():
    proc = start("1")
    try:
        swift = call(proc, "lsp_diagnostics", {"path": "testdata/warm_lsp/python/bad.py"}, 2)
        # real python first
        check(swift.get("isError") is False, f"python diagnostics failed: {text_of(swift)}")
        payload = swift.get("structuredContent") or {}
        diags = payload.get("diagnostics") or []
        check(len(diags) >= 1, f"expected a python diagnostic, got {text_of(swift)}")
        check(any(d.get("severity") == "error" for d in diags), text_of(swift))

        cpp = call(proc, "lsp_diagnostics", {"path": "testdata/warm_lsp/cpp/bad.cpp"}, 3)
        check(cpp.get("isError") is False, f"cpp diagnostics failed: {text_of(cpp)}")
        cpp_diags = (cpp.get("structuredContent") or {}).get("diagnostics") or []
        check(len(cpp_diags) >= 1, f"expected a cpp diagnostic, got {text_of(cpp)}")

        # Warm reuse: second call must also return the planted error.
        again = call(proc, "lsp_diagnostics", {"path": "testdata/warm_lsp/python/bad.py"}, 4)
        check(again.get("isError") is False, text_of(again))
        check(len((again.get("structuredContent") or {}).get("diagnostics") or []) >= 1, text_of(again))

        stub = call(proc, "lsp_diagnostics", {"path": "NoSuch.swift"}, 5)
        check(stub.get("isError") is True, f"swift must stay parked: {stub}")
        check("parked" in text_of(stub), text_of(stub))

        gd = call(proc, "lsp_diagnostics", {"path": "testdata/warm_lsp/gdscript/bad.gd"}, 6)
        check(gd.get("isError") is False, f"gdscript diagnostics failed: {text_of(gd)}")
        gd_diags = (gd.get("structuredContent") or {}).get("diagnostics") or []
        check(len(gd_diags) >= 1, f"expected a gdscript diagnostic, got {text_of(gd)}")
        check("Godot.app" not in text_of(gd), text_of(gd))
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_seatbelt_denies_outside_workspace():
    profile = warm_lsp_mcp.seatbelt_profile(ROOT)
    # Not tempfile.gettempdir(): a parent shell may point TMPDIR at the workspace.
    outside = Path("/private/tmp/warm-lsp-escape-probe")
    denied = subprocess.run(
        ["sandbox-exec", "-p", profile, "/usr/bin/touch", str(outside)],
        capture_output=True,
        text=True,
    )
    check(denied.returncode != 0, f"touch outside workspace was allowed: {denied.stderr}")
    allowed = subprocess.run(
        ["sandbox-exec", "-p", profile, "/bin/ls", str(ROOT / "testdata" / "warm_lsp")],
        capture_output=True,
        text=True,
    )
    check(allowed.returncode == 0, f"workspace read denied: {allowed.stderr}")
    home_list = subprocess.run(
        ["sandbox-exec", "-p", profile, "/bin/ls", str(Path.home())],
        capture_output=True,
        text=True,
    )
    check(home_list.returncode != 0, "listing $HOME was allowed")


def test_path_inside():
    root = (ROOT / "testdata" / "warm_lsp").resolve()
    inside_file = warm_lsp_mcp.path_inside(root, "python/bad.py")
    check(inside_file == root / "python" / "bad.py", f"expected path inside workspace: {inside_file}")

    # Traversal attempt outside workspace
    try:
        warm_lsp_mcp.path_inside(root, "../../scripts/warm_lsp_mcp.py")
        check(False, "should have raised ValueError for escaping path")
    except ValueError as e:
        check("escapes the workspace" in str(e), f"unexpected error message: {e}")

    # Traversal with absolute path outside
    try:
        warm_lsp_mcp.path_inside(root, "/etc/passwd")
        check(False, "should have raised ValueError for absolute escaping path")
    except ValueError as e:
        check("escapes the workspace" in str(e), f"unexpected error message: {e}")

    # Workspace at root directory '/'
    root_dir = Path("/").resolve()
    inside_root = warm_lsp_mcp.path_inside(root_dir, "etc/passwd")
    check(inside_root == root_dir / "etc" / "passwd", f"expected path inside root: {inside_root}")


def test_godot_defaults_to_godoer_fork():
    keys = ("LMP_GODOT_BIN", "GODOER_GODOT_BIN", "GODOT_BIN")
    saved = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    try:
        path = warm_lsp_mcp.godot_bin()
        check(path == warm_lsp_mcp.DEFAULT_GODOER_GODOT.resolve(), f"default godot is {path}")
        check("Godot.app" not in str(path), str(path))
        check(path.name.startswith("godot.macos.editor"), path.name)
    except warm_lsp_mcp.LspError:
        # Default Godoer Godot binary might not exist on non-dev environments
        pass
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    test_path_inside()
    test_godot_defaults_to_godoer_fork()
    test_flag_off()
    test_seatbelt_denies_outside_workspace()
    test_stub_and_real()
    print("warm_lsp tests passed")


if __name__ == "__main__":
    main()
