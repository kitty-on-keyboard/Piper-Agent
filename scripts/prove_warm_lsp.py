#!/usr/bin/env python3
"""Cold vs warm diagnostic latency for the warm-lsp MCP server.

Writes nothing but stdout. Numbers are this process, this machine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N = 10
CASES = (
    ("cpp", "testdata/warm_lsp/cpp/bad.cpp", "//"),
    ("python", "testdata/warm_lsp/python/bad.py", "#"),
    ("gdscript", "testdata/warm_lsp/gdscript/bad.gd", "#"),
)


def pct(samples, p):
    ordered = sorted(samples)
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, int(round((p / 100) * (len(ordered) - 1)))))
    return ordered[index]


def start_server():
    env = os.environ.copy()
    env["LMP_WARM_LSP"] = "1"
    for key in ("LMP_GODOT_BIN", "GODOER_GODOT_BIN", "GODOT_BIN"):
        env.pop(key, None)
    return subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "warm_lsp_mcp.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=str(ROOT),
        env=env,
        text=True,
    )


def rpc(proc, msg):
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("server closed stdout")
    return json.loads(line)


def handshake(proc):
    rpc(
        proc,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "prove", "version": "0"},
            },
        },
    )
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
    proc.stdin.flush()


def call(proc, msg_id, name, arguments):
    started = time.perf_counter()
    msg = rpc(
        proc,
        {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    result = msg.get("result") or {}
    if result.get("isError"):
        text = "\n".join(block.get("text", "") for block in result.get("content") or [])
        raise RuntimeError(f"{name} failed: {text}")
    return elapsed_ms, result


def diag_count(result) -> int:
    return len((result.get("structuredContent") or {}).get("diagnostics") or [])


def rss_of(pid) -> int | None:
    """RSS of this process plus descendants. basedpyright's Node child is the bulk."""
    if not pid:
        return None
    out = subprocess.run(["ps", "-ax", "-o", "pid=,ppid=,rss="], capture_output=True, text=True)
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            continue
        rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
    by_parent: dict[int, list[int]] = {}
    rss = {proc: size for proc, _parent, size in rows}
    for proc, parent, _size in rows:
        by_parent.setdefault(parent, []).append(proc)
    if pid not in rss:
        return None
    total = 0
    stack = [pid]
    seen = set()
    while stack:
        current = stack.pop()
        if current in seen or current not in rss:
            continue
        seen.add(current)
        total += rss[current]
        stack.extend(by_parent.get(current, []))
    return total


def main():
    proc = start_server()
    msg_id = 2
    try:
        handshake(proc)
        rows = []
        for language, path, mark in CASES:
            base = (ROOT / path).read_text(encoding="utf-8")
            cold = []
            for index in range(N):
                call(proc, msg_id, "lsp_restart", {"language": language})
                msg_id += 1
                # A fresh comment forces a real publish. An identical buffer is not a cold spawn.
                text = base + f"{mark} cold {index}\n"
                elapsed, result = call(proc, msg_id, "lsp_diagnostics", {"path": path, "text": text})
                msg_id += 1
                if diag_count(result) < 1:
                    raise RuntimeError(f"{language} cold returned no diagnostic")
                cold.append(elapsed)
            warm = []
            idle_rss = None
            for index in range(N):
                text = base + f"{mark} warm {index}\n"
                elapsed, result = call(proc, msg_id, "lsp_diagnostics", {"path": path, "text": text})
                msg_id += 1
                if diag_count(result) < 1:
                    raise RuntimeError(f"{language} warm returned no diagnostic")
                warm.append(elapsed)
                if index == 0:
                    _, status = call(proc, msg_id, "lsp_status", {})
                    msg_id += 1
                    body = json.loads(
                        "\n".join(block.get("text", "") for block in status.get("content") or [])
                    )
                    pids = [item.get("pid") for item in body.get("running") or [] if item.get("language") == language]
                    idle_rss = rss_of(pids[0]) if pids else None
            _, status = call(proc, msg_id, "lsp_status", {})
            msg_id += 1
            body = json.loads("\n".join(block.get("text", "") for block in status.get("content") or []))
            pids = [item.get("pid") for item in body.get("running") or [] if item.get("language") == language]
            busy_rss = rss_of(pids[0]) if pids else None
            rows.append(
                {
                    "language": language,
                    "cold_p50_ms": round(pct(cold, 50), 1),
                    "cold_p95_ms": round(pct(cold, 95), 1),
                    "warm_p50_ms": round(pct(warm, 50), 1),
                    "warm_p95_ms": round(pct(warm, 95), 1),
                    "rss_after_first_warm_kb": idle_rss,
                    "rss_after_n_kb": busy_rss,
                    "n": N,
                    "cold_ms": [round(v, 1) for v in cold],
                    "warm_ms": [round(v, 1) for v in warm],
                    "godot_bin": body.get("godot_bin"),
                }
            )
        print(json.dumps({"rows": rows}, indent=2))
    finally:
        proc.kill()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
