#!/usr/bin/env python3
"""Correct reference MCP server. The board must show this one all-green."""
import json, sys

TOOLS = [{"name": "echo", "title": "Echo",
          "description": "Echo the input back",
          "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}},
                          "required": ["text"]}}]

initialized = False


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def err(rid, code, message):
    send({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}})


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except Exception:
        err(None, -32700, "Parse error")
        continue

    method = msg.get("method")
    has_id = "id" in msg
    rid = msg.get("id")

    if not has_id:                       # notification: never respond
        if method == "notifications/initialized":
            initialized = True
        continue

    if method == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "reference", "version": "1.0.0"}}})
    elif method == "ping":
        send({"jsonrpc": "2.0", "id": rid, "result": {}})
    elif not initialized:
        err(rid, -32002, "Server not initialized")
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
    elif method == "tools/call":
        p = msg.get("params") or {}
        name = p.get("name")
        if name != "echo":
            err(rid, -32602, f"Unknown tool: {name}")
        else:
            txt = (p.get("arguments") or {}).get("text", "")
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "content": [{"type": "text", "text": str(txt)}], "isError": False}})
    else:
        err(rid, -32601, f"Method not found: {method}")
