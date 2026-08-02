#!/usr/bin/env python3
"""Deliberately broken MCP server. The board must catch every one of these.

Planted defects:
  1. prints a human banner to stdout       -> stdout-clean must FAIL
  2. replies to notifications              -> notif / unknown-notif must FAIL
  3. coerces string ids to int 0           -> strid must FAIL
  4. returns -32601 as -1                  -> -32601 must FAIL
  5. dies on malformed JSON                -> -32700 must FAIL
  6. serves tools/list before initialize   -> pre-init must FAIL
  7. unknown tool returns a happy result   -> badtool must FAIL
"""
import json, sys

print("MCP server starting up!")          # defect 1
sys.stdout.flush()

TOOLS = [{"name": "echo", "inputSchema": {"type": "object", "properties": {}}}]


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)                # defect 5: no try/except, dies
    method = msg.get("method")
    rid = msg.get("id", 0)
    if isinstance(rid, str):
        rid = 0                           # defect 3

    if method == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "serverInfo": {"name": "bad", "version": "0"}}})
    elif method == "ping":
        send({"jsonrpc": "2.0", "id": rid, "result": {}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})  # defect 6
    elif method == "tools/call":
        send({"jsonrpc": "2.0", "id": rid, "result": {                   # defect 7
            "content": [{"type": "text", "text": "sure, whatever"}]}})
    else:
        # defect 2 (answers notifications too) + defect 4 (wrong code)
        send({"jsonrpc": "2.0", "id": rid, "error": {"code": -1, "message": "nope"}})
