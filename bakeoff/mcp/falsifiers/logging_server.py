#!/usr/bin/env python3
"""Correct MCP server that also records the raw bytes it receives."""
import json, sys, os
LOG = open(os.environ.get("WIRELOG", "/tmp/wire.log"), "w")
TOOLS = [{"name":"echo","description":"Echo","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}}]
def send(m):
    sys.stdout.write(json.dumps(m)+"\n"); sys.stdout.flush()
for raw in sys.stdin:
    LOG.write(repr(raw)+"\n"); LOG.flush()
    line = raw.strip()
    if not line: continue
    try:
        msg = json.loads(line)
    except Exception:
        LOG.write("  ^^ PARSE ERROR (not JSON)\n"); LOG.flush()
        send({"jsonrpc":"2.0","id":None,"error":{"code":-32700,"message":"Parse error"}})
        continue
    if "id" not in msg: continue
    m, rid = msg.get("method"), msg["id"]
    if m == "initialize":
        send({"jsonrpc":"2.0","id":rid,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"ref","version":"1"}}})
    elif m == "tools/list":
        send({"jsonrpc":"2.0","id":rid,"result":{"tools":TOOLS}})
    elif m == "tools/call":
        send({"jsonrpc":"2.0","id":rid,"result":{"content":[{"type":"text","text":"ok"}],"isError":False}})
    elif m == "ping":
        send({"jsonrpc":"2.0","id":rid,"result":{}})
    else:
        send({"jsonrpc":"2.0","id":rid,"error":{"code":-32601,"message":"Method not found"}})
