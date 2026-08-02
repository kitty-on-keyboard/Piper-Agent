#!/usr/bin/env python3
"""Correct MCP server that logs verbosely to stderr, as real servers do."""
import json, sys
TOOLS=[{"name":"echo","description":"E","inputSchema":{"type":"object","properties":{}}}]
def send(m):
    sys.stdout.write(json.dumps(m)+"\n"); sys.stdout.flush()
sys.stderr.write("startup diagnostics\n"*8000)   # ~150 KB, exceeds the 64 KB pipe buffer
sys.stderr.flush()
for raw in sys.stdin:
    line=raw.strip()
    if not line: continue
    try: msg=json.loads(line)
    except Exception: continue
    if "id" not in msg: continue
    m,rid=msg.get("method"),msg["id"]
    if m=="initialize": send({"jsonrpc":"2.0","id":rid,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"chatty","version":"1"}}})
    elif m=="tools/list": send({"jsonrpc":"2.0","id":rid,"result":{"tools":TOOLS}})
    elif m=="tools/call": send({"jsonrpc":"2.0","id":rid,"result":{"content":[{"type":"text","text":"ok"}]}})
    else: send({"jsonrpc":"2.0","id":rid,"error":{"code":-32601,"message":"nf"}})
