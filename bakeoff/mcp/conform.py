#!/usr/bin/env python3
"""MCP server conformance harness.

Feeds newline-delimited JSON-RPC at a candidate server on stdin and grades the
stdout stream against the MCP spec. Each check is independent so one failure
does not mask the rest.
"""
import json, subprocess, sys, time

TIMEOUT = 6.0


def run(binary, lines):
    """Send `lines` (list of raw strings) and collect stdout lines."""
    p = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    payload = "".join(l + "\n" for l in lines)
    try:
        out, _ = p.communicate(payload, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()
        return [l for l in out.splitlines() if l.strip()], True
    return [l for l in out.splitlines() if l.strip()], False


def objs(lines):
    got = []
    for l in lines:
        try:
            got.append(json.loads(l))
        except Exception:
            got.append({"__unparseable__": l})
    return got


INIT = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                   "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                              "clientInfo": {"name": "harness", "version": "1"}}})
INITED = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})


def check(name, binary, send, verdict):
    lines, timed_out = run(binary, send)
    got = objs(lines)
    try:
        ok, note = verdict(got, timed_out)
    except Exception as ex:
        ok, note = False, f"harness exc: {ex}"
    return name, ok, note


def first_result(got, rid):
    for m in got:
        if m.get("id") == rid:
            return m
    return None


CHECKS = []


def define(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@define("initialize returns protocolVersion+serverInfo")
def c1(binary):
    def v(got, to):
        r = first_result(got, 1)
        if not r or "result" not in r:
            return False, "no result for id=1"
        res = r["result"]
        missing = [k for k in ("protocolVersion", "capabilities", "serverInfo") if k not in res]
        if missing:
            return False, "missing " + ",".join(missing)
        return True, res.get("protocolVersion")
    return check("init", binary, [INIT], v)


@define("notification gets NO response")
def c2(binary):
    def v(got, to):
        # after init, send the initialized notification; only the init reply may appear
        extra = [m for m in got if m.get("id") != 1]
        if extra:
            return False, f"replied to notification: {json.dumps(extra[0])[:70]}"
        return True, "silent"
    return check("notif", binary, [INIT, INITED], v)


@define("unknown notification gets NO response")
def c3(binary):
    def v(got, to):
        extra = [m for m in got if m.get("id") != 1]
        if extra:
            return False, f"replied: {json.dumps(extra[0])[:70]}"
        return True, "silent"
    return check("unknown-notif", binary, [INIT, INITED,
                 json.dumps({"jsonrpc": "2.0", "method": "notifications/cancelled",
                             "params": {"requestId": 99}})], v)


@define("string id echoed back as same string")
def c4(binary):
    def v(got, to):
        for m in got:
            if m.get("id") == "abc-1":
                return True, "preserved"
        ids = [repr(m.get("id")) for m in got]
        return False, f"ids seen: {ids}"
    return check("strid", binary, [INIT, INITED,
                 json.dumps({"jsonrpc": "2.0", "id": "abc-1", "method": "tools/list"})], v)


@define("tools/list returns tools array")
def c5(binary):
    def v(got, to):
        r = first_result(got, 2)
        if not r or "result" not in r:
            return False, "no result"
        t = r["result"].get("tools")
        if not isinstance(t, list):
            return False, "tools not a list"
        return True, f"{len(t)} tools"
    return check("tools/list", binary, [INIT, INITED,
                 json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})], v)


@define("unknown method -> -32601")
def c6(binary):
    def v(got, to):
        r = first_result(got, 3)
        if not r:
            return False, "no response at all"
        if "error" not in r:
            return False, "returned a result, not an error"
        code = r["error"].get("code")
        return (code == -32601), f"code={code}"
    return check("-32601", binary, [INIT, INITED,
                 json.dumps({"jsonrpc": "2.0", "id": 3, "method": "no/such/method"})], v)


@define("malformed JSON -> -32700 and survives")
def c7(binary):
    def v(got, to):
        parse_err = [m for m in got if m.get("error", {}).get("code") == -32700]
        survived = first_result(got, 4) is not None
        if not parse_err:
            return False, "no -32700 emitted"
        if not survived:
            return False, "-32700 ok but server died after"
        return True, "recovers"
    return check("-32700", binary, [INIT, INITED, "{ this is not json",
                 json.dumps({"jsonrpc": "2.0", "id": 4, "method": "tools/list"})], v)


@define("tools/call unknown tool -> error or isError")
def c8(binary):
    def v(got, to):
        r = first_result(got, 5)
        if not r:
            return False, "no response"
        if "error" in r:
            return True, f"error {r['error'].get('code')}"
        if r.get("result", {}).get("isError") is True:
            return True, "isError:true"
        return False, "silently succeeded on unknown tool"
    return check("badtool", binary, [INIT, INITED,
                 json.dumps({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                             "params": {"name": "definitely_not_a_tool", "arguments": {}}})], v)


@define("tools/call result has content[]")
def c9(binary, toolname=None):
    def v(got, to):
        r = first_result(got, 6)
        if not r or "result" not in r:
            return False, f"no result ({json.dumps(got[-1])[:60] if got else 'nothing'})"
        c = r["result"].get("content")
        if not isinstance(c, list):
            return False, "no content array"
        return True, f"content[{len(c)}]"
    # discover a real tool name first
    lines, _ = run(binary, [INIT, INITED, json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})])
    name, args = None, {}
    for m in objs(lines):
        if m.get("id") == 2 and "result" in m:
            tl = m["result"].get("tools") or []
            if tl:
                name = tl[0].get("name")
                props = (tl[0].get("inputSchema") or {}).get("properties") or {}
                for k, spec in props.items():
                    t = spec.get("type")
                    args[k] = 1 if t in ("number", "integer") else ("x" if t == "string" else True)
    if not name:
        return ("tools/call", False, "could not discover a tool")
    return check("tools/call", binary, [INIT, INITED,
                 json.dumps({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                             "params": {"name": name, "arguments": args}})], v)


@define("ping -> empty result")
def c10(binary):
    def v(got, to):
        r = first_result(got, 7)
        if not r:
            return False, "no response"
        return ("result" in r), ("ok" if "result" in r else "error returned")
    return check("ping", binary, [INIT, INITED,
                 json.dumps({"jsonrpc": "2.0", "id": 7, "method": "ping"})], v)


@define("request before initialize is rejected")
def c11(binary):
    def v(got, to):
        r = first_result(got, 8)
        if r and "error" in r:
            return True, f"rejected {r['error'].get('code')}"
        if r and "result" in r:
            return False, "served tools/list before initialize"
        return False, "no response"
    return check("pre-init", binary,
                 [json.dumps({"jsonrpc": "2.0", "id": 8, "method": "tools/list"})], v)


@define("stdout carries only JSON-RPC")
def c12(binary):
    def v(got, to):
        bad = [m for m in got if "__unparseable__" in m]
        if bad:
            return False, f"non-JSON on stdout: {bad[0]['__unparseable__'][:60]!r}"
        return True, "clean"
    return check("stdout-clean", binary, [INIT, INITED,
                 json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})], v)


def main():
    bins = sys.argv[1:]
    names = [n for n, _ in CHECKS]
    rows = {}
    for b in bins:
        label = b.split("/")[-1]
        rows[label] = []
        for n, fn in CHECKS:
            _, ok, note = fn(b)
            rows[label].append((n, ok, note))

    w = max(len(n) for n in names) + 2
    hdr = " " * w + "".join(f"{l:<6}" for l in rows)
    print(hdr)
    for i, n in enumerate(names):
        line = f"{n:<{w}}"
        for l in rows:
            line += f"{'PASS' if rows[l][i][1] else 'FAIL':<6}"
        print(line)
    print()
    for l in rows:
        score = sum(1 for _, ok, _ in rows[l] if ok)
        print(f"--- {l}: {score}/{len(names)}")
        for n, ok, note in rows[l]:
            if not ok:
                print(f"      FAIL {n}: {note}")
            else:
                print(f"      pass {n}: {note}")


if __name__ == "__main__":
    main()
