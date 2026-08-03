#!/usr/bin/env python3
"""Drive the sidecar over the real lmp/* protocol, without the editor.

Sends exactly what extension/src/client.ts sends -- including the nested `settings`
object -- so anything that breaks here breaks in the editor too. Approval cards are
printed in full and auto-approved, which stands in for a human clicking Approve.

    python3 scripts/drive.py --workspace /path/to/ws --mission "fix the failing test"

Steering, which is the thing worth exercising here -- it is the one part of the protocol
with no test that can prove it against a real model:

    # say something mid-run, once the 3rd turn lands
    --say 3:"actually, use the other approach"

    # and/or continue the conversation after the run ends, as many times as you like
    --then "now do the same for the other module"

Loads the model, so it is subject to the one-MLX-process-at-a-time rule in
docs/HANDOFF_AGENT.md: run it alone, in the foreground, to completion.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

DEFAULT_MODEL = "/Users/dev/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit"
DEFAULT_SIDECAR = "build/src/surface/lmp_sidecar"

# Qwen3's recommended thinking-mode operating point (S5.9). These are also the pinned
# defaults in src/model/backend.hpp and extension/package.json; sending them explicitly is
# what proves the settings actually reach the sampler.
QWEN_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                 "repetition_penalty": 1.05, "seed": 0}

ap = argparse.ArgumentParser()
ap.add_argument("--workspace", required=True)
ap.add_argument("--mission", required=True)
ap.add_argument("--model", default=os.environ.get("LMP_QWEN_DIR", DEFAULT_MODEL))
ap.add_argument("--sidecar", default=DEFAULT_SIDECAR)
ap.add_argument("--mode", default="agent", choices=["plan", "debug", "agent"])
ap.add_argument("--deadline", type=float, default=1800.0)
ap.add_argument("--max-iterations", type=int, default=80)
ap.add_argument("--wall-clock", type=int, default=1800)
ap.add_argument("--sandbox-tier", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--auto", action="store_true",
                help="send the autonomy flags the eval harness sends (risk-routed "
                     "approvals) instead of require_approval, so a hand-run and a "
                     "scored run can be compared like for like")
ap.add_argument("--deny", action="store_true",
                help="deny every approval card instead of approving it")
ap.add_argument("--say", action="append", default=[], metavar="TURN:TEXT",
                help="send TEXT as an lmp/message once TURN turns have gone by "
                     "(repeatable). Exercises mid-run steering.")
ap.add_argument("--then", action="append", default=[], metavar="TEXT",
                help="on run_end, continue the conversation with TEXT instead of "
                     "shutting down (repeatable, in order).")
args = ap.parse_args()

# {turn_number: [text, ...]}
say_at = {}
for spec in args.say:
    turn, _, text = spec.partition(":")
    if not text:
        ap.error(f"--say wants TURN:TEXT, got {spec!r}")
    say_at.setdefault(int(turn), []).append(text)
follow_ups = list(args.then)

proc = subprocess.Popen(
    [os.path.abspath(args.sidecar)], cwd=args.workspace,
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, bufsize=1,
)

start = time.monotonic()
lock = threading.Lock()
next_id = [1]


def T():
    return f"[{time.monotonic() - start:7.1f}s]"


def send(obj):
    with lock:
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()


def new_id():
    next_id[0] += 1
    return str(next_id[0])


def pump_stderr():
    for line in proc.stderr:
        print(f"{T()} STDERR {line.rstrip()}", flush=True)


threading.Thread(target=pump_stderr, daemon=True).start()

answer, thinking = [], []
approvals = 0
turns = 0
run_id = "1"


def say(text):
    """An lmp/message. Steering if a run is turning, a follow-up if not -- the sidecar
    decides which, and says so in `started_run`."""
    print(f"{T()} >>> SAY {text!r}", flush=True)
    send({"jsonrpc": "2.0", "id": new_id(), "method": "lmp/message",
          "params": {"run_id": run_id, "text": text}})


for raw in proc.stdout:
    raw = raw.strip()
    if not raw:
        continue
    if time.monotonic() - start > args.deadline:
        print(f"{T()} DEADLINE; cancelling", flush=True)
        send({"jsonrpc": "2.0", "id": new_id(), "method": "lmp/cancel",
              "params": {"run_id": run_id}})
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        print(f"{T()} UNPARSEABLE {raw[:300]}", flush=True)
        continue

    method, params = msg.get("method"), msg.get("params") or {}

    if method == "lmp/ready":
        print(f"{T()} ready, protocol {params.get('protocol_version')}", flush=True)
        send({"jsonrpc": "2.0", "id": "1", "method": "lmp/start", "params": {
            "mission": args.mission,
            "settings": {
                "model_dir": args.model, "workspace_root": os.path.abspath(args.workspace),
                "mode": args.mode, "sampling": dict(QWEN_SAMPLING, seed=args.seed),
                "max_iterations": args.max_iterations,
                "wall_clock_seconds": args.wall_clock,
                "sandbox_tier": args.sandbox_tier,
                "require_approval": not args.auto,
                "auto_approve_exec": bool(args.auto),
                "auto_approve_writes": bool(args.auto),
                "system_prompt": "",
                "context_budget_tokens": 96000,
            },
        }})
    elif method == "lmp/token":
        (thinking if params.get("channel") == "thinking" else answer).append(
            params.get("text", ""))
    elif method == "lmp/turn":
        turns += 1
        run_id = params.get("run_id", run_id)
        print(f"{T()} TURN #{turns} outcome={params.get('outcome')} "
              f"tool={params.get('tool_name')!r} status={params.get('tool_status')}",
              flush=True)
        print(f"          args: {str(params.get('tool_args'))[:200]}", flush=True)
        print(f"          summary: {str(params.get('summary'))[:400]}", flush=True)
        for text in say_at.pop(turns, []):
            say(text)
    elif method == "lmp/checklist":
        items = json.loads(params.get("items_json") or "[]")
        done = sum(1 for i in items if i.get("done"))
        print(f"{T()} CHECKLIST {done}/{len(items)}", flush=True)
        for item in items:
            print(f"          [{'x' if item.get('done') else ' '}] {item.get('text')}",
                  flush=True)
    elif method == "lmp/approval_request":
        approvals += 1
        caps = [k for k, v in (params.get("capabilities") or {}).items() if v is True]
        approved = not args.deny
        print(f"{T()} *** APPROVAL #{approvals} tool={params.get('tool_name')} "
              f"risk={params.get('risk')} -> {'APPROVE' if approved else 'DENY'}", flush=True)
        print(f"          preview: {params.get('preview')}", flush=True)
        print(f"          caps: {caps or '(none)'} "
              f"parse={(params.get('capabilities') or {}).get('parse_status')}", flush=True)
        send({"jsonrpc": "2.0", "id": new_id(), "method": "lmp/approve",
              "params": {"request_id": params.get("request_id"), "approved": approved}})
    elif method == "lmp/verification":
        print(f"{T()} VERIFY {'PASS' if params.get('passed') else 'FAIL'} "
              f"falsifiable={params.get('falsifiable')} {params.get('contract')!r}", flush=True)
    elif method == "lmp/perf":
        s = params.get("sample") or {}
        print(f"{T()} perf ttft={s.get('ttft_ms', 0):.0f}ms "
              f"decode={s.get('decode_tok_per_s', 0):.1f} tok/s "
              f"ctx={s.get('context_used')}", flush=True)
    elif method == "lmp/run_end":
        print(f"{T()} RUN_END reason={params.get('termination_reason')!r} "
              f"iterations={params.get('iterations')} completed={params.get('completed')} "
              f"unfinished_items={params.get('unfinished_items')}", flush=True)
        if follow_ups:
            # A follow-up, not a new mission: same context, same loaded weights.
            answer.clear()
            thinking.clear()
            say(follow_ups.pop(0))
        else:
            send({"jsonrpc": "2.0", "id": new_id(), "method": "lmp/shutdown", "params": {}})
    elif method is not None:
        print(f"{T()} <-- {method}", flush=True)
    elif "error" in msg:
        # Replies were dropped on the floor before, so a refused request looked exactly
        # like a request that worked.
        print(f"{T()} ERROR {msg['error'].get('message')}", flush=True)
    elif "result" in msg:
        print(f"{T()} result {json.dumps(msg['result'])}", flush=True)

proc.wait(timeout=30)
print(f"\n{T()} sidecar exited {proc.returncode}; approval cards: {approvals}", flush=True)
if thinking:
    print(f"\n--- THINKING ---\n{''.join(thinking)[:2000]}", flush=True)
if answer:
    print(f"\n--- ANSWER ---\n{''.join(answer)[:3000]}", flush=True)
sys.exit(0)
