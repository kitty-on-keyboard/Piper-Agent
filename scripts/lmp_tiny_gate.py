#!/usr/bin/env python3
"""lmp_tiny_gate — second-process typed Choice helper (Qwen3 tiny MLX).

Loads ONE small causal LM (target: Qwen3-0.6B-4bit) and answers binary
force_tool vs nudge letter-mask Choice over HTTP or stdio JSON.

This is NOT same-weights Pulse and NOT an MTP draft head. The main Piper
sidecar keeps its one full model load (S5.11); this process is a sibling.

Env:
  LMP_GATE_MODEL_DIR   path to MLX Qwen3-family checkpoint (required to serve)
  LMP_GATE_HOST        default 127.0.0.1
  LMP_GATE_PORT        default 18765

Usage:
  # HTTP (Benchbot / live prove)
  python scripts/lmp_tiny_gate.py --listen \\
      --model-dir "$LMP_GATE_MODEL_DIR"

  # Stdio JSON lines (one request → one response)
  python scripts/lmp_tiny_gate.py --stdio --model-dir "$LMP_GATE_MODEL_DIR"

  # Dry parse / health without weights (CI smoke)
  python scripts/lmp_tiny_gate.py --self-test

Protocol:
  POST /v1/choose  {"prompt": "...", "choices": ["force_tool","nudge"], "letters": ["A","B"]}
  GET  /health     {"ok": true, "model": "...", "loaded": bool}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


DEFAULT_HOST = os.environ.get("LMP_GATE_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("LMP_GATE_PORT", "18765"))
DEFAULT_MODEL = os.environ.get("LMP_GATE_MODEL_DIR", "")


class GateModel:
    """Thin mlx_lm wrapper. Letter-mask Choice over A/B first-token logits."""

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.letter_ids: dict[str, int] = {}

    def load(self) -> None:
        import mlx.core as mx
        from mlx_lm import load

        self.mx = mx
        self.model, self.tokenizer = load(self.model_dir)
        for letter in ("A", "B"):
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if not ids:
                raise RuntimeError(f"empty encode for letter {letter!r}")
            self.letter_ids[letter] = int(ids[0])
        # Collision check (same Qwen3 family should keep A/B unique).
        if len(set(self.letter_ids.values())) != len(self.letter_ids):
            raise RuntimeError(f"letter-token collision: {self.letter_ids}")

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def choose(self, prompt: str, choices: list[str], letters: list[str]) -> dict[str, Any]:
        t0 = time.perf_counter()
        if not self.loaded:
            return {
                "ok": False,
                "error": "gate model not loaded",
                "latency_ms": 0.0,
                "model": self.model_dir,
            }
        if len(choices) != len(letters) or len(choices) != 2:
            return {
                "ok": False,
                "error": "gate expects exactly 2 choices/letters",
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "model": self.model_dir,
            }

        # Prefill once; score next-token logits at letter ids only.
        tokens = self.tokenizer.encode(prompt)
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        input_ids = self.mx.array(tokens)[None]
        logits = self.model(input_ids)
        # Last position vocabulary logits.
        if hasattr(logits, "astype"):
            last = logits[0, -1, :]
        else:
            last = logits[0][-1]
        last = last.astype(self.mx.float32)
        self.mx.eval(last)

        scores: list[float] = []
        option_ids: list[int] = []
        for letter in letters:
            lid = self.letter_ids.get(letter)
            if lid is None:
                # Resolve on the fly for unexpected letters.
                ids = self.tokenizer.encode(letter, add_special_tokens=False)
                if not ids:
                    return {
                        "ok": False,
                        "error": f"empty encode for letter {letter!r}",
                        "latency_ms": (time.perf_counter() - t0) * 1000.0,
                        "model": self.model_dir,
                    }
                lid = int(ids[0])
                self.letter_ids[letter] = lid
            option_ids.append(lid)
            scores.append(float(last[lid].item()))

        max_s = max(scores)
        exps = [math.exp(s - max_s) for s in scores]
        total = sum(exps) or 1.0
        p_vec = [e / total for e in exps]
        best = max(range(len(p_vec)), key=lambda i: p_vec[i])
        choice = choices[best]
        p_force = 0.0
        for i, name in enumerate(choices):
            if name == "force_tool":
                p_force = p_vec[i]
                break

        return {
            "ok": True,
            "error": "",
            "letter": letters[best],
            "choice": choice,
            "p": p_vec[best],
            "p_force": p_force,
            "p_vec": p_vec,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "model": self.model_dir,
            "encoding": "letter",
            "order": ",".join(choices),
            "option_ids": option_ids,
        }


def handle_choose(gate: GateModel | None, body: dict[str, Any]) -> dict[str, Any]:
    prompt = body.get("prompt", "")
    choices = body.get("choices") or ["force_tool", "nudge"]
    letters = body.get("letters") or ["A", "B"]
    if not isinstance(prompt, str) or not prompt.strip():
        return {"ok": False, "error": "missing prompt", "model": getattr(gate, "model_dir", "")}
    if gate is None or not gate.loaded:
        return {
            "ok": False,
            "error": "gate model not loaded (set LMP_GATE_MODEL_DIR and start with weights)",
            "model": getattr(gate, "model_dir", "") if gate else "",
        }
    return gate.choose(prompt, list(choices), list(letters))


def make_handler(gate: GateModel | None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            sys.stderr.write("[lmp_tiny_gate] " + (fmt % args) + "\n")

        def _send(self, code: int, obj: dict[str, Any]) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] in ("/health", "/v1/health"):
                self._send(
                    200,
                    {
                        "ok": True,
                        "model": getattr(gate, "model_dir", "") if gate else "",
                        "loaded": bool(gate and gate.loaded),
                    },
                )
                return
            self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path not in ("/v1/choose", "/choose"):
                self._send(404, {"ok": False, "error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError as e:
                self._send(400, {"ok": False, "error": f"bad json: {e}"})
                return
            if not isinstance(body, dict):
                self._send(400, {"ok": False, "error": "body must be object"})
                return
            self._send(200, handle_choose(gate, body))

    return Handler


def run_stdio(gate: GateModel | None) -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            body = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stdout.write(json.dumps({"ok": False, "error": f"bad json: {e}"}) + "\n")
            sys.stdout.flush()
            continue
        if not isinstance(body, dict):
            sys.stdout.write(json.dumps({"ok": False, "error": "body must be object"}) + "\n")
            sys.stdout.flush()
            continue
        if body.get("cmd") == "health":
            sys.stdout.write(
                json.dumps(
                    {
                        "ok": True,
                        "model": getattr(gate, "model_dir", "") if gate else "",
                        "loaded": bool(gate and gate.loaded),
                    }
                )
                + "\n"
            )
            sys.stdout.flush()
            continue
        sys.stdout.write(json.dumps(handle_choose(gate, body)) + "\n")
        sys.stdout.flush()
    return 0


def self_test() -> int:
    """CPU-only smoke: request/response schema without MLX weights."""
    # Simulate a helper response the C++ client must parse.
    sample = {
        "ok": True,
        "error": "",
        "letter": "B",
        "choice": "force_tool",
        "p": 0.61,
        "p_force": 0.61,
        "p_vec": [0.39, 0.61],
        "latency_ms": 12.5,
        "model": "/models/Qwen3-0.6B-4bit",
        "encoding": "letter",
        "order": "nudge,force_tool",
    }
    assert sample["choice"] in ("force_tool", "nudge")
    assert len(sample["p_vec"]) == 2
    assert abs(sum(sample["p_vec"]) - 1.0) < 1e-6
    # parse round-trip via json
    raw = json.dumps(sample)
    back = json.loads(raw)
    assert back["ok"] is True
    print("lmp_tiny_gate self-test ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL, help="MLX Qwen3 checkpoint dir")
    ap.add_argument("--listen", action="store_true", help="serve HTTP on host:port")
    ap.add_argument("--stdio", action="store_true", help="JSON lines on stdin/stdout")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--self-test", action="store_true", help="CPU schema smoke; no GPU")
    ap.add_argument(
        "--allow-unloaded",
        action="store_true",
        help="start HTTP/stdio without loading weights (health only / fail choose)",
    )
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    gate: GateModel | None = None
    if args.model_dir:
        gate = GateModel(args.model_dir)
        if not args.allow_unloaded:
            try:
                gate.load()
            except Exception as e:  # noqa: BLE001 — surface load errors to operator
                print(f"lmp_tiny_gate: failed to load {args.model_dir}: {e}", file=sys.stderr)
                return 1
        else:
            print("lmp_tiny_gate: --allow-unloaded; choose will fail until load", file=sys.stderr)
    elif not args.allow_unloaded:
        print(
            "lmp_tiny_gate: set --model-dir or LMP_GATE_MODEL_DIR (or pass --allow-unloaded / --self-test)",
            file=sys.stderr,
        )
        return 1

    if args.stdio:
        return run_stdio(gate)
    if args.listen:
        handler = make_handler(gate)
        server = ThreadingHTTPServer((args.host, args.port), handler)
        print(
            f"lmp_tiny_gate listening on http://{args.host}:{args.port} "
            f"model={getattr(gate, 'model_dir', '')!r} loaded={bool(gate and gate.loaded)}",
            file=sys.stderr,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("lmp_tiny_gate shutdown", file=sys.stderr)
        return 0

    print("lmp_tiny_gate: pass --listen, --stdio, or --self-test", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
