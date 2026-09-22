#!/usr/bin/env python3
"""One-shot proof that paste seams are gone for answer + wake + packet emit.

Exit 0 only when:
  1. `piper answer` writes the known-right answer.json shape
  2. `.piper/orch_webhook` is discovered without CLI/env paste
  3. `piper packet` emits a packet `load_packet` accepts
  4. piper_ui wake-file helper matches the worker path convention
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import piper_ui  # noqa: E402
import piper_worker as w  # noqa: E402


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def main():
    with tempfile.TemporaryDirectory(prefix="piper-det-harness-") as tmp:
        # 1. answer
        if w.main(["answer", "allow", "--dir", tmp]) != w.EXIT_OK:
            return fail("piper answer allow")
        with open(os.path.join(tmp, "answer.json"), encoding="utf-8") as fh:
            body = json.load(fh)
        if body != {"text": "allow"}:
            return fail(f"answer payload {body!r}")

        # 2. wake file discovery (no panel copy)
        url = "http://127.0.0.1:8765/wake"
        path = w.write_orch_webhook_file(tmp, url)
        if not os.path.isfile(path):
            return fail("wake file missing")
        os.environ.pop("LMP_ORCH_WEBHOOK", None)
        got = w.resolve_orch_webhook(
            None,
            {"orch_webhook": "", "cwd": tmp, "task_dir": tmp,
             "result_path": os.path.join(tmp, "result.json")},
        )
        if got != url:
            return fail(f"resolve_orch_webhook got {got!r}")

        # 3. packet emit + load
        model = os.path.join(tmp, "model")
        os.makedirs(model)
        out = os.path.join(tmp, "task.json")
        rc = w.main([
            "packet", "--id", "prove-1", "--cwd", tmp, "--prompt", "Prove seam.",
            "--out", out, "--model-dir", model, "--check", "true",
        ])
        if rc != w.EXIT_OK:
            return fail(f"piper packet rc={rc}")
        packet = w.load_packet(out)
        if packet["id"] != "prove-1" or packet["cwd"] != os.path.abspath(tmp):
            return fail(f"loaded packet mismatch: {packet!r}")

        # 4. UI helper writes the same relative path the worker reads
        ui_path = piper_ui.write_wake_url_file(tmp, url)
        if os.path.normpath(ui_path) != os.path.normpath(path):
            return fail(f"ui path {ui_path!r} != worker path {path!r}")

    print("prove_deterministic_harness: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
