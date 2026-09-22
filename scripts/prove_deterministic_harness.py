#!/usr/bin/env python3
"""One-shot proof that paste seams are gone for answer + wake + packet + review.

Exit 0 only when:
  1. `piper answer` writes the known-right answer.json shape
  2. `.piper/orch_webhook` is discovered without CLI/env paste
  3. `piper packet` emits a packet `load_packet` accepts
  4. piper_ui wake-file helper matches the worker path convention
  5. `piper review` prints a deterministic PASS card from result.json
  6. UI answer path shares write_answer_file (approved → allow)
  7. `piper status` reports ask/done without freehand cat/jq
  8. `piper await` returns when ask appears (no sleep-loop paste)
  9. `piper mcp-list` + `packet --trust-mcp` write known-right trust_mcp
 10. `piper progress` appends the known-right progress line
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time

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

        # 5. review card from result.json (no freehand rubric)
        result_path = os.path.join(tmp, "result.json")
        with open(result_path, "w", encoding="utf-8") as fh:
            json.dump(w.result_shell(
                task_id="prove-1",
                cwd=tmp,
                model_dir=model,
                status="ok",
                message="Harness proof.",
                files_touched=["scripts/prove_deterministic_harness.py"],
                diff_stat={"insertions": 1, "deletions": 0, "files": 1},
            ), fh)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rev_rc = w.main(["review", "--result", result_path])
        if rev_rc != w.EXIT_OK:
            return fail(f"piper review rc={rev_rc}")
        card = buf.getvalue()
        if "verdict:  PASS" not in card or "prove-1" not in card:
            return fail(f"review card missing PASS/prove-1: {card!r}")

        # 6. shared answer writer + UI synonym (approved → allow)
        if w.build_answer_payload(action="approved") != {"text": "allow"}:
            return fail("approved must normalize to allow")
        if w.build_answer_payload(action="denied") != {"text": "deny"}:
            return fail("denied must normalize to deny")
        shared = w.write_answer_file(tmp, w.build_answer_payload(action="approved"))
        with open(shared, encoding="utf-8") as fh:
            if json.load(fh) != {"text": "allow"}:
                return fail("shared write_answer_file shape mismatch")

        # 7. status card (no freehand cat/jq)
        st_dir = os.path.join(tmp, "status_ws")
        os.makedirs(st_dir)
        with open(os.path.join(st_dir, "awaiting_user.json"), "w", encoding="utf-8") as fh:
            json.dump({"question": "Allow overwrite?", "options": "allow,deny"}, fh)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            st_rc = w.main(["status", "--dir", st_dir])
        if st_rc != w.EXIT_OK:
            return fail(f"piper status ask rc={st_rc}")
        if "state:    ask" not in buf.getvalue() or "Allow overwrite?" not in buf.getvalue():
            return fail(f"status ask card: {buf.getvalue()!r}")

        # 8. await returns when ask appears
        await_dir = os.path.join(tmp, "await_ws")
        os.makedirs(await_dir)

        def _later():
            time.sleep(0.1)
            with open(os.path.join(await_dir, "awaiting_user.json"), "w", encoding="utf-8") as fh:
                json.dump({"question": "Go?", "options": "allow,deny"}, fh)

        threading.Thread(target=_later, daemon=True).start()
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            aw_rc = w.main(["await", "--dir", await_dir, "--timeout-s", "2",
                            "--interval-s", "0.05"])
        if aw_rc != w.EXIT_OK:
            return fail(f"piper await rc={aw_rc}")
        if "state:    ask" not in buf2.getvalue():
            return fail(f"await card: {buf2.getvalue()!r}")

        # 9. mcp-list + packet --trust-mcp
        mcp_ws = os.path.join(tmp, "mcp_ws")
        os.makedirs(mcp_ws)
        with open(os.path.join(mcp_ws, ".mcp.json"), "w", encoding="utf-8") as fh:
            json.dump({"mcpServers": {"godoer": {"command": "godoer"}}}, fh)
        buf3 = io.StringIO()
        with contextlib.redirect_stdout(buf3):
            if w.main(["mcp-list", "--cwd", mcp_ws]) != w.EXIT_OK:
                return fail("piper mcp-list")
        if buf3.getvalue().strip() != "godoer":
            return fail(f"mcp-list output {buf3.getvalue()!r}")
        trust_out = os.path.join(tmp, "trust_task.json")
        if w.main([
            "packet", "--id", "trust-1", "--cwd", mcp_ws, "--prompt", "Use Godoer.",
            "--out", trust_out, "--model-dir", model, "--trust-mcp", "godoer",
        ]) != w.EXIT_OK:
            return fail("piper packet --trust-mcp")
        trusted = w.load_packet(trust_out)
        if trusted.get("trust_mcp") != ["godoer"]:
            return fail(f"trust_mcp {trusted.get('trust_mcp')!r}")

        # 10. progress log
        prog_root = os.path.join(tmp, "prog_ws")
        os.makedirs(prog_root)
        buf4 = io.StringIO()
        with contextlib.redirect_stdout(buf4):
            if w.main(["progress", "--id", "slice-001", "pass",
                       "--note", "prove", "--dir", prog_root]) != w.EXIT_OK:
                return fail("piper progress")
        prog_path = os.path.join(prog_root, ".piper", "progress.log")
        with open(prog_path, encoding="utf-8") as fh:
            if fh.read() != "slice-001 | pass | prove\n":
                return fail("progress.log shape mismatch")

    print("prove_deterministic_harness: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
