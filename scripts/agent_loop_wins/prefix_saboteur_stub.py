# Prefix-saboteur stub (PR1 optional tease-out).
#
# Forces the scenarios Phase A will A/B against and records expected kv_reuse.reason
# labels. Full live runs need a Mac agent + real model; this documents the matrix and
# exercises summarize_events.py against a synthetic journal.

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Expected Reset reasons for each saboteur scenario (measurement labels only).
SCENARIOS = [
    {"name": "noop_plan_lock", "expect_reason": "tools_guidance_changed", "expect_changed": "0"},
    {"name": "real_allowlist_change", "expect_reason": "tools_guidance_changed", "expect_changed": "1"},
    {"name": "compact_no_shadow", "expect_reason": "compact_no_shadow", "expect_changed": "0"},
    {"name": "compact_with_shadow", "expect_reason": "checkpoint_restore", "expect_changed": "0"},
    {"name": "one_byte_system_poke", "expect_reason": "ledger_mismatch", "expect_changed": "0"},
]


def synthetic_journal() -> list[dict]:
    """Minimal events.jsonl that summarize_events.py can histogram."""
    return [
        {
            "kind": "tools_refresh",
            "trigger": "plan_lock",
            "guidance_hash_before": "aaa",
            "guidance_hash_after": "aaa",
            "changed": "0",
            "spec_count_before": "3",
            "spec_count_after": "3",
            "noop": "0",
        },
        {
            "kind": "kv_reuse",
            "mode": "Reset",
            "reused_tokens": "0",
            "prompt_tokens": "1000",
            "reason": "tools_guidance_changed",
            "stable_prefix_tokens": "800",
            "shadow_armed": "0",
        },
        {
            "kind": "generation",
            "ttft_ms": "12.5",
            "prefill_reused_tokens": "0",
            "spec_blocks": "0",
            "spec_drafted": "0",
            "spec_accepted": "0",
            "accept_at_depth": "0,0,0",
            "draft_len_hist": "0,0,0",
            "grammar_empty_mask": "0",
            "grammar_phase_end": "done",
        },
        {
            "kind": "tool_result",
            "tool": "read_file",
            "status": "ToolError",
            "error_class": "exec",
            "error_code": "1",
            "summary": "missing",
        },
    ]


def main() -> int:
    script = Path(__file__).resolve().parents[2] / "scripts" / "agent_loop_wins" / "summarize_events.py"
    if not script.is_file():
        print(f"missing summarize script: {script}", file=sys.stderr)
        return 2

    print("prefix-saboteur expected reason matrix:")
    for s in SCENARIOS:
        print(f"  {s['name']}: reason={s['expect_reason']} tools_changed={s['expect_changed']}")

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "events.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for ev in synthetic_journal():
                f.write(json.dumps(ev) + "\n")
        proc = subprocess.run(
            [sys.executable, str(script), str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            return proc.returncode
        if "tools_guidance_changed" not in proc.stdout:
            print("summarize_events.py missing Reset reason histogram", file=sys.stderr)
            return 1
        if "error_class" not in proc.stdout and "exec" not in proc.stdout:
            print("summarize_events.py missing ToolError class histogram", file=sys.stderr)
            return 1
    print("prefix-saboteur stub: synthetic journal OK")
    print("NOTE: live Mac agent confirmation still required for §2.7 checklist items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
