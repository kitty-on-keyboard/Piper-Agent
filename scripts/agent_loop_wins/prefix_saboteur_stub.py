# Prefix-saboteur stub (PR1 optional tease-out + Phase A1 matrix).
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

# Expected labels post-A1. noop_plan_lock must NOT rewrite / must not attribute a tools Reset.
SCENARIOS = [
    {
        "name": "noop_plan_lock",
        "expect_reason": "extend_or_non_tools_reset",
        "expect_changed": "0",
        "expect_noop": "1",
    },
    {
        "name": "real_allowlist_change",
        "expect_reason": "tools_guidance_changed",
        "expect_changed": "1",
        "expect_noop": "0",
    },
    {
        "name": "compact_no_shadow",
        "expect_reason": "compact_no_shadow",
        "expect_changed": "0",
        "expect_noop": "0",
    },
    {
        "name": "compact_with_shadow",
        "expect_reason": "checkpoint_restore",
        "expect_changed": "0",
        "expect_noop": "0",
    },
    {
        "name": "one_byte_system_poke",
        "expect_reason": "ledger_mismatch",
        "expect_changed": "0",
        "expect_noop": "0",
    },
]


def synthetic_journal() -> list[dict]:
    """Minimal events.jsonl that summarize_events.py can histogram (post-A1 shape)."""
    return [
        {
            "kind": "tools_refresh",
            "trigger": "plan_lock",
            "guidance_hash_before": "aaa",
            "guidance_hash_after": "aaa",
            "changed": "0",
            "spec_count_before": "3",
            "spec_count_after": "3",
            "noop": "1",
        },
        {
            "kind": "kv_reuse",
            "mode": "Extend",
            "reused_tokens": "900",
            "prompt_tokens": "1000",
            "reason": "prefix_match",
            "stable_prefix_tokens": "800",
            "shadow_armed": "0",
        },
        {
            "kind": "tools_refresh",
            "trigger": "plan_lock",
            "guidance_hash_before": "aaa",
            "guidance_hash_after": "bbb",
            "changed": "1",
            "spec_count_before": "3",
            "spec_count_after": "2",
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
            "prefill_reused_tokens": "900",
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def a1_baseline_journal() -> list[dict]:
    """Kill-switch off: identical plan-lock refresh still rewrites (noop=0)."""
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
            "ttft_ms": "120.0",
            "decode_tok_per_s": "40.0",
            "prefill_reused_tokens": "0",
            "spec_abandoned": "0",
            "grammar_empty_mask": "0",
            "grammar_forced_tokens": "0",
        },
        {"kind": "tool_result", "tool": "read_file", "status": "Ok", "error_class": ""},
    ]


def a1_treatment_journal() -> list[dict]:
    """A1 on: identical refresh no-ops; prefix extends."""
    return [
        {
            "kind": "tools_refresh",
            "trigger": "plan_lock",
            "guidance_hash_before": "aaa",
            "guidance_hash_after": "aaa",
            "changed": "0",
            "spec_count_before": "3",
            "spec_count_after": "3",
            "noop": "1",
        },
        {
            "kind": "kv_reuse",
            "mode": "Extend",
            "reused_tokens": "900",
            "prompt_tokens": "1000",
            "reason": "prefix_match",
            "stable_prefix_tokens": "800",
            "shadow_armed": "0",
        },
        {
            "kind": "generation",
            "ttft_ms": "80.0",
            "decode_tok_per_s": "40.0",
            "prefill_reused_tokens": "900",
            "spec_abandoned": "0",
            "grammar_empty_mask": "0",
            "grammar_forced_tokens": "0",
        },
        {"kind": "tool_result", "tool": "read_file", "status": "Ok", "error_class": ""},
    ]


def main() -> int:
    script = Path(__file__).resolve().parents[2] / "scripts" / "agent_loop_wins" / "summarize_events.py"
    if not script.is_file():
        print(f"missing summarize script: {script}", file=sys.stderr)
        return 2

    print("prefix-saboteur expected reason matrix (post-A1):")
    for s in SCENARIOS:
        print(
            f"  {s['name']}: reason={s['expect_reason']} "
            f"tools_changed={s['expect_changed']} noop={s['expect_noop']}"
        )

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        journal = td_path / "events.jsonl"
        write_jsonl(journal, synthetic_journal())
        proc = subprocess.run(
            [sys.executable, str(script), str(journal)],
            check=False,
            capture_output=True,
            text=True,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            return proc.returncode

        base = td_path / "a1_baseline.jsonl"
        treat = td_path / "a1_treatment.jsonl"
        write_jsonl(base, a1_baseline_journal())
        write_jsonl(treat, a1_treatment_journal())
        ab = subprocess.run(
            [sys.executable, str(script), str(base), str(treat)],
            check=False,
            capture_output=True,
            text=True,
        )
        print(ab.stdout)
        if ab.returncode != 0:
            print(ab.stderr, file=sys.stderr)
            return ab.returncode
        if "tools_refresh.noop: 0 -> 1" not in ab.stdout:
            print("A/B compare did not show A1 noop 0 -> 1", file=sys.stderr)
            return 1
        if "tools_guidance_changed" not in ab.stdout:
            print("A/B compare missing reset_reasons", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
