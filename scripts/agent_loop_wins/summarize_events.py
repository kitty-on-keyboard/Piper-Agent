#!/usr/bin/env python3
"""Summarize agent-loop wins journal fields from an events.jsonl.

Reads tier-A attribution events emitted by PR1 measurement logging:
  kv_reuse, tools_refresh, generation (spec / grammar), tool_result.

Usage:
  python3 scripts/agent_loop_wins/summarize_events.py path/to/events.jsonl

Local multi-MB dumps: keep under piper-bench/agent_loop_wins/ (gitignored).
Enable per-block firehose with LMP_AGENT_LOOP_TRACE=1.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_events(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"skip line {line_no}: {e}", file=sys.stderr)
    return out


def as_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def as_int(v: Any) -> int | None:
    f = as_float(v)
    return int(f) if f is not None else None


def parse_hist(s: Any) -> list[int]:
    if not s or not isinstance(s, str):
        return []
    parts = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return parts


def mean_or_dash(xs: list[float]) -> str:
    if not xs:
        return "-"
    return f"{statistics.mean(xs):.3f}"


def print_counter(title: str, c: Counter[str], limit: int = 20) -> None:
    print(f"\n{title}")
    if not c:
        print("  (none)")
        return
    for k, n in c.most_common(limit):
        print(f"  {k}: {n}")


def summarize(events: list[dict[str, Any]]) -> None:
    ttfts: list[float] = []
    reused: list[float] = []
    prompts: list[float] = []
    reuse_pairs: list[tuple[float, float]] = []
    reset_reasons: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    refresh_changed = 0
    refresh_total = 0
    refresh_triggers: Counter[str] = Counter()
    error_classes: Counter[str] = Counter()
    tool_status: Counter[str] = Counter()
    accept_depth: Counter[int] = Counter()
    draft_lens: Counter[int] = Counter()
    grammar_empty = 0
    grammar_phases: Counter[str] = Counter()
    spec_blocks = 0
    spec_accepted = 0
    spec_drafted = 0

    for ev in events:
        kind = ev.get("kind", "")
        if kind == "generation":
            t = as_float(ev.get("ttft_ms"))
            if t is not None:
                ttfts.append(t)
            r = as_float(ev.get("prefill_reused_tokens"))
            if r is not None:
                reused.append(r)
            for i, n in enumerate(parse_hist(ev.get("accept_at_depth"))):
                if n:
                    accept_depth[i] += n
            for i, n in enumerate(parse_hist(ev.get("draft_len_hist"))):
                if n:
                    draft_lens[i] += n
            ge = as_int(ev.get("grammar_empty_mask"))
            if ge:
                grammar_empty += ge
            gp = ev.get("grammar_phase_end") or ""
            if gp:
                grammar_phases[str(gp)] += 1
            sb = as_int(ev.get("spec_blocks")) or 0
            sa = as_int(ev.get("spec_accepted")) or 0
            sd = as_int(ev.get("spec_drafted")) or 0
            spec_blocks += sb
            spec_accepted += sa
            spec_drafted += sd
        elif kind == "kv_reuse":
            mode = str(ev.get("mode", ""))
            modes[mode] += 1
            r = as_float(ev.get("reused_tokens"))
            p = as_float(ev.get("prompt_tokens"))
            if r is not None and p is not None:
                reuse_pairs.append((r, p))
            if mode == "Reset":
                reset_reasons[str(ev.get("reason") or "unknown")] += 1
        elif kind == "tools_refresh":
            refresh_total += 1
            refresh_triggers[str(ev.get("trigger") or "other")] += 1
            if str(ev.get("changed")) == "1":
                refresh_changed += 1
        elif kind == "tool_result":
            tool_status[str(ev.get("status") or "")] += 1
            ec = str(ev.get("error_class") or "")
            if ec:
                error_classes[ec] += 1

    print("=== agent-loop wins summary ===")
    print(f"events: {len(events)}")
    print(f"mean ttft_ms: {mean_or_dash(ttfts)} (n={len(ttfts)})")
    if reuse_pairs:
        rates = [r / p if p > 0 else 0.0 for r, p in reuse_pairs]
        print(f"mean reuse rate (reused/prompt): {mean_or_dash(rates)} (n={len(rates)})")
    print(f"mean prefill_reused_tokens: {mean_or_dash(reused)} (n={len(reused)})")
    print(f"spec totals: blocks={spec_blocks} drafted={spec_drafted} accepted={spec_accepted}")
    if spec_drafted:
        print(f"  accept ratio: {spec_accepted / spec_drafted:.3f}")
    print(f"grammar_empty_mask sum: {grammar_empty}")

    print_counter("kv_reuse.mode", modes)
    print_counter("kv_reuse.reason (Reset only)", reset_reasons)
    print(
        f"\ntools_refresh: total={refresh_total} changed={refresh_changed} "
        f"changed_rate={'(n/a)' if not refresh_total else f'{refresh_changed / refresh_total:.3f}'}"
    )
    print_counter("tools_refresh.trigger", refresh_triggers)
    print_counter("tool_result.status", tool_status)
    print_counter("tool_result.error_class", error_classes)
    print_counter("grammar_phase_end", grammar_phases)
    print_counter(
        "accept_at_depth (count by depth)",
        Counter({str(k): v for k, v in sorted(accept_depth.items())}),
    )
    print_counter(
        "draft_len_hist (count by draft_len)",
        Counter({str(k): v for k, v in sorted(draft_lens.items())}),
    )

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("events_jsonl", type=Path, help="path to events.jsonl")
    args = ap.parse_args()
    if not args.events_jsonl.is_file():
        print(f"not a file: {args.events_jsonl}", file=sys.stderr)
        return 2
    summarize(load_events(args.events_jsonl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
