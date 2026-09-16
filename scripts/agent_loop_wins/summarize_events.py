#!/usr/bin/env python3
"""Summarize agent-loop wins journal fields from an events.jsonl.

Reads tier-A attribution events emitted by PR1 measurement logging:
  kv_reuse, tools_refresh, generation (spec / grammar), tool_result.

Usage:
  python3 scripts/agent_loop_wins/summarize_events.py path/to/events.jsonl
  python3 scripts/agent_loop_wins/summarize_events.py baseline.jsonl treatment.jsonl

A/B pairing (one flag, same model/seed/corpus):
  A1: unset vs LMP_A1_NOOP_TOOLS_REFRESH=0

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


def collect(events: list[dict[str, Any]]) -> dict[str, Any]:
    ttfts: list[float] = []
    decode_tps: list[float] = []
    reused: list[float] = []
    reuse_pairs: list[tuple[float, float]] = []
    reset_reasons: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    refresh_changed = 0
    refresh_noop = 0
    refresh_total = 0
    refresh_triggers: Counter[str] = Counter()
    error_classes: Counter[str] = Counter()
    tool_status: Counter[str] = Counter()
    accept_depth: Counter[int] = Counter()
    draft_lens: Counter[int] = Counter()
    grammar_empty = 0
    grammar_forced = 0
    grammar_phases: Counter[str] = Counter()
    spec_blocks = 0
    spec_accepted = 0
    spec_drafted = 0
    spec_abandoned = 0
    nudged_why: Counter[str] = Counter()
    degenerate_events = 0
    run_end_metrics: dict[str, Any] | None = None

    for ev in events:
        kind = ev.get("kind", "")
        if kind == "generation":
            t = as_float(ev.get("ttft_ms"))
            if t is not None:
                ttfts.append(t)
            d = as_float(ev.get("decode_tok_per_s"))
            if d is not None:
                decode_tps.append(d)
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
            gf = as_int(ev.get("grammar_forced_tokens"))
            if gf:
                grammar_forced += gf
            gp = ev.get("grammar_phase_end") or ""
            if gp:
                grammar_phases[str(gp)] += 1
            spec_blocks += as_int(ev.get("spec_blocks")) or 0
            spec_accepted += as_int(ev.get("spec_accepted")) or 0
            spec_drafted += as_int(ev.get("spec_drafted")) or 0
            spec_abandoned += as_int(ev.get("spec_abandoned")) or 0
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
            if str(ev.get("noop")) == "1":
                refresh_noop += 1
        elif kind == "tool_result":
            tool_status[str(ev.get("status") or "")] += 1
            ec = str(ev.get("error_class") or "")
            if ec:
                error_classes[ec] += 1
        elif kind == "degenerate_text":
            degenerate_events += 1
        elif kind == "nudged":
            nudged_why[str(ev.get("why") or "")] += 1
        elif kind == "run_end":
            run_end_metrics = {
                "degenerate_text_count": as_int(ev.get("degenerate_text_count")),
                "text_only_turns": as_int(ev.get("text_only_turns")),
                "tool_error_count": as_int(ev.get("tool_error_count")),
                "nudged_count": as_int(ev.get("nudged_count")),
                "nudged_by_why.loop_cut": as_int(ev.get("nudged_by_why.loop_cut")),
                "nudged_by_why.no_progress": as_int(ev.get("nudged_by_why.no_progress")),
                "nudged_by_why.no_tool_recovery": as_int(
                    ev.get("nudged_by_why.no_tool_recovery")
                ),
                "termination_reason": ev.get("termination_reason"),
            }

    reuse_rates = [r / p if p > 0 else 0.0 for r, p in reuse_pairs]
    return {
        "n_events": len(events),
        "ttfts": ttfts,
        "decode_tps": decode_tps,
        "reused": reused,
        "reuse_rates": reuse_rates,
        "reset_reasons": reset_reasons,
        "modes": modes,
        "refresh_total": refresh_total,
        "refresh_changed": refresh_changed,
        "refresh_noop": refresh_noop,
        "refresh_triggers": refresh_triggers,
        "error_classes": error_classes,
        "tool_status": tool_status,
        "accept_depth": accept_depth,
        "draft_lens": draft_lens,
        "grammar_empty": grammar_empty,
        "grammar_forced": grammar_forced,
        "grammar_phases": grammar_phases,
        "spec_blocks": spec_blocks,
        "spec_accepted": spec_accepted,
        "spec_drafted": spec_drafted,
        "spec_abandoned": spec_abandoned,
        "nudged_why": nudged_why,
        "degenerate_events": degenerate_events,
        "run_end_metrics": run_end_metrics,
    }


def mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def pct_delta(base: float | None, treat: float | None) -> str:
    if base is None or treat is None or base == 0:
        return "-"
    return f"{100.0 * (treat - base) / base:+.2f}%"


def summarize(events: list[dict[str, Any]], title: str = "agent-loop wins summary") -> dict[str, Any]:
    m = collect(events)
    print(f"=== {title} ===")
    print(f"events: {m['n_events']}")
    print(f"mean ttft_ms: {mean_or_dash(m['ttfts'])} (n={len(m['ttfts'])})")
    print(f"mean decode_tok_per_s: {mean_or_dash(m['decode_tps'])} (n={len(m['decode_tps'])})")
    if m["reuse_rates"]:
        print(
            f"mean reuse rate (reused/prompt): {mean_or_dash(m['reuse_rates'])} "
            f"(n={len(m['reuse_rates'])})"
        )
    print(f"mean prefill_reused_tokens: {mean_or_dash(m['reused'])} (n={len(m['reused'])})")
    print(
        f"spec totals: blocks={m['spec_blocks']} drafted={m['spec_drafted']} "
        f"accepted={m['spec_accepted']} abandoned={m['spec_abandoned']}"
    )
    if m["spec_drafted"]:
        print(f"  accept ratio: {m['spec_accepted'] / m['spec_drafted']:.3f}")
    print(f"grammar_empty_mask sum: {m['grammar_empty']}")
    print(f"grammar_forced_tokens sum: {m['grammar_forced']}")

    print_counter("kv_reuse.mode", m["modes"])
    print_counter("kv_reuse.reason (Reset only)", m["reset_reasons"])
    rt = m["refresh_total"]
    changed_rate = "(n/a)" if not rt else f"{m['refresh_changed'] / rt:.3f}"
    noop_rate = "(n/a)" if not rt else f"{m['refresh_noop'] / rt:.3f}"
    print(
        f"\ntools_refresh: total={rt} changed={m['refresh_changed']} "
        f"noop={m['refresh_noop']} "
        f"changed_rate={changed_rate} "
        f"noop_rate={noop_rate}"
    )
    print_counter("tools_refresh.trigger", m["refresh_triggers"])
    print_counter("tool_result.status", m["tool_status"])
    print_counter("tool_result.error_class", m["error_classes"])
    print_counter("grammar_phase_end", m["grammar_phases"])
    print_counter(
        "accept_at_depth (count by depth)",
        Counter({str(k): v for k, v in sorted(m["accept_depth"].items())}),
    )
    print_counter(
        "draft_len_hist (count by draft_len)",
        Counter({str(k): v for k, v in sorted(m["draft_lens"].items())}),
    )
    print(f"\ndegenerate_text events: {m['degenerate_events']}")
    print_counter("nudged.why", m["nudged_why"])
    if m["run_end_metrics"]:
        print("\nrun_end loop metrics:")
        for k, v in m["run_end_metrics"].items():
            print(f"  {k}: {v}")
    return m


def compare(baseline: dict[str, Any], treatment: dict[str, Any]) -> None:
    print("\n=== A/B delta (treatment vs baseline) ===")
    print(f"ttft_ms: {pct_delta(mean(baseline['ttfts']), mean(treatment['ttfts']))}")
    print(
        f"decode_tok_per_s: {pct_delta(mean(baseline['decode_tps']), mean(treatment['decode_tps']))}"
    )
    print(
        f"prefill_reused_tokens: {pct_delta(mean(baseline['reused']), mean(treatment['reused']))}"
    )
    print(
        f"reuse_rate: {pct_delta(mean(baseline['reuse_rates']), mean(treatment['reuse_rates']))}"
    )
    print(
        f"grammar_empty_mask: {baseline['grammar_empty']} -> {treatment['grammar_empty']}"
    )
    print(
        f"grammar_forced_tokens: {baseline['grammar_forced']} -> {treatment['grammar_forced']}"
    )
    print(
        f"spec_abandoned: {baseline['spec_abandoned']} -> {treatment['spec_abandoned']}"
    )
    print(f"tools_refresh.noop: {baseline['refresh_noop']} -> {treatment['refresh_noop']}")
    print(
        f"tools_refresh.changed: {baseline['refresh_changed']} -> {treatment['refresh_changed']}"
    )
    print(
        f"degenerate_text events: {baseline['degenerate_events']} -> "
        f"{treatment['degenerate_events']}"
    )
    print("reset_reasons baseline: " + (str(dict(baseline["reset_reasons"])) or "{}"))
    print("reset_reasons treatment: " + (str(dict(treatment["reset_reasons"])) or "{}"))
    print("error_class baseline: " + (str(dict(baseline["error_classes"])) or "{}"))
    print("error_class treatment: " + (str(dict(treatment["error_classes"])) or "{}"))
    print("tool_status baseline: " + (str(dict(baseline["tool_status"])) or "{}"))
    print("tool_status treatment: " + (str(dict(treatment["tool_status"])) or "{}"))
    print("nudged.why baseline: " + (str(dict(baseline["nudged_why"])) or "{}"))
    print("nudged.why treatment: " + (str(dict(treatment["nudged_why"])) or "{}"))
    print(
        "Worth: primary metric must beat the plan's noise floor. "
        "Stability: TTFT/ToolError/empty-mask/abandoned/reset-reasons within noise."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline_jsonl", type=Path, help="baseline events.jsonl")
    ap.add_argument(
        "treatment_jsonl",
        nargs="?",
        type=Path,
        help="optional treatment events.jsonl for an A/B delta",
    )
    args = ap.parse_args()
    if not args.baseline_jsonl.is_file():
        print(f"not a file: {args.baseline_jsonl}", file=sys.stderr)
        return 2
    base = summarize(load_events(args.baseline_jsonl), "baseline")
    if args.treatment_jsonl is None:
        return 0
    if not args.treatment_jsonl.is_file():
        print(f"not a file: {args.treatment_jsonl}", file=sys.stderr)
        return 2
    treat = summarize(load_events(args.treatment_jsonl), "treatment")
    compare(base, treat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
