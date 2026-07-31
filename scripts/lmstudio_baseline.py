#!/usr/bin/env python3
"""Derive LM Studio's decode/prefill throughput on this machine from its own server logs.

The logs are the only record of LM Studio running THIS checkpoint on THIS hardware, so
they are the baseline LM_Pipe has to beat. Timestamps are 1-second resolution, which is
why this reports a median over many requests rather than any single number: a 2-second
window on a 40-token completion carries ~50% error, but the median over n=100+ does not.

Per request the log gives:
    Prompt cache restore: cached_tokens=C uncached_tokens=U
    Prompt processing progress: 0.0%     <- prefill starts
    Prompt processing progress: 100.0%   <- prefill ends, decode starts
    Generated prediction: { ... "completion_tokens": N ... }   <- decode ends

    prefill tok/s = U / (t100 - t0)
    decode  tok/s = N / (t_pred - t100)

Requests whose measured window is 0 s are dropped, not counted as infinity.

Usage: scripts/lmstudio_baseline.py [--logs DIR] [--model SUBSTR]
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys
from datetime import datetime

TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]")
CACHE = re.compile(r"Prompt cache restore: cached_tokens=(\d+) uncached_tokens=(\d+)")
PROG = re.compile(r"Prompt processing progress: ([\d.]+)%")
PRED = re.compile(r"Generated prediction:")
COMPL = re.compile(r'"completion_tokens":\s*(\d+)')


def ts(line):
    m = TS.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")


def scan(path, model):
    """Walk one log file as a state machine, yielding (prefill, decode) samples."""
    prefills, decodes = [], []
    uncached = None
    t0 = t100 = None
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if model and model not in line and TS.match(line):
                # Model-tagged lines that belong to another model reset nothing;
                # untagged lines (JSON bodies) still need to flow through.
                if "][INFO][" in line or "][WARN][" in line:
                    continue
            t = ts(line)
            m = CACHE.search(line)
            if m:
                uncached = int(m.group(2))
                t0 = t100 = None
                continue
            m = PROG.search(line)
            if m and t is not None:
                pct = float(m.group(1))
                if pct == 0.0:
                    t0 = t
                elif pct == 100.0:
                    t100 = t
                    if t0 is not None and uncached:
                        dt = (t100 - t0).total_seconds()
                        if dt > 0:
                            prefills.append((uncached / dt, dt))
                        # One cache-restore line == one prefill. Consuming the count
                        # here stops a later progress pair from being scored against a
                        # stale token count.
                        uncached = None
                        t0 = None
                continue
            if PRED.search(line) and t is not None:
                if t100 is not None:
                    scan.pending = (t, t100)
                else:
                    scan.pending = None
                continue
            m = COMPL.search(line)
            if m and getattr(scan, "pending", None):
                t_pred, t_end_prefill = scan.pending
                scan.pending = None
                n = int(m.group(1))
                dt = (t_pred - t_end_prefill).total_seconds()
                if dt > 0 and n > 0:
                    decodes.append((n / dt, dt))
    return prefills, decodes


def report(name, samples, min_dt=0.0):
    # A 1-second clock over a 1-second window is +-50% error. Filtering to longer
    # windows trades sample count for per-sample accuracy; both views are printed
    # because neither alone is honest about this data.
    xs = sorted(r for r, dt in samples if dt >= min_dt)
    if not xs:
        print(f"  {name}: no samples")
        return None
    q = statistics.quantiles(xs, n=4) if len(xs) >= 4 else [xs[0], statistics.median(xs), xs[-1]]
    print(
        f"  {name}: n={len(xs):<5} median={statistics.median(xs):8.1f} tok/s "
        f"p25={q[0]:8.1f} p75={q[2]:8.1f} min={xs[0]:7.1f} max={xs[-1]:7.1f}"
    )
    return statistics.median(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=os.path.expanduser("~/.lmstudio/server-logs"))
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--min-dt", type=float, default=5.0, dest="min_dt")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.logs, "*", "*.log")))
    if not files:
        print(f"no logs under {args.logs}", file=sys.stderr)
        return 1

    prefills, decodes = [], []
    for f in files:
        p, d = scan(f, args.model)
        prefills += p
        decodes += d

    print(f"LM Studio baseline  ({len(files)} log files, model~={args.model})")
    print("all windows:")
    report("prefill", prefills)
    report("decode ", decodes)
    print(f"windows >= {args.min_dt:.0f}s (lower clock error, fewer samples):")
    report("prefill", prefills, args.min_dt)
    report("decode ", decodes, args.min_dt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
