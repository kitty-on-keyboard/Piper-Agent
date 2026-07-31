#!/usr/bin/env python3
"""Measure mlx-lm's prefill/decode on this machine, as a live apples-to-apples reference.

LM Studio's server logs are a 1-second-resolution record of a stack we cannot instrument.
This is the same stack -- LM Studio ships its own interpreter with mlx + mlx_lm, and it is
on this disk -- so it can be swept and ablated directly. Run it with that interpreter:

  ~/.lmstudio/extensions/backends/vendor/_amphibian/\
app-mlx-generate-mac14-arm64@29/bin/python scripts/mlxlm_reference.py

Prefill throughput is strongly dependent on prompt length and on prefill_step_size, so a
single number is not a baseline. --prompts sweeps the first; --chunks sweeps the second.

Usage: mlxlm_reference.py [--prompts 512,2048,8192] [--chunks 2048] [--max-new N] [--runs N]
"""

import argparse
import statistics
import sys
import time

MODEL = "/Users/dev/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit"

# The same filler lmp_diag's bench builds, so the two measure the same token count of the
# same kind of text. Real prose, not a repeated token: the tokenizer must not collapse it.
FILLER = (
    "The build system compiles each translation unit separately, then the "
    "linker resolves symbols across them and emits one binary. Line {}.\n"
)


def build_prompt(tokenizer, want):
    s = ""
    n = 0
    i = 0
    while n < want:
        s += FILLER.format(i)
        i += 1
        n = len(tokenizer.encode(s))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="512,2048,8192")
    ap.add_argument("--chunks", default="2048")
    ap.add_argument("--max-new", type=int, default=32, dest="max_new")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate_step
    except ImportError as e:
        print(f"needs the LM Studio interpreter (mlx + mlx_lm): {e}", file=sys.stderr)
        return 1

    print(f"mlx {mx.__version__}, model={args.model.split('/')[-1]}")
    model, tokenizer = load(args.model)

    prompts = [int(x) for x in args.prompts.split(",")]
    chunks = [int(x) for x in args.chunks.split(",")]

    print(f"\n{'prompt':>8} {'chunk':>6} {'prefill tok/s':>14} {'decode tok/s':>13}")
    for want in prompts:
        text = build_prompt(tokenizer, want)
        ids = tokenizer.encode(text)
        for chunk in chunks:
            pre, dec = [], []
            for _ in range(args.runs):
                # Prefill and decode have to be separated by hand: generate_step yields
                # its first token only after the whole prompt is processed, so the time to
                # that first yield is prefill, and everything after it is decode.
                mx.reset_peak_memory()
                t0 = time.perf_counter()
                n = 0
                t_first = None
                for _tok, _lp in generate_step(
                    mx.array(ids), model, max_tokens=args.max_new,
                    prefill_step_size=chunk,
                ):
                    if t_first is None:
                        mx.synchronize()
                        t_first = time.perf_counter()
                    n += 1
                mx.synchronize()
                t_end = time.perf_counter()
                pre.append(len(ids) / (t_first - t0))
                if n > 1:
                    dec.append((n - 1) / (t_end - t_first))
            print(f"{len(ids):>8} {chunk:>6} {statistics.median(pre):>14.1f} "
                  f"{statistics.median(dec):>13.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
