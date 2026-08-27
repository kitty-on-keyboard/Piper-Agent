#!/usr/bin/env python3
"""Histogram the MLX primitives in ONE unevaluated decode step.

Why this exists: four passes of this project were misled by timing instruments
(see the warning at the top of tests/model/diag_moe.cpp). A primitive histogram
is not a timing -- it is what the graph literally contains -- so it cannot be
wrong about the thing it measures, only incomplete. Two stacks that claim to run
"the same ops on the same shapes" must produce the same histogram; wherever they
do not, the difference is real work one of them is doing and the other is not.

Two modes:

  --dot FILE      histogram a dot file already written by mx::export_to_dot
                  (this is how the LM_Pipe side gets here: `lmp_diag graph`)

  --reference     load mlx-lm on this machine, prefill a cache, build one decode
                  step WITHOUT evaluating it, and histogram that. Needs the
                  LM Studio interpreter:
                    ~/.lmstudio/extensions/backends/vendor/_amphibian/\
app-mlx-generate-mac14-arm64@29/bin/python

  --compare A B   diff two histograms (json files written by --json)

The prompt length matters: the graph for a decode step is the same shape at any
context length, but prefill has to run first so the cache is populated and the
step is a real decode step rather than a first-token forward.
"""

import argparse
import os
import collections
import json
import re
import sys

MODEL = os.environ.get("LMP_QWEN_DIR", "")

LABEL = re.compile(r'\[label\s*="([^"]+)"')


def histogram_dot(text):
    counts = collections.Counter(LABEL.findall(text))
    return dict(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dot")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--json")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompt-tokens", type=int, default=512, dest="prompt_tokens")
    ap.add_argument("--top", type=int, default=0)
    args = ap.parse_args()

    if args.compare:
        a = json.load(open(args.compare[0]))
        b = json.load(open(args.compare[1]))
        name_a = args.compare[0].split("/")[-1]
        name_b = args.compare[1].split("/")[-1]
        keys = sorted(set(a) | set(b))
        print(f"{'primitive':<28}{name_a:>14}{name_b:>14}{'diff':>10}")
        ta = tb = 0
        for k in keys:
            va, vb = a.get(k, 0), b.get(k, 0)
            ta += va
            tb += vb
            mark = "" if va == vb else "   <--"
            print(f"{k:<28}{va:>14}{vb:>14}{va - vb:>10}{mark}")
        print(f"{'TOTAL':<28}{ta:>14}{tb:>14}{ta - tb:>10}")
        return 0

    if args.dot:
        hist = histogram_dot(open(args.dot).read())
    elif args.reference:
        hist = build_reference(args.model, args.prompt_tokens)
    else:
        ap.error("one of --dot, --reference, --compare")

    total = sum(hist.values())
    items = sorted(hist.items(), key=lambda kv: -kv[1])
    if args.top:
        items = items[: args.top]
    for k, v in items:
        print(f"{v:>8}  {k}")
    print(f"{total:>8}  TOTAL")
    if args.json:
        json.dump(hist, open(args.json, "w"), indent=1, sort_keys=True)
    return 0


def build_reference(model_path, prompt_tokens):
    import io

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = load(model_path)

    filler = (
        "The build system compiles each translation unit separately, then the "
        "linker resolves symbols across them and emits one binary. Line {}.\n"
    )
    s, n, i = "", 0, 0
    while n < prompt_tokens:
        s += filler.format(i)
        i += 1
        n = len(tokenizer.encode(s))
    ids = tokenizer.encode(s)
    print(f"prompt {len(ids)} tokens", file=sys.stderr)

    cache = make_prompt_cache(model)
    logits = model(mx.array(ids)[None], cache=cache)
    mx.eval(logits)
    mx.eval([c.state for c in cache])

    # One decode step, built but NOT evaluated. This is the graph the reference's
    # decode loop submits per token.
    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)
    y = model(tok[None], cache=cache)

    f = io.StringIO()
    mx.export_to_dot(f, y)
    return histogram_dot(f.getvalue())


if __name__ == "__main__":
    sys.exit(main())
