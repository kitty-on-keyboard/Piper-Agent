#!/usr/bin/env python3
"""Generates a miniature Qwen-SHAPED tokenizer.json for the gate (G0).

WHY THIS EXISTS. docs/PHASES.md names four surviving mutants and says three of them
(grammar.cpp:105, agent.cpp:168, agent.cpp:287) are blocked on the same thing: there is no
vocabulary in the gate, so TurnGrammar and Agent::step are only ever exercised by
`realmodel` tests, which are excluded. Everything that needs to assert "the mask forbade
this id" or "the turn stopped on grammar acceptance" needs a vocab, and loading the real
one costs 19 GB and a GPU.

WHY IT IS GENERATED RATHER THAN COMMITTED. QwenTokenizer::load refuses any vocabulary
outside the Qwen3 size band [140000, 260000] -- family verification, S5.2, and the reason
v1 did not silently accept a Gemma tokenizer. Meeting that band honestly means ~140k
entries, which is a few megabytes of JSON. Generating it at build time keeps the property
(the fixture goes through the SAME verification production does) without committing a blob
nobody will ever read.

WHAT MAKES IT USABLE RATHER THAN JUST BIG:
  * the 256 byte-level tokens under GPT-2's bytes_to_unicode mapping, so arbitrary bytes
    encode;
  * merges for the ASCII that source code and tool calls are actually made of;
  * every structural token TurnGrammar and ChatTemplate name, as added_tokens so BPE can
    never reach them -- the property S5.4 depends on;
  * a multi-byte UTF-8 character split across two tokens, because "streaming decode is
    byte-identical to batch" is a claim about exactly that case and 944 of the real
    checkpoint's tokens are byte fragments;
  * ids that DELIBERATELY do not match the real checkpoint's, so any code that assumes a
    literal id fails here instead of in production.

THE FOREIGN SHAPES. `--shape` also emits the two vocabularies family verification must
REFUSE. Until 2026-08-08 that test loaded a real 31 MB Gemma tokenizer off this machine;
the checkpoint is gone (Sean never ran it, and this is a Qwen-only product) and keeping a
model on disk as a test fixture was the wrong dependency anyway -- it made the guard a
`realmodel` test, so it never ran in CI.

  * `foreign` reproduces that Gemma tokenizer's measured shape: 262,144 base entries plus
    the 24 added tokens it actually shipped, so the upper bound is exercised at 262,168 --
    the real number that motivated the ceiling. Its specials are Gemma's real ones, which
    matter because they are NEAR-MISSES: `<|tool_call>` and `<|think|>` against Qwen's
    `<tool_call>` and `<think>`. A name-sniffing loader accepts those; that is v1's bug.
  * `unmarked` sits INSIDE the band and simply has no structural tokens, which isolates
    the second probe. The Gemma test never reached it -- the size check short-circuited --
    so that half of load() went unasserted for as long as it existed.
"""

import json
import os
import sys

# GPT-2 / Qwen byte-level alphabet: the printable ranges map to themselves, everything
# else is displaced into a private area so every byte has a printable representative.
def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


SPECIALS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    # Vision framing. Present on both real checkpoints (they are VL models), and needed
    # here so ChatTemplate's image framing -- where the pad run lands, which is the offset
    # the embedding splice addresses -- is pinned in the GATE rather than only against a
    # 19 GB checkpoint. They are deliberately NOT part of SpecialIds::complete(): a
    # text-only checkpoint must still load.
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
]

# Gemma-4's 24 added tokens, read off the checkpoint before it was deleted. Kept verbatim
# rather than invented: the point of this list is that `<|tool_call>` is not
# `<tool_call>` and `<|think|>` is not `<think>`, and a plausible-looking substitute
# would lose exactly the near-miss the refusal exists to catch.
FOREIGN_SPECIALS = [
    "<pad>", "<eos>", "<bos>", "<unk>", "<mask>",
    "<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>",
    "<|tool_response>", "<tool_response|>", '<|"|>',
    "<|think|>", "<|channel>", "<channel|>", "<|turn>", "<turn|>",
    "<|image>", "<|audio>", "<|image|>", "<|audio|>",
    "<image|>", "<audio|>", "<|video|>",
]

# The band QwenTokenizer::load enforces. Sitting just above the floor keeps the file as
# small as honesty allows.
VOCAB_FLOOR = 140_000

# Gemma-4's measured base vocabulary. + 24 added = 262,168, which is what the ceiling of
# 260,000 was chosen to exclude.
FOREIGN_VOCAB_FLOOR = 262_144

SHAPES = ("qwen", "foreign", "unmarked")


def build(shape="qwen"):
    b2u = bytes_to_unicode()
    vocab = {}
    nxt = 0

    def add(tok):
        nonlocal nxt
        if tok not in vocab:
            vocab[tok] = nxt
            nxt += 1

    for b in range(256):
        add(b2u[b])

    # Merges over the characters real prompts are made of: identifiers, digits, the
    # punctuation of XML tool calls and of source code, and the space marker.
    alphabet = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "_<>/=\"'(){}[],.:;-+*!?#\n"
    ) + b2u[ord(" ")]
    merges = []
    for a in alphabet:
        for c in alphabet:
            merged = a + c
            add(merged)
            merges.append(f"{a} {c}")

    # Filler, so the vocabulary reaches the size band the family check requires. Named so
    # that a filler token showing up in a decode is obviously a bug and not a near-miss.
    floor = FOREIGN_VOCAB_FLOOR if shape == "foreign" else VOCAB_FLOOR
    i = 0
    while nxt < floor:
        add(f"ġunused{i}")
        i += 1

    # `unmarked` ships none, which is the whole point of it: in-band size, no structural
    # tokens, so load() has to fail on the second probe or not at all.
    specials = {"qwen": SPECIALS, "foreign": FOREIGN_SPECIALS, "unmarked": []}[shape]
    added = [{"content": s, "id": nxt + k} for k, s in enumerate(specials)]

    return {
        "version": "1.0",
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {
                        "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    },
                    "behavior": "Isolated",
                    "invert": False,
                },
                {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False},
            ],
        },
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
    }


def main():
    args = sys.argv[1:]
    shape = "qwen"
    if len(args) == 2:
        shape, args = args[0], args[1:]
    if len(args) != 1 or shape not in SHAPES:
        print(f"usage: make_mini_vocab.py [{'|'.join(SHAPES)}] <out/tokenizer.json>",
              file=sys.stderr)
        return 2
    out = args[0]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    doc = build(shape)
    with open(out, "w") as f:
        json.dump(doc, f)
    n = len(doc["model"]["vocab"]) + len(doc["added_tokens"])
    print(f"make_mini_vocab[{shape}]: {n} entries -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
