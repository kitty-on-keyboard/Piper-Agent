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
]

# The band QwenTokenizer::load enforces. Sitting just above the floor keeps the file as
# small as honesty allows.
VOCAB_FLOOR = 140_000


def build():
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
    i = 0
    while nxt < VOCAB_FLOOR:
        add(f"ġunused{i}")
        i += 1

    added = [{"content": s, "id": nxt + k} for k, s in enumerate(SPECIALS)]

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
    if len(sys.argv) != 2:
        print("usage: make_mini_vocab.py <out/tokenizer.json>", file=sys.stderr)
        return 2
    out = sys.argv[1]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    doc = build()
    with open(out, "w") as f:
        json.dump(doc, f)
    n = len(doc["model"]["vocab"]) + len(doc["added_tokens"])
    print(f"make_mini_vocab: {n} entries -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
