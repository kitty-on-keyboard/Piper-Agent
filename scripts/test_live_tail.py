#!/usr/bin/env python3
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from piper_ui import read_appended


def main():
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "live.jsonl")
    rows, offset = read_appended(path, 0)
    assert rows == [] and offset == 0

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"seq": 1, "kind": "delta", "text": "ab"}) + "\n")
        fh.write('{"seq": 2')  # partial line
    rows, offset = read_appended(path, 0)
    assert len(rows) == 1 and rows[0]["seq"] == 1
    assert offset > 0

    with open(path, "a", encoding="utf-8") as fh:
        fh.write(', "kind": "turn"}\n')
    rows, offset = read_appended(path, offset)
    assert len(rows) == 1 and rows[0]["kind"] == "turn" and rows[0]["seq"] == 2

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"seq": 1, "kind": "delta"}) + "\n")
    rows, offset = read_appended(path, offset)
    assert len(rows) == 1 and rows[0]["kind"] == "delta"
    assert offset == os.path.getsize(path)
    print("ok")


if __name__ == "__main__":
    main()
