#!/usr/bin/env python3
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from piper_worker import LiveJournal


def lines(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "live.jsonl")
    journal = LiveJournal(path)
    journal.on_notification("lmp/token", {"channel": "thinking", "text": "abc"})
    journal.on_notification("lmp/token", {"channel": "thinking", "text": "def"})
    journal.on_notification("lmp/token", {"channel": "answer", "text": "Z"})
    journal.on_notification("lmp/turn", {
        "tool_name": "read_file",
        "tool_args": json.dumps({"path": "src/a.js"}),
        "tool_status": "ok",
        "summary": "y" * 300,
        "read_bytes": 12,
        "edit_bytes": 0,
        "think_tokens": 10,
        "text_tokens": 4,
        "tool_tokens": 6,
    })
    journal.on_notification("lmp/turn", {
        "tool_name": "write_file",
        "tool_args": json.dumps({"path": "src/b.js", "command": "ignored"}),
        "tool_status": "ok",
        "summary": "wrote",
        "read_bytes": 0,
        "edit_bytes": 40,
        "think_tokens": 1,
        "text_tokens": 1,
        "tool_tokens": 1,
    })
    journal.on_notification("lmp/token", {"channel": "thinking", "text": "q" * 64})
    journal.on_notification("lmp/notice", {"code": "nope"})
    journal.close()

    got = lines(path)
    assert [row["seq"] for row in got] == list(range(1, len(got) + 1))
    assert got[0] == {"seq": 1, "kind": "delta", "channel": "thinking", "text": "abcdef"}
    assert got[1] == {"seq": 2, "kind": "delta", "channel": "answer", "text": "Z"}
    assert got[2]["kind"] == "turn"
    assert got[2]["tool"] == "read_file"
    assert got[2]["path"] == "src/a.js"
    assert got[2]["summary"] == "y" * 240
    assert got[2]["read_bytes"] == 12
    assert got[2]["edit_bytes"] == 0
    assert got[2]["think_tokens"] == 10
    assert got[2]["text_tokens"] == 4
    assert got[2]["tool_tokens"] == 6
    assert got[3]["kind"] == "turn" and got[3]["tool"] == "write_file"
    assert got[3]["path"] == "src/b.js"
    assert got[4] == {"seq": 5, "kind": "write", "tool": "write_file", "path": "src/b.js", "edit_bytes": 40}
    assert got[5] == {"seq": 6, "kind": "delta", "channel": "thinking", "text": "q" * 64}
    assert len(got) == 6
    print("ok")


if __name__ == "__main__":
    main()
