#!/usr/bin/env python3
"""Headless Piper worker: one mission, one result.json, no IDE.

  piper worker run --task /path/to/task.json
  piper run --task /path/to/task.json
  piper --help
  python3 scripts/piper_worker.py self-test

Install (Homebrew PATH link): ./scripts/install_piper_link.sh
  or: cmake --build --preset dev --target install-piper-link
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import agent_eval as ae  # noqa: E402

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_TIMEOUT = 2
EXIT_INVALID = 3
MESSAGE_CAP = 2000
MODES = ("plan", "debug", "agent")


class PacketError(Exception):
    """Invalid task packet or missing model — maps to exit 3."""


def resolved_sidecar():
    override = os.environ.get("LMP_SIDECAR", "").strip()
    return override or ae.SIDECAR


def bind_sidecar():
    ae.SIDECAR = resolved_sidecar()
    return ae.SIDECAR


def as_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("0", "false", "no", "off"):
        return False
    if text in ("1", "true", "yes", "on"):
        return True
    return default


def resolve_task_file(task_arg):
    path = os.path.abspath(os.path.expanduser(task_arg))
    if os.path.isdir(path):
        candidate = os.path.join(path, "task.json")
        if not os.path.isfile(candidate):
            raise PacketError(f"no task.json in directory {path}")
        return candidate
    if os.path.isfile(path):
        return path
    raise PacketError(f"task packet not found: {path}")


def load_packet(task_arg):
    """Parse task.json (+ optional prompt.md). Raises PacketError."""
    task_path = resolve_task_file(task_arg)
    task_dir = os.path.dirname(task_path)
    try:
        with open(task_path, encoding="utf-8") as fh:
            raw = fh.read()
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PacketError(f"task.json is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise PacketError(f"cannot read task packet: {exc}") from exc
    if not isinstance(data, dict):
        raise PacketError("task.json must be a JSON object")

    task_id = data.get("id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise PacketError("task.json missing non-empty string `id`")

    cwd = data.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        raise PacketError("task.json missing `cwd`")
    cwd = os.path.expanduser(cwd)
    if not os.path.isabs(cwd):
        raise PacketError(f"cwd must be an absolute path, got {cwd!r}")
    if not os.path.isdir(cwd):
        raise PacketError(f"cwd is not a directory: {cwd}")

    prompt = data.get("prompt")
    if prompt is None:
        prompt_path = os.path.join(task_dir, "prompt.md")
        if os.path.isfile(prompt_path):
            with open(prompt_path, encoding="utf-8") as fh:
                prompt = fh.read()
    if not isinstance(prompt, str) or not prompt.strip():
        raise PacketError("task.json missing `prompt` (or sibling prompt.md)")

    model_dir = data.get("model_dir") or os.environ.get("LMP_QWEN_DIR", "")
    if not isinstance(model_dir, str) or not model_dir.strip():
        raise PacketError("missing model_dir (set task.json `model_dir` or LMP_QWEN_DIR)")
    model_dir = os.path.abspath(os.path.expanduser(model_dir.strip()))
    if not os.path.isdir(model_dir):
        raise PacketError(f"model_dir is not a directory: {model_dir}")

    mode = data.get("mode", "agent")
    if mode not in MODES:
        raise PacketError(f"mode must be one of {MODES}, got {mode!r}")

    timeout_s = data.get("timeout_s", 900)
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise PacketError("timeout_s must be a number")
    timeout_s = float(timeout_s)
    if timeout_s <= 0:
        raise PacketError("timeout_s must be positive")

    result_path = data.get("result_path")
    if result_path:
        if not isinstance(result_path, str):
            raise PacketError("result_path must be a string")
        result_path = os.path.expanduser(result_path)
        if not os.path.isabs(result_path):
            result_path = os.path.abspath(os.path.join(task_dir, result_path))
    else:
        result_path = os.path.join(task_dir, "result.json")

    orch_webhook = data.get("orch_webhook")
    if orch_webhook is not None and not isinstance(orch_webhook, str):
        orch_webhook = None

    trust_mcp = data.get("trust_mcp") or []
    if not isinstance(trust_mcp, list):
        raise PacketError("trust_mcp must be an array of server names")
    trust_mcp = [str(x) for x in trust_mcp if isinstance(x, str) and x.strip()]

    return {
        "id": task_id.strip(),
        "cwd": os.path.abspath(cwd),
        "prompt": prompt,
        "model_dir": model_dir,
        "mode": mode,
        "auto_approve_exec": as_bool(data.get("auto_approve_exec"), True),
        "auto_approve_writes": as_bool(data.get("auto_approve_writes"), True),
        "auto_approve_irreversible": as_bool(data.get("auto_approve_irreversible"), False),
        "timeout_s": timeout_s,
        "result_path": result_path,
        "orch_webhook": (orch_webhook or "").strip(),
        "trust_mcp": trust_mcp,
        "commit_think": as_bool(data.get("commit_think"), True),
        "shadow_compact": as_bool(data.get("shadow_compact"), True),
        "task_path": task_path,
        "task_dir": task_dir,
        "raw_bytes": raw.encode("utf-8"),
    }


def apply_feature_flags(packet):
    os.environ["LMP_COMMIT_THINK"] = "1" if packet["commit_think"] else "0"
    os.environ["LMP_SHADOW_COMPACT"] = "1" if packet["shadow_compact"] else "0"


def collect_files_touched(log_path, cwd):
    ordered = []
    seen = set()
    if not log_path or not os.path.isfile(log_path):
        return ordered
    try:
        with open(log_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = ev.get("kind")
                candidates = []
                if kind == "write":
                    if ev.get("changed") == "0" or ev.get("changed") == 0:
                        continue
                    path = ev.get("path") or ev.get("normalised") or ""
                    if path:
                        candidates.append(path)
                elif kind == "tool_call":
                    for key, value in ev.items():
                        if not isinstance(value, str):
                            continue
                        bare = key[4:] if key.startswith("arg.") else key
                        if bare not in {
                            "path", "file", "filepath", "file_path", "target",
                            "uri", "resource", "scene",
                        }:
                            continue
                        if not value or len(value) > 512 or "\n" in value:
                            continue
                        if value[:1] in "{[":
                            continue
                        if "://" in value and not value.startswith("file://"):
                            continue
                        candidates.append(
                            value[7:] if value.startswith("file://") else value
                        )
                elif kind == "tool_result":
                    path = ev.get("path") or ev.get("normalised") or ""
                    if path:
                        candidates.append(path)
                for path in candidates:
                    if not path:
                        continue
                    rel = path
                    if os.path.isabs(path) and cwd:
                        try:
                            cand = os.path.relpath(path, cwd)
                        except ValueError:
                            cand = path
                        else:
                            if not cand.startswith(".."):
                                rel = cand
                    rel = rel.replace("\\", "/")
                    if rel not in seen:
                        seen.add(rel)
                        ordered.append(rel)
    except Exception as exc:
        print(f"piper: failed to parse event log: {exc}", file=sys.stderr)
        return []
    return ordered


def log_has_remote_tool_write(log_path):
    if not log_path or not os.path.isfile(log_path):
        return False
    try:
        with open(log_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("kind") == "workspace_freshness" and ev.get("why") == "remote_tool":
                    return True
    except OSError:
        return False
    return False


def collect_git_changed_paths(cwd):
    ordered = []
    seen = set()
    if not cwd:
        return ordered
    try:
        probe = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ordered
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return ordered
    try:
        names = subprocess.run(
            ["git", "-C", cwd, "diff", "--name-only"],
            capture_output=True, text=True, timeout=30,
        )
        status = subprocess.run(
            ["git", "-C", cwd, "status", "--porcelain"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ordered
    if names.returncode == 0:
        for line in names.stdout.splitlines():
            rel = line.strip().strip('"')
            if not rel or rel in seen:
                continue
            seen.add(rel)
            ordered.append(rel.replace("\\", "/"))
    if status.returncode == 0:
        for line in status.stdout.splitlines():
            if len(line) > 3 and line.startswith("??"):
                rel = line[3:].strip().strip('"').replace("\\", "/")
                if not rel or rel in seen:
                    continue
                full = os.path.join(cwd, rel)
                if os.path.isfile(full):
                    seen.add(rel)
                    ordered.append(rel)
    return ordered


def collect_git(cwd, out_dir):
    diff_stat = {"insertions": 0, "deletions": 0, "files": 0}
    git_diff_path = None
    try:
        probe = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return diff_stat, None
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return diff_stat, None
    try:
        numstat = subprocess.run(
            ["git", "-C", cwd, "diff", "--numstat"],
            capture_output=True, text=True, timeout=30,
        )
        unified = subprocess.run(
            ["git", "-C", cwd, "diff"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return diff_stat, None
    insertions = deletions = files = 0
    for line in numstat.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        files += 1
        if parts[0] != "-":
            try:
                insertions += int(parts[0])
            except ValueError:
                pass
        if parts[1] != "-":
            try:
                deletions += int(parts[1])
            except ValueError:
                pass
    diff_stat = {"insertions": insertions, "deletions": deletions, "files": files}
    if unified.stdout:
        git_diff_path = os.path.join(out_dir, "git.diff")
        with open(git_diff_path, "w", encoding="utf-8") as fh:
            fh.write(unified.stdout)
    return diff_stat, git_diff_path


def merge_files_touched(dest, extra):
    seen = set(dest)
    for path in extra:
        if path and path not in seen:
            seen.add(path)
            dest.append(path)
    return dest


def is_stalled_termination(reason):
    return reason in ("max_turns", "stalled", "stalled_no_turn")


def archive_prior_events(result_dir):
    """Copy existing events.jsonl/ndjson to a timestamped immutable file, then remove live names."""
    jsonl = os.path.join(result_dir, "events.jsonl")
    ndjson = os.path.join(result_dir, "events.ndjson")
    src = jsonl if os.path.isfile(jsonl) else (ndjson if os.path.isfile(ndjson) else None)
    if src is None:
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest = os.path.join(result_dir, f"events-{stamp}.jsonl")
    if os.path.exists(dest):
        dest = os.path.join(result_dir, f"events-{stamp}-1.jsonl")
    try:
        shutil.copy2(src, dest)
    except OSError:
        return None
    for path in (jsonl, ndjson):
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass
    return dest


def _first_last_slice(text, budget=480):
    if len(text) <= budget:
        return text
    if budget < 32:
        return text[:budget]
    ellipsis = " … "
    head_budget = (budget - len(ellipsis)) // 2
    tail_budget = budget - len(ellipsis) - head_budget
    head_end = head_budget
    nl = text.rfind("\n", 0, head_budget)
    if nl >= head_budget // 3:
        head_end = nl
    tail_start = len(text) - tail_budget
    tnl = text.find("\n", tail_start)
    if tnl != -1 and tnl + 1 < len(text) and (len(text) - (tnl + 1)) >= tail_budget // 3:
        tail_start = tnl + 1
    out = text[:head_end].rstrip(" \t\r") + ellipsis + text[tail_start:].lstrip(" \t\r\n")
    return out[:budget]


def compose_message(answer, status, reason, files, error, *, finish_summary="", completed=False):
    if finish_summary and str(finish_summary).strip():
        return str(finish_summary).strip()[:MESSAGE_CAP]
    text = (answer or "").strip()
    if text:
        if completed and status == "ok":
            return text[:MESSAGE_CAP]
        return _first_last_slice(text, min(480, MESSAGE_CAP))
    if error:
        return str(error)[:MESSAGE_CAP]
    bits = [f"status={status}"]
    if reason:
        bits.append(f"reason={reason}")
    if files:
        bits.append("files: " + ", ".join(files[:12]))
    return "; ".join(bits)[:MESSAGE_CAP]


def write_result(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def post_orch_webhook(url, payload, timeout=5.0):
    if not url:
        return False
    if not (url.startswith("http://") or url.startswith("https://")):
        print(f"piper: warning: refusing webhook URL with non-http(s) scheme: {url}", file=sys.stderr)
        return False
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:
        print(f"piper: warning: failed to post '{payload.get('kind')}' wake event to {url}: {exc}", file=sys.stderr)
        return False



# --- Deterministic harness helpers (no paste / no freehand JSON) ------------

ORCH_WEBHOOK_RELPATH = os.path.join(".piper", "orch_webhook")

DETACHED_NO_WAKE_MSG = (
    "piper: detached launch needs a wake URL. "
    "Start piper_ui (writes .piper/orch_webhook), "
    "or pass --orch-webhook / task orch_webhook / LMP_ORCH_WEBHOOK. "
    "Or stay attached."
)


def orch_webhook_path(root):
    return os.path.join(os.path.abspath(root), ORCH_WEBHOOK_RELPATH)


def write_orch_webhook_file(root, url):
    """Persist the wake URL so parents never copy it from a panel."""
    url = (url or "").strip()
    if not url:
        raise ValueError("orch webhook URL must be non-empty")
    path = orch_webhook_path(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(url + "\n")
    os.replace(tmp, path)
    return path


def read_orch_webhook_file(*roots):
    """Return first non-empty wake URL found under the given roots."""
    seen = set()
    for root in roots:
        if not root:
            continue
        root = os.path.abspath(root)
        if root in seen:
            continue
        seen.add(root)
        path = orch_webhook_path(root)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                url = fh.read().strip()
        except OSError:
            continue
        if url.startswith("http://") or url.startswith("https://"):
            return url
    return ""


def resolve_orch_webhook(cli_url=None, packet=None, search_roots=None):
    """CLI > task.json > env > .piper/orch_webhook file. Never invent a host."""
    for candidate in (
        (cli_url or "").strip(),
        ((packet or {}).get("orch_webhook") or "").strip(),
        (os.environ.get("LMP_ORCH_WEBHOOK") or "").strip(),
    ):
        if candidate:
            return candidate
    roots = list(search_roots or [])
    if packet:
        roots.extend([packet.get("cwd"), packet.get("task_dir"),
                      os.path.dirname(packet.get("result_path") or "")])
    return read_orch_webhook_file(*roots)


def build_answer_payload(action=None, text=None):
    """Known-right answer.json body. Models judge; harness owns the shape."""
    if action is not None and text is not None:
        raise ValueError("pass either action or text, not both")
    if action is not None:
        key = str(action).strip().lower()
        if key in ("allow", "allowed", "approve", "approved", "yes", "y", "true", "1"):
            return {"text": "allow"}
        if key in ("deny", "denied", "refuse", "refused", "no", "n", "false", "0"):
            return {"text": "deny"}
        raise ValueError(f"unknown answer action {action!r}; use allow or deny")
    if text is None:
        raise ValueError("answer requires allow/deny or --text")
    if not isinstance(text, str):
        raise ValueError("answer text must be a string")
    if not text.strip():
        raise ValueError("answer text must be non-empty")
    return {"text": text}


def resolve_answer_dir(dir_arg=None, task_arg=None):
    if dir_arg:
        return os.path.abspath(os.path.expanduser(dir_arg))
    if task_arg:
        packet = load_packet(task_arg)
        return os.path.dirname(os.path.abspath(packet["result_path"]))
    return os.path.abspath(".")


def write_answer_file(directory, payload):
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "answer.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def emit_task_packet(*, task_id, cwd, prompt, out_path, model_dir=None,
                     check=None, timeout_s=600, result_path=None,
                     auto_approve_irreversible=True):
    """Emit a correctly shaped task.json — no freehand JSON from an LLM."""
    cwd = os.path.abspath(os.path.expanduser(cwd))
    if not os.path.isdir(cwd):
        raise PacketError(f"cwd is not a directory: {cwd}")
    if not isinstance(task_id, str) or not task_id.strip():
        raise PacketError("id must be a non-empty string")
    if not isinstance(prompt, str) or not prompt.strip():
        raise PacketError("prompt must be a non-empty string")
    out_path = os.path.abspath(os.path.expanduser(out_path))
    packet = {
        "id": task_id.strip(),
        "cwd": cwd,
        "prompt": prompt,
        "auto_approve_exec": True,
        "auto_approve_writes": True,
        "auto_approve_irreversible": bool(auto_approve_irreversible),
        "timeout_s": float(timeout_s),
    }
    if model_dir:
        packet["model_dir"] = os.path.abspath(os.path.expanduser(model_dir))
    if check:
        packet["check"] = check
    if result_path:
        packet["result_path"] = os.path.abspath(os.path.expanduser(result_path))
    else:
        packet["result_path"] = os.path.join(os.path.dirname(out_path), "result.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(packet, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, out_path)
    return out_path, packet


def cmd_answer(args):
    try:
        if getattr(args, "text", None) is not None:
            payload = build_answer_payload(text=args.text)
        else:
            action = getattr(args, "action", None)
            if not action:
                print("piper answer: need allow|deny or --text", file=sys.stderr)
                return EXIT_INVALID
            payload = build_answer_payload(action=action)
        directory = resolve_answer_dir(getattr(args, "dir", None), getattr(args, "task", None))
        path = write_answer_file(directory, payload)
    except (ValueError, PacketError) as exc:
        print(f"piper answer: {exc}", file=sys.stderr)
        return EXIT_INVALID
    print(path)
    return EXIT_OK


def cmd_packet(args):
    prompt = args.prompt
    if args.prompt_file:
        try:
            with open(os.path.expanduser(args.prompt_file), encoding="utf-8") as fh:
                prompt = fh.read()
        except OSError as exc:
            print(f"piper packet: cannot read --prompt-file: {exc}", file=sys.stderr)
            return EXIT_INVALID
    if not prompt or not str(prompt).strip():
        print("piper packet: need --prompt or --prompt-file", file=sys.stderr)
        return EXIT_INVALID
    out_path = args.out or os.path.join(os.getcwd(), "task.json")
    try:
        path, _packet = emit_task_packet(
            task_id=args.id,
            cwd=args.cwd,
            prompt=prompt,
            out_path=out_path,
            model_dir=args.model_dir,
            check=args.check,
            timeout_s=args.timeout_s,
            result_path=args.result_path,
            auto_approve_irreversible=not args.no_auto_approve_irreversible,
        )
    except PacketError as exc:
        print(f"piper packet: {exc}", file=sys.stderr)
        return EXIT_INVALID
    print(path)
    return EXIT_OK


def read_task_roots(task_arg):
    """Lightweight task.json read for wake discovery — no model_dir required."""
    path = os.path.abspath(os.path.expanduser(task_arg))
    if not os.path.isfile(path):
        raise PacketError(f"task packet not found: {path}")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise PacketError(f"task.json is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PacketError("task.json must be an object")
    roots = [os.path.dirname(path)]
    for key in ("cwd", "task_dir"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            roots.append(os.path.abspath(os.path.expanduser(val.strip())))
    result_path = data.get("result_path")
    if isinstance(result_path, str) and result_path.strip():
        roots.append(os.path.dirname(os.path.abspath(os.path.expanduser(result_path.strip()))))
    # Preserve orch_webhook field if present without validating the rest.
    return data, roots


def cmd_wake_url(args):
    roots = []
    packet = None
    if getattr(args, "dir", None):
        roots.append(args.dir)
    if getattr(args, "task", None):
        try:
            packet, task_roots = read_task_roots(args.task)
        except PacketError as exc:
            print(f"piper wake-url: {exc}", file=sys.stderr)
            return EXIT_INVALID
        roots.extend(task_roots)
        url = resolve_orch_webhook(packet=packet, search_roots=roots)
    else:
        roots.append(os.getcwd())
        url = resolve_orch_webhook(search_roots=roots)
    if not url:
        print("piper wake-url: no wake URL in env or .piper/orch_webhook", file=sys.stderr)
        return EXIT_INVALID
    print(url)
    return EXIT_OK




def load_result_file(result_path):
    """Load result.json if present; return None when missing or unreadable."""
    if not result_path or not os.path.isfile(result_path):
        return None
    try:
        with open(result_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def review_verdict(result, exit_code):
    """Cheap parent verdict: PASS / FAIL / STALLED / DIED."""
    if result is None:
        return "DIED"
    status = str(result.get("status") or "").lower()
    if status == "ok" and exit_code == EXIT_OK:
        return "PASS"
    if status == "stalled":
        return "STALLED"
    return "FAIL"


def format_diff_stat(diff_stat):
    if not isinstance(diff_stat, dict):
        return str(diff_stat or "-")
    ins = diff_stat.get("insertions", 0) or 0
    dels = diff_stat.get("deletions", 0) or 0
    files = diff_stat.get("files", 0) or 0
    return f"+{ins} -{dels} ({files} files)"


def format_review_card(result, *, exit_code, result_path):
    """Deterministic review card — parents should not freehand the rubric."""
    verdict = review_verdict(result, exit_code)
    if result is None:
        return (
            "── piper review ─────────────────────────\n"
            f"result:   {result_path or '(none)'}\n"
            f"exit:     {exit_code}\n"
            f"verdict:  {verdict}  (no result.json)\n"
            "─────────────────────────────────────────"
        )
    files = result.get("files_touched") or []
    if isinstance(files, list):
        files_s = ", ".join(str(x) for x in files) if files else "(none)"
    else:
        files_s = str(files)
    test = result.get("test") or {}
    if isinstance(test, dict) and test.get("ran"):
        test_s = f"exit={test.get('exit_code')} cmd={test.get('command')!r}"
    elif isinstance(test, dict):
        test_s = "not run"
    else:
        test_s = str(test)
    msg = str(result.get("message") or "").strip().replace("\n", " ")
    if len(msg) > 160:
        msg = msg[:157] + "..."
    return (
        "── piper review ─────────────────────────\n"
        f"task:     {result.get('task_id') or '(unknown)'}\n"
        f"status:   {result.get('status')}\n"
        f"exit:     {exit_code}\n"
        f"message:  {msg or '(empty)'}\n"
        f"files:    {files_s}\n"
        f"diff:     {format_diff_stat(result.get('diff_stat'))}\n"
        f"test:     {test_s}\n"
        f"git_diff: {result.get('git_diff_path') or '(none)'}\n"
        f"verdict:  {verdict}\n"
        "─────────────────────────────────────────"
    )


def resolve_result_path(result_arg=None, task_arg=None):
    if result_arg:
        return os.path.abspath(os.path.expanduser(result_arg))
    if task_arg:
        packet = load_packet(task_arg)
        return os.path.abspath(packet["result_path"])
    return os.path.abspath("result.json")


def cmd_review(args):
    try:
        result_path = resolve_result_path(
            getattr(args, "result", None), getattr(args, "task", None)
        )
    except PacketError as exc:
        print(f"piper review: {exc}", file=sys.stderr)
        return EXIT_INVALID
    result = load_result_file(result_path)
    # When reviewing an existing result, treat missing status/ok as the card source of truth.
    exit_code = EXIT_OK
    if result is None:
        exit_code = EXIT_ERROR
    elif str(result.get("status") or "").lower() == "ok":
        exit_code = EXIT_OK
    elif str(result.get("status") or "").lower() == "timeout":
        exit_code = EXIT_TIMEOUT
    else:
        exit_code = EXIT_ERROR
    card = format_review_card(result, exit_code=exit_code, result_path=result_path)
    if getattr(args, "json", False):
        print(json.dumps({
            "result_path": result_path,
            "exit_code": exit_code,
            "verdict": review_verdict(result, exit_code),
            "result": result,
            "card": card,
        }, indent=2))
    else:
        print(card)
    return EXIT_OK if result is not None else EXIT_ERROR


def cmd_dispatch(args):
    """dispatch → wait → print review card. Cloud still decides next slice."""
    try:
        packet = load_packet(args.task)
    except PacketError as exc:
        print(f"piper dispatch: {exc}", file=sys.stderr)
        return EXIT_INVALID
    result_path = packet["result_path"]
    run_argv = ["run", "--task", args.task]
    if getattr(args, "jsonl", False):
        run_argv.append("--jsonl")
    if getattr(args, "orch_webhook", None):
        run_argv.extend(["--orch-webhook", args.orch_webhook])
    if getattr(args, "auto_approve_all", False):
        run_argv.append("--auto-approve-all")
    elif getattr(args, "auto_approve_irreversible", False):
        run_argv.append("--auto-approve-irreversible")
    # Always attached: dispatch owns wait. Detach is out of scope for this helper.
    code = main(run_argv)
    result = load_result_file(result_path)
    card = format_review_card(result, exit_code=code, result_path=result_path)
    if getattr(args, "json", False):
        print(json.dumps({
            "result_path": result_path,
            "exit_code": code,
            "verdict": review_verdict(result, code),
            "result": result,
            "card": card,
        }, indent=2))
    else:
        print(card)
    return code


def empty_test_block():
    return {"ran": False, "exit_code": None, "command": None, "output_tail": None}


def result_shell(*, task_id, cwd, model_dir, status, message, wall_seconds=0,
                 turns=0, generated_tokens=0, files_touched=None, diff_stat=None,
                 git_diff_path=None, log_path=None, error=None):
    return {
        "task_id": task_id,
        "status": status,
        "message": message,
        "cwd": cwd,
        "model_dir": model_dir,
        "wall_seconds": wall_seconds,
        "turns": turns,
        "generated_tokens": generated_tokens,
        "files_touched": files_touched or [],
        "diff_stat": diff_stat or {"insertions": 0, "deletions": 0, "files": 0},
        "git_diff_path": git_diff_path,
        "test": empty_test_block(),
        "log_path": log_path,
        "error": error,
    }


class LiveJournal:
    """Append-only live.jsonl writer fed by sidecar notifications."""

    _WRITE_TOOLS = frozenset({"write_file", "apply_patch", "commit_think_block"})

    def __init__(self, path: str) -> None:
        self._fh = open(path, "w", encoding="utf-8")
        self._seq = 0
        self._buffer = ""
        self._channel = ""

    def _write_line(self, obj: dict) -> None:
        self._seq += 1
        obj["seq"] = self._seq
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()

    def _flush_buffer(self) -> None:
        if self._buffer:
            self._write_line({
                "kind": "delta",
                "channel": self._channel,
                "text": self._buffer,
            })
            self._buffer = ""

    def on_notification(self, method: str, params) -> None:
        if method == "lmp/token":
            channel = params.get("channel") or ""
            text = params.get("text") or ""
            if channel not in ("thinking", "answer"):
                return
            if not text:
                return
            if channel != self._channel:
                self._flush_buffer()
                self._channel = channel
            self._buffer += text
            if len(self._buffer) >= 64:
                self._flush_buffer()
        elif method == "lmp/turn":
            self._flush_buffer()
            self._handle_turn(params)
        # Other methods: no flush, no write.

    def _handle_turn(self, params) -> None:
        tool_name = params.get("tool_name") or ""
        tool_status = params.get("tool_status") or ""
        summary = str(params.get("summary") or "")[:240]
        read_bytes = int(params.get("read_bytes") or 0)
        edit_bytes = int(params.get("edit_bytes") or 0)
        think_tokens = int(params.get("think_tokens") or 0)
        text_tokens = int(params.get("text_tokens") or 0)
        tool_tokens = int(params.get("tool_tokens") or 0)

        path, command = self._parse_tool_args(params.get("tool_args"))

        self._write_line({
            "kind": "turn",
            "tool": tool_name,
            "path": path,
            "command": command,
            "status": tool_status,
            "summary": summary,
            "read_bytes": read_bytes,
            "edit_bytes": edit_bytes,
            "think_tokens": think_tokens,
            "text_tokens": text_tokens,
            "tool_tokens": tool_tokens,
        })

        if edit_bytes > 0 or tool_name in self._WRITE_TOOLS:
            self._write_line({
                "kind": "write",
                "tool": tool_name,
                "path": path,
                "edit_bytes": edit_bytes,
            })

    @staticmethod
    def _parse_tool_args(raw):
        """Extract (path, command) from tool_args which may be a JSON string or dict."""
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return "", ""
            if not isinstance(data, dict):
                return "", ""
        elif isinstance(raw, dict):
            data = raw
        else:
            return "", ""

        path_keys = ("path", "file", "filepath", "file_path", "target")
        cmd_keys = ("command", "cmd")
        path = ""
        for k in path_keys:
            v = data.get(k)
            if isinstance(v, str) and v:
                path = v
                break
        command = ""
        for k in cmd_keys:
            v = data.get(k)
            if isinstance(v, str) and v:
                command = v
                break
        return path, command

    def flush(self) -> None:
        self._flush_buffer()

    def close(self) -> None:
        self._flush_buffer()
        self._fh.close()


def run_mission(task_arg, jsonl=False, orch_webhook=None):
    """Drive one packet. Returns process exit code; always tries to write result.json."""
    result_path = None
    packet = None
    try:
        packet = load_packet(task_arg)
    except PacketError as exc:
        print(f"piper: {exc}", file=sys.stderr)
        # Best-effort result next to a readable task file.
        try:
            task_path = resolve_task_file(task_arg)
            fallback = os.path.join(os.path.dirname(task_path), "result.json")
            write_result(fallback, result_shell(
                task_id="", cwd="", model_dir="", status="error",
                message=str(exc), error=str(exc),
            ))
        except (PacketError, OSError):
            pass
        return EXIT_INVALID

    result_path = packet["result_path"]
    sidecar = bind_sidecar()
    if not os.path.isfile(sidecar):
        err = f"no sidecar at {sidecar}; build lmp_sidecar first (or set LMP_SIDECAR)"
        print(f"piper: {err}", file=sys.stderr)
        write_result(result_path, result_shell(
            task_id=packet["id"], cwd=packet["cwd"], model_dir=packet["model_dir"],
            status="error", message=err, error=err,
        ))
        hook = resolve_orch_webhook(orch_webhook, packet)
        if hook:
            post_orch_webhook(hook, {
                "kind": "stalled",
                "task_id": packet["id"],
                "run_id": "",
                "cwd": packet["cwd"],
                "result_path": result_path,
                "seq": 0,
                "status": "error",
                "question": "",
            })
        return EXIT_ERROR

    apply_feature_flags(packet)
    result_dir = os.path.dirname(os.path.abspath(result_path))
    os.makedirs(result_dir, exist_ok=True)
    archive_prior_events(result_dir)
    journal = LiveJournal(os.path.join(result_dir, "live.jsonl"))
    harness_dir = tempfile.mkdtemp(prefix=f"piper-worker-{packet['id']}-")
    event_log = os.path.join(harness_dir, "events.jsonl")
    durable_log = os.path.join(result_dir, "events.jsonl")
    answer_parts = []

    def on_notification(method, params):
        if jsonl:
            sys.stdout.write(json.dumps({"method": method, "params": params}) + "\n")
            sys.stdout.flush()
        if method == "lmp/token" and params.get("channel") == "answer":
            answer_parts.append(params.get("text") or "")
        journal.on_notification(method, params)

    meta = {
        "name": packet["id"],
        "mission": packet["prompt"],
        "mode": packet["mode"],
        "wall_clock_seconds": packet["timeout_s"],
        "auto_approve_exec": packet["auto_approve_exec"],
        "auto_approve_writes": packet.get("auto_approve_writes", True),
        "deny_irreversible": not packet.get("auto_approve_irreversible", False),
    }
    contract = ae.TaskContract(
        check="", task_json_path=packet["task_path"],
        task_json_bytes=packet["raw_bytes"], protected=(),
    )
    sampling = dict(ae.DEFAULT_SAMPLING)
    sampling["seed"] = ae.DEFAULT_SEED

    state = None
    protocol_error = None
    try:
        state = ae.drive_sidecar(
            meta, packet["model_dir"], packet["cwd"], harness_dir, sampling,
            contract, extra_env=None, on_notification=on_notification, quiet=True,
        )
    except Exception as exc:
        protocol_error = exc
    finally:
        journal.close()
        if os.path.isfile(event_log):
            try:
                shutil.copy2(event_log, durable_log)
                # Compatibility alias for older bakeoff paths.
                try:
                    shutil.copy2(event_log, os.path.join(result_dir, "events.ndjson"))
                except OSError:
                    pass
            except OSError:
                durable_log = None
        else:
            durable_log = None
        shutil.rmtree(harness_dir, ignore_errors=True)

    if protocol_error is not None:
        err = f"sidecar protocol failure: {protocol_error}"
        print(f"piper: {err}", file=sys.stderr)
        write_result(result_path, result_shell(
            task_id=packet["id"], cwd=packet["cwd"], model_dir=packet["model_dir"],
            status="error", message=err, error=err, log_path=durable_log,
        ))
        return EXIT_ERROR

    files_touched = collect_files_touched(durable_log, packet["cwd"])
    diff_stat, git_diff_path = collect_git(packet["cwd"], result_dir)
    if packet.get("trust_mcp") or (
        not files_touched and log_has_remote_tool_write(durable_log)
    ):
        merge_files_touched(files_touched, collect_git_changed_paths(packet["cwd"]))
    answer = "".join(answer_parts)
    deadline = bool(state.get("deadline_killed"))
    denied_irr = bool(state.get("denied_irreversible"))
    detail = state.get("denied_irreversible_detail")
    reason = state.get("reason") or ""

    if deadline:
        status, exit_code = "timeout", EXIT_TIMEOUT
        error = f"wall clock exceeded ({packet['timeout_s']}s)"
    elif denied_irr:
        status, exit_code = "error", EXIT_ERROR
        error = (
            "irreversible tool denied (orchestrator must escalate): "
            + str(detail or "irreversible tool")
        )
    elif state.get("completed"):
        status, exit_code = "ok", EXIT_OK
        error = None
    elif is_stalled_termination(reason):
        status, exit_code = "stalled", EXIT_ERROR
        error = f"agent did not complete ({reason})"
    else:
        status, exit_code = "error", EXIT_ERROR
        sig = state.get("sidecar_signal_name")
        error = f"agent did not complete ({reason or 'no_run_end'}"
        if sig:
            error += f", {sig}"
        error += ")"

    message = compose_message(
        answer, status, reason, files_touched, error,
        completed=bool(state.get("completed")) and status == "ok",
    )
    write_result(result_path, result_shell(
        task_id=packet["id"], cwd=packet["cwd"], model_dir=packet["model_dir"],
        status=status, message=message,
        wall_seconds=state.get("seconds") or 0,
        turns=state.get("iterations") or 0,
        generated_tokens=state.get("generated_tokens") or 0,
        files_touched=files_touched, diff_stat=diff_stat,
        git_diff_path=git_diff_path, log_path=durable_log, error=error,
    ))
    hook = resolve_orch_webhook(orch_webhook, packet)
    if hook:
        is_completed = bool(state.get("completed")) and (status == "ok")
        hook_payload = {
            "kind": "done" if is_completed else "stalled",
            "task_id": packet["id"],
            "run_id": state.get("run_id") or "",
            "cwd": packet["cwd"],
            "result_path": result_path,
            "seq": 0,
            "status": status,
            "question": "",
        }
        post_orch_webhook(hook, hook_payload)
    return exit_code


class WorkerParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"piper: {message}", file=sys.stderr)
        sys.exit(EXIT_INVALID)


HELP_CONTRACT = (
    "Piper worker wake standard: parent owns the horizon; events ask/done/stalled/died; pass --orch-webhook or stay attached. "
    "Two legal ways to own the horizon: stay attached (parent waits on files/exit, no webhook) or detach "
    "(screen, nohup, background, requiring a wake URL via --orch-webhook, task.json orch_webhook, or LMP_ORCH_WEBHOOK). "
    "A run that writes result.json POSTs done if completed, stalled if stopped/failed. Process exit with no result.json POSTs died. "
    "Silent detached workers are refused."
)

PARENT_CONTRACT_BANNER = (
    "piper: you are the parent. Stay attached and read the exit and result.json.\n"
    "Do not detach unless you pass --orch-webhook. No default URL. Events:\n"
    "ask (piper answer allow|deny|--text), done, stalled (not success), died (do not relaunch).\n"
    "See PIPER.md if present.\n"
)

AGENT_WAKE_FALLBACK = """# Piper Local Worker — Orchestrator Guide & Parent Runbook

## Overview
**Core Principle:** Cloud directs, local writes, cloud verifies, repeat.

Piper is a fast headless coding worker running locally on Apple Silicon (via MLX). It executes edits, shell commands, and file operations inside `cwd` with zero cloud output token cost.

The cloud orchestrator (Cursor, Claude, Gemini, Antigravity, or custom script) acts as the high-level brain:
- Maintains the long-horizon plan and acceptance criteria.
- Decomposes complex tasks into bounded packets (1–3 files per packet).
- Directs Piper by writing task packets (`task.json`).
- Verifies outcomes (diffs, test execution, acceptance checks).
- Loops until the entire mission is verified complete.

## The Long-Horizon Execution Loop
```
┌─────────────────────────┐         task.json            ┌──────────────────────────┐
│   Cloud Orchestrator    │ ───────────────────────────► │       Piper Worker       │
│  (Cursor, Claude, etc.) │                              │  (MLX on Apple Silicon)  │
│                         │ ◄─────────────────────────── │                          │
│ plan · review · verify  │     result.json + diff       │  edits · tools · loop    │
└─────────────────────────┘                              └──────────────────────────┘
             │                                                         │
             └────────── repeat until horizon acceptance passes ───────┘
```

### Roles

| Who | Owns | Does not own |
|---|---|---|
| **Cloud Orchestrator** | Goal decomposition, file-level direction, acceptance criteria, high-level review, troubleshooting, "are we done?" | Bulk code generation tokens, local tool thrash |
| **Piper Worker** | Edits, tool execution, test commands, local iteration inside `cwd` | Long-horizon judgment, multi-repo strategy |

Local models work best on scoped packets: **packets must be specific**, and **every turn gets an orchestrator review**. Trust outcomes (diff + tests + `result.json`), not vibes.

### Step-by-Step Procedure

1. **Frame the Horizon**
   Define the overarching objective and an acceptance checklist (e.g. unit tests pass, new command works, UI renders).

2. **Slice into Discrete Packets**
   Pick the smallest incremental step towards the goal.
   - Scope each slice to 1–3 files.
   - Explicitly list which files to EDIT, CREATE, and DO NOT TOUCH.

3. **Write `task.json`**
   Write a task packet in the workspace or a task directory:
   ```json
   {
     "id": "slice-001",
     "cwd": "/absolute/path/to/workspace",
     "mode": "agent",
     "model_dir": "/Users/dev/Desktop/Models/Qwen3.6-35B-A3B-MLX-4bit",
     "prompt": "## Horizon Context\\nBuilding feature X.\\n\\n## This Slice Only\\nAdd validator in src/validator.cpp and test in tests/test_validator.cpp.\\n\\n## Files\\n- EDIT: src/validator.cpp\\n- CREATE: tests/test_validator.cpp\\n- DO NOT TOUCH: src/core.cpp\\n\\n## Done When\\n- Unit test passes with `ctest -R test_validator`",
     "auto_approve_exec": true,
     "auto_approve_writes": true,
     "auto_approve_irreversible": true,
     "timeout_s": 600,
     "max_iterations": 30,
     "check": "ctest -R test_validator",
     "result_path": "/absolute/path/to/result.json"
   }
   ```
   - **`max_iterations`**: turn budget sent to the agent loop. Default **30**; default **60** when `trust_mcp` is set (Godoer-heavy). Raise in the packet for long slices — no rebuild.
   - **`check`**: operator acceptance command. Also becomes `verify_contract` during the run. A green post-run check yields `status=ok` / wake `done` even if the loop hit `max_turns` without `completed=true` (not a crash). Timeouts and irreversible denials stay failures.

4. **Dispatch Piper**
   Run the CLI command:
   ```bash
   piper run --task /path/to/task.json
   # or equivalently:
   piper worker run --task /path/to/task.json
   # For autonomous unattended execution without prompts or pauses:
   piper run --task /path/to/task.json --auto-approve-irreversible
   # Or approve all (exec + writes + irreversible):
   piper run --task /path/to/task.json --auto-approve-all
   ```
   - **Attached mode (standard)**: Process waits and exits when the slice completes.
     - `0`: Completed normally.
     - `1`: Worker error.
     - `2`: Execution timed out (`timeout_s`).
     - `3`: Invalid task packet or missing wake URL for detached run.
   - **Detached mode**: If launching in the background (nohup, screen), you MUST pass `--orch-webhook <URL>`. Silent background launches without a webhook are refused.
   - **Irreversible tools**: Destructive tools or project managers (like `godot_project`, `delete_file`, or whole-file overwrites) escalate to `gate: irreversible`. Set `"auto_approve_irreversible": true` or pass `--auto-approve-irreversible` / `--auto-approve-all` for unattended runs; otherwise Piper pauses and writes `awaiting_user.json` for `answer.json`.

5. **Review `result.json` & Inspect Changes**
   Prefer the thin helper (no freehand rubric):
   ```bash
   piper dispatch --task /path/to/task.json   # run attached → wait → print review card
   piper review --result /path/to/result.json # card only, when you already waited
   ```
   Piper writes a structured result upon completion:
   ```json
   {
     "task_id": "slice-001",
     "status": "ok",
     "message": "Implemented validation logic and verified with unit test.",
     "files_touched": ["src/validator.cpp", "tests/test_validator.cpp"],
     "diff_stat": "+52 -2",
     "git_diff_path": "/path/to/slice.diff"
   }
   ```

   **Review Rubric (Keep it cheap):**
   - **Status**: Is `status == "ok"`? If `"error"` or `"stalled"`, inspect the message. Green `test.exit_code=0` with incomplete loop is still `ok` when `check` was set.
   - **Files Touched**: Are changes confined to expected paths? Reject drive-by edits.
   - **Diff**: Skim git diff for regressions or unnecessary churn.
   - **Acceptance**: Prefer `result.test` from the packet `check`; re-run only if you need a second reading.

6. **Iterate or Complete**
   - **Pass**: If acceptance criteria for the slice pass, dispatch the next slice.
   - **Fail**: Send a narrowed/clarified packet, or perform that specific edit yourself.
   - **Done**: When all acceptance checklist items are verified, complete the mission.

---

## Piper worker wake standard

Any agent that starts Piper is the parent. Piper does not come find you.
A human must not copy a URL from a panel. A timer that checks the folder is
not the wake. It is a fallback, and it is how a finished or stalled run sits
until someone asks.

## Who owns the wake

The agent that launches `lmp_sidecar --worker` or `piper worker run` already
has a way to be woken, or it stays attached to the process. Piper does not
know who that agent is. There is no default host. Grok, Gemini, a script,
and a human CI job each pass their own URL.

Turn-based agents cannot stay attached. They pass a wake URL or they do not
detach.

## How to launch

Stay attached only if your process waits and reads the exit.

Otherwise, before start:

- `--orch-webhook URL`, or
- task field `orch_webhook`, or
- env `LMP_ORCH_WEBHOOK`

Detached with no URL: the CLI exits before the sidecar starts. `--help`
says this in one paragraph. Read that before the first launch. The help
text is the contract, not this chat.

## Events

One short JSON POST. No prompt bodies. Failed POST does not fail the
mission. No URL means no POST.

| kind | when | parent does |
| --- | --- | --- |
| `ask` | `awaiting_user.json` written, or an irreversible call is paused | write `answer.json` as `{"text":"..."}`. Do not restart. `allow` or `deny` for an irreversible call. Guidance for a real question. |
| `done` | `result.json` written and the slice completed | read the files. Send the next slice or stop. |
| `stalled` | `result.json` written and the harness stopped the run (`stalled`, `max_turns`, not completed) | read what landed. Do not treat it as success. Next slice or stop. |
| `died` | process exited and no `result.json` was written | launch parent sends this. Tell the user. Do not relaunch blindly. |

`stalled` is its own kind. Do not hide it inside `done` with `status: error`.
A parent that only handles `done` will miss a stall, which is the bug this
standard exists to kill. `result.json` uses the same `status: "stalled"` for
`max_turns` / no-progress stalls so parents need not parse error strings.

Body:

```json
{
  "kind": "stalled",
  "task_id": "mc-live-2",
  "run_id": "r-...",
  "cwd": "/absolute/path",
  "result_path": "/absolute/path/result.json",
  "seq": 0,
  "status": "stalled",
  "question": ""
}
```

`ask` fills `question` and `seq`. `done` and `stalled` fill `status`
(`ok` / `error` / `timeout` / `stalled`).
`died` fills `cwd` and `task_id`.

Do not POST every turn. Do not POST tool output.

## What the parent must not do

- Do not poll Piper, and do not poll the cloud from Piper.
- Do not ask the human if the sidecar is still there. The event is the notice.
- Do not bake another agent's webhook into the binary.
- Do not start a silent screen or nohup without the URL.

A folder watch is allowed only as a backup when the platform has no wake
URL yet. It is not the standard, and it must not be the path a second
agent copies.

## Done when

- `--help` names this standard in one paragraph.
- A detached launch with no URL exits before the sidecar starts.
- A run that writes `result.json` POSTs `done` if `status=ok` (model completed,
  `plan_ready`, or a green packet `check` after an incomplete loop stop such as
  `max_turns`), `stalled` if it did not.
- The launch parent POSTs `died` if the sidecar exits with no result.
- An irreversible call and `ask_user` both POST `ask` and wait.
"""

CURSOR_RULE_CONTENT = """---
description: Piper local worker orchestration and parent contract
alwaysApply: true
---

piper: you are the parent. You have access to `piper`, a fast local coding worker CLI running on Apple Silicon (MLX).
Philosophy: Cloud directs, local writes, cloud verifies, repeat.
Use Piper to execute small, bounded slices of long-horizon tasks until the greater task is complete.

## How to Orchestrate with Piper

1. **Decompose into Small Slices**:
   Break the horizon goal into small slices touching 1–3 files per slice.
   Never ask Piper to solve an entire complex task in a single prompt.

2. **Prepare a Task Packet (`task.json`)**:
   Write a packet file with this structure:
   ```json
   {
     "id": "slice-001",
     "cwd": "/absolute/path/to/workspace",
     "prompt": "## Goal\\nImplement X.\\n\\n## Files\\n- EDIT: src/a.cpp\\n- CREATE: tests/test_a.cpp\\n- DO NOT TOUCH: other files\\n\\n## Done When\\n- Tests compile and pass.",
     "auto_approve_exec": true,
     "auto_approve_irreversible": true,
     "timeout_s": 600
   }
   ```

3. **Dispatch Piper CLI**:
   Run and stay attached to the process:
   `piper run --task path/to/task.json` (or `piper worker run --task ...`)
   - For unattended execution (no pauses on irreversible tools): pass `--auto-approve-irreversible` or `--auto-approve-all`.
   - Exit code `0` = success, non-zero = error / timeout.
   - If running detached/background, pass `--orch-webhook <URL>`.

4. **Review Results**:
   Inspect `result.json` written by Piper:
   - Check `status` ("ok" vs "error"/"stalled").
   - Review `files_touched` and `git_diff` — verify changes match instructions.
   - Run slice acceptance tests.

5. **Loop Until Horizon Complete**:
   - If slice passed: dispatch the next slice.
   - If slice failed: write a narrower prompt or fix minor issues directly.
   - Repeat until all acceptance criteria for the greater task pass.

6. **Parent Contract & Events**:
   - Stay attached and read exit code + `result.json`.
   - Events: `ask` (write `answer.json`), `done`, `stalled` (not success; harness stopped), `died` (crashed).
   - See `PIPER.md` for full specification and review rubric.
"""


def init_project(target_dir="."):
    target_dir = os.path.abspath(os.path.expanduser(target_dir))
    os.makedirs(target_dir, exist_ok=True)

    wake_doc_path = os.path.join(ROOT, "docs", "AGENT_WAKE.md")
    piper_md_content = None
    if os.path.isfile(wake_doc_path):
        try:
            with open(wake_doc_path, "r", encoding="utf-8") as fh:
                piper_md_content = fh.read()
        except OSError:
            piper_md_content = None
    if not piper_md_content:
        piper_md_content = AGENT_WAKE_FALLBACK

    piper_md_path = os.path.join(target_dir, "PIPER.md")
    with open(piper_md_path, "w", encoding="utf-8") as fh:
        fh.write(piper_md_content)

    cursor_rules_dir = os.path.join(target_dir, ".cursor", "rules")
    os.makedirs(cursor_rules_dir, exist_ok=True)
    cursor_rule_path = os.path.join(cursor_rules_dir, "piper-parent.mdc")
    with open(cursor_rule_path, "w", encoding="utf-8") as fh:
        fh.write(CURSOR_RULE_CONTENT)

    print(
        "piper init: PIPER.md and .cursor/rules/piper-parent.mdc. "
        "Godoer briefs left alone. Stay attached, or pass --orch-webhook."
    )
    return EXIT_OK


def build_parser():
    parser = WorkerParser(
        prog="piper",
        description="Piper headless worker — one agent, CLI interface for orchestrators.\n\n" + HELP_CONTRACT,
        epilog="VSIX for humans, CLI for agents. One MLX process at a time.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    def add_run_flags(p):
        p.add_argument("--task", required=True,
                       help="task.json path, or a directory containing task.json")
        p.add_argument("--jsonl", action="store_true",
                       help="stream lmp/* notifications as ndjson on stdout")
        p.add_argument("--orch-webhook", default=None,
                       help="webhook URL to wake cloud orchestrator on ask/done/died events")
        p.add_argument("--auto-approve-irreversible", action="store_true",
                       help="auto-approve irreversible tool calls (destroys data / overwrite)")
        p.add_argument("--auto-approve-all", action="store_true",
                       help="auto-approve all tool calls (exec + writes + irreversible)")
        p.add_argument("--detach", action="store_true",
                       help="detach from launch session (requires --orch-webhook, task.json orch_webhook, LMP_ORCH_WEBHOOK, or .piper/orch_webhook)")

    def add_serve_flags(p):
        p.add_argument("--socket", default=None, help="unix domain socket path")
        p.add_argument("--idle-timeout", type=float, default=3600.0, help="idle timeout in seconds")

    run_p = sub.add_parser("run", help="run one task packet (synonym for worker run)",
                           description=HELP_CONTRACT,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_flags(run_p)

    init_p = sub.add_parser("init", help="initialize PIPER.md and .cursor/rules/piper-parent.mdc",
                            description="Initialize PIPER.md and .cursor/rules/piper-parent.mdc without touching Godoer briefs.",
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    init_p.add_argument("target_dir", nargs="?", default=".",
                        help="target project directory (default: current directory)")

    serve_p = sub.add_parser("serve", help="run keep-warm worker daemon")
    add_serve_flags(serve_p)

    worker_p = sub.add_parser("worker", help="worker commands")
    worker_sub = worker_p.add_subparsers(dest="worker_cmd")
    worker_run = worker_sub.add_parser("run", help="run one task packet",
                                       description=HELP_CONTRACT,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_flags(worker_run)
    worker_serve = worker_sub.add_parser("serve", help="run keep-warm worker daemon")
    add_serve_flags(worker_serve)
    worker_init = worker_sub.add_parser("init", help="initialize PIPER.md and .cursor/rules/piper-parent.mdc",
                                        description="Initialize PIPER.md and .cursor/rules/piper-parent.mdc without touching Godoer briefs.",
                                        formatter_class=argparse.RawDescriptionHelpFormatter)
    worker_init.add_argument("target_dir", nargs="?", default=".",
                             help="target project directory (default: current directory)")

    sub.add_parser("self-test", help="fake-sidecar protocol tests (no model)")

    answer_p = sub.add_parser(
        "answer",
        help="write answer.json for an ask (allow/deny/--text; no freehand JSON)",
        description="Write a correctly shaped answer.json next to result.json. "
                    "Use this instead of pasting JSON into a file.",
    )
    answer_p.add_argument("action", nargs="?", choices=("allow", "deny"),
                          help="approve or deny an irreversible ask")
    answer_p.add_argument("--text", default=None,
                          help="free-text answer for a real question (writes {\"text\":...})")
    answer_p.add_argument("--dir", default=None,
                          help="directory that holds answer.json (default: cwd)")
    answer_p.add_argument("--task", default=None,
                          help="task.json whose result_path directory receives answer.json")

    packet_p = sub.add_parser(
        "packet",
        help="emit a correctly shaped task.json (no freehand packet JSON)",
        description="Emit task.json with the known-right fields. Models invent goals; "
                    "the harness owns packet shape.",
    )
    packet_p.add_argument("--id", required=True, help="task id")
    packet_p.add_argument("--cwd", required=True, help="absolute workspace cwd")
    packet_p.add_argument("--prompt", default=None, help="prompt string")
    packet_p.add_argument("--prompt-file", default=None, help="read prompt from file")
    packet_p.add_argument("--out", default=None, help="output path (default: ./task.json)")
    packet_p.add_argument("--model-dir", default=None, help="optional model_dir")
    packet_p.add_argument("--check", default=None, help="optional acceptance command")
    packet_p.add_argument("--timeout-s", type=float, default=600.0, help="timeout_s (default 600)")
    packet_p.add_argument("--result-path", default=None, help="optional result_path")
    packet_p.add_argument("--no-auto-approve-irreversible", action="store_true",
                          help="leave auto_approve_irreversible false")

    wake_p = sub.add_parser(
        "wake-url",
        help="print the resolved orch wake URL (env or .piper/orch_webhook)",
        description="Resolve the wake URL without copying it from a dashboard panel.",
    )
    wake_p.add_argument("--dir", default=None, help="workspace root containing .piper/orch_webhook")
    wake_p.add_argument("--task", default=None, help="task.json used to locate search roots")

    review_p = sub.add_parser(
        "review",
        help="print a deterministic review card from result.json",
        description="Print the cheap parent review card. No freehand rubric.",
    )
    review_p.add_argument("--result", default=None, help="path to result.json (default: ./result.json)")
    review_p.add_argument("--task", default=None, help="task.json whose result_path is reviewed")
    review_p.add_argument("--json", action="store_true", help="emit machine-readable JSON including the card")

    dispatch_p = sub.add_parser(
        "dispatch",
        help="run a task attached, wait, print review card",
        description="Thin orchestrator helper: dispatch → wait → print review card. "
                    "Cloud still decides the next slice.",
    )
    add_run_flags(dispatch_p)
    # Dispatch stays attached; drop --detach from this helper's UX by not advertising it.
    # (Flag may still exist via add_run_flags; we ignore it and never pass --detach.)
    dispatch_p.add_argument("--json", action="store_true",
                            help="emit machine-readable JSON including the review card")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    command = getattr(args, "command", None)
    if command is None:
        parser.print_help()
        return EXIT_INVALID
    if command == "self-test":
        return self_test()
    if command == "answer":
        return cmd_answer(args)
    if command == "packet":
        return cmd_packet(args)
    if command == "wake-url":
        return cmd_wake_url(args)
    if command == "review":
        return cmd_review(args)
    if command == "dispatch":
        return cmd_dispatch(args)

    if command == "init" or (command == "worker" and getattr(args, "worker_cmd", None) == "init"):
        return init_project(getattr(args, "target_dir", "."))

    if command == "serve" or (command == "worker" and getattr(args, "worker_cmd", None) == "serve"):
        sidecar = resolved_sidecar()
        if not os.path.isfile(sidecar) or not os.access(sidecar, os.X_OK):
            print(f"piper: sidecar binary not found at {sidecar}", file=sys.stderr)
            return EXIT_ERROR
        cmd = [sidecar, "--worker", "--serve"]
        if getattr(args, "socket", None):
            cmd.extend(["--socket", args.socket])
        if getattr(args, "idle_timeout", None):
            cmd.extend(["--idle-timeout", str(args.idle_timeout)])
        os.execv(sidecar, cmd)

    if command == "worker":
        if getattr(args, "worker_cmd", None) not in ("run", "serve", "init"):
            parser.error("expected `piper worker run --task PATH` or `piper worker serve` or `piper worker init`")

    try:
        packet = load_packet(args.task)
    except PacketError as exc:
        print(f"piper: {exc}", file=sys.stderr)
        return EXIT_INVALID

    if getattr(args, "auto_approve_all", False):
        packet["auto_approve_exec"] = True
        packet["auto_approve_writes"] = True
        packet["auto_approve_irreversible"] = True
    elif getattr(args, "auto_approve_irreversible", False):
        packet["auto_approve_irreversible"] = True

    webhook_url = resolve_orch_webhook(
        getattr(args, "orch_webhook", None),
        packet,
        search_roots=[packet.get("cwd"), packet.get("task_dir"),
                      os.path.dirname(packet.get("result_path") or ""), os.getcwd()],
    )

    flag = os.environ.get("LMP_DAEMONIZE", "")
    is_detached = bool(getattr(args, "detach", False)) or (flag == "1") or (
        flag != "0" and (ae.stdin_is_devnull() or ae.stdout_is_regular_file())
    )

    if is_detached and not webhook_url:
        print(DETACHED_NO_WAKE_MSG, file=sys.stderr)
        return EXIT_INVALID

    sys.stderr.write(PARENT_CONTRACT_BANNER)
    sys.stderr.flush()
    os.environ["LMP_BANNER_PRINTED"] = "1"

    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    if is_detached:
        ae.detach_from_launch_session()

    result_path = packet["result_path"]
    if os.path.isfile(result_path):
        try:
            os.remove(result_path)
        except OSError:
            pass

    sidecar = resolved_sidecar()
    use_cpp = os.environ.get("LMP_USE_CPP_WORKER", "1") != "0"
    if use_cpp and os.path.isfile(sidecar) and os.access(sidecar, os.X_OK) and not sidecar.endswith(".py"):
        cmd = [sidecar, "--worker", "--task", args.task]
        if getattr(args, "jsonl", False):
            cmd.append("--jsonl")
        if getattr(args, "auto_approve_irreversible", False):
            cmd.append("--auto-approve-irreversible")
        if getattr(args, "auto_approve_all", False):
            cmd.append("--auto-approve-all")
        if webhook_url:
            cmd.extend(["--orch-webhook", webhook_url])

        proc = subprocess.Popen(cmd)

        def _forward_sig(signum, frame):
            try:
                proc.send_signal(signum)
            except OSError:
                pass

        old_int = signal.signal(signal.SIGINT, _forward_sig)
        old_term = signal.signal(signal.SIGTERM, _forward_sig)
        try:
            ret = proc.wait()
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)

        if not os.path.isfile(result_path):
            if webhook_url:
                died_payload = {
                    "kind": "died",
                    "task_id": packet["id"],
                    "run_id": "",
                    "cwd": packet["cwd"],
                    "result_path": result_path,
                    "seq": 0,
                    "status": "",
                    "question": "",
                }
                post_orch_webhook(webhook_url, died_payload)
        return ret

    try:
        ret = run_mission(args.task, jsonl=bool(getattr(args, "jsonl", False)), orch_webhook=webhook_url)
    finally:
        if webhook_url and not os.path.isfile(result_path):
            died_payload = {
                "kind": "died",
                "task_id": packet["id"],
                "run_id": "",
                "cwd": packet["cwd"],
                "result_path": result_path,
                "seq": 0,
                "status": "",
                "question": "",
            }
            post_orch_webhook(webhook_url, died_payload)
    return ret


# --- self-test (fake sidecar, no model) ------------------------------------

FAKE_OK = r"""#!%s
import json, os, sys

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

emit({"jsonrpc": "2.0", "method": "lmp/ready", "params": {}})
for raw in sys.stdin:
    msg = json.loads(raw)
    method = msg.get("method")
    if method == "lmp/start":
        log = os.environ.get("LMP_EVENT_LOG")
        if log:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"kind": "write", "path": "a.py",
                                     "changed": "1", "tool": "write"}) + "\n")
        emit({"jsonrpc": "2.0", "method": "lmp/token",
              "params": {"channel": "answer", "text": "Added hello() to a.py."}})
        emit({"jsonrpc": "2.0", "method": "lmp/turn",
              "params": {"tool_name": "write", "tool_args": "a.py",
                         "tool_status": "ok", "text_tokens": 8}})
        emit({"jsonrpc": "2.0", "method": "lmp/run_end",
              "params": {"completed": True, "termination_reason": "completed",
                         "iterations": 1}})
    elif method == "lmp/shutdown":
        break
"""

FAKE_TIMEOUT = r"""#!%s
import json, sys, time

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

emit({"jsonrpc": "2.0", "method": "lmp/ready", "params": {}})
for raw in sys.stdin:
    msg = json.loads(raw)
    if msg.get("method") == "lmp/start":
        time.sleep(300)
    elif msg.get("method") == "lmp/shutdown":
        break
"""

FAKE_IRREVERSIBLE = r"""#!%s
import json, sys

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

ended = False
def end(reason):
    global ended
    if ended:
        return
    ended = True
    emit({"jsonrpc": "2.0", "method": "lmp/run_end",
          "params": {"completed": False, "termination_reason": reason,
                     "iterations": 1}})

emit({"jsonrpc": "2.0", "method": "lmp/ready", "params": {}})
for raw in sys.stdin:
    msg = json.loads(raw)
    method = msg.get("method")
    if method == "lmp/start":
        emit({"jsonrpc": "2.0", "method": "lmp/approval_request",
              "params": {"request_id": "r1", "run_id": "1", "tool_name": "bash",
                         "preview": "rm -rf /", "command": "rm -rf /",
                         "irreversible": True, "can_remember": False, "risk": 0.9}})
    elif method == "lmp/approve":
        approved = (msg.get("params") or {}).get("approved")
        end("denied" if not approved else "completed")
    elif method == "lmp/cancel":
        end("cancelled")
    elif method == "lmp/shutdown":
        break
"""


def _write_exec(path, template):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(template % sys.executable)
    os.chmod(path, 0o755)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


@contextlib.contextmanager
def _sidecar_env(path):
    old = os.environ.get("LMP_SIDECAR")
    old_qwen = os.environ.get("LMP_QWEN_DIR")
    os.environ["LMP_SIDECAR"] = path
    saved = ae.SIDECAR
    try:
        yield
    finally:
        ae.SIDECAR = saved
        if old is None:
            os.environ.pop("LMP_SIDECAR", None)
        else:
            os.environ["LMP_SIDECAR"] = old
        if old_qwen is None:
            os.environ.pop("LMP_QWEN_DIR", None)
        else:
            os.environ["LMP_QWEN_DIR"] = old_qwen


def self_test():
    os.environ["LMP_DAEMONIZE"] = "0"
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    with tempfile.TemporaryDirectory(prefix="piper-worker-st-") as tmp:
        model_dir = os.path.join(tmp, "model")
        os.makedirs(model_dir)

        # 1. Valid packet → ok, files_touched, message.
        ws = os.path.join(tmp, "smoke")
        os.makedirs(ws)
        with open(os.path.join(ws, "a.py"), "w", encoding="utf-8") as fh:
            fh.write("# workspace\n")
        fake = os.path.join(tmp, "fake_ok.py")
        _write_exec(fake, FAKE_OK)
        task = os.path.join(ws, "task.json")
        _write_json(task, {
            "id": "selftest-ok", "cwd": ws, "model_dir": model_dir,
            "prompt": "Add hello() to a.py.", "timeout_s": 30,
        })
        with _sidecar_env(fake):
            code = run_mission(ws, jsonl=False)
        result_path = os.path.join(ws, "result.json")
        check(code == EXIT_OK, f"ok packet must exit 0, got {code}")
        check(os.path.isfile(result_path), "ok packet must write result.json")
        result = {}
        if os.path.isfile(result_path):
            with open(result_path, encoding="utf-8") as fh:
                result = json.load(fh)
        check(result.get("status") == "ok", f"status=ok, got {result.get('status')!r}")
        check("a.py" in (result.get("files_touched") or []),
              f"files_touched must include a.py, got {result.get('files_touched')!r}")
        check(bool(result.get("message")), "message must be non-empty")

        # 2. timeout_s=5 + hanging sidecar → exit 2.
        hang_ws = os.path.join(tmp, "hang")
        os.makedirs(hang_ws)
        fake_to = os.path.join(tmp, "fake_timeout.py")
        _write_exec(fake_to, FAKE_TIMEOUT)
        hang_task = os.path.join(hang_ws, "task.json")
        _write_json(hang_task, {
            "id": "selftest-timeout", "cwd": hang_ws, "model_dir": model_dir,
            "prompt": "Never finish.", "timeout_s": 5,
        })
        began = time.monotonic()
        with _sidecar_env(fake_to):
            code = run_mission(hang_task, jsonl=False)
        took = time.monotonic() - began
        hang_result = {}
        hang_path = os.path.join(hang_ws, "result.json")
        if os.path.isfile(hang_path):
            with open(hang_path, encoding="utf-8") as fh:
                hang_result = json.load(fh)
        check(code == EXIT_TIMEOUT, f"timeout must exit 2, got {code}")
        check(hang_result.get("status") == "timeout",
              f"status=timeout, got {hang_result.get('status')!r}")
        check(took < 30, f"timeout must land promptly, took {took:.1f}s")

        # 3. Invalid packets → exit 3 (no sidecar required).
        bogus = os.path.join(tmp, "bogus.json")
        with open(bogus, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        check(run_mission(bogus) == EXIT_INVALID, "invalid JSON must exit 3")

        missing_cwd = os.path.join(tmp, "missing_cwd.json")
        _write_json(missing_cwd, {
            "id": "x", "cwd": os.path.join(tmp, "no-such-cwd"),
            "model_dir": model_dir, "prompt": "hi",
        })
        check(run_mission(missing_cwd) == EXIT_INVALID, "missing cwd must exit 3")

        missing_model = os.path.join(tmp, "missing_model.json")
        _write_json(missing_model, {
            "id": "x", "cwd": ws, "prompt": "hi",
        })
        os.environ.pop("LMP_QWEN_DIR", None)
        check(run_mission(missing_model) == EXIT_INVALID, "missing model_dir must exit 3")

        # 4. Irreversible approval → denied, error set, no hang.
        irr_ws = os.path.join(tmp, "irr")
        os.makedirs(irr_ws)
        fake_irr = os.path.join(tmp, "fake_irr.py")
        _write_exec(fake_irr, FAKE_IRREVERSIBLE)
        irr_task = os.path.join(irr_ws, "task.json")
        _write_json(irr_task, {
            "id": "selftest-irr", "cwd": irr_ws, "model_dir": model_dir,
            "prompt": "Delete everything.", "timeout_s": 30,
        })
        began = time.monotonic()
        with _sidecar_env(fake_irr):
            code = run_mission(irr_task, jsonl=False)
        took = time.monotonic() - began
        irr_result = {}
        irr_path = os.path.join(irr_ws, "result.json")
        if os.path.isfile(irr_path):
            with open(irr_path, encoding="utf-8") as fh:
                irr_result = json.load(fh)
        check(code == EXIT_ERROR, f"irreversible deny must exit 1, got {code}")
        check(irr_result.get("status") == "error",
              f"irreversible status=error, got {irr_result.get('status')!r}")
        check(bool(irr_result.get("error")), "irreversible must set error")
        err_text = str(irr_result.get("error") or "")
        check("irreversible" in err_text.lower() or "escalate" in err_text.lower(),
              f"error must mention irreversible/escalate, got {err_text!r}")
        check(took < 20, f"irreversible must not hang, took {took:.1f}s")

        # 5. Detached without webhook URL -> rejected with exit 3
        import io
        stderr_buf = io.StringIO()
        with contextlib.redirect_stderr(stderr_buf):
            code = main(["worker", "run", "--task", task, "--detach"])
        check(code == EXIT_INVALID, f"detached without webhook must exit 3, got {code}")
        stderr_det = stderr_buf.getvalue()
        check("piper_ui" in stderr_det and ".piper/orch_webhook" in stderr_det,
              f"detached without webhook must point at piper_ui / wake file, got {stderr_det!r}")

        # 6. Webhook notification on done
        import http.server
        import threading
        received_payloads = []

        class WebhookHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    received_payloads.append(json.loads(body.decode("utf-8")))
                except Exception:
                    pass
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')

            def log_message(self, format, *args):
                pass

        httpd = http.server.HTTPServer(("127.0.0.1", 0), WebhookHandler)
        server_port = httpd.server_port
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        webhook_url = f"http://127.0.0.1:{server_port}/wake"

        with _sidecar_env(fake):
            code = run_mission(task, jsonl=False, orch_webhook=webhook_url)
        check(code == EXIT_OK, f"webhook mission must exit 0, got {code}")
        check(len(received_payloads) >= 1, f"expected at least 1 webhook event, got {len(received_payloads)}")
        if received_payloads:
            p = received_payloads[-1]
            check(p.get("kind") == "done", f"expected kind=done, got {p.get('kind')!r}")
            check(p.get("task_id") == "selftest-ok", f"expected task_id=selftest-ok, got {p.get('task_id')!r}")
            check(p.get("status") == "ok", f"expected status=ok, got {p.get('status')!r}")

        # 7. Sidecar dies -> parent POSTs 'died' event
        fake_crash = os.path.join(tmp, "fake_crash.sh")
        with open(fake_crash, "w") as fh:
            fh.write("#!/bin/sh\nexit 42\n")
        os.chmod(fake_crash, 0o755)

        received_payloads.clear()
        crash_ws = os.path.join(tmp, "crash")
        os.makedirs(crash_ws)
        crash_task = os.path.join(crash_ws, "task.json")
        _write_json(crash_task, {
            "id": "selftest-crash", "cwd": crash_ws, "model_dir": model_dir,
            "prompt": "Crash now.", "timeout_s": 10,
        })
        with _sidecar_env(fake_crash):
            old_use_cpp = os.environ.get("LMP_USE_CPP_WORKER")
            os.environ["LMP_USE_CPP_WORKER"] = "1"
            try:
                code = main(["worker", "run", "--task", crash_task, "--orch-webhook", webhook_url])
            finally:
                if old_use_cpp is None:
                    os.environ.pop("LMP_USE_CPP_WORKER", None)
                else:
                    os.environ["LMP_USE_CPP_WORKER"] = old_use_cpp

        check(code == 42, f"crashed sidecar must return 42, got {code}")
        check(len(received_payloads) == 1, f"expected 1 died webhook event, got {len(received_payloads)}")
        if received_payloads:
            dp = received_payloads[0]
            check(dp.get("kind") == "died", f"expected kind=died, got {dp.get('kind')!r}")
            check(dp.get("task_id") == "selftest-crash", f"expected task_id=selftest-crash, got {dp.get('task_id')!r}")
            check(dp.get("status") == "", f"died event status must be empty, got {dp.get('status')!r}")

        # 8. Failed / non-completed task writing result.json -> POSTs 'stalled' event
        received_payloads.clear()
        with _sidecar_env(fake_irr):
            code = run_mission(irr_task, jsonl=False, orch_webhook=webhook_url)
        check(code == EXIT_ERROR, f"irr mission must exit 1, got {code}")
        check(len(received_payloads) >= 1, f"expected at least 1 webhook event, got {len(received_payloads)}")
        if received_payloads:
            sp = received_payloads[-1]
            check(sp.get("kind") == "stalled", f"expected kind=stalled, got {sp.get('kind')!r}")
            check(sp.get("task_id") == "selftest-irr", f"expected task_id=selftest-irr, got {sp.get('task_id')!r}")
            check(sp.get("status") == "error", f"expected status=error, got {sp.get('status')!r}")

        # 9. piper init: PIPER.md and .cursor/rules/piper-parent.mdc created, Godoer briefs untouched
        godoer_ws = os.path.join(tmp, "godoer_project")
        os.makedirs(godoer_ws)
        agents_md = os.path.join(godoer_ws, "AGENTS.md")
        gemini_md = os.path.join(godoer_ws, "GEMINI.md")
        claude_md = os.path.join(godoer_ws, "CLAUDE.md")
        for f, content in [(agents_md, "godoer agents"), (gemini_md, "godoer gemini"), (claude_md, "godoer claude")]:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(content)

        init_stdout = io.StringIO()
        with contextlib.redirect_stdout(init_stdout):
            init_code = main(["init", godoer_ws])
        check(init_code == EXIT_OK, f"piper init must exit 0, got {init_code}")
        expected_msg = "piper init: PIPER.md and .cursor/rules/piper-parent.mdc. Godoer briefs left alone. Stay attached, or pass --orch-webhook."
        check(expected_msg in init_stdout.getvalue(),
              f"piper init must print contract line, got {init_stdout.getvalue()!r}")

        piper_md = os.path.join(godoer_ws, "PIPER.md")
        cursor_mdc = os.path.join(godoer_ws, ".cursor", "rules", "piper-parent.mdc")
        check(os.path.isfile(piper_md), "PIPER.md must exist after init")
        check(os.path.isfile(cursor_mdc), ".cursor/rules/piper-parent.mdc must exist after init")

        with open(piper_md, "r", encoding="utf-8") as fh:
            p_content = fh.read()
        check("Piper worker wake standard" in p_content, "PIPER.md must contain wake standard")

        with open(cursor_mdc, "r", encoding="utf-8") as fh:
            c_content = fh.read()
        check("piper: you are the parent." in c_content, "cursor rule must contain parent paragraph")

        with open(agents_md, "r", encoding="utf-8") as fh:
            check(fh.read() == "godoer agents", "AGENTS.md must be untouched")
        with open(gemini_md, "r", encoding="utf-8") as fh:
            check(fh.read() == "godoer gemini", "GEMINI.md must be untouched")
        with open(claude_md, "r", encoding="utf-8") as fh:
            check(fh.read() == "godoer claude", "CLAUDE.md must be untouched")

        # Init twice does not duplicate or corrupt
        with contextlib.redirect_stdout(io.StringIO()):
            init_code2 = main(["init", godoer_ws])
        check(init_code2 == EXIT_OK, f"second init must exit 0, got {init_code2}")
        with open(piper_md, "r", encoding="utf-8") as fh:
            check(fh.read() == p_content, "second init must not duplicate PIPER.md")
        with open(cursor_mdc, "r", encoding="utf-8") as fh:
            check(fh.read() == c_content, "second init must not duplicate cursor rule")

        # Test worker run prints parent paragraph to stderr even when PIPER.md is missing
        missing_piper_ws = os.path.join(tmp, "missing_piper")
        os.makedirs(missing_piper_ws)
        task_no_piper = os.path.join(missing_piper_ws, "task.json")
        _write_json(task_no_piper, {
            "id": "test-no-piper", "cwd": missing_piper_ws, "model_dir": model_dir,
            "prompt": "Test parent banner.", "timeout_s": 30,
        })
        run_stderr = io.StringIO()
        old_use_cpp = os.environ.get("LMP_USE_CPP_WORKER")
        os.environ["LMP_USE_CPP_WORKER"] = "0"
        try:
            with _sidecar_env(fake), contextlib.redirect_stderr(run_stderr):
                main(["worker", "run", "--task", task_no_piper])
        finally:
            if old_use_cpp is None:
                os.environ.pop("LMP_USE_CPP_WORKER", None)
            else:
                os.environ["LMP_USE_CPP_WORKER"] = old_use_cpp
        check(
            "piper: you are the parent. Stay attached and read the exit and result.json." in run_stderr.getvalue(),
            f"worker run must print parent paragraph to stderr even without PIPER.md, got {run_stderr.getvalue()!r}"
        )

        # 10. Non-HTTP/HTTPS webhook URL -> refused
        refused_buf = io.StringIO()
        with contextlib.redirect_stderr(refused_buf):
            res = post_orch_webhook("file:///etc/passwd", {"kind": "done"})
        check(res is False, "file:// webhook URL must be refused")
        check("refusing webhook URL with non-http(s) scheme" in refused_buf.getvalue(),
              f"expected non-http warning in stderr, got {refused_buf.getvalue()!r}")

        # 11. collect_files_touched handles errors on stderr without stdout clutter
        corrupt_log = os.path.join(tmp, "unreadable_events.ndjson")
        with open(corrupt_log, "w", encoding="utf-8") as fh:
            fh.write("line\n")
        os.chmod(corrupt_log, 0o000)
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                res_touched = collect_files_touched(corrupt_log, tmp)
        finally:
            os.chmod(corrupt_log, 0o644)
        check(res_touched == [], f"corrupt log must return [], got {res_touched!r}")
        check(stdout_buf.getvalue() == "", f"collect_files_touched must not print to stdout, got {stdout_buf.getvalue()!r}")
        check("piper: failed to parse event log:" in stderr_buf.getvalue(),
              f"expected 'piper: failed to parse event log:' in stderr, got {stderr_buf.getvalue()!r}")

        # 12. PR-S1: MCP path harvest, stalled status helper, incomplete message trim
        mcp_log = os.path.join(tmp, "mcp_events.ndjson")
        with open(mcp_log, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "kind": "tool_call", "tool": "godot_world",
                "arg.file": "scenes/Main.tscn",
            }) + "\n")
            fh.write(json.dumps({
                "kind": "workspace_freshness", "why": "remote_tool",
                "tool": "godot_world", "writes": "1",
            }) + "\n")
            fh.write(json.dumps({
                "kind": "tool_result", "tool": "godot_world",
                "status": "ok", "path": "world/player.gd",
            }) + "\n")
        mcp_touched = collect_files_touched(mcp_log, tmp)
        check(mcp_touched == ["scenes/Main.tscn", "world/player.gd"],
              f"MCP paths must be harvested, got {mcp_touched!r}")
        check(log_has_remote_tool_write(mcp_log),
              "remote_tool freshness must be detected")
        check(is_stalled_termination("max_turns"), "max_turns is stalled")
        check(is_stalled_termination("stalled"), "stalled is stalled")
        check(not is_stalled_termination("ended"), "ended is not stalled")
        diary = ("Turn narration line: still poking at the scene.\n" * 20)
        diary += "Final narration before the turn budget expires."
        trimmed = compose_message(
            diary, "stalled", "max_turns", ["a.tscn"], "agent did not complete (max_turns)",
            completed=False,
        )
        check(len(trimmed) <= 480, f"incomplete message must be short, got {len(trimmed)}")
        check(" … " in trimmed, f"incomplete message must use first/last ellipsis, got {trimmed!r}")
        check("Final narration" in trimmed, f"incomplete message must keep last text, got {trimmed!r}")
        finished = compose_message(
            diary, "stalled", "max_turns", [], "err",
            finish_summary="Shipped the scene edit.", completed=False,
        )

        # 13. Bakeoff journal hygiene: archive prior events.jsonl before overwrite
        bake_dir = os.path.join(tmp, "bakeoff_try")
        os.makedirs(bake_dir)
        live = os.path.join(bake_dir, "events.jsonl")
        with open(live, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": "run_end", "seq": 1}) + "\n")
        archived = archive_prior_events(bake_dir)
        check(archived is not None and os.path.isfile(archived),
              f"archive_prior_events must write a timestamped copy, got {archived!r}")
        check(not os.path.isfile(live), "live events.jsonl must be cleared after archive")
        check(os.path.basename(archived).startswith("events-") and archived.endswith(".jsonl"),
              f"archive name must be events-<UTC>.jsonl, got {archived!r}")
        check(finished == "Shipped the scene edit.",
              f"finish summary must win, got {finished!r}")
        merged = merge_files_touched(["a.tscn"], ["a.tscn", "b.gd"])
        check(merged == ["a.tscn", "b.gd"], f"merge_files_touched unique union, got {merged!r}")

        # 14. piper answer writes known-right answer.json (no freehand paste)
        ans_dir = os.path.join(tmp, "answer_dir")
        os.makedirs(ans_dir)
        check(main(["answer", "allow", "--dir", ans_dir]) == EXIT_OK, "answer allow must exit 0")
        with open(os.path.join(ans_dir, "answer.json"), encoding="utf-8") as fh:
            ans_body = json.load(fh)
        check(ans_body == {"text": "allow"}, f"answer allow payload, got {ans_body!r}")
        check(main(["answer", "deny", "--dir", ans_dir]) == EXIT_OK, "answer deny must exit 0")
        with open(os.path.join(ans_dir, "answer.json"), encoding="utf-8") as fh:
            ans_body = json.load(fh)
        check(ans_body == {"text": "deny"}, f"answer deny payload, got {ans_body!r}")
        check(main(["answer", "--text", "use option B", "--dir", ans_dir]) == EXIT_OK,
              "answer --text must exit 0")
        with open(os.path.join(ans_dir, "answer.json"), encoding="utf-8") as fh:
            ans_body = json.load(fh)
        check(ans_body == {"text": "use option B"}, f"answer text payload, got {ans_body!r}")
        check(main(["answer", "--dir", ans_dir]) == EXIT_INVALID,
              "answer without action/text must exit 3")

        # 15. piper packet emits loadable task.json
        pkt_out = os.path.join(tmp, "emitted_task.json")
        check(
            main(["packet", "--id", "emit-1", "--cwd", tmp, "--prompt", "Ship X.",
                  "--out", pkt_out, "--model-dir", model_dir, "--check", "true"]) == EXIT_OK,
            "packet emit must exit 0",
        )
        loaded = load_packet(pkt_out)
        check(loaded["id"] == "emit-1", f"emitted id, got {loaded.get('id')!r}")
        check(loaded["cwd"] == os.path.abspath(tmp), f"emitted cwd, got {loaded.get('cwd')!r}")
        check("Ship X." in loaded["prompt"], f"emitted prompt, got {loaded.get('prompt')!r}")

        # 16. .piper/orch_webhook file is discovered (no panel copy)
        wake_root = os.path.join(tmp, "wake_ws")
        os.makedirs(wake_root)
        wake_url = "http://127.0.0.1:9/wake"
        written = write_orch_webhook_file(wake_root, wake_url)
        check(os.path.isfile(written), "wake file must exist")
        os.environ.pop("LMP_ORCH_WEBHOOK", None)
        resolved = resolve_orch_webhook(
            None,
            {"orch_webhook": "", "cwd": wake_root, "task_dir": wake_root,
             "result_path": os.path.join(wake_root, "result.json")},
        )
        check(resolved == wake_url, f"file wake URL must resolve, got {resolved!r}")
        wake_stdout = io.StringIO()
        with contextlib.redirect_stdout(wake_stdout):
            wake_rc = main(["wake-url", "--dir", wake_root])
        check(wake_rc == EXIT_OK, f"wake-url must exit 0, got {wake_rc}")
        check(wake_stdout.getvalue().strip() == wake_url,
              f"wake-url must print file URL, got {wake_stdout.getvalue()!r}")

        # 16b. wake-url --task must not require model_dir (packet emit is optional there)
        pkt_no_model = os.path.join(tmp, "wake_task_no_model.json")
        check(
            main(["packet", "--id", "wake-nm", "--cwd", wake_root, "--prompt", "x",
                  "--out", pkt_no_model]) == EXIT_OK,
            "packet without model_dir must exit 0",
        )
        wake_stdout2 = io.StringIO()
        with contextlib.redirect_stdout(wake_stdout2):
            wake_rc2 = main(["wake-url", "--task", pkt_no_model])
        check(wake_rc2 == EXIT_OK, f"wake-url --task without model_dir must exit 0, got {wake_rc2}")
        check(wake_stdout2.getvalue().strip() == wake_url,
              f"wake-url --task must print file URL, got {wake_stdout2.getvalue()!r}")

        # 17. piper_ui wake-file helper matches worker discovery path
        try:
            import piper_ui as pui
            ui_wake = pui.write_wake_url_file(wake_root, "http://127.0.0.1:8765/wake")
            check(ui_wake == written or os.path.isfile(ui_wake),
                  f"ui wake file path must match harness convention, got {ui_wake!r}")
            with open(ui_wake, encoding="utf-8") as fh:
                check(fh.read().strip() == "http://127.0.0.1:8765/wake",
                      "ui wake file must contain the URL")
        except Exception as exc:
            check(False, f"piper_ui wake helper must import/run: {exc}")

        # 18. review card is deterministic from result.json (no freehand rubric)
        rev_dir = os.path.join(tmp, "review_dir")
        os.makedirs(rev_dir)
        rev_path = os.path.join(rev_dir, "result.json")
        with open(rev_path, "w", encoding="utf-8") as fh:
            json.dump(result_shell(
                task_id="rev-1", cwd=rev_dir, model_dir=model_dir, status="ok",
                message="Shipped the helper.", files_touched=["a.py"],
                diff_stat={"insertions": 3, "deletions": 1, "files": 1},
            ), fh)
        rev_stdout = io.StringIO()
        with contextlib.redirect_stdout(rev_stdout):
            rev_rc = main(["review", "--result", rev_path])
        check(rev_rc == EXIT_OK, f"review must exit 0, got {rev_rc}")
        rev_out = rev_stdout.getvalue()
        check("verdict:  PASS" in rev_out, f"review card must PASS, got {rev_out!r}")
        check("rev-1" in rev_out and "a.py" in rev_out, f"review card must show task/files, got {rev_out!r}")
        check(build_answer_payload(action="approved") == {"text": "allow"},
              "approved must normalize to allow")
        check(build_answer_payload(action="denied") == {"text": "deny"},
              "denied must normalize to deny")

        # 19. dispatch → wait → review card (fake sidecar)
        disp_ws = os.path.join(tmp, "dispatch_ws")
        os.makedirs(disp_ws)
        disp_task = os.path.join(disp_ws, "task.json")
        with open(disp_task, "w", encoding="utf-8") as fh:
            json.dump({
                "id": "disp-1",
                "cwd": disp_ws,
                "prompt": "Ship review card.",
                "model_dir": model_dir,
                "result_path": os.path.join(disp_ws, "result.json"),
                "timeout_s": 30,
                "auto_approve_exec": True,
                "auto_approve_writes": True,
                "auto_approve_irreversible": True,
            }, fh)
        disp_stdout = io.StringIO()
        with _sidecar_env(fake), contextlib.redirect_stdout(disp_stdout):
            disp_rc = main(["dispatch", "--task", disp_task])
        check(disp_rc == EXIT_OK, f"dispatch must exit 0, got {disp_rc}")
        disp_out = disp_stdout.getvalue()
        check("verdict:  PASS" in disp_out, f"dispatch must print PASS card, got {disp_out!r}")
        check(os.path.isfile(os.path.join(disp_ws, "result.json")),
              "dispatch must leave result.json")

        httpd.shutdown()

    for line in failures:
        print(f"  FAIL: {line}")
    print(f"  piper_worker self-test: 20 scenario(s), {len(failures)} failure(s)")
    return EXIT_ERROR if failures else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
