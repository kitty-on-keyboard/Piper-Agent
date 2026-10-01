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

# Worker telemetry (MLX breadcrumbs, heartbeats, `piper:` notices) goes to a per-slice
# file beside result.json, not into the parent's output: ~1 KB per model turn pushed
# the review card past tool-output caps. One file per run, the previous run's kept.
WORKER_STDERR_LOG = "worker.stderr.log"
WORKER_STDERR_PREV_LOG = "worker.stderr.prev.log"


def worker_stderr_log_path(result_path):
    return os.path.join(os.path.dirname(os.path.abspath(result_path)), WORKER_STDERR_LOG)


def open_worker_stderr_log(result_path):
    """Fresh log fd for this run, or None to inherit stderr.

    The previous run's log is rotated to worker.stderr.prev.log rather than appended
    to, so a card never mixes two runs. The engine writes the fd directly, so its
    last breadcrumb survives a SIGKILL. LMP_WORKER_STDERR=inherit keeps the old
    behaviour for a human watching `piper run` in a terminal.
    """
    if os.environ.get("LMP_WORKER_STDERR", "").strip().lower() == "inherit":
        return None
    path = worker_stderr_log_path(result_path)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            os.replace(path, os.path.join(os.path.dirname(path), WORKER_STDERR_PREV_LOG))
        return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_APPEND, 0o644)
    except OSError as exc:
        print(f"piper: cannot open worker log {path}: {exc}; worker stderr stays inline",
              file=sys.stderr)
        return None


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
        packet, _roots = read_task_roots(task_arg)
        task_path = resolve_task_json_path(task_arg)
        return os.path.dirname(result_path_from_light_packet(packet, task_path))
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


def mcp_json_path(cwd):
    return os.path.join(os.path.abspath(cwd), ".mcp.json")


def list_mcp_servers(cwd):
    """Return ordered server names from workspace `.mcp.json` (empty if absent)."""
    path = mcp_json_path(cwd)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise PacketError(f".mcp.json is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PacketError(".mcp.json must be an object")
    servers = data.get("mcpServers")
    if servers is None:
        return []
    if not isinstance(servers, dict):
        raise PacketError(".mcp.json mcpServers must be an object")
    return [str(name) for name in servers.keys() if str(name).strip()]


def normalize_trust_mcp(names):
    """Deduplicate trust_mcp names while preserving first-seen order."""
    out = []
    seen = set()
    for raw in names or []:
        name = str(raw).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def validate_trust_mcp_names(cwd, names):
    """Consent stays explicit: named servers must exist in cwd `.mcp.json`."""
    wanted = normalize_trust_mcp(names)
    if not wanted:
        return []
    available = list_mcp_servers(cwd)
    if not available:
        raise PacketError(
            f"trust_mcp names server(s), but .mcp.json was not found or has no "
            f"mcpServers in {os.path.abspath(cwd)}"
        )
    missing = [n for n in wanted if n not in available]
    if missing:
        avail_s = ", ".join(available)
        miss_s = ", ".join(missing)
        raise PacketError(
            f"trust_mcp names unknown server(s): {miss_s} "
            f"(available: {avail_s})"
        )
    return wanted


def default_slice_task_path(cwd, task_id):
    """Default packet location: <cwd>/.piper/slices/<id>/task.json."""
    sid = str(task_id or "").strip()
    if not sid:
        raise PacketError("id must be a non-empty string")
    return os.path.join(
        os.path.abspath(os.path.expanduser(cwd)), ".piper", "slices", sid, "task.json"
    )


def resolve_emit_model_dir(model_dir=None):
    """Bake a model at emit time. Dispatch must not be the first failure."""
    raw = model_dir if model_dir else os.environ.get("LMP_QWEN_DIR", "")
    if not isinstance(raw, str) or not raw.strip():
        raise PacketError("missing model_dir (pass --model-dir or set LMP_QWEN_DIR)")
    path = os.path.abspath(os.path.expanduser(raw.strip()))
    if not os.path.isdir(path):
        raise PacketError(f"model_dir is not a directory: {path}")
    return path


def write_active_slice(cwd, task_id, task_path, result_path):
    """Point the watch UI at the slice just emitted. Parents do not author this."""
    root = os.path.abspath(os.path.expanduser(cwd))
    path = os.path.join(root, ".piper", "active.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "id": str(task_id).strip(),
        "task_path": os.path.abspath(task_path),
        "result_path": os.path.abspath(result_path),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def append_orch_event(result_path, event):
    """Append one orch.jsonl line beside result.json. Parents do not author this."""
    directory = os.path.dirname(os.path.abspath(result_path))
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "orch.jsonl")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    return path


def emit_task_packet(*, task_id, cwd, prompt, out_path, model_dir=None,
                     check=None, timeout_s=600, result_path=None,
                     auto_approve_irreversible=True, trust_mcp=None,
                     prompt_file=None, max_iterations=None, check_timeout_s=None):
    """Emit a correctly shaped task.json — no freehand JSON from an LLM."""
    cwd = os.path.abspath(os.path.expanduser(cwd))
    if not os.path.isdir(cwd):
        raise PacketError(f"cwd is not a directory: {cwd}")
    if not isinstance(task_id, str) or not task_id.strip():
        raise PacketError("id must be a non-empty string")
    if not isinstance(prompt, str) or not prompt.strip():
        raise PacketError("prompt must be a non-empty string")
    resolved_model = resolve_emit_model_dir(model_dir)
    out_path = os.path.abspath(os.path.expanduser(out_path))
    packet = {
        "id": task_id.strip(),
        "cwd": cwd,
        "prompt": prompt,
        "model_dir": resolved_model,
        "auto_approve_exec": True,
        "auto_approve_writes": True,
        "auto_approve_irreversible": bool(auto_approve_irreversible),
        "timeout_s": float(timeout_s),
    }
    if check:
        packet["check"] = check
    if check_timeout_s is not None:
        if isinstance(check_timeout_s, bool) or not isinstance(check_timeout_s, (int, float)) \
                or check_timeout_s <= 0:
            raise PacketError("check_timeout_s must be a positive number")
        packet["check_timeout_s"] = float(check_timeout_s)
    if max_iterations is not None:
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) \
                or max_iterations < 1:
            raise PacketError("max_iterations must be a positive integer")
        packet["max_iterations"] = max_iterations
    if result_path:
        packet["result_path"] = os.path.abspath(os.path.expanduser(result_path))
    else:
        packet["result_path"] = os.path.join(os.path.dirname(out_path), "result.json")
    if trust_mcp:
        packet["trust_mcp"] = validate_trust_mcp_names(cwd, trust_mcp)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if prompt_file:
        src = os.path.abspath(os.path.expanduser(prompt_file))
        dest = os.path.join(os.path.dirname(out_path), "prompt.md")
        if src != dest:
            shutil.copyfile(src, dest)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(packet, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, out_path)
    write_active_slice(cwd, packet["id"], out_path, packet["result_path"])
    return out_path, packet


PROGRESS_DEFAULT_RELPATH = os.path.join(".piper", "progress.log")


def format_progress_line(slice_id, verdict, note=""):
    """Known-right one-line progress memory — parents should not freehand the shape."""
    sid = str(slice_id or "").strip()
    if not sid:
        raise ValueError("progress requires a non-empty slice id")
    ver = str(verdict or "").strip().lower()
    aliases = {
        "pass": "pass", "ok": "pass", "passed": "pass", "success": "pass",
        "fail": "fail", "failed": "fail", "error": "fail",
        "stalled": "stalled", "stall": "stalled",
        "timeout": "timeout", "timedout": "timeout", "timed-out": "timeout",
        "died": "died", "skip": "skip", "skipped": "skip",
    }
    if ver not in aliases:
        raise ValueError(
            f"unknown progress verdict {verdict!r}; "
            "use pass|fail|stalled|timeout|died|skip"
        )
    note_s = " ".join(str(note or "").strip().split())
    if note_s:
        return f"{sid} | {aliases[ver]} | {note_s}"
    return f"{sid} | {aliases[ver]}"


def append_progress_log(path, line):
    """Append one progress line (creates parent dirs). Returns absolute path."""
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    text = line if line.endswith("\n") else line + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(text)
    return path


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
    out_path = args.out or default_slice_task_path(args.cwd, args.id)
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
            trust_mcp=getattr(args, "trust_mcp", None),
            prompt_file=args.prompt_file,
            max_iterations=getattr(args, "max_iterations", None),
            check_timeout_s=getattr(args, "check_timeout_s", None),
        )
    except PacketError as exc:
        print(f"piper packet: {exc}", file=sys.stderr)
        return EXIT_INVALID
    print(path)
    return EXIT_OK


def cmd_mcp_list(args):
    """List `.mcp.json` server names — no freehand guessing for trust_mcp."""
    cwd = os.path.abspath(os.path.expanduser(args.cwd or "."))
    if not os.path.isdir(cwd):
        print(f"piper mcp-list: cwd is not a directory: {cwd}", file=sys.stderr)
        return EXIT_INVALID
    try:
        names = list_mcp_servers(cwd)
    except PacketError as exc:
        print(f"piper mcp-list: {exc}", file=sys.stderr)
        return EXIT_INVALID
    path = mcp_json_path(cwd)
    if getattr(args, "json", False):
        print(json.dumps({
            "cwd": cwd,
            "mcp_json": path if os.path.isfile(path) else None,
            "servers": names,
        }, indent=2))
    else:
        if not names:
            print(f"piper mcp-list: no mcpServers in {path}", file=sys.stderr)
            return EXIT_INVALID
        for name in names:
            print(name)
    return EXIT_OK


def cmd_ui(args):
    """Start the watch UI. Same process as scripts/piper_ui.py."""
    import piper_ui
    cwd = os.path.abspath(os.path.expanduser(getattr(args, "cwd", None) or "."))
    if not os.path.isdir(cwd):
        print(f"piper ui: cwd is not a directory: {cwd}", file=sys.stderr)
        return EXIT_INVALID
    piper_ui.run_server(
        workspace_dir=cwd,
        port=int(getattr(args, "port", 8765) or 8765),
        open_browser=not getattr(args, "no_open", False),
    )
    return EXIT_OK


def cmd_progress(args):
    """Append one progress line — vision loop memory without freehand paste."""
    try:
        line = format_progress_line(args.id, args.verdict, getattr(args, "note", "") or "")
    except ValueError as exc:
        print(f"piper progress: {exc}", file=sys.stderr)
        return EXIT_INVALID
    if getattr(args, "file", None):
        path = os.path.abspath(os.path.expanduser(args.file))
    else:
        root = os.path.abspath(os.path.expanduser(args.dir or "."))
        path = os.path.join(root, PROGRESS_DEFAULT_RELPATH)
    try:
        written = append_progress_log(path, line)
    except OSError as exc:
        print(f"piper progress: cannot write {path}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if getattr(args, "json", False):
        print(json.dumps({"path": written, "line": line}, indent=2))
    else:
        print(written)
        print(line)
    return EXIT_OK


def read_task_roots(task_arg):
    """Lightweight task.json read for wake discovery — no model_dir required.

    Accepts a task.json file or a directory containing task.json (same as run/dispatch).
    """
    path = os.path.abspath(os.path.expanduser(task_arg))
    if os.path.isdir(path):
        candidate = os.path.join(path, "task.json")
        if os.path.isfile(candidate):
            path = candidate
        else:
            raise PacketError(f"task packet not found: {candidate}")
    elif not os.path.isfile(path):
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
    """Cheap parent verdict: PASS / UNVERIFIED / FAIL / STALLED / DIED.

    UNVERIFIED: the model finished but nothing checked it (no `--check`). A model's
    own "done" is not a pass.
    """
    if result is None:
        return "DIED"
    status = str(result.get("status") or "").lower()
    if status == "ok" and exit_code == EXIT_OK:
        test = result.get("test")
        if isinstance(test, dict) and test.get("ran"):
            return "PASS"
        return "UNVERIFIED"
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


CARD_CHECK_LINES = 8
CARD_LOG_PIPER_LINES = 3
CARD_DIED_LOG_LINES = 8
CARD_LINE_CAP = 160


def _card_clip(line, cap=CARD_LINE_CAP):
    line = line.rstrip()
    return line if len(line) <= cap else line[:cap - 3] + "..."


def _last_nonempty_lines(text, limit):
    lines = [ln for ln in str(text or "").splitlines() if ln.strip()]
    return lines[-limit:] if limit > 0 else []


def _read_log_lines(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().splitlines()
    except OSError:
        return []


def _card_block(label, lines):
    """`label` on the first line, the rest aligned under it, each behind `| `."""
    out = ""
    for i, line in enumerate(lines):
        head = f"{label:<10}" if i == 0 else " " * 10
        out += f"{head}| {_card_clip(line)}\n"
    return out


def format_review_card(result, *, exit_code, result_path):
    """Deterministic review card — parents should not freehand the rubric.

    Compact but complete: on anything but PASS it carries the evidence (error, the
    failing check's tail, the worker's own `piper:` notices) so the parent does not
    have to open result.json or the log to learn why.
    """
    verdict = review_verdict(result, exit_code)
    log_path = worker_stderr_log_path(result_path) if result_path else ""
    if result is None:
        card = (
            "── piper review ─────────────────────────\n"
            f"result:   {result_path or '(none)'}\n"
            f"exit:     {exit_code}\n"
        )
        log_lines = _read_log_lines(log_path) if log_path else []
        if log_lines:
            card += f"log:      {log_path}\n"
            card += _card_block("log tail:", _last_nonempty_lines("\n".join(log_lines),
                                                                   CARD_DIED_LOG_LINES))
        return card + (
            f"verdict:  {verdict}  (no result.json)\n"
            "─────────────────────────────────────────"
        )
    files = result.get("files_touched") or []
    if isinstance(files, list):
        files_s = ", ".join(str(x) for x in files) if files else "(none)"
    else:
        files_s = str(files)
    test = result.get("test") or {}
    check_lines = []
    triage_lines = []
    if isinstance(test, dict) and test.get("ran"):
        test_s = f"{format_check_outcome(test)} cmd={test.get('command')!r}"
        if test.get("exit_code") != 0 or test.get("timed_out") or test.get("could_not_run"):
            triage_lines = format_triage_lines(test)
            check_lines = _last_nonempty_lines(test.get("output_tail"),
                                               CARD_CHECK_LINES - len(triage_lines))
    elif isinstance(test, dict):
        test_s = "not run"
    else:
        test_s = str(test)
    msg = str(result.get("message") or "").strip().replace("\n", " ")
    if len(msg) > 160:
        msg = msg[:157] + "..."
    loop_s = format_loop_line(result)
    passed = verdict == "PASS"
    error_s = str(result.get("error") or "").strip().replace("\n", " ")
    notices = []
    if not passed and log_path:
        notices = [ln for ln in _read_log_lines(log_path) if ln.startswith("piper:")]
        notices = notices[-CARD_LOG_PIPER_LINES:]
    return (
        "── piper review ─────────────────────────\n"
        f"task:     {result.get('task_id') or '(unknown)'}\n"
        f"status:   {result.get('status')}\n"
        f"exit:     {exit_code}\n"
        + (f"error:    {_card_clip(error_s)}\n" if error_s and not passed else "")
        + f"run:      {format_run_line(result)}\n"
        f"message:  {msg or '(empty)'}\n"
        f"files:    {files_s}\n"
        f"diff:     {format_diff_stat(result.get('diff_stat'))}\n"
        f"test:     {test_s}\n"
        + _card_block("check:", check_lines)
        + "".join(triage_lines)
        + (f"loop:     {loop_s}\n" if loop_s else "")
        + _card_block("worker:", notices)
        + f"git_diff: {result.get('git_diff_path') or '(none)'}\n"
        f"verdict:  {verdict}\n"
        "─────────────────────────────────────────"
    )


def format_check_outcome(test):
    """exit=N, or what stopped the check when it never produced a real exit code."""
    seconds = test.get("seconds")
    if test.get("timed_out"):
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
            return f"timed out after {seconds:.0f}s"
        return "timed out"
    if test.get("could_not_run"):
        return f"could not run (exit {test.get('exit_code')})"
    return f"exit={test.get('exit_code')}"


def format_triage_lines(test):
    """At most three lines from the check's log_triage block: failing tests, the first
    located primary diagnostic, and where the full log is."""
    lines = []
    triage = test.get("triage")
    if isinstance(triage, dict):
        failing = [str(t) for t in (triage.get("failing_tests") or []) if t]
        if failing:
            lines.append(f"failing:  {_card_clip(', '.join(failing))}\n")
        for diag in triage.get("primary") or []:
            if not isinstance(diag, dict) or not diag.get("path"):
                continue
            where = str(diag["path"])
            line_no = diag.get("line")
            if isinstance(line_no, int) and not isinstance(line_no, bool) and line_no > 0:
                where += f":{line_no}"
            message = str(diag.get("message") or "").strip()
            lines.append(f"at:       {_card_clip(where + (': ' + message if message else ''))}\n")
            break
    if test.get("output_path"):
        lines.append(f"log:      {test['output_path']}\n")
    return lines


def format_run_line(result):
    turns = result.get("turns")
    turns_s = str(turns) if isinstance(turns, int) and not isinstance(turns, bool) else "?"
    wall = result.get("wall_seconds")
    wall_s = f"{wall:.0f}s" if isinstance(wall, (int, float)) and not isinstance(wall, bool) else "?"
    return f"turns={turns_s} wall={wall_s}"


def format_loop_line(result):
    """How the loop stopped, when that is not the normal ending or a check vouched for it.

    A pass promoted by the check rests on the check alone, so the card says so.
    """
    reason = str(result.get("termination_reason") or "")
    promoted = bool(result.get("promoted_by_check"))
    if not promoted and reason in ("", "ended", "plan_ready"):
        return ""
    line = reason or "no_run_end"
    turns = result.get("turns")
    if isinstance(turns, int) and not isinstance(turns, bool):
        line += f" after {turns} turn{'s' if turns != 1 else ''}"
    if promoted:
        line += "; ok because check passed"
    return line


def resolve_result_path(result_arg=None, task_arg=None):
    if result_arg:
        return os.path.abspath(os.path.expanduser(result_arg))
    if task_arg:
        packet, _roots = read_task_roots(task_arg)
        task_path = resolve_task_json_path(task_arg)
        return result_path_from_light_packet(packet, task_path)
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
    append_orch_event(result_path, {
        "kind": "dispatch",
        "id": packet["id"],
        "text": f"dispatched {packet['id']}",
    })
    run_args = argparse.Namespace(
        task=args.task,
        jsonl=bool(getattr(args, "jsonl", False)),
        orch_webhook=getattr(args, "orch_webhook", None),
        auto_approve_all=bool(getattr(args, "auto_approve_all", False)),
        auto_approve_irreversible=bool(getattr(args, "auto_approve_irreversible", False))
        and not bool(getattr(args, "auto_approve_all", False)),
        detach=False,
    )
    # Attached by construction: dispatch owns the wait and prints the card, so
    # neither LMP_DAEMONIZE=1 nor how this process's stdio happens to be wired
    # can turn it into a detached launch.
    code = cmd_run(run_args, attached_only=True)
    result = load_result_file(result_path)
    card = format_review_card(result, exit_code=code, result_path=result_path)
    append_orch_event(result_path, {
        "kind": "review",
        "id": packet["id"],
        "verdict": review_verdict(result, code),
        "text": card,
    })
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


def _default_worker_socket():
    env = os.environ.get("LMP_WORKER_SOCKET", "").strip()
    if env:
        return env
    home = os.environ.get("HOME", "")
    if home:
        return os.path.join(home, ".piper", "worker.sock")
    return "/tmp/piper_worker.sock"


def _is_daemon_alive(socket_path):
    if not socket_path or not os.path.exists(socket_path):
        return False
    import socket
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect(socket_path)
        s.sendall(b'{"method":"ping"}\n')
        resp = s.recv(1024)
        s.close()
        data = json.loads(resp.decode("utf-8"))
        return data.get("status") == "ok"
    except Exception:
        try:
            s.close()
        except Exception:
            pass
        return False


def _forward_task_to_daemon(sock_path, task_path):
    import socket
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    req = {
        "method": "run",
        "task": os.path.abspath(task_path),
        "auto_approve_irreversible": True,
        "auto_approve_all": True,
    }
    s.sendall((json.dumps(req) + "\n").encode("utf-8"))
    accum = b""
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        accum += chunk
    s.close()
    return accum.decode("utf-8", errors="replace")


def cmd_distill(args):
    """Distill telemetry incidents into root-cause diagnosis using local MLX model."""
    incidents_path = os.path.abspath(args.incidents)
    if not os.path.isfile(incidents_path):
        print(f"piper distill: incidents file not found: {incidents_path}", file=sys.stderr)
        return EXIT_INVALID

    try:
        with open(incidents_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"piper distill: invalid incidents JSON: {exc}", file=sys.stderr)
        return EXIT_INVALID

    incidents = data.get("incidents", [])
    if not incidents:
        print("piper distill: zero incidents recorded. All systems green.")
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({"status": "clean", "diagnoses": []}, f, indent=2)
        return EXIT_OK

    sock_path = getattr(args, "socket", None) or _default_worker_socket()
    daemon_online = _is_daemon_alive(sock_path)

    allow_cold = getattr(args, "allow_cold", False)
    if not daemon_online and not allow_cold:
        msg = (
            f"piper distill: keep-warm daemon is not active on {sock_path}.\n"
            f"Start the warm daemon with: piper worker serve\n"
            f"Or pass '--allow-cold' to explicitly permit cold loading weights into RAM."
        )
        print(msg, file=sys.stderr)
        out_path = args.out or os.path.join(os.path.dirname(incidents_path), "distilled_diagnosis.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "status": "skipped_daemon_offline",
                "message": msg,
                "diagnoses": [],
            }, f, indent=2)
        return EXIT_OK

    model_dir = args.model_dir or os.environ.get("LMP_QWEN_DIR") or "/Users/dev/Desktop/Models/Qwen3.8-27B-MLX-4bit"
    if not daemon_online and not os.path.isdir(model_dir):
        fallback = "/Users/dev/Desktop/Models/Qwen3.6-35B-A3B-MLX-4bit"
        if os.path.isdir(fallback):
            model_dir = fallback
        else:
            print(f"piper distill: model directory not found: {model_dir}", file=sys.stderr)
            return EXIT_INVALID

    mlx_python = None
    if not daemon_online:
        for cand in [sys.executable, "/Users/dev/.local/share/godoer-venv/bin/python", shutil.which("python3")]:
            if cand and os.path.isfile(cand):
                try:
                    res = subprocess.run([cand, "-c", "import mlx_lm"], capture_output=True, text=True)
                    if res.returncode == 0:
                        mlx_python = cand
                        break
                except Exception:
                    continue

        if not mlx_python:
            print("piper distill: mlx_lm python environment not found", file=sys.stderr)
            return EXIT_ERROR

    lock_file = None
    if not daemon_online:
        import fcntl
        lock_path = "/tmp/piper_distill.lock"
        try:
            lock_file = open(lock_path, "w")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            print("piper distill: another distillation process is active; skipping cold load to prevent memory contention.", file=sys.stderr)
            if lock_file:
                lock_file.close()
            return EXIT_OK

    mode_label = f"warm daemon ({sock_path})" if daemon_online else f"cold {os.path.basename(model_dir)}"
    print(f"piper distill: analyzing {len(incidents)} incident(s) via {mode_label}...")

    diagnoses = []
    try:
        for inc in incidents:
            inc_id = inc.get("id", "INC-UNK")
            kind = inc.get("kind", "UNKNOWN")
            msg = inc.get("message", "")
            file_path = inc.get("file", "")
            line = inc.get("line", "")
            ctx = inc.get("source_context", "")

            prompt = f"""<|im_start|>system
You are a senior Godot 4 engine expert and flight-recorder analyst.
Your task is to analyze factual runtime telemetry incidents and provide a grounded root-cause diagnosis.
CRITICAL RULES:
1. You must cite the exact incident ID [{inc_id}].
2. Do NOT invent, assume, or hallucinate bugs not listed in the incident.
3. Provide:
   - Root Cause: exactly why the line threw this error in Godot 4.
   - Recommended Fix: the concrete GDScript code replacement or configuration change.
<|im_end|>
<|im_start|>user
INCIDENT:
- ID: {inc_id}
- Kind: {kind}
- Error: {msg}
- File: {file_path} (Line {line})

SOURCE CODE CONTEXT:
```gdscript
{ctx}
```

Diagnose [{inc_id}] and provide the exact fix.
<|im_end|>
<|im_start|>assistant
"""
            raw_output = ""
            if daemon_online:
                temp_slice_dir = tempfile.mkdtemp(prefix="piper_distill_")
                task_file = os.path.join(temp_slice_dir, "task.json")
                res_file = os.path.join(temp_slice_dir, "result.json")
                task_data = {
                    "id": f"distill-{inc_id.lower()}",
                    "cwd": os.path.dirname(incidents_path),
                    "prompt": prompt,
                    "model_dir": model_dir,
                    "auto_approve_exec": True,
                    "auto_approve_writes": True,
                    "auto_approve_irreversible": True,
                    "timeout_s": 300,
                    "result_path": res_file,
                }
                with open(task_file, "w", encoding="utf-8") as f:
                    json.dump(task_data, f, indent=2)

                _forward_task_to_daemon(sock_path, task_file)

                if os.path.isfile(res_file):
                    try:
                        with open(res_file, encoding="utf-8") as f:
                            res_obj = json.load(f)
                        raw_output = str(res_obj.get("message") or res_obj.get("error") or "").strip()
                    except Exception as exc:
                        raw_output = f"Error reading daemon result: {exc}"
                else:
                    raw_output = "Daemon finished but no result.json was produced"
                shutil.rmtree(temp_slice_dir, ignore_errors=True)
            else:
                code = f"""
import json, sys
from mlx_lm import load, generate

model, tok = load({json.dumps(model_dir)})
out = generate(model, tok, prompt={json.dumps(prompt)}, max_tokens={args.max_tokens}, verbose=False)
print(out)
"""
                run = subprocess.run([mlx_python, "-c", code], capture_output=True, text=True)
                if run.returncode != 0:
                    print(f"piper distill: error analyzing {inc_id}: {run.stderr}", file=sys.stderr)
                    diagnoses.append({
                        "incident_id": inc_id,
                        "status": "error",
                        "diagnosis": run.stderr.strip()
                    })
                    continue
                raw_output = run.stdout.strip()

            clean_diagnosis = raw_output
            if "</think>" in raw_output:
                clean_diagnosis = raw_output.split("</think>")[-1].strip()

            diagnoses.append({
                "incident_id": inc_id,
                "kind": kind,
                "file": file_path,
                "line": line,
                "error": msg,
                "diagnosis": clean_diagnosis,
                "raw_output": raw_output,
            })
    finally:
        if lock_file:
            import fcntl
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            except Exception:
                pass

    report = {
        "status": "diagnosed",
        "incidents_count": len(incidents),
        "model": "warm-daemon" if daemon_online else os.path.basename(model_dir),
        "diagnoses": diagnoses,
    }

    out_path = args.out or os.path.join(os.path.dirname(incidents_path), "distilled_diagnosis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print(f" PIPER FLIGHT-RECORDER DISTILLATION CARD ({report['model']})")
    print("=" * 65)
    for d in diagnoses:
        print(f"\n[{d['incident_id']}] {d.get('file', '')}:{d.get('line', '')} - {d.get('error', '')}")
        print("-" * 65)
        print(d.get("diagnosis", ""))
    print("=" * 65)
    print(f"Saved distilled report to: {out_path}\n")

    return EXIT_OK


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
    "explicitly with --detach (or LMP_DAEMONIZE=1) plus a wake URL via --orch-webhook, task.json orch_webhook, LMP_ORCH_WEBHOOK, or .piper/orch_webhook. "
    "Piper never infers detach from stdio: nohup/screen/background launches must pass --detach. "
    "A run that writes result.json POSTs done if completed, stalled if stopped/failed. Process exit with no result.json POSTs died. "
    "Silent detached workers are refused."
)

PARENT_CONTRACT_BANNER = (
    "piper: you are the parent. Stay attached and read the exit and result.json.\n"
    "Detach only with --detach plus a wake URL. No default URL. Events:\n"
    "ask (piper answer allow|deny|--text), done, stalled (not success), died (do not relaunch).\n"
    "See PIPER.md if present.\n"
)

AGENT_WAKE_FALLBACK = """# Piper Local Worker — Orchestrator Guide & Parent Runbook

## Overview
**Core Principle:** Cloud directs, local writes, cloud verifies, repeat.

Piper is a fast headless coding worker running locally on Apple Silicon (via MLX). It executes edits, shell commands, and file operations inside `cwd` with zero cloud output token cost.

The cloud orchestrator (Cursor, Claude, Gemini, Antigravity, or custom script) acts as the high-level brain:
- Maintains the long-horizon plan and acceptance criteria.
- Decomposes complex tasks into bounded slices (1–3 files per slice).
- Directs Piper with a slice brief (`prompt.md`). The harness emits `task.json`.
- Verifies outcomes with the review card, the packet `check`, and `result.json`.
- Loops until the entire mission is verified complete.

## The Long-Horizon Execution Loop
```
┌─────────────────────────┐    prompt.md → task.json    ┌──────────────────────────┐
│   Cloud Orchestrator    │ ───────────────────────────► │       Piper Worker       │
│  (Cursor, Claude, etc.) │                              │  (MLX on Apple Silicon)  │
│                         │ ◄─────────────────────────── │                          │
│ plan · review · verify  │   review card + result.json  │  edits · tools · loop    │
└─────────────────────────┘                              └──────────────────────────┘
             │                                                         │
             └────────── repeat until horizon acceptance passes ───────┘
```

### Roles

| Who | Owns | Does not own |
|---|---|---|
| **Cloud Orchestrator** | Goal decomposition, file-level direction, acceptance criteria, high-level review, troubleshooting, "are we done?" | Packet JSON shape, bulk code generation tokens, local tool thrash |
| **Piper Worker** | Edits, tool execution, test commands, local iteration inside `cwd` | Long-horizon judgment, multi-repo strategy |

Local models work best on scoped slices: **the brief must be specific**, and **every slice gets an orchestrator review**. Trust the review card, the `check`, and `result.json`.

### Step-by-Step Procedure

1. **Frame the Horizon**
   Define the overarching objective and an acceptance checklist (e.g. unit tests pass, new command works, UI renders).

2. **Write `prompt.md` for this slice**
   Pick the smallest incremental step. Scope it to 1–3 files. The cloud writes this file only:
   - Horizon context (short).
   - This slice only — one outcome.
   - Files: EDIT, CREATE, and DO NOT TOUCH.
   - The acceptance command.
   - When to stop if stuck.

3. **Emit `task.json`**
   Do not hand-write the packet. The harness owns the shape.
   ```bash
   piper mcp-list --cwd /abs/ws                 # only when the slice needs MCP
   piper packet --id slice-001 --cwd /abs/ws \\
     --prompt-file prompt.md --check "ctest -R test_validator"
   piper packet ... --trust-mcp godoer          # names must exist in cwd .mcp.json
   piper ui --cwd /abs/ws                       # watch; follows .piper/active.json
   ```
   Do not pass `--out`. The emitter writes `<cwd>/.piper/slices/<id>/task.json`, copies `prompt.md` beside it, and records `.piper/active.json`. It writes `id`, `cwd`, `prompt`, `model_dir` (from `--model-dir` or `LMP_QWEN_DIR`; missing model exits 3), auto-approve flags, `timeout_s` (default 600), and `result_path` (sibling `result.json` unless `--result-path` is set). Optional `--check`.
   - **`check`**: operator acceptance command. Also becomes `verify_contract` during the run. A green post-run check yields `status=ok` / wake `done` when the loop stopped short without breaking (`max_turns`, `stalled`, or a text ending with work left open); the card's `loop:` line then says the pass rests on the check alone. A crash (`backend_error`), a cancel, a timeout or an unanswered irreversible ask is never promoted. A red check turns a completed run into `error`. The post-run check runs on the same 300 s clock as the in-loop check (`piper packet --check-timeout-s N` to change it). A check that timed out or could not run (exit 126/127) is reported as such (`test.timed_out`, `test.could_not_run`) and never counts as green. `test.output_tail` is a log_triage digest of the output (failing locators kept), `test.triage` names the runner, failing tests and primary diagnostics, and `test.output_path` points at the full log (`check.log`) when the digest dropped anything.
   - **`max_iterations`**: turn budget sent to the agent loop. Default **30**; default **60** when `trust_mcp` is set (Godoer-heavy). Raise it for a long slice with `piper packet --max-iterations N` — no rebuild.
   - **`trust_mcp`**: explicit server names from `piper mcp-list`. No guessed JSON array.

4. **Dispatch**
   Default parent launch. Attached: wait, then print the review card.
   ```bash
   piper dispatch --task /abs/ws/.piper/slices/slice-001/task.json
   # Unattended irreversible tools (godot_project, delete_file, whole-file overwrite):
   piper dispatch --task /abs/ws/.piper/slices/slice-001/task.json --auto-approve-irreversible
   # Or approve exec + writes + irreversible:
   piper dispatch --task /abs/ws/.piper/slices/slice-001/task.json --auto-approve-all
   ```
   Exit codes:
   - `0`: Completed normally.
   - `1`: Worker error.
   - `2`: Execution timed out (`timeout_s`).
   - `3`: Invalid task packet or missing wake URL for a detached run.

   Lower-level attached run, when you are not using the review card helper: `piper run --task task.json` (same as `piper worker run --task task.json`). Keep weights warm across slices with `piper worker serve`, then `piper worker run`. `piper dispatch` never detaches: it is attached by construction, whatever its stdio or `LMP_DAEMONIZE` say.

   In Claude Code, run `piper dispatch` with `run_in_background: true` and read the card from the task output when the completion notice arrives. A foreground Bash call is capped at 10 minutes.

   Piper never guesses detach from stdio. A background launch (nohup, screen, `&`) must say so with `piper run --detach` (or `LMP_DAEMONIZE=1`) and needs a wake URL before start; without `--detach` the run stays attached to whatever launched it. Resolve the URL with `piper wake-url`. Do not copy a URL from the panel.
   - `--orch-webhook URL`, or
   - task field `orch_webhook`, or
   - env `LMP_ORCH_WEBHOOK`, or
   - file `.piper/orch_webhook` (written by `piper_ui`)

   `--detach` without a URL exits before the sidecar starts.

   Irreversible tools pause unless auto-approve was set. Answer with `piper answer`. Do not write `answer.json`.

5. **Review**
   Read the card `piper dispatch` printed. If you already waited some other way:
   ```bash
   piper review --task task.json              # or: piper review --result result.json
   piper status --dir /path/to/slice          # idle/ask/done/stalled/error; no cat/jq
   piper await --dir /path/to/slice           # wait for ask/done; no sleep-loop
   ```
   On `ask`: `piper answer allow`, `piper answer deny`, or `piper answer --text "..."`. Do not restart the process.

   The card is the rubric. A pass is `status == "ok"`, `files_touched` inside the brief, a proportional diff, and a green `check` (`result.test`). `"stalled"` is not a pass. Green `test.exit_code=0` after an incomplete loop stop is still `ok` when `check` was set, and the card's `loop:` line names the stop.

   Verdicts: `PASS` (ok and a green check), `UNVERIFIED` (the model finished but no check ran — review the diff yourself or re-dispatch with `--check`; never treat it as a pass), `FAIL`, `STALLED`, `DIED`. On anything but `PASS` the card carries the evidence: `error:`, the failing check's last lines (`check:`), and the worker's own `piper:` notices (`worker:`, e.g. an ask question). Worker telemetry goes to `worker.stderr.log` beside `result.json` (the previous run's is `worker.stderr.prev.log`), not into the dispatch output; a `DIED` card shows that log's tail. `LMP_WORKER_STDERR=inherit` keeps it inline for a human at a terminal.

6. **Record and continue**
   ```bash
   piper progress --id slice-001 pass --note "validator + test"
   ```
   Verdicts: `pass`, `fail`, `stalled`, `timeout`, `died`, `skip`. The line is appended to `.piper/progress.log`.
   - **Pass**: emit the next slice.
   - **Fail**: a narrower brief, or do that edit yourself.
   - **Done**: when the horizon checklist is green, stop.

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
detach. `piper dispatch` stays attached and does not need a URL. Detach is
declared with `--detach` (or `LMP_DAEMONIZE=1`), never inferred from stdin or
stdout: agent tool runners hand every foreground command `/dev/null` and a
file, so a guess made there refused or orphaned attached runs.

## How to launch

Stay attached only if your process waits and reads the exit. That is `piper dispatch` and `piper run`.

Otherwise, before start, resolve the URL with `piper wake-url` (do not copy it from a panel):

- `--orch-webhook URL`, or
- task field `orch_webhook`, or
- env `LMP_ORCH_WEBHOOK`, or
- file `.piper/orch_webhook` (written automatically by `piper_ui` — **do not copy a URL from the panel**)

`--detach` with no URL: the CLI exits before the sidecar starts. `--help`
says this in one paragraph. Read that before the first launch. The help
text is the contract, not this chat.

## Events

One short JSON POST. No prompt bodies. Failed POST does not fail the
mission. No URL means no POST.

| kind | when | parent does |
| --- | --- | --- |
| `ask` | `awaiting_user.json` written, or an irreversible call is paused | `piper answer allow`, `piper answer deny`, or `piper answer --text "..."` (writes `answer.json`). Do not restart. Do not freehand the JSON. |
| `done` | `result.json` written and the slice completed | `piper review` (or the dispatch card). Send the next slice or stop. |
| `stalled` | `result.json` written, the harness stopped the run (`stalled`, `max_turns`, not completed), and the check did not pass | read what landed. Do not treat it as success. Next slice or stop. |
| `died` | process exited and no `result.json` was written | launch parent sends this. Tell the user. Do not relaunch blindly. |

`stalled` is its own kind. Do not hide it inside `done` with `status: error`.
A parent that only handles `done` will miss a stall, which is the bug this
standard exists to kill. `result.json` uses the same `status: "stalled"` for
`max_turns` / no-progress stalls whose check did not pass, so parents need not
parse error strings.

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

- Do not hand-write `task.json`, `answer.json`, or a progress-log line.
- Do not poll Piper, and do not poll the cloud from Piper.
- Do not ask the human if the sidecar is still there. The event is the notice.
- Do not bake another agent's webhook into the binary.
- Do not start a silent screen or nohup without the URL.

A folder watch is allowed only as a backup when the platform has no wake
URL yet. It is not the standard, and it must not be the path a second
agent copies.

## Done when

- `--help` names this standard in one paragraph.
- A `--detach` launch with no URL exits before the sidecar starts.
- Detach is never inferred from stdio: `piper dispatch` under stdin=/dev/null and a
  regular-file stdout waits for the run and prints the card.
- A run that writes `result.json` POSTs `done` if `status=ok` (model completed,
  `plan_ready`, or a green packet `check` after an incomplete loop stop such as
  `max_turns`), `stalled` if it did not. A crash or cancel is never `done`.
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

2. **Write `prompt.md` only**:
   Horizon, this slice, EDIT / CREATE / DO NOT TOUCH, the check command, and when to stop.
   Do not hand-write `task.json`. The harness owns packet shape.

3. **Emit the packet**:
   `piper packet --id slice-001 --cwd /abs/workspace --prompt-file prompt.md --check "…"`
   Do not pass `--out`. The harness writes `<cwd>/.piper/slices/<id>/task.json`, copies `prompt.md` beside it, bakes `model_dir` from `--model-dir` or `LMP_QWEN_DIR`, and records `.piper/active.json`.
   Optional MCP consent (still explicit): `piper mcp-list --cwd /abs/workspace` then
   `piper packet ... --trust-mcp godoer` (names must exist in workspace `.mcp.json`).

4. **Dispatch**:
   `piper dispatch --task` the path `piper packet` printed (attached run → wait → review card).
   Watch the run: `piper ui --cwd /abs/workspace`.
   Unattended irreversible: `--auto-approve-irreversible` or `--auto-approve-all`.
   Lower-level attached run: `piper run --task` that same path.
   In Claude Code: `piper dispatch` with `run_in_background: true`, then read the card from the task output.
   Detached/background is declared, never inferred from stdio: `piper run --detach` plus a wake URL
   (`--orch-webhook`, task field, env, or `.piper/orch_webhook` from `piper ui`).
   Resolve it with `piper wake-url`. Do not copy a URL from the panel.

5. **Review**:
   Use the card from `piper dispatch`, or `piper review --task path/to/task.json`.
   Status without freehand cat/jq: `piper status --dir …` / `piper await --dir …`.

6. **Loop Until Horizon Complete**:
   - If slice passed: `piper progress --id slice-001 pass --note "…"` then next slice.
   - If slice failed: narrower packet, or fix that spot yourself.
   - Repeat until horizon acceptance is green.

7. **Parent Contract & Events**:
   - Stay attached and read the exit code and the review card.
   - Events: `ask` → `piper answer allow|deny|--text` (do not freehand `answer.json`);
     `done`, `stalled` (not success), `died` (do not relaunch blindly).
   - See `PIPER.md` for the full specification.
"""


ORCH_SKILL_RELS = (
    os.path.join(".cursor", "skills", "piper-orchestration", "SKILL.md"),
    os.path.join(".agents", "skills", "piper-orchestration", "SKILL.md"),
)


def orchestration_skill_text():
    """Canonical parent skill shipped in this checkout. Empty if unreadable."""
    path = os.path.join(ROOT, ORCH_SKILL_RELS[0])
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


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

    skill_text = orchestration_skill_text()
    installed = []
    if skill_text:
        for rel in ORCH_SKILL_RELS:
            dest = os.path.join(target_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w", encoding="utf-8") as fh:
                fh.write(skill_text)
            installed.append(rel)
    else:
        print(
            "piper init: orchestration skill missing from this checkout; "
            "PIPER.md and the parent rule were still written.",
            file=sys.stderr,
        )

    parts = ["PIPER.md", ".cursor/rules/piper-parent.mdc"]
    parts.extend(installed)
    print(
        "piper init: " + ", ".join(parts) + ". "
        "Godoer briefs left alone. Stay attached, or pass --orch-webhook."
    )
    return EXIT_OK



def result_path_from_light_packet(packet, task_path):
    """Resolve result_path from a light packet read (no model_dir required).

    Relative paths are joined to the task.json directory — same as load_packet.
    """
    task_dir = os.path.dirname(os.path.abspath(task_path))
    raw = packet.get("result_path")
    if isinstance(raw, str) and raw.strip():
        expanded = os.path.expanduser(raw.strip())
        if os.path.isabs(expanded):
            return os.path.abspath(expanded)
        return os.path.abspath(os.path.join(task_dir, expanded))
    return os.path.join(task_dir, "result.json")


def resolve_task_json_path(task_arg):
    """Return absolute task.json path; accept file or directory containing task.json."""
    path = os.path.abspath(os.path.expanduser(task_arg))
    if os.path.isdir(path):
        candidate = os.path.join(path, "task.json")
        if not os.path.isfile(candidate):
            raise PacketError(f"task packet not found: {candidate}")
        return candidate
    if not os.path.isfile(path):
        raise PacketError(f"task packet not found: {path}")
    return path


def resolve_status_dir(dir_arg=None, task_arg=None):
    """Directory that holds awaiting_user.json / result.json / answer.json."""
    if dir_arg:
        return os.path.abspath(os.path.expanduser(dir_arg))
    if task_arg:
        packet, _roots = read_task_roots(task_arg)
        task_path = resolve_task_json_path(task_arg)
        return os.path.dirname(result_path_from_light_packet(packet, task_path))
    return os.path.abspath(".")


def read_json_object(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def collect_run_status(directory):
    """Deterministic workspace status — parents should not freehand cat/jq.

    Priority: ask (awaiting_user.json) > finished (result.json) > idle.
    """
    directory = os.path.abspath(directory)
    awaiting_path = os.path.join(directory, "awaiting_user.json")
    result_path = os.path.join(directory, "result.json")
    answer_path = os.path.join(directory, "answer.json")
    awaiting = read_json_object(awaiting_path)
    result = read_json_object(result_path)
    answer_present = os.path.isfile(answer_path)
    if awaiting is not None:
        question = str(awaiting.get("question") or "").strip()
        return {
            "state": "ask",
            "directory": directory,
            "awaiting_path": awaiting_path,
            "result_path": result_path if os.path.isfile(result_path) else None,
            "answer_present": answer_present,
            "question": question,
            "options": awaiting.get("options"),
            "run_id": awaiting.get("run_id"),
            "seq": awaiting.get("seq"),
            "result": result,
        }
    if result is not None:
        status = str(result.get("status") or "").lower()
        if status == "ok":
            state = "done"
        elif status == "stalled":
            state = "stalled"
        elif status == "timeout":
            state = "timeout"
        else:
            state = "error"
        return {
            "state": state,
            "directory": directory,
            "awaiting_path": None,
            "result_path": result_path,
            "answer_present": answer_present,
            "question": None,
            "options": None,
            "run_id": result.get("run_id") or result.get("task_id"),
            "seq": None,
            "result": result,
            "status": status,
            "message": result.get("message"),
            "task_id": result.get("task_id"),
        }
    return {
        "state": "idle",
        "directory": directory,
        "awaiting_path": None,
        "result_path": None,
        "answer_present": answer_present,
        "question": None,
        "options": None,
        "run_id": None,
        "seq": None,
        "result": None,
    }


def format_status_card(info):
    state = info.get("state") or "idle"
    lines = [
        "── piper status ─────────────────────────",
        f"state:    {state}",
        f"dir:      {info.get('directory')}",
    ]
    if state == "ask":
        q = str(info.get("question") or "").strip().replace("\n", " ")
        if len(q) > 160:
            q = q[:157] + "..."
        lines.append(f"question: {q or '(empty)'}")
        if info.get("options") is not None:
            lines.append(f"options:  {info.get('options')}")
        lines.append(f"awaiting: {info.get('awaiting_path')}")
        lines.append("next:     piper answer allow|deny|--text ...")
    elif state in ("done", "stalled", "timeout", "error"):
        lines.append(f"task:     {info.get('task_id') or '(unknown)'}")
        lines.append(f"status:   {info.get('status')}")
        msg = str(info.get("message") or "").strip().replace("\n", " ")
        if len(msg) > 160:
            msg = msg[:157] + "..."
        lines.append(f"message:  {msg or '(empty)'}")
        lines.append(f"result:   {info.get('result_path')}")
        lines.append("next:     piper review --result …")
    else:
        lines.append("next:     piper dispatch --task …  (or wait for a run)")
    lines.append("─────────────────────────────────────────")
    return "\n".join(lines)


def status_exit_code(info):
    state = info.get("state")
    if state == "ask":
        return EXIT_OK
    if state == "done":
        return EXIT_OK
    if state == "idle":
        return EXIT_OK
    if state == "timeout":
        return EXIT_TIMEOUT
    if state in ("stalled", "error"):
        return EXIT_ERROR
    return EXIT_ERROR


def cmd_status(args):
    try:
        directory = resolve_status_dir(getattr(args, "dir", None), getattr(args, "task", None))
    except PacketError as exc:
        print(f"piper status: {exc}", file=sys.stderr)
        return EXIT_INVALID
    info = collect_run_status(directory)
    if getattr(args, "json", False):
        print(json.dumps(info, indent=2))
    else:
        print(format_status_card(info))
    return status_exit_code(info)


def cmd_await(args):
    """Poll until ask/done/stalled/error/timeout — no hand-rolled sleep loops."""
    try:
        directory = resolve_status_dir(getattr(args, "dir", None), getattr(args, "task", None))
    except PacketError as exc:
        print(f"piper await: {exc}", file=sys.stderr)
        return EXIT_INVALID
    timeout_s = float(getattr(args, "timeout_s", 600.0) or 600.0)
    interval_s = float(getattr(args, "interval_s", 0.25) or 0.25)
    if timeout_s < 0:
        print("piper await: --timeout-s must be >= 0", file=sys.stderr)
        return EXIT_INVALID
    if interval_s <= 0:
        print("piper await: --interval-s must be > 0", file=sys.stderr)
        return EXIT_INVALID
    terminal = {"ask", "done", "stalled", "error", "timeout"}
    deadline = time.time() + timeout_s
    last_state = None
    while True:
        info = collect_run_status(directory)
        state = info.get("state")
        if state != last_state and getattr(args, "verbose", False):
            print(f"piper await: state={state}", file=sys.stderr)
            last_state = state
        if state in terminal:
            if getattr(args, "json", False):
                print(json.dumps(info, indent=2))
            else:
                print(format_status_card(info))
            return status_exit_code(info)
        remaining = deadline - time.time()
        if remaining <= 0:
            print("piper await: timed out still idle", file=sys.stderr)
            if getattr(args, "json", False):
                info = collect_run_status(directory)
                info["state"] = "idle"
                info["await_timeout"] = True
                print(json.dumps(info, indent=2))
            return EXIT_TIMEOUT
        time.sleep(min(interval_s, remaining))



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

    init_p = sub.add_parser("init", help="initialize PIPER.md, parent rule, and orchestration skills",
                            description="Initialize PIPER.md, .cursor/rules/piper-parent.mdc, and "
                                        "Cursor/Codex orchestration skills without touching Godoer briefs.",
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
    worker_init = worker_sub.add_parser("init", help="initialize PIPER.md, parent rule, and orchestration skills",
                                        description="Initialize PIPER.md, .cursor/rules/piper-parent.mdc, and "
                                                    "Cursor/Codex orchestration skills without touching Godoer briefs.",
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
    packet_p.add_argument(
        "--out", default=None,
        help="output path (default: <cwd>/.piper/slices/<id>/task.json)",
    )
    packet_p.add_argument(
        "--model-dir", default=None,
        help="model directory (default: LMP_QWEN_DIR; required at emit time)",
    )
    packet_p.add_argument("--check", default=None, help="optional acceptance command")
    packet_p.add_argument(
        "--check-timeout-s", type=float, default=None,
        help="post-run check clock (default 300, the same clock as the in-loop check)",
    )
    packet_p.add_argument("--timeout-s", type=float, default=600.0, help="timeout_s (default 600)")
    packet_p.add_argument(
        "--max-iterations", type=int, default=None,
        help="turn budget (default 30, or 60 with --trust-mcp)",
    )
    packet_p.add_argument("--result-path", default=None, help="optional result_path")
    packet_p.add_argument("--trust-mcp", action="append", default=None,
                          help="explicit MCP server name to trust (repeatable; must exist in cwd .mcp.json)")
    packet_p.add_argument("--no-auto-approve-irreversible", action="store_true",
                          help="leave auto_approve_irreversible false")

    mcp_list_p = sub.add_parser(
        "mcp-list",
        help="list server names from workspace .mcp.json (for explicit trust_mcp)",
        description="Print mcpServers keys from cwd/.mcp.json. Consent stays explicit: "
                    "use names with piper packet --trust-mcp. No freehand guessing.",
    )
    mcp_list_p.add_argument("--cwd", default=".", help="workspace root containing .mcp.json")
    mcp_list_p.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    progress_p = sub.add_parser(
        "progress",
        help="append one progress log line (no freehand progress paste)",
        description="Append `slice | verdict | note` to .piper/progress.log (or --file). "
                    "Vision-loop memory without pasting log snippets into chat.",
    )
    progress_p.add_argument("--id", required=True, help="slice / task id")
    progress_p.add_argument("verdict",
                            help="slice outcome: pass|fail|stalled|timeout|died|skip")
    progress_p.add_argument("--note", default="", help="optional short note")
    progress_p.add_argument("--dir", default=None,
                            help="workspace root for .piper/progress.log (default: cwd)")
    progress_p.add_argument("--file", default=None, help="explicit progress log path")
    progress_p.add_argument("--json", action="store_true", help="emit machine-readable JSON")

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
                    "Cloud still decides the next slice. Always attached; detach is not supported here.",
    )
    dispatch_p.add_argument("--task", required=True,
                            help="task.json path, or a directory containing task.json")
    dispatch_p.add_argument("--jsonl", action="store_true",
                            help="stream lmp/* notifications as ndjson on stdout")
    dispatch_p.add_argument("--orch-webhook", default=None,
                            help="webhook URL to wake cloud orchestrator on ask/done/died events")
    dispatch_p.add_argument("--auto-approve-irreversible", action="store_true",
                            help="auto-approve irreversible tool calls (destroys data / overwrite)")
    dispatch_p.add_argument("--auto-approve-all", action="store_true",
                            help="auto-approve all tool calls (exec + writes + irreversible)")
    dispatch_p.add_argument("--json", action="store_true",
                            help="emit machine-readable JSON including the review card")

    status_p = sub.add_parser(
        "status",
        help="print deterministic run status (idle/ask/done/stalled/error)",
        description="Read awaiting_user.json / result.json and print a status card. "
                    "Parents should not freehand cat/jq the workspace.",
    )
    status_p.add_argument("--dir", default=None, help="directory holding awaiting_user.json/result.json")
    status_p.add_argument("--task", default=None, help="task.json whose result_path directory is inspected")
    status_p.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    await_p = sub.add_parser(
        "await",
        help="wait until ask/done/stalled/error (no hand-rolled sleep loops)",
        description="Poll the workspace until a terminal state appears, then print status. "
                    "Use this instead of pasting sleep/poll snippets into chat.",
    )
    await_p.add_argument("--dir", default=None, help="directory holding awaiting_user.json/result.json")
    await_p.add_argument("--task", default=None, help="task.json whose result_path directory is watched")
    await_p.add_argument("--timeout-s", type=float, default=600.0, help="max seconds to wait (default 600)")
    await_p.add_argument("--interval-s", type=float, default=0.25, help="poll interval seconds (default 0.25)")
    await_p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    await_p.add_argument("--verbose", action="store_true", help="print state transitions on stderr")

    ui_p = sub.add_parser(
        "ui",
        help="watch the active slice in a local web view",
        description="Start the orchestration watch UI on a workspace. "
                    "It follows .piper/active.json and .piper/slices/.",
    )
    ui_p.add_argument("--cwd", default=".", help="workspace to watch (default: .)")
    ui_p.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    ui_p.add_argument("--no-open", action="store_true", help="do not open a browser")

    distill_p = sub.add_parser(
        "distill",
        help="distill runtime telemetry incidents into a grounded root-cause diagnosis",
        description="Ingest an incidents.json packet and synthesize root-cause diagnoses using the local model with strict citations.",
    )
    distill_p.add_argument("--incidents", required=True, help="path to incidents.json")
    distill_p.add_argument("--model-dir", default=None, help="path to local MLX model directory (or LMP_QWEN_DIR)")
    distill_p.add_argument("--out", default=None, help="output path for distilled diagnosis JSON")
    distill_p.add_argument("--max-tokens", type=int, default=1500, help="max tokens per diagnosis")
    distill_p.add_argument("--socket", default=None, help="unix domain socket path for keep-warm daemon")
    distill_p.add_argument("--allow-cold", action="store_true", help="allow loading model cold into RAM if daemon is not running")

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
    if command == "mcp-list":
        return cmd_mcp_list(args)
    if command == "progress":
        return cmd_progress(args)
    if command == "wake-url":
        return cmd_wake_url(args)
    if command == "review":
        return cmd_review(args)
    if command == "dispatch":
        return cmd_dispatch(args)
    if command == "status":
        return cmd_status(args)
    if command == "await":
        return cmd_await(args)
    if command == "ui":
        return cmd_ui(args)
    if command == "distill":
        return cmd_distill(args)

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

    return cmd_run(args)


def cmd_run(args, *, attached_only=False):
    """Run one packet. Attached unless the caller explicitly asked to detach.

    Detach is declared, never inferred: `--detach` or LMP_DAEMONIZE=1. Under agent
    tool runners stdin is /dev/null and stdout is a regular file on every call, so
    reading intent from stdio turned every foreground dispatch into a refused or
    orphaned run. `attached_only` (dispatch) ignores both switches: dispatch owns
    the wait and prints the card.
    """
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

    is_detached = (not attached_only) and (
        bool(getattr(args, "detach", False)) or os.environ.get("LMP_DAEMONIZE") == "1"
    )

    if is_detached and not webhook_url:
        print(DETACHED_NO_WAKE_MSG, file=sys.stderr)
        return EXIT_INVALID

    if not attached_only:
        # dispatch prints the card instead; the banner is noise in the parent's output.
        sys.stderr.write(PARENT_CONTRACT_BANNER)
        sys.stderr.flush()
    os.environ["LMP_BANNER_PRINTED"] = "1"

    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    if is_detached:
        ae.daemonize()

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

        # The engine is always attached to this harness; when the harness itself
        # detached it already forked above. Pin that so an inherited
        # LMP_DAEMONIZE=1 cannot make the engine refuse an attached run.
        child_env = dict(os.environ, LMP_DAEMONIZE="0")
        stderr_fd = open_worker_stderr_log(result_path)
        try:
            proc = subprocess.Popen(cmd, env=child_env, stderr=stderr_fd)
        finally:
            if stderr_fd is not None:
                os.close(stderr_fd)

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


FAKE_SLOW_OK = FAKE_OK.replace(
    'emit({"jsonrpc": "2.0", "method": "lmp/run_end",',
    'import time; time.sleep(1.2)\n        emit({"jsonrpc": "2.0", "method": "lmp/run_end",',
)

# Stand-in for the C++ engine (no .py suffix, so the harness takes its Popen
# branch): writes a green result.json after ~1 s and records the LMP_DAEMONIZE
# it inherited, so the self-test can see the harness pinned the child attached.
FAKE_CPP_WORKER = r"""#!%s
import json, os, sys, time
task = sys.argv[sys.argv.index("--task") + 1]
if os.path.isdir(task):
    task = os.path.join(task, "task.json")
with open(task, encoding="utf-8") as fh:
    packet = json.load(fh)
time.sleep(1.2)
result_path = packet.get("result_path") or os.path.join(os.path.dirname(task), "result.json")
with open(result_path, "w", encoding="utf-8") as fh:
    json.dump({"task_id": packet["id"], "status": "ok", "message": "done",
               "turns": 1, "wall_seconds": 1.2,
               "child_daemonize": os.environ.get("LMP_DAEMONIZE"),
               "files_touched": [], "diff_stat": {"insertions": 0, "deletions": 0, "files": 0},
               "test": {"ran": True, "exit_code": 0, "command": "true", "output_tail": ""}}, fh)
"""


# Stand-in engine that writes MLX-style breadcrumbs to stderr every "turn", like
# src/model/mlx_backend.cpp does (~1 KB per turn), then a green result.
FAKE_CPP_NOISY = r"""#!%s
import json, os, sys
task = sys.argv[sys.argv.index("--task") + 1]
with open(task, encoding="utf-8") as fh:
    packet = json.load(fh)
for turn in range(30):
    for at in ("generate_enter", "prefill_start", "prefill_chunk_begin", "prefill_chunk_end",
               "prefill_done", "decode_begin", "decode_end"):
        sys.stderr.write("mem at=%%s tokens=%%d active=16716315128 cache=507924 "
                         "peak=19684121116 sum=16716823052\n" %% (at, 5000 + turn))
sys.stderr.write("piper: task %%s finished with status 'ok' (exit 0, 1.0s, 30 turns)\n" %% packet["id"])
with open(packet["result_path"], "w", encoding="utf-8") as fh:
    json.dump({"task_id": packet["id"], "status": "ok", "message": "done", "turns": 30,
               "wall_seconds": 1.0, "files_touched": ["a.py"],
               "diff_stat": {"insertions": 1, "deletions": 0, "files": 1},
               "test": {"ran": True, "exit_code": 0, "command": "true", "output_tail": ""}}, fh)
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
    """Fake-sidecar scenarios, hermetic: own cwd, no inherited wake URL or detach switch.

    Wake-URL discovery searches the process cwd, so running from a checkout that has
    a `.piper/orch_webhook` used to change what `--detach` did. Detach is never
    pinned off here: the stdio cases below must see the real launch policy.
    """
    scrubbed = ("LMP_DAEMONIZE", "LMP_ORCH_WEBHOOK", "LMP_WORKER_STDERR")
    saved_env = {key: os.environ.get(key) for key in scrubbed}
    saved_cwd = os.getcwd()
    for key in scrubbed:
        os.environ.pop(key, None)
    try:
        with tempfile.TemporaryDirectory(prefix="piper-worker-cwd-") as cwd:
            os.chdir(cwd)
            return _self_test_scenarios()
    finally:
        os.chdir(saved_cwd)
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _self_test_scenarios():
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

        # 8b. dispatch is attached by construction. Agent tool runners (Claude Code's
        # Bash tool) launch with stdin=/dev/null and stdout to a regular file; that
        # used to read as "detached" -> exit 3, or a double fork and exit 0 with no
        # card. Run it exactly that way, with LMP_DAEMONIZE unset, for every wake-URL
        # source and both engine branches.
        fake_slow = os.path.join(tmp, "fake_slow.py")
        _write_exec(fake_slow, FAKE_SLOW_OK)
        fake_cpp = os.path.join(tmp, "fake_cpp_worker")
        _write_exec(fake_cpp, FAKE_CPP_WORKER)
        stdio_env = {k: v for k, v in os.environ.items()
                     if k not in ("LMP_DAEMONIZE", "LMP_ORCH_WEBHOOK")}
        stdio_env["LMP_USE_CPP_WORKER"] = "1"
        stdio_env["LMP_QWEN_DIR"] = model_dir
        for engine_name, engine in (("py", fake_slow), ("cpp", fake_cpp)):
            for url_source in ("none", "env", "file"):
                label = f"dispatch[{engine_name},{url_source}]"
                sws = os.path.join(tmp, f"stdio_{engine_name}_{url_source}")
                os.makedirs(sws)
                with open(os.path.join(sws, "a.py"), "w", encoding="utf-8") as fh:
                    fh.write("# workspace\n")
                stask = os.path.join(sws, "task.json")
                task_id = f"stdio-{engine_name}-{url_source}"
                _write_json(stask, {
                    "id": task_id, "cwd": sws, "model_dir": model_dir,
                    "prompt": "Add hello() to a.py.", "timeout_s": 30,
                })
                env = dict(stdio_env, LMP_SIDECAR=engine)
                if url_source == "env":
                    env["LMP_ORCH_WEBHOOK"] = webhook_url
                elif url_source == "file":
                    write_orch_webhook_file(sws, webhook_url)
                received_payloads.clear()
                out_file = os.path.join(sws, "dispatch.out")
                started = time.monotonic()
                with open(out_file, "w", encoding="utf-8") as out_fh:
                    proc = subprocess.run(
                        [sys.executable, os.path.abspath(__file__), "dispatch", "--task", stask],
                        stdin=subprocess.DEVNULL, stdout=out_fh, stderr=subprocess.STDOUT,
                        env=env, cwd=sws, timeout=60,
                    )
                elapsed = time.monotonic() - started
                with open(out_file, encoding="utf-8") as fh:
                    disp_text = fh.read()
                check(proc.returncode == EXIT_OK, f"{label} must exit 0, got {proc.returncode}: {disp_text!r}")
                check(elapsed >= 1.0, f"{label} must wait for the run, returned after {elapsed:.2f}s")
                check("detached pid=" not in disp_text, f"{label} must not detach: {disp_text!r}")
                # The .py fallback never runs a check, so its pass is UNVERIFIED.
                want_verdict = "verdict:  PASS" if engine_name == "cpp" else "verdict:  UNVERIFIED"
                check(want_verdict in disp_text, f"{label} must print the card at exit: {disp_text!r}")
                kinds = [p.get("kind") for p in received_payloads if p.get("task_id") == task_id]
                check("died" not in kinds, f"{label} must not POST died, got {kinds!r}")
                if engine_name == "py" and url_source != "none":
                    check(kinds.count("done") == 1, f"{label} must POST exactly one done, got {kinds!r}")
                if engine_name == "cpp":
                    cpp_result = load_result_file(os.path.join(sws, "result.json")) or {}
                    check(cpp_result.get("child_daemonize") == "0",
                          f"{label} must pin the engine attached, got {cpp_result.get('child_daemonize')!r}")

        # 8c. an explicit --detach really detaches, whatever the stdio (piped here):
        # the launcher returns at once with `detached pid=`, the run finishes orphaned.
        dws = os.path.join(tmp, "explicit_detach")
        os.makedirs(dws)
        dtask = os.path.join(dws, "task.json")
        _write_json(dtask, {
            "id": "explicit-detach", "cwd": dws, "model_dir": model_dir,
            "prompt": "Add hello() to a.py.", "timeout_s": 30,
        })
        started = time.monotonic()
        # Pipes on stdin and stdout, so no stdio probe could fire: only the flag can
        # make this detach (the old helper re-probed stdio and ran it attached).
        launcher = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "run", "--task", dtask,
             "--detach", "--orch-webhook", webhook_url],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=dict(stdio_env, LMP_SIDECAR=fake_cpp), cwd=dws, text=True,
        )
        launcher.stdin.close()
        launch_rc = launcher.wait(timeout=30)
        launch_s = time.monotonic() - started
        # The orphan holds the pipe until its run ends, so EOF here also means the
        # detached run is gone before the tempdir is.
        detach_text = launcher.stdout.read()
        launcher.stdout.close()
        check(launch_rc == EXIT_OK, f"--detach launcher must exit 0, got {launch_rc}")
        check(launch_s < 1.0, f"--detach launcher must return at once, took {launch_s:.2f}s")
        check("detached pid=" in detach_text, f"--detach must detach under piped stdio: {detach_text!r}")
        check(os.path.isfile(os.path.join(dws, "result.json")),
              "detached run must still finish and write result.json")

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
        expected_msg = (
            "piper init: PIPER.md, .cursor/rules/piper-parent.mdc, "
            ".cursor/skills/piper-orchestration/SKILL.md, "
            ".agents/skills/piper-orchestration/SKILL.md. "
            "Godoer briefs left alone. Stay attached, or pass --orch-webhook."
        )
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
        check("piper wake-url" in c_content, "cursor rule must teach piper wake-url")

        skill_bodies = []
        for rel in ORCH_SKILL_RELS:
            skill_path = os.path.join(godoer_ws, rel)
            check(os.path.isfile(skill_path), f"{rel} must exist after init")
            with open(skill_path, "r", encoding="utf-8") as fh:
                skill_bodies.append(fh.read())
        check(skill_bodies[0] == skill_bodies[1], "Cursor and Codex skills must match")
        skill_body = skill_bodies[0]
        check("piper packet" in skill_body and "piper dispatch" in skill_body,
              "init skill must teach packet/dispatch")
        check("write answer.json" not in skill_body and "writes `answer.json`" not in skill_body,
              "init skill must not teach freehand answer.json")

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
        check(loaded["model_dir"] == os.path.abspath(model_dir),
              f"emitted model_dir, got {loaded.get('model_dir')!r}")
        with open(os.path.join(tmp, ".piper", "active.json"), encoding="utf-8") as fh:
            active = json.load(fh)
        check(active.get("id") == "emit-1" and active.get("task_path") == os.path.abspath(pkt_out),
              f"active.json must point at emitted packet, got {active!r}")
        with open(pkt_out, encoding="utf-8") as fh:
            check("max_iterations" not in json.load(fh),
                  "packet without --max-iterations must leave the default to the worker")
        iters_out = os.path.join(tmp, "emitted_iters.json")
        check(
            main(["packet", "--id", "emit-iters", "--cwd", tmp, "--prompt", "Ship X.",
                  "--out", iters_out, "--model-dir", model_dir,
                  "--max-iterations", "45"]) == EXIT_OK,
            "packet --max-iterations must exit 0",
        )
        with open(iters_out, encoding="utf-8") as fh:
            iters_raw = json.load(fh)
        check(iters_raw.get("max_iterations") == 45,
              f"packet --max-iterations must emit max_iterations, got {iters_raw!r}")
        with contextlib.redirect_stderr(io.StringIO()):
            bad_iters = main(["packet", "--id", "emit-iters0", "--cwd", tmp, "--prompt", "X.",
                              "--out", os.path.join(tmp, "emitted_iters0.json"),
                              "--model-dir", model_dir, "--max-iterations", "0"])
        check(bad_iters == EXIT_INVALID, f"--max-iterations 0 must exit 3, got {bad_iters}")
        clock_out = os.path.join(tmp, "emitted_clock.json")
        check(main(["packet", "--id", "emit-clock", "--cwd", tmp, "--prompt", "X.", "--out", clock_out,
                    "--model-dir", model_dir, "--check", "true", "--check-timeout-s", "900"]) == EXIT_OK,
              "packet --check-timeout-s must exit 0")
        with open(clock_out, encoding="utf-8") as fh:
            check(json.load(fh).get("check_timeout_s") == 900.0,
                  "packet --check-timeout-s must emit check_timeout_s")

        # 15b. default out path, prompt copy, model from env, missing model exits 3
        def_ws = os.path.join(tmp, "default_slice_ws")
        os.makedirs(def_ws)
        brief = os.path.join(def_ws, "brief.md")
        with open(brief, "w", encoding="utf-8") as fh:
            fh.write("## This Slice Only\nDo the thing.\n")
        saved_model = os.environ.get("LMP_QWEN_DIR")
        os.environ["LMP_QWEN_DIR"] = model_dir
        def_stdout = io.StringIO()
        with contextlib.redirect_stdout(def_stdout):
            def_rc = main(["packet", "--id", "slice-001", "--cwd", def_ws,
                           "--prompt-file", brief, "--check", "true"])
        expected_task = os.path.join(def_ws, ".piper", "slices", "slice-001", "task.json")
        check(def_rc == EXIT_OK, f"default packet path must exit 0, got {def_rc}")
        check(def_stdout.getvalue().strip() == expected_task,
              f"default packet path, got {def_stdout.getvalue()!r}")
        check(os.path.isfile(os.path.join(def_ws, ".piper", "slices", "slice-001", "prompt.md")),
              "prompt.md must be copied beside the default packet")
        def_loaded = load_packet(expected_task)
        check(def_loaded["model_dir"] == os.path.abspath(model_dir),
              f"env model must be baked in, got {def_loaded.get('model_dir')!r}")
        os.environ.pop("LMP_QWEN_DIR", None)
        with contextlib.redirect_stderr(io.StringIO()):
            miss_rc = main(["packet", "--id", "no-model", "--cwd", def_ws, "--prompt", "x"])
        check(miss_rc == EXIT_INVALID, f"packet without model must exit 3, got {miss_rc}")
        if saved_model is None:
            os.environ.pop("LMP_QWEN_DIR", None)
        else:
            os.environ["LMP_QWEN_DIR"] = saved_model

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

        # 16b. wake-url --task must not require model_dir (emit is not this path)
        pkt_no_model = os.path.join(tmp, "wake_task_no_model.json")
        with open(pkt_no_model, "w", encoding="utf-8") as fh:
            json.dump({"id": "wake-nm", "cwd": wake_root, "prompt": "x",
                       "result_path": os.path.join(wake_root, "result.json")}, fh)
        wake_stdout2 = io.StringIO()
        with contextlib.redirect_stdout(wake_stdout2):
            wake_rc2 = main(["wake-url", "--task", pkt_no_model])
        check(wake_rc2 == EXIT_OK, f"wake-url --task without model_dir must exit 0, got {wake_rc2}")
        check(wake_stdout2.getvalue().strip() == wake_url,
              f"wake-url --task must print file URL, got {wake_stdout2.getvalue()!r}")

        # 16c. status/review/await --task must not require model_dir
        status_root = os.path.join(tmp, "status_no_model")
        os.makedirs(status_root, exist_ok=True)
        pkt_status = os.path.join(tmp, "status_task_no_model.json")
        with open(pkt_status, "w", encoding="utf-8") as fh:
            json.dump({"id": "status-nm", "cwd": status_root, "prompt": "x",
                       "result_path": os.path.join(status_root, "result.json")}, fh)
        # Clear any leftover error result from earlier scenarios that share tmp.
        with open(os.path.join(status_root, "result.json"), "w", encoding="utf-8") as fh:
            json.dump({"status": "ok", "task_id": "status-nm", "message": "done"}, fh)
        st_out = io.StringIO()
        with contextlib.redirect_stdout(st_out):
            st_rc = main(["status", "--task", pkt_status])
        check(st_rc == EXIT_OK, f"status --task without model_dir must exit 0, got {st_rc}")
        check("state:    done" in st_out.getvalue(),
              f"status --task card must show done, got {st_out.getvalue()!r}")
        rev_out = io.StringIO()
        with contextlib.redirect_stdout(rev_out):
            rev_rc = main(["review", "--task", pkt_status])
        check(rev_rc == EXIT_OK, f"review --task without model_dir must exit 0, got {rev_rc}")
        aw_out = io.StringIO()
        with contextlib.redirect_stdout(aw_out):
            aw_rc = main(["await", "--task", pkt_status, "--timeout-s", "1", "--interval-s", "0.1"])
        check(aw_rc == EXIT_OK, f"await --task without model_dir must exit 0, got {aw_rc}")
        check("state:    done" in aw_out.getvalue(),
              f"await --task card must show done, got {aw_out.getvalue()!r}")

        # 16d. answer --task must not require model_dir
        ans_out = io.StringIO()
        with contextlib.redirect_stdout(ans_out):
            ans_rc = main(["answer", "allow", "--task", pkt_status])
        check(ans_rc == EXIT_OK, f"answer --task without model_dir must exit 0, got {ans_rc}")
        ans_path = os.path.join(status_root, "answer.json")
        check(os.path.isfile(ans_path), "answer --task must write answer.json beside result")

        # 16e. relative result_path joins to task.json dir (not process cwd)
        rel_root = os.path.join(tmp, "rel_task_dir")
        os.makedirs(rel_root, exist_ok=True)
        rel_pkt = os.path.join(rel_root, "task.json")
        with open(rel_pkt, "w", encoding="utf-8") as fh:
            json.dump({
                "id": "rel",
                "cwd": rel_root,
                "prompt": "x",
                "result_path": "onlyhere/result.json",
            }, fh)
        onlyhere = os.path.join(rel_root, "onlyhere")
        os.makedirs(onlyhere, exist_ok=True)
        with open(os.path.join(onlyhere, "result.json"), "w", encoding="utf-8") as fh:
            json.dump({"status": "ok", "task_id": "rel", "message": "rel-ok"}, fh)
        other = os.path.join(tmp, "other_cwd")
        os.makedirs(other, exist_ok=True)
        prev = os.getcwd()
        try:
            os.chdir(other)
            rel_out = io.StringIO()
            with contextlib.redirect_stdout(rel_out):
                rel_rc = main(["status", "--task", rel_pkt])
        finally:
            os.chdir(prev)
        check(rel_rc == EXIT_OK, f"status relative result_path must exit 0, got {rel_rc}")
        check("state:    done" in rel_out.getvalue(),
              f"status relative result_path must find done, got {rel_out.getvalue()!r}")

        # 16f. wake-url --task accepts a directory containing task.json
        wake_dir = os.path.join(tmp, "wake_dir_packet")
        os.makedirs(os.path.join(wake_dir, ".piper"), exist_ok=True)
        with open(os.path.join(wake_dir, ".piper", "orch_webhook"), "w", encoding="utf-8") as fh:
            fh.write(wake_url + "\n")
        with open(os.path.join(wake_dir, "task.json"), "w", encoding="utf-8") as fh:
            json.dump({"id": "wake-dir", "cwd": wake_dir, "prompt": "x",
                       "result_path": os.path.join(wake_dir, "result.json")}, fh)
        wake_dir_out = io.StringIO()
        with contextlib.redirect_stdout(wake_dir_out):
            wake_dir_rc = main(["wake-url", "--task", wake_dir])
        check(wake_dir_rc == EXIT_OK, f"wake-url --task DIR must exit 0, got {wake_dir_rc}")
        check(wake_dir_out.getvalue().strip() == wake_url,
              f"wake-url --task DIR must print URL, got {wake_dir_out.getvalue()!r}")

        # 16g. dispatch must not advertise --detach
        help_out = io.StringIO()
        with contextlib.redirect_stdout(help_out):
            try:
                main(["dispatch", "--help"])
            except SystemExit as exc:
                check(exc.code in (0, None), f"dispatch --help exit {exc.code}")
        check("  --detach" not in help_out.getvalue(),
              "dispatch --help must not advertise a --detach flag")

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
        check("verdict:  UNVERIFIED" in rev_out,
              f"an ok result with no check must be UNVERIFIED, got {rev_out!r}")
        check("rev-1" in rev_out and "a.py" in rev_out, f"review card must show task/files, got {rev_out!r}")
        check("loop:" not in rev_out, f"a normal ending must not print a loop line, got {rev_out!r}")

        # 18b. a pass that rests on the check alone says so; a stall names its stop
        promo = result_shell(
            task_id="promo-1", cwd=rev_dir, model_dir=model_dir, status="ok",
            message="Halfway there.", turns=30,
        )
        promo.update({"termination_reason": "max_turns", "promoted_by_check": True,
                      "test": {"ran": True, "exit_code": 0, "command": "make test",
                               "output_tail": "ok"}})
        promo_card = format_review_card(promo, exit_code=EXIT_OK, result_path=rev_path)
        check("loop:     max_turns after 30 turns; ok because check passed" in promo_card,
              f"promoted card must carry the loop line, got {promo_card!r}")
        check("verdict:  PASS" in promo_card, f"promoted card must PASS, got {promo_card!r}")
        stall = dict(promo, status="stalled", promoted_by_check=False,
                     error="agent did not complete (max_turns)")
        stall["test"] = dict(promo["test"], exit_code=1)
        stall_card = format_review_card(stall, exit_code=EXIT_ERROR, result_path=rev_path)
        check("loop:     max_turns after 30 turns\n" in stall_card,
              f"stalled card must name the stop, got {stall_card!r}")
        check("verdict:  STALLED" in stall_card, f"stalled card must be STALLED, got {stall_card!r}")
        check(build_answer_payload(action="approved") == {"text": "allow"},
              "approved must normalize to allow")
        check(build_answer_payload(action="denied") == {"text": "deny"},
              "denied must normalize to deny")

        # 18c. non-PASS cards carry their evidence; PASS stays short
        ev_dir = os.path.join(tmp, "evidence")
        os.makedirs(ev_dir)
        ev_path = os.path.join(ev_dir, "result.json")
        with open(os.path.join(ev_dir, WORKER_STDERR_LOG), "w", encoding="utf-8") as fh:
            fh.write("mem at=decode_end tokens=5396 active=1\n"
                     "piper: awaiting_user.json written (seq 7). Question: which file?\n"
                     "mem at=decode_end tokens=5400 active=1\n"
                     "piper: task ev finished with status 'error' (exit 1, 9.0s, 4 turns)\n")
        base = result_shell(task_id="ev", cwd=ev_dir, model_dir=model_dir, status="ok",
                            message="All done, tests pass.", turns=4, wall_seconds=9.0)
        red = dict(base, status="error", error="check command failed (exit code 1)",
                   test={"ran": True, "exit_code": 1, "command": "python3 test_calc.py",
                         "output_tail": "Traceback (most recent call last):\n"
                                        "  File \"test_calc.py\", line 3, in <module>\n"
                                        "    assert add(2, 3) == 5\n"
                                        "AssertionError: expected 5, got None\n"})
        red_card = format_review_card(red, exit_code=EXIT_ERROR, result_path=ev_path)
        check("error:    check command failed (exit code 1)" in red_card,
              f"red card must carry the error, got {red_card!r}")
        check("check:    | Traceback" in red_card and "| AssertionError: expected 5, got None" in red_card,
              f"red card must carry the failing check's tail, got {red_card!r}")
        check("run:      turns=4 wall=9s" in red_card, f"card must carry turns/wall, got {red_card!r}")
        check("worker:   | piper: awaiting_user.json written" in red_card
              and "| piper: task ev finished" in red_card,
              f"non-PASS card must bring back the worker's piper: notices, got {red_card!r}")
        check("mem at=" not in red_card, f"telemetry must stay off the card, got {red_card!r}")
        check("verdict:  FAIL" in red_card, f"red card must FAIL, got {red_card!r}")
        check(len(red_card.splitlines()) <= 25, f"card must stay compact, got {len(red_card.splitlines())} lines")

        timeout_res = dict(red, error="check command failed (exit code -1)",
                           test={"ran": True, "exit_code": -1, "command": "make test",
                                 "output_tail": "running 312 tests\n\n[timeout after 60.000000s]"})
        to_card = format_review_card(timeout_res, exit_code=EXIT_ERROR, result_path=ev_path)
        check("| [timeout after 60.000000s]" in to_card,
              f"a timed-out check must show its timeout marker, got {to_card!r}")

        stalled_res = dict(base, status="stalled", error="agent did not complete (max_turns)",
                           termination_reason="max_turns", turns=30,
                           test={"ran": True, "exit_code": 2, "command": "make test",
                                 "output_tail": "FAILED test_x"})
        st_card = format_review_card(stalled_res, exit_code=EXIT_ERROR, result_path=ev_path)
        check("error:    agent did not complete (max_turns)" in st_card,
              f"stalled card must say why, got {st_card!r}")
        check("verdict:  STALLED" in st_card, f"stalled card must be STALLED, got {st_card!r}")

        green = dict(base, test={"ran": True, "exit_code": 0, "command": "make test",
                                 "output_tail": "ok"})
        pass_card = format_review_card(green, exit_code=EXIT_OK, result_path=ev_path)
        check("verdict:  PASS" in pass_card, f"green card must PASS, got {pass_card!r}")
        check("error:" not in pass_card and "check:" not in pass_card and "worker:" not in pass_card,
              f"PASS card must not carry failure evidence, got {pass_card!r}")
        check(pass_card == (
            "── piper review ─────────────────────────\n"
            "task:     ev\n"
            "status:   ok\n"
            "exit:     0\n"
            "run:      turns=4 wall=9s\n"
            "message:  All done, tests pass.\n"
            "files:    (none)\n"
            "diff:     +0 -0 (0 files)\n"
            "test:     exit=0 cmd='make test'\n"
            "git_diff: (none)\n"
            "verdict:  PASS\n"
            "─────────────────────────────────────────"
        ), f"PASS card must stay byte-stable, got {pass_card!r}")

        # A red check's log_triage block: failing tests, the first located primary
        # diagnostic, and the spooled full log -- at most three lines.
        triaged = dict(red, test=dict(red["test"], output_path=os.path.join(ev_dir, "check.log"),
                                      seconds=4.2, timed_out=False, could_not_run=False,
                                      triage={"runner": "pytest", "passed": 3, "failed": 1,
                                              "failing_tests": ["tests/test_calc.py::test_add"],
                                              "primary": [{"path": "", "line": 0, "message": "noise"},
                                                          {"path": "calc.py", "line": 2,
                                                           "message": "AssertionError: expected 5"}],
                                              "paths": ["calc.py"]}))
        tri_card = format_review_card(triaged, exit_code=EXIT_ERROR, result_path=ev_path)
        check("failing:  tests/test_calc.py::test_add\n" in tri_card,
              f"red card must name failing tests, got {tri_card!r}")
        check("at:       calc.py:2: AssertionError: expected 5\n" in tri_card,
              f"red card must name the first located diagnostic, got {tri_card!r}")
        check(f"log:      {os.path.join(ev_dir, 'check.log')}\n" in tri_card,
              f"red card must point at the full check log, got {tri_card!r}")
        check(len(tri_card.splitlines()) <= 25, f"triaged card must stay compact, got {tri_card!r}")
        timed = dict(red, error="check timed out after 300s",
                     test={"ran": True, "exit_code": -1, "command": "make test", "timed_out": True,
                           "could_not_run": False, "seconds": 300.1,
                           "output_tail": "running\n[timeout after 300.000000s]"})
        timed_card = format_review_card(timed, exit_code=EXIT_ERROR, result_path=ev_path)
        check("test:     timed out after 300s cmd='make test'" in timed_card,
              f"a timed-out check must say so, not exit=-1, got {timed_card!r}")
        never = dict(red, error="check could not run (exit 127)",
                     test={"ran": True, "exit_code": 127, "command": "maek test", "timed_out": False,
                           "could_not_run": True, "seconds": 0.01,
                           "output_tail": "sh: maek: command not found"})
        never_card = format_review_card(never, exit_code=EXIT_ERROR, result_path=ev_path)
        check("test:     could not run (exit 127) cmd='maek test'" in never_card,
              f"a check that never ran must say so, got {never_card!r}")

        # DIED: no result.json, so the log is the only evidence. Only this run's log,
        # never the previous run's (rotated to worker.stderr.prev.log).
        died_dir = os.path.join(tmp, "died")
        os.makedirs(died_dir)
        died_result = os.path.join(died_dir, "result.json")
        with open(os.path.join(died_dir, WORKER_STDERR_PREV_LOG), "w", encoding="utf-8") as fh:
            fh.write("piper: task old finished with status 'ok' (exit 0, 3.0s, 2 turns)\n")
        with open(os.path.join(died_dir, WORKER_STDERR_LOG), "w", encoding="utf-8") as fh:
            fh.write("mem at=generate_enter tokens=9000 active=1\n"
                     "libc++abi: terminating due to uncaught exception: [metal] OOM\n")
        died_card = format_review_card(None, exit_code=-9, result_path=died_result)
        check("verdict:  DIED" in died_card, f"no result must be DIED, got {died_card!r}")
        check(f"log:      {os.path.join(died_dir, WORKER_STDERR_LOG)}" in died_card,
              f"DIED card must name the log, got {died_card!r}")
        check("[metal] OOM" in died_card, f"DIED card must show the crash breadcrumbs, got {died_card!r}")
        check("task old finished" not in died_card,
              f"DIED card must not show the previous run's log, got {died_card!r}")

        # 18d. dispatch output stays a card: telemetry lands in worker.stderr.log,
        # and each run starts a fresh log (the previous one is rotated, not appended).
        noisy = os.path.join(tmp, "fake_cpp_noisy")
        _write_exec(noisy, FAKE_CPP_NOISY)
        nws = os.path.join(tmp, "noisy_ws")
        os.makedirs(nws)
        ntask = os.path.join(nws, "task.json")
        _write_json(ntask, {"id": "noisy", "cwd": nws, "model_dir": model_dir,
                            "prompt": "Add hello() to a.py.", "timeout_s": 30,
                            "result_path": os.path.join(nws, "result.json")})
        noisy_env = {k: v for k, v in os.environ.items()
                     if k not in ("LMP_DAEMONIZE", "LMP_ORCH_WEBHOOK", "LMP_WORKER_STDERR")}
        noisy_env.update(LMP_SIDECAR=noisy, LMP_USE_CPP_WORKER="1")
        for attempt in (1, 2):
            nout = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "dispatch", "--task", ntask],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=noisy_env, cwd=nws, timeout=60,
            )
            check(nout.returncode == EXIT_OK, f"noisy dispatch #{attempt} must exit 0, got {nout.returncode}")
            check(len(nout.stdout) < 1500,
                  f"dispatch output must be a compact card, got {len(nout.stdout)} bytes: {nout.stdout[:300]!r}")
            check(b"verdict:  PASS" in nout.stdout, f"noisy dispatch must PASS, got {nout.stdout!r}")
        nlog = _read_log_lines(os.path.join(nws, WORKER_STDERR_LOG))
        check(sum(1 for ln in nlog if ln.startswith("mem at=")) == 210,
              f"telemetry must land in worker.stderr.log, got {len(nlog)} lines")
        check(sum(1 for ln in nlog if ln.startswith("piper: task noisy finished")) == 1,
              "each run must start a fresh worker.stderr.log")
        check(os.path.isfile(os.path.join(nws, WORKER_STDERR_PREV_LOG)),
              "the previous run's log must be kept as worker.stderr.prev.log")
        inherit_env = dict(noisy_env, LMP_WORKER_STDERR="inherit")
        iout = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "dispatch", "--task", ntask],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=inherit_env, cwd=nws, timeout=60,
        )
        check(iout.stdout.count(b"mem at=") == 210,
              "LMP_WORKER_STDERR=inherit must keep worker stderr inline")

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
        check("verdict:  UNVERIFIED" in disp_out,
              f"dispatch of a no-check packet must print an UNVERIFIED card, got {disp_out!r}")
        check(os.path.isfile(os.path.join(disp_ws, "result.json")),
              "dispatch must leave result.json")
        orch_path = os.path.join(disp_ws, "orch.jsonl")
        check(os.path.isfile(orch_path), "dispatch must append orch.jsonl")
        orch_lines = []
        with open(orch_path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    orch_lines.append(json.loads(raw))
        check(any(line.get("kind") == "dispatch" for line in orch_lines),
              f"orch.jsonl must record dispatch, got {orch_lines!r}")
        check(any(line.get("kind") == "review" and line.get("verdict") == "UNVERIFIED"
                  for line in orch_lines),
              f"orch.jsonl must record the review verdict, got {orch_lines!r}")

        # 20. status card from awaiting_user.json / result.json (no freehand cat/jq)
        st_dir = os.path.join(tmp, "status_dir")
        os.makedirs(st_dir)
        idle_stdout = io.StringIO()
        with contextlib.redirect_stdout(idle_stdout):
            idle_rc = main(["status", "--dir", st_dir])
        check(idle_rc == EXIT_OK, f"status idle must exit 0, got {idle_rc}")
        check("state:    idle" in idle_stdout.getvalue(),
              f"status idle card, got {idle_stdout.getvalue()!r}")
        with open(os.path.join(st_dir, "awaiting_user.json"), "w", encoding="utf-8") as fh:
            json.dump({"question": "Allow irreversible delete?", "options": "allow,deny",
                       "run_id": "r1", "seq": 7}, fh)
        ask_stdout = io.StringIO()
        with contextlib.redirect_stdout(ask_stdout):
            ask_rc = main(["status", "--dir", st_dir])
        check(ask_rc == EXIT_OK, f"status ask must exit 0, got {ask_rc}")
        ask_out = ask_stdout.getvalue()
        check("state:    ask" in ask_out and "Allow irreversible delete?" in ask_out,
              f"status ask card, got {ask_out!r}")
        check("piper answer" in ask_out, f"status ask must point at piper answer, got {ask_out!r}")
        os.remove(os.path.join(st_dir, "awaiting_user.json"))
        with open(os.path.join(st_dir, "result.json"), "w", encoding="utf-8") as fh:
            json.dump(result_shell(
                task_id="st-1", cwd=st_dir, model_dir=model_dir, status="ok",
                message="Done.", files_touched=["z.py"],
            ), fh)
        done_stdout = io.StringIO()
        with contextlib.redirect_stdout(done_stdout):
            done_rc = main(["status", "--dir", st_dir])
        check(done_rc == EXIT_OK, f"status done must exit 0, got {done_rc}")
        check("state:    done" in done_stdout.getvalue(),
              f"status done card, got {done_stdout.getvalue()!r}")

        # 21. await returns when ask appears (no hand-rolled sleep snippet)
        await_dir = os.path.join(tmp, "await_dir")
        os.makedirs(await_dir)

        def _write_ask_later():
            time.sleep(0.15)
            with open(os.path.join(await_dir, "awaiting_user.json"), "w", encoding="utf-8") as fh:
                json.dump({"question": "Continue?", "options": "allow,deny",
                           "run_id": "r2", "seq": 1}, fh)

        thr = threading.Thread(target=_write_ask_later, daemon=True)
        thr.start()
        await_stdout = io.StringIO()
        with contextlib.redirect_stdout(await_stdout):
            await_rc = main(["await", "--dir", await_dir, "--timeout-s", "2",
                             "--interval-s", "0.05"])
        thr.join(timeout=2)
        check(await_rc == EXIT_OK, f"await ask must exit 0, got {await_rc}")
        check("state:    ask" in await_stdout.getvalue(),
              f"await must print ask card, got {await_stdout.getvalue()!r}")
        # idle timeout
        empty_await = os.path.join(tmp, "await_empty")
        os.makedirs(empty_await)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            to_rc = main(["await", "--dir", empty_await, "--timeout-s", "0.2",
                          "--interval-s", "0.05"])
        check(to_rc == EXIT_TIMEOUT, f"await idle timeout must exit 2, got {to_rc}")

        # 22. mcp-list + packet --trust-mcp (explicit consent, no hand JSON array)
        mcp_ws = os.path.join(tmp, "mcp_ws")
        os.makedirs(mcp_ws)
        with open(os.path.join(mcp_ws, ".mcp.json"), "w", encoding="utf-8") as fh:
            json.dump({
                "mcpServers": {
                    "godoer": {"command": "godoer", "args": ["mcp"]},
                    "memory": {"command": "npx", "args": ["-y", "server-memory"]},
                }
            }, fh)
        mcp_out = io.StringIO()
        with contextlib.redirect_stdout(mcp_out):
            mcp_rc = main(["mcp-list", "--cwd", mcp_ws])
        check(mcp_rc == EXIT_OK, f"mcp-list must exit 0, got {mcp_rc}")
        listed = [ln.strip() for ln in mcp_out.getvalue().splitlines() if ln.strip()]
        check(listed == ["godoer", "memory"], f"mcp-list names, got {listed!r}")
        trust_out = os.path.join(tmp, "trust_task.json")
        check(
            main(["packet", "--id", "trust-1", "--cwd", mcp_ws, "--prompt", "Use Godoer.",
                  "--out", trust_out, "--model-dir", model_dir, "--trust-mcp", "godoer"]) == EXIT_OK,
            "packet --trust-mcp must exit 0",
        )
        trusted = load_packet(trust_out)
        check(trusted.get("trust_mcp") == ["godoer"],
              f"trust_mcp field, got {trusted.get('trust_mcp')!r}")
        with contextlib.redirect_stderr(io.StringIO()):
            bad_trust = main(["packet", "--id", "trust-bad", "--cwd", mcp_ws, "--prompt", "x",
                              "--out", os.path.join(tmp, "bad_trust.json"),
                              "--model-dir", model_dir, "--trust-mcp", "nope"])
        check(bad_trust == EXIT_INVALID, f"unknown trust_mcp must exit 3, got {bad_trust}")

        # 23. progress log append (no freehand progress paste)
        prog_root = os.path.join(tmp, "prog_ws")
        os.makedirs(prog_root)
        prog_stdout = io.StringIO()
        with contextlib.redirect_stdout(prog_stdout):
            prog_rc = main(["progress", "--id", "slice-001", "pass",
                            "--note", "validator + test", "--dir", prog_root])
        check(prog_rc == EXIT_OK, f"progress must exit 0, got {prog_rc}")
        prog_path = os.path.join(prog_root, ".piper", "progress.log")
        check(os.path.isfile(prog_path), "progress.log must exist")
        with open(prog_path, encoding="utf-8") as fh:
            prog_body = fh.read()
        check("slice-001 | pass | validator + test\n" in prog_body,
              f"progress line shape, got {prog_body!r}")
        check(format_progress_line("s2", "ok", "note") == "s2 | pass | note",
              "progress ok alias must normalize to pass")
        with contextlib.redirect_stderr(io.StringIO()):
            bad_prog = main(["progress", "--id", "s3", "maybe", "--dir", prog_root])
        check(bad_prog == EXIT_INVALID, f"bad progress verdict must exit 3, got {bad_prog}")

        # 24. piper init cursor rule points at helpers (no freehand answer.json teaching)
        init_ws = os.path.join(tmp, "init_helpers")
        os.makedirs(init_ws)
        with contextlib.redirect_stdout(io.StringIO()):
            check(main(["init", init_ws]) == EXIT_OK, "init helpers workspace must exit 0")
        rule_path = os.path.join(init_ws, ".cursor", "rules", "piper-parent.mdc")
        with open(rule_path, encoding="utf-8") as fh:
            rule_body = fh.read()
        check("piper answer allow|deny|--text" in rule_body,
              "init rule must teach piper answer")
        check("piper packet" in rule_body and "piper dispatch" in rule_body,
              "init rule must teach packet/dispatch")
        check("piper progress" in rule_body, "init rule must teach progress")
        check('write `answer.json`' not in rule_body and "write answer.json" not in rule_body,
              "init rule must not teach freehand answer.json paste")

        httpd.shutdown()

    for line in failures:
        print(f"  FAIL: {line}")
    print(f"  piper_worker self-test: 30 scenario(s), {len(failures)} failure(s)")
    return EXIT_ERROR if failures else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
