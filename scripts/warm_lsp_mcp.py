#!/usr/bin/env python3
"""Warm LSP behind one MCP server.

Speaks newline-delimited MCP on stdio and Content-Length LSP to stock
language servers. LMP_WARM_LSP=1 starts them. Any other value registers the
tools and returns a tool error without spawning.

The MCP process itself is outside Seatbelt, the same as every other trusted
server Piper hosts. Each language server is exec'd under sandbox-exec with
seatbelt_profile(): workspace read/write, toolchain paths outside $HOME, and
no home file contents.
"""

from __future__ import annotations

import json
import os
import queue
import selectors
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

PROTOCOL = "2025-06-18"
SERVER_NAME = "warm-lsp"
INIT_TIMEOUT_S = 30.0
DIAG_TIMEOUT_S = 20.0
STDERR_CAP = 200

CPP_EXTS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
PY_EXTS = {".py", ".pyi"}
# sourcekitd aborts under this Seatbelt profile. GDScript uses the Godoer fork.
STUBS = {
    ".swift": (
        "sourcekit-lsp is parked: under the warm-lsp Seatbelt profile sourcekitd "
        "aborts with 'Service is invalid' and publishes no diagnostics"
    ),
}

# Sean's Godoer fork. Not /Applications/Godot.app. Override with
# LMP_GODOT_BIN, GODOER_GODOT_BIN, or GODOT_BIN when the checkout moves.
DEFAULT_GODOER_GODOT = Path("/Users/dev/godot-godoer/bin/godot.macos.editor.arm64")


def enabled() -> bool:
    return os.environ.get("LMP_WARM_LSP") == "1"


def workspace_root() -> Path:
    return Path.cwd().resolve()


def sb_escape(path: str) -> str:
    return path.replace("\\", "\\\\").replace('"', '\\"')


def ancestor_literals(root: Path) -> list[str]:
    """Directory nodes above the workspace. realpath lstats each of them.

    The workspace lives under $HOME. Denying home reads without allowing
    metadata on these literals makes clangd fail with 'Failed to resolve
    path'. literal, not subpath: siblings of the workspace stay unreadable.
    """
    resolved = root.resolve()
    out = []
    cur = Path(resolved.anchor)
    for part in resolved.parts[1:-1]:
        cur = cur / part
        out.append(str(cur))
    return out


def godot_bin() -> Path:
    """The Godoer editor binary. Explicit env wins; otherwise the fork path."""
    for key in ("LMP_GODOT_BIN", "GODOER_GODOT_BIN", "GODOT_BIN"):
        raw = os.environ.get(key, "").strip()
        if raw:
            path = Path(raw).expanduser()
            if path.is_file():
                return path.resolve()
            raise LspError(f"{key}={raw} is not an executable file")
    if DEFAULT_GODOER_GODOT.is_file():
        return DEFAULT_GODOER_GODOT.resolve()
    raise LspError(
        "Godoer Godot was not found at "
        f"{DEFAULT_GODOER_GODOT}. Set GODOER_GODOT_BIN. "
        "Stock /Applications/Godot.app is not used."
    )


def _godot_bin_status() -> str:
    try:
        return str(godot_bin())
    except LspError:
        return ""


def godot_engine_root(binary: Path) -> Path:
    """Checkout that contains the editor binary, so Seatbelt can read it."""
    resolved = binary.resolve()
    if resolved.parent.name == "bin":
        return resolved.parent.parent
    return resolved.parent


def seatbelt_profile(root: Path | None = None, extra_reads: list[Path] | None = None) -> str:
    """Deny-by-default home reads. Last matching file-read* rule wins.

    A later allow of file-read* does not override an earlier deny of the
    narrower file-read-data filter, so the home deny is file-read* and the
    workspace allow is file-read* too.
    """
    root = (root or workspace_root()).resolve()
    home = str(Path.home())
    lines = [
        "(version 1)",
        "(allow default)",
        '(deny network*)',
        '(allow network-outbound (remote ip "localhost:*"))',
        '(allow network-inbound (local ip "localhost:*"))',
        '(allow network-bind (local ip "localhost:*"))',
        f'(deny file-read* (subpath "{sb_escape(home)}"))',
    ]
    for literal in ancestor_literals(root):
        lines.append(f'(allow file-read-metadata (literal "{sb_escape(literal)}"))')
    lines.append(f'(allow file-read* (subpath "{sb_escape(str(root))}"))')
    for extra in extra_reads or []:
        # The Godoer checkout lives under $HOME, outside the workspace. This
        # allow is that tree only. It does not reopen the rest of $HOME.
        escaped = sb_escape(str(extra.resolve()))
        lines.append(f'(allow file-read* (subpath "{escaped}"))')
        lines.append(f'(allow file-map-executable (subpath "{escaped}"))')
        lines.append(f'(allow process-exec* (subpath "{escaped}"))')
    lines.append("(deny file-write*)")
    lines.append(f'(allow file-write* (subpath "{sb_escape(str(root))}"))')
    lines.append('(allow file-write-data (literal "/dev/null"))')
    lines.append('(allow file-write-data (literal "/dev/stdout"))')
    lines.append('(allow file-write-data (literal "/dev/stderr"))')
    return "\n".join(lines) + "\n"


def path_inside(root: Path, raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    # Security: Ensure path resides strictly inside root to prevent path traversal attacks
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes the workspace: {raw}")
    return resolved


def file_uri(path: Path) -> str:
    return "file://" + quote(str(path.resolve()))


def uri_path(uri: str) -> str:
    parsed = urlparse(uri)
    return unquote(parsed.path)


def language_of(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in CPP_EXTS:
        return "cpp"
    if ext in PY_EXTS:
        return "python"
    if ext == ".swift":
        return "swift"
    if ext == ".gd":
        return "gdscript"
    return None


def nearest_root(path: Path, markers: tuple[str, ...], stop: Path) -> Path:
    cur = path if path.is_dir() else path.parent
    stop = stop.resolve()
    while True:
        if any((cur / marker).is_file() for marker in markers):
            return cur
        if cur == stop or cur.parent == cur:
            return path.parent if path.is_file() else path
        cur = cur.parent


def _free_local_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def pyright_site() -> Path | None:
    lib = workspace_root() / ".piper" / "lsp-py" / "lib"
    if not lib.is_dir():
        return None
    sites = sorted(lib.glob("python*/site-packages"))
    return sites[-1] if sites else None


def python_bin() -> str:
    brew = "/opt/homebrew/bin/python3"
    if os.path.isfile(brew) and os.access(brew, os.X_OK):
        return brew
    return sys.executable


class LspError(Exception):
    pass


class LspSession:
    def __init__(
        self,
        language: str,
        root: Path,
        argv: list[str],
        extra_env: dict[str, str],
        tcp_port: int | None = None,
        read_roots: list[Path] | None = None,
    ):
        self.language = language
        self.root = root
        self.argv = argv
        self.extra_env = extra_env
        self.tcp_port = tcp_port
        self.read_roots = list(read_roots or [])
        self.proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        self._out = None
        self._stream = None
        self.stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[int, queue.Queue] = {}
        self._pending_lock = threading.Lock()
        self._notes: queue.Queue = queue.Queue()
        self._next_id = 1
        self._versions: dict[str, int] = {}
        self._texts: dict[str, str] = {}
        self._last_diags: dict[str, list] = {}
        self.sync_full = True
        self._buf = b""
        self._reader: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self.started = time.monotonic()

    def stderr_tail(self) -> str:
        with self._stderr_lock:
            return "\n".join(self.stderr_lines[-20:])

    def start(self) -> None:
        scratch = workspace_root() / ".piper"
        tmp = scratch / "lsp-tmp"
        cache = scratch / "lsp-cache"
        tmp.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/opt/homebrew/bin"),
            "HOME": os.environ.get("HOME", ""),
            "USER": os.environ.get("USER", ""),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "TMPDIR": str(tmp),
            "TMP": str(tmp),
            "TEMP": str(tmp),
            "XDG_CACHE_HOME": str(cache),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
        env.update(self.extra_env)
        profile = seatbelt_profile(workspace_root(), self.read_roots)
        cmd = ["sandbox-exec", "-p", profile, *self.argv]
        if self.tcp_port is None:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.root),
                env=env,
                start_new_session=True,
            )
            assert self.proc.stdout and self.proc.stderr and self.proc.stdin
            self._out = self.proc.stdin
            self._stream = self.proc.stdout
        else:
            # Godot LSP is TCP. stdout is editor progress, not frames.
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                cwd=str(self.root),
                env=env,
                start_new_session=True,
            )
            assert self.proc.stderr
        self._stderr_thread = threading.Thread(target=self._read_stderr, name=f"lsp-{self.language}-err", daemon=True)
        self._stderr_thread.start()
        if self.tcp_port is not None:
            self._connect_tcp()
            self._out = self._sock.makefile("wb")
            self._stream = self._sock.makefile("rb")
        self._reader = threading.Thread(target=self._read_stdout, name=f"lsp-{self.language}", daemon=True)
        self._reader.start()
        self._initialize()

    def _initialize(self) -> None:
        root_uri = file_uri(self.root)
        lang_opts: dict = {}
        if self.language == "python":
            lang_opts = {
                "python": {
                    "analysis": {
                        "diagnosticMode": "openFilesOnly",
                        "typeCheckingMode": "basic",
                    }
                }
            }
        result = self.request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": root_uri,
                "rootPath": str(self.root),
                "workspaceFolders": [{"uri": root_uri, "name": self.root.name}],
                "capabilities": {
                    "textDocument": {
                        "synchronization": {"didOpen": True, "didChange": True, "dynamicRegistration": False},
                        "publishDiagnostics": {},
                    },
                    "workspace": {"configuration": True},
                },
                "initializationOptions": lang_opts,
            },
            INIT_TIMEOUT_S,
        )
        if result.get("error"):
            raise LspError(f"initialize failed: {result['error']}")
        caps = (result.get("result") or {}).get("capabilities") or {}
        sync = caps.get("textDocumentSync")
        if isinstance(sync, dict):
            kind = sync.get("change", 1)
        elif isinstance(sync, int):
            kind = sync
        else:
            kind = 1
        # 2 is incremental. A full-text change with no range then never republishes.
        self.sync_full = kind != 2
        self.notify("initialized", {})

    def _connect_tcp(self) -> None:
        deadline = time.monotonic() + INIT_TIMEOUT_S
        last = "not started"
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise LspError(
                    f"godot exited {self.proc.returncode} before LSP port {self.tcp_port} opened"
                    + (f"\n{self.stderr_tail()}" if self.stderr_tail() else "")
                )
            try:
                sock = socket.create_connection(("127.0.0.1", self.tcp_port or 0), timeout=0.4)
                self._sock = sock
                return
            except OSError as exc:
                last = str(exc)
                time.sleep(0.2)
        raise LspError(
            f"godot LSP did not listen on 127.0.0.1:{self.tcp_port} ({last})"
            + (f"\n{self.stderr_tail()}" if self.stderr_tail() else "")
        )

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def pid(self) -> int | None:
        return self.proc.pid if self.proc is not None else None

    def request(self, method: str, params: dict, timeout: float) -> dict:
        if not self.alive():
            raise LspError(f"{self.language} language server is not running")
        with self._write_lock:
            msg_id = self._next_id
            self._next_id += 1
            box: queue.Queue = queue.Queue(maxsize=1)
            with self._pending_lock:
                self._pending[msg_id] = box
            self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        try:
            return box.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise LspError(
                f"{self.language} language server did not answer {method} within {timeout:.0f}s"
                + (f"\n{self.stderr_tail()}" if self.stderr_tail() else "")
            ) from None

    def notify(self, method: str, params: dict) -> None:
        with self._write_lock:
            self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def diagnostics(self, path: Path, text: str) -> list[dict]:
        uri = file_uri(path)
        previous = self._texts.get(uri)
        version = self._versions.get(uri, 0) + 1
        self._versions[uri] = version
        if self.language == "python":
            language_id = "python"
        elif self.language == "gdscript":
            language_id = "gdscript"
        elif path.suffix.lower() == ".c":
            language_id = "c"
        else:
            language_id = "cpp"
        if version == 1:
            self.notify(
                "textDocument/didOpen",
                {"textDocument": {"uri": uri, "languageId": language_id, "version": version, "text": text}},
            )
        else:
            if self.sync_full:
                change: dict = {"text": text}
            else:
                old = self._texts.get(uri, "")
                if old.endswith("\n") or old == "":
                    end = {"line": old.count("\n"), "character": 0}
                else:
                    lines = old.split("\n")
                    end = {"line": len(lines) - 1, "character": len(lines[-1])}
                change = {
                    "range": {"start": {"line": 0, "character": 0}, "end": end},
                    "text": text,
                }
            self.notify(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": version},
                    "contentChanges": [change],
                },
            )
        self._texts[uri] = text
        # clangd does not republish when the buffer is unchanged, and waiting
        # for a publish that will not arrive hits the kill timeout.
        unchanged = previous is not None and previous == text
        return self._wait_publish(uri, DIAG_TIMEOUT_S, unchanged=unchanged)

    def _wait_publish(self, uri: str, timeout: float, unchanged: bool = False) -> list[dict]:
        deadline = time.monotonic() + timeout
        started = time.monotonic()
        latest: list[dict] | None = None
        last = time.monotonic()
        while time.monotonic() < deadline:
            if not self.alive():
                raise LspError(
                    f"{self.language} language server exited before diagnostics"
                    + (f"\n{self.stderr_tail()}" if self.stderr_tail() else "")
                )
            remaining = deadline - time.monotonic()
            try:
                note = self._notes.get(timeout=min(0.25, remaining))
            except queue.Empty:
                if latest is not None and time.monotonic() - last >= 0.35:
                    return latest
                if (
                    unchanged
                    and uri in self._last_diags
                    and time.monotonic() - started >= 0.05
                ):
                    return self._last_diags[uri]
                continue
            if note.get("method") != "textDocument/publishDiagnostics":
                continue
            params = note.get("params") or {}
            if params.get("uri") != uri:
                continue
            latest = list(params.get("diagnostics") or [])
            self._last_diags[uri] = latest
            last = time.monotonic()
            if latest:
                # A non-empty publish is the result. A follow-up empty would
                # be a clear, and planted errors do not clear themselves.
                return latest
        if latest is not None:
            return latest
        self.close()
        raise LspError(
            f"{self.language} language server published no diagnostics within {timeout:.0f}s"
            + (f"\n{self.stderr_tail()}" if self.stderr_tail() else "")
        )

    def _send(self, msg: dict) -> None:
        out = self._out
        if out is None:
            raise LspError("language server stdin is closed")
        body = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        try:
            out.write(header + body)
            out.flush()
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise LspError(f"{self.language} language server pipe broke: {exc}") from exc

    def _read_stdout(self) -> None:
        stream = self._stream
        proc = self.proc
        if stream is None or proc is None:
            return
        sel = selectors.DefaultSelector()
        sel.register(stream, selectors.EVENT_READ)
        try:
            while self.proc is not None and proc.poll() is None:
                events = sel.select(timeout=0.2)
                if not events:
                    continue
                chunk = os.read(stream.fileno(), 65536)
                if not chunk:
                    break
                self._buf += chunk
                self._drain_frames()
        except OSError:
            pass
        finally:
            sel.close()

    def _drain_frames(self) -> None:
        while True:
            sep = self._buf.find(b"\r\n\r\n")
            sep_len = 4
            if sep < 0:
                sep = self._buf.find(b"\n\n")
                sep_len = 2
            if sep < 0:
                return
            header = self._buf[:sep].decode("ascii", "replace")
            length = None
            for line in header.splitlines():
                if line.lower().startswith("content-length:"):
                    length = int(line.split(":", 1)[1].strip())
            if length is None:
                self._buf = self._buf[sep + sep_len :]
                continue
            start = sep + sep_len
            if len(self._buf) < start + length:
                return
            raw = self._buf[start : start + length]
            self._buf = self._buf[start + length :]
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            self._dispatch(msg)

    def _dispatch(self, msg: dict) -> None:
        method = msg.get("method")
        msg_id = msg.get("id")
        if method and msg_id is not None:
            self._reply_server(msg_id, method, msg.get("params") or {})
            return
        if msg_id is not None and method is None:
            with self._pending_lock:
                box = self._pending.pop(msg_id, None)
            if box is not None:
                box.put(msg)
            return
        if method:
            self._notes.put(msg)

    def _reply_server(self, msg_id, method: str, params: dict) -> None:
        if method == "workspace/configuration":
            items = params.get("items") or []
            result = [{} for _ in items] or [{}]
        elif method == "workspace/workspaceFolders":
            result = [{"uri": file_uri(self.root), "name": self.root.name}]
        else:
            result = None
        with self._write_lock:
            try:
                self._send({"jsonrpc": "2.0", "id": msg_id, "result": result})
            except LspError:
                pass

    def _read_stderr(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                text = line.decode("utf-8", "replace").rstrip()
                with self._stderr_lock:
                    self.stderr_lines.append(text)
                    if len(self.stderr_lines) > STDERR_CAP:
                        del self.stderr_lines[: len(self.stderr_lines) - STDERR_CAP]
        except OSError:
            pass


class Hub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[tuple[str, str], LspSession] = {}

    def close(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()

    def status(self) -> dict:
        with self._lock:
            running = []
            for (language, root), session in self._sessions.items():
                running.append(
                    {
                        "language": language,
                        "root": root,
                        "alive": session.alive(),
                        "pid": session.pid(),
                    }
                )
        return {
            "enabled": enabled(),
            "flag": "LMP_WARM_LSP",
            "running": running,
            "godot_bin": _godot_bin_status(),
            "stubs": {"swift": STUBS[".swift"]},
        }

    def restart(self, language: str | None) -> dict:
        if not enabled():
            raise LspError("LMP_WARM_LSP is off; no language server was started")
        with self._lock:
            keys = [
                key
                for key in self._sessions
                if language in (None, "", key[0])
            ]
            sessions = [self._sessions.pop(key) for key in keys]
        for session in sessions:
            session.close()
        return {"restarted": [list(key) for key in keys]}

    def diagnostics(self, raw_path: str, text: str | None) -> dict:
        if not enabled():
            raise LspError("LMP_WARM_LSP is off; no language server was started")
        root = workspace_root()
        path = path_inside(root, raw_path)
        ext = path.suffix.lower()
        if ext in STUBS:
            raise LspError(STUBS[ext])
        language = language_of(path)
        if language is None:
            raise LspError(f"no language server for {ext or path.name}")
        if text is None:
            if not path.is_file():
                raise LspError(f"file not found: {path}")
            text = path.read_text(encoding="utf-8")
        session = self._session_for(language, path)
        raw = session.diagnostics(path, text)
        diags = [_shape(path, item) for item in raw]
        return {"path": str(path), "uri": file_uri(path), "diagnostics": diags}

    def _session_for(self, language: str, path: Path) -> LspSession:
        read_roots: list[Path] = []
        tcp_port: int | None = None
        if language == "cpp":
            project = nearest_root(path, ("compile_commands.json", "compile_flags.txt"), workspace_root())
            argv = [
                "/usr/bin/clangd",
                "--background-index=0",
                "--clang-tidy=0",
                "--header-insertion=never",
            ]
            if (project / "compile_commands.json").is_file() or (project / "compile_flags.txt").is_file():
                argv.append(f"--compile-commands-dir={project}")
            extra: dict[str, str] = {}
        elif language == "python":
            project = nearest_root(path, ("pyrightconfig.json", "pyproject.toml"), workspace_root())
            site = pyright_site()
            if site is None:
                raise LspError(
                    "basedpyright is not installed. "
                    "uv venv --python /opt/homebrew/bin/python3 .piper/lsp-py && "
                    "uv pip install --python .piper/lsp-py/bin/python basedpyright"
                )
            argv = [
                python_bin(),
                "-c",
                "import sys; from basedpyright.langserver import main; "
                "sys.argv=['basedpyright-langserver','--stdio']; raise SystemExit(main())",
            ]
            extra = {"PYTHONPATH": str(site)}
        elif language == "gdscript":
            project = nearest_root(path, ("project.godot",), workspace_root())
            binary = godot_bin()
            port = _free_local_port()
            home = workspace_root() / ".piper" / "lsp-home"
            home.mkdir(parents=True, exist_ok=True)
            argv = [
                str(binary),
                "--headless",
                "--editor",
                "--lsp-port",
                str(port),
                "--path",
                str(project),
            ]
            extra = {"HOME": str(home)}
            read_roots = [godot_engine_root(binary)]
            tcp_port = port
        else:
            raise LspError(STUBS.get("." + language, f"{language} is not wired"))
        key = (language, str(project))
        with self._lock:
            session = self._sessions.get(key)
            if session is not None and session.alive():
                return session
            if session is not None:
                session.close()
            session = LspSession(
                language, project, argv, extra, tcp_port=tcp_port, read_roots=read_roots
            )
            try:
                session.start()
            except Exception:
                session.close()
                raise
            self._sessions[key] = session
            return session


def _shape(path: Path, item: dict) -> dict:
    severity = {1: "error", 2: "warning", 3: "information", 4: "hint"}.get(item.get("severity"), "error")
    code = item.get("code")
    if isinstance(code, dict):
        code = code.get("value")
    rng = item.get("range") or {}
    start = rng.get("start") or {}
    end = rng.get("end") or {}
    return {
        "path": str(path),
        "range": {
            "start": {"line": int(start.get("line", 0)), "character": int(start.get("character", 0))},
            "end": {"line": int(end.get("line", 0)), "character": int(end.get("character", 0))},
        },
        "severity": severity,
        "code": "" if code is None else str(code),
        "message": str(item.get("message") or ""),
    }


def _summary(payload: dict) -> str:
    lines = []
    for diag in payload.get("diagnostics") or []:
        start = (diag.get("range") or {}).get("start") or {}
        line = int(start.get("line", 0)) + 1
        col = int(start.get("character", 0)) + 1
        lines.append(
            f"{diag.get('path')}:{line}:{col}: {diag.get('severity')}: {diag.get('message')}"
            + (f" [{diag.get('code')}]" if diag.get("code") else "")
        )
    if not lines:
        return f"no diagnostics for {payload.get('path')}"
    return "\n".join(lines)


HUB = Hub()


def tool_result(text: str, structured: dict | None = None, is_error: bool = False) -> dict:
    result = {"content": [{"type": "text", "text": text}], "isError": is_error}
    if structured is not None:
        result["structuredContent"] = structured
    return result


TOOLS = [
    {
        "name": "lsp_status",
        "description": "Report whether warm LSP is enabled and which language servers are running.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "lsp_diagnostics",
        "description": (
            "Diagnostics from a warm language server for one file. "
            "Pass path inside the workspace. Pass text to diagnose an unsaved buffer; "
            "omit text to read the file from disk. Swift and GDScript are stubs until Phase 3."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path inside the workspace."},
                "text": {"type": "string", "description": "Optional unsaved buffer text."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "lsp_restart",
        "description": "Kill warm language servers so the next diagnostics call starts them again.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "description": "cpp or python. Omit to restart every running server.",
                }
            },
        },
    },
]


def call_tool(name: str, arguments: dict) -> dict:
    try:
        if name == "lsp_status":
            payload = HUB.status()
            return tool_result(json.dumps(payload), payload, is_error=False)
        if name == "lsp_diagnostics":
            path = arguments.get("path")
            if not isinstance(path, str) or not path.strip():
                raise LspError("path is required")
            text = arguments.get("text")
            if text is not None and not isinstance(text, str):
                raise LspError("text must be a string")
            payload = HUB.diagnostics(path, text)
            return tool_result(_summary(payload), payload, is_error=False)
        if name == "lsp_restart":
            language = arguments.get("language")
            if language is not None and not isinstance(language, str):
                raise LspError("language must be a string")
            payload = HUB.restart(language)
            return tool_result(json.dumps(payload), payload, is_error=False)
        return tool_result(f"unknown tool {name}", is_error=True)
    except (LspError, ValueError, OSError) as exc:
        return tool_result(str(exc), is_error=True)


def reply(msg_id, result=None, error=None) -> None:
    if msg_id is None:
        return
    if error is not None:
        body = {"jsonrpc": "2.0", "id": msg_id, "error": error}
    else:
        body = {"jsonrpc": "2.0", "id": msg_id, "result": result}
    sys.stdout.write(json.dumps(body) + "\n")
    sys.stdout.flush()


def handle(msg: dict, initialized: bool) -> bool:
    method = msg.get("method")
    msg_id = msg.get("id")
    if method == "notifications/initialized":
        return True
    if method in ("notifications/cancelled", "notifications/progress"):
        return initialized
    if not initialized and method not in ("initialize", "ping"):
        reply(msg_id, error={"code": -32600, "message": "server not initialized"})
        return initialized
    if method == "initialize":
        params = msg.get("params") or {}
        version = params.get("protocolVersion") or PROTOCOL
        if version not in ("2025-06-18", "2025-03-26", "2024-11-05"):
            version = PROTOCOL
        reply(
            msg_id,
            result={
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": "0.1.0"},
                "instructions": (
                    "After editing C++, Python, or GDScript, call lsp_diagnostics with the file path. "
                    "GDScript uses the Godoer fork (GODOER_GODOT_BIN, else "
                    "/Users/dev/godot-godoer/bin/godot.macos.editor.arm64), not stock Godot.app. "
                    "LMP_WARM_LSP must be 1 or the tool returns an error and starts nothing. "
                    "Swift is parked."
                ),
            },
        )
        return initialized
    if method == "ping":
        reply(msg_id, result={})
        return initialized
    if method == "tools/list":
        reply(msg_id, result={"tools": TOOLS})
        return initialized
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name") or ""
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        reply(msg_id, result=call_tool(name, arguments))
        return initialized
    reply(msg_id, error={"code": -32601, "message": f"method not found: {method}"})
    return initialized


def serve() -> None:
    initialized = False
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(msg, dict):
            continue
        initialized = handle(msg, initialized) or initialized
    HUB.close()


if __name__ == "__main__":
    try:
        serve()
    finally:
        HUB.close()
