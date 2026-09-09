## 2026-09-09 - Shell Argument Injection in Container Invocation and Non-HTTP Webhook Scheme Processing
**Vulnerability:**
1. In `container_command` (`src/tools/container.cpp`), paths such as `cwd` and `workspace_root` were unquoted when constructing container invocation strings passed to `/bin/sh -c`, enabling command/argument injection if paths contained spaces or shell control characters.
2. In `post_orch_webhook` (`scripts/piper_worker.py`), webhook URLs were not validated for scheme, allowing `file://` or non-HTTP URLs to be processed by `urllib.request.urlopen`.

**Learning:**
1. When constructing shell commands meant to be run via `/bin/sh -c`, any directory paths or workspace configuration parameters must be shell-quoted, even if they originate from internal configuration structs.
2. Webhook dispatchers that accept external or task packet URLs must enforce strict scheme allowlists (`http://`, `https://`) to avoid SSRF or local file protocol exploitation.

**Prevention:**
1. Always wrap dynamically inserted path parameters in shell-quoting functions (e.g. single-quote escaping) before appending to shell command strings.
2. Validate and restrict network request destinations to expected transport schemes (`http://` / `https://`) before executing request handlers.
