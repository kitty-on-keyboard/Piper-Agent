## 2026-09-09 - Shell Argument Injection in Container Invocation and Non-HTTP Webhook Scheme Processing
**Vulnerability:**
1. In `container_command` (`src/tools/container.cpp`), paths such as `cwd` and `workspace_root` were unquoted when constructing container invocation strings passed to `/bin/sh -c`, enabling command/argument injection if paths contained spaces or shell control characters.
2. In `post_orch_webhook` (`scripts/piper_worker.py`), webhook URLs were not validated for scheme, allowing `file://` or non-HTTP URLs to be processed by `urllib.request.urlopen`.

**Learning:**
1. When constructing shell commands meant to be run via `/bin/sh -c`, any directory paths or workspace configuration parameters must be shell-quoted, even if they originate from internal configuration structs.
2. Webhook dispatchers that accept external or task packet URLs must enforce strict scheme allowlists (`http://` / `https://`) to avoid SSRF or local file protocol exploitation.

**Prevention:**
1. Always wrap dynamically inserted path parameters in shell-quoting functions (e.g. single-quote escaping) before appending to shell command strings.
2. Validate and restrict network request destinations to expected transport schemes (`http://` / `https://`) before executing request handlers.

## 2026-09-10 - Newline Injection in Command Allowlist Serialization
**Vulnerability:**
`remember(command)` in `extension/src/sidebar.ts` allowed strings with embedded newlines (`\n` or `\r`) to be saved into `allowedCommands` settings array. When `settingsFromConfig()` in `extension/src/extension.ts` joined `allowedCommands` into a newline-delimited wire string `allowed_commands` for the C++ sidecar, the sidecar split the string on newlines, parsing the injected command line as a separate allowed command and automatically approving its execution without user consent.

**Learning:**
1. When serializing an array of string values into a newline-delimited wire string for process IPC, input validation must reject newlines at ingestion time (`remember()`), and serialization logic must filter out any array elements containing newlines before joining.

**Prevention:**
1. Reject `\r` and `\n` in inputs intended for newline-delimited wire protocols.
2. Sanitize and filter array elements before joining with delimiters during protocol serialization.
