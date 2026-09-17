## 2026-03-31 - Command Injection via std::system in Container Probe
**Vulnerability:** `probe()` in `src/tools/container.cpp` invoked `std::system` with string concatenation (`"command -v " + binary + " ..."`). If a binary argument contained shell metacharacters, arbitrary commands could be executed with host privileges.
**Learning:** Shell invocation via `std::system` passes raw command strings through `/bin/sh`, evaluating shell metacharacters instead of isolating argument boundaries.
**Prevention:** Always use process creation APIs (`posix_spawnp`, `execve`, or Subprocess wrappers) without a shell when probing or executing binaries with dynamic parameters.

## 2026-03-31 - Unquoted Container Image Parameter Command Injection
**Vulnerability:** `container_command()` in `src/tools/container.cpp` concatenated `rt.image` (and `rt.binary`) directly into a command string without shell quoting. If `LMP_CONTAINER_IMAGE` contained shell metacharacters (e.g., `;`, `$()`, `&&`), arbitrary commands would be executed on the host when constructing container sandboxing invocations.
**Learning:** Building command line strings for `/bin/sh -c` requires strict shell quoting for every dynamic parameter, including environment variable overrides like image tags/digests.
**Prevention:** Always wrap dynamic command arguments in POSIX single-quote escaping (`shell_quote()`) before assembling shell command strings.

## 2024-11-23 - Enhance environment variable sanitization
**Vulnerability:** Subprocesses might inherit sensitive information if parent environment variables contained tokens not previously caught, like "PAT" (Personal Access Token), "AUTH", or "JWT".
**Learning:** The existing filtering logic was strong but the blocklist of suffixes for denying sensitive environment variables was limited.
**Prevention:** Expanded the `kSuffix` array in `src/mcp/spawn_env.cpp` to include "PAT", "AUTH", and "JWT" to provide broader coverage against leaking common credential suffixes to child processes.

## 2026-03-31 - Expand environment variable suffix filtering for credentials
**Vulnerability:** Parent environment variables prefixed with inherited prefixes (such as `LC_` or `XDG_`) containing credentials with suffixes like `APIKEY`, `PASSPHRASE`, `PRIVATEKEY`, `COOKIE`, `SESSID`, or `SESSION` (e.g. `LC_APIKEY`) bypassed `denied_parent_key` and leaked into child MCP process environments.
**Learning:** Suffix filtering must include all common secret and session identifier patterns to prevent credentials using allowlisted prefix namespaces from being inherited.
**Prevention:** Added `APIKEY`, `PASSPHRASE`, `PRIVATEKEY`, `COOKIE`, `SESSID`, and `SESSION` to the `kSuffix` deny list in `src/mcp/spawn_env.cpp`.
