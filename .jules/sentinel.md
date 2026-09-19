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

## 2026-03-31 - Include bearer tokens, certificates, and signatures in spawn environment deny list
**Vulnerability:** Parent environment variables carrying bearer tokens, client certificates, or digital signatures with allowlisted prefixes (such as `LC_BEARER`, `XDG_CERT`, `LC_CERTIFICATE`, `XDG_SIGNATURE`) bypassed `denied_parent_key` and leaked into child MCP server process environments.
**Learning:** Common credential type terms like `BEARER`, `CERT`, `CERTIFICATE`, and `SIGNATURE` must be explicitly listed in suffix filtering rules to prevent sensitive authentication tokens and keys in allowlisted namespaces from being inherited across process boundaries.
**Prevention:** Added `BEARER`, `CERT`, `CERTIFICATE`, and `SIGNATURE` to the `kSuffix` deny list in `src/mcp/spawn_env.cpp`.

## 2026-03-31 - Include private keys, passcodes, and tickets in spawn environment deny list
**Vulnerability:** Parent environment variables carrying private keys, passcodes, or kerberos/auth tickets with allowlisted prefixes (such as `LC_PRIVKEY`, `XDG_PASSCODE`, `LC_TICKET`) bypassed `denied_parent_key` and leaked into child MCP server process environments.
**Learning:** Shorthand and alternative credential terms like `PRIVKEY`, `PASSCODE`, and `TICKET` must be explicitly listed in suffix filtering rules to prevent sensitive secrets using allowlisted namespace prefixes from being inherited.
**Prevention:** Added `PRIVKEY`, `PASSCODE`, and `TICKET` to the `kSuffix` deny list in `src/mcp/spawn_env.cpp`.

## 2026-03-31 - Expand environment variable suffix filtering for token IDs and auth credentials
**Vulnerability:** Parent environment variables carrying auth credentials, token IDs, or secret keys under allowlisted prefixes (such as `LC_CRED`, `XDG_CREDS`, `LC_TOKEN_ID`, `XDG_SECRET_KEY`, `LC_AUTH_TOKEN`, `XDG_ACCESS_TOKEN`) bypassed `denied_parent_key` and leaked into child MCP server process environments.
**Learning:** Token identifier and access credential suffixes like `CRED`, `CREDS`, `TOKEN_ID`, `SECRET_KEY`, `AUTH_TOKEN`, and `ACCESS_TOKEN` must be explicitly included in suffix filtering rules to prevent sensitive authentication tokens and key pairs from being inherited.
**Prevention:** Added `CRED`, `CREDS`, `TOKEN_ID`, `SECRET_KEY`, `AUTH_TOKEN`, and `ACCESS_TOKEN` to the `kSuffix` deny list in `src/mcp/spawn_env.cpp`.
