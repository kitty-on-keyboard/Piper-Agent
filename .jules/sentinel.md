## 2026-03-31 - Command Injection via std::system in Container Probe
**Vulnerability:** `probe()` in `src/tools/container.cpp` invoked `std::system` with string concatenation (`"command -v " + binary + " ..."`). If a binary argument contained shell metacharacters, arbitrary commands could be executed with host privileges.
**Learning:** Shell invocation via `std::system` passes raw command strings through `/bin/sh`, evaluating shell metacharacters instead of isolating argument boundaries.
**Prevention:** Always use process creation APIs (`posix_spawnp`, `execve`, or Subprocess wrappers) without a shell when probing or executing binaries with dynamic parameters.

## 2026-03-31 - Unquoted Container Image Parameter Command Injection
**Vulnerability:** `container_command()` in `src/tools/container.cpp` concatenated `rt.image` (and `rt.binary`) directly into a command string without shell quoting. If `LMP_CONTAINER_IMAGE` contained shell metacharacters (e.g., `;`, `$()`, `&&`), arbitrary commands would be executed on the host when constructing container sandboxing invocations.
**Learning:** Building command line strings for `/bin/sh -c` requires strict shell quoting for every dynamic parameter, including environment variable overrides like image tags/digests.
**Prevention:** Always wrap dynamic command arguments in POSIX single-quote escaping (`shell_quote()`) before assembling shell command strings.
