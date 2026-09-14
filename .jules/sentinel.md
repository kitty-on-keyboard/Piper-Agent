## 2026-03-31 - Command Injection via std::system in Container Probe
**Vulnerability:** `probe()` in `src/tools/container.cpp` invoked `std::system` with string concatenation (`"command -v " + binary + " ..."`). If a binary argument contained shell metacharacters, arbitrary commands could be executed with host privileges.
**Learning:** Shell invocation via `std::system` passes raw command strings through `/bin/sh`, evaluating shell metacharacters instead of isolating argument boundaries.
**Prevention:** Always use process creation APIs (`posix_spawnp`, `execve`, or Subprocess wrappers) without a shell when probing or executing binaries with dynamic parameters.
