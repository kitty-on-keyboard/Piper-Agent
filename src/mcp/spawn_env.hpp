#pragma once
//
// The environment an MCP child actually sees.
//
// Subprocess used to copy the sidecar's full `environ` and append the server's config
// `env` map. After spawn the child then held IDE/shell secrets (*_TOKEN, AWS_*,
// SSH_AUTH_SOCK, LMP_*, ...) even when the config only needed PATH. That is Piper's
// spawn policy, not an IDE CVE -- and it is the wrong default for a process that runs
// outside Seatbelt.
//
// Policy (same for trusted and untrusted servers; `trusted` skips per-call cards, it
// does not mean "inherit AWS creds"):
//
//   1. Inherit from the parent only an allowlist: PATH, HOME, USER, LOGNAME, SHELL,
//      TMPDIR/TMP/TEMP, LANG, TZ, TERM, LC_*, XDG_*, and a handful of TLS CA bundle
//      paths. Language path injectors (PYTHONPATH, NODE_PATH, VIRTUAL_ENV) and DISPLAY
//      are not on it -- Godoer gets cwd from working_dir and GODOT_BIN from config env;
//      npx/uvx need PATH and HOME.
//   2. Drop parent keys that look like secrets even if they matched an allow prefix
//      (case-insensitive): *TOKEN/*SECRET/*KEY/*PASSWORD, AWS_/AZURE_/LMP_/SSH_/...,
//      SSH_AUTH_SOCK, proxy vars, kubeconfig, and so on.
//   3. Always apply the server's explicit env afterwards, replacing same keys. That is
//      the operator-intended secret path. Config may set AWS_* ; parent inheritance
//      may not. Duplicate keys are collapsed so getenv (first match on Unix) sees the
//      config value, not a stale parent copy sitting in front of it.
//
#include <string>
#include <utility>
#include <vector>

namespace lmp::mcp {

// `parent` is a nullptr-terminated KEY=VALUE array in the shape of `environ`. Null
// parent is treated as empty. Entries without '=' are skipped. Result is sorted
// KEY=VALUE strings, one per surviving key.
[[nodiscard]] std::vector<std::string> build_child_environ(
    const char* const* parent,
    const std::vector<std::pair<std::string, std::string>>& extra);

} // namespace lmp::mcp
