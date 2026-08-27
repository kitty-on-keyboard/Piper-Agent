#pragma once
//
// The MCP half of a run's settings: reading the server list off the wire, and connecting
// it into the tool registry (M1).
//
// Split out of sidecar.cpp because it is a distinct responsibility over untrusted input
// and wants tests of its own. `trusted` in particular decides whether a tool that runs
// OUTSIDE the sandbox may be called without an approval card, and a settings parser that
// decides that has no business being untestable inside an anonymous namespace.
//
#include <string>
#include <vector>

#include "src/platform/clock.hpp"
#include "src/platform/event_log.hpp"
#include "src/tools/mcp_host.hpp"
#include "src/tools/registry.hpp"

namespace lmp::surface {

// Reads `params.mcp_servers` off a start message. `signature` receives a stable identity
// for the list, so a caller can tell whether the configured set has changed since the
// registry it built was populated.
//
// Total: a malformed message, a missing field and a non-array all yield an empty list
// rather than an error, because "no MCP servers" is the overwhelmingly common case and
// must not be a failure mode.
[[nodiscard]] std::vector<tools::McpServerConfig> parse_mcp_servers(const std::string& message,
                                                                    std::string& signature);

// Connects each server into `registry`, appending one `mcp_server` event per server.
// Never throws: a server that will not start leaves its tools absent and the run goes on.
void connect_mcp_servers(tools::McpHost& host,
                         const std::vector<tools::McpServerConfig>& servers,
                         tools::Registry& registry, platform::EventLogWriter& log,
                         const platform::Clock& clock);

// Reads `<workspace_root>/.mcp.json` -- the project-carried server list that Claude Code,
// Cursor, Antigravity and Gemini CLI all read, and that `godoer connect` writes. LM_Pipe
// read only the editor's settings, so a project correctly configured for every other agent
// on the machine arrived here with no servers at all and nothing said why.
//
// TRUST IS NEVER INHERITED FROM THIS FILE, whatever it says. A settings entry is the
// operator typing into their own machine's config; a `.mcp.json` arrives with a clone, and
// honouring `trusted` from it would let any repository run a command outside Seatbelt with
// no approval card at run start. `trusted_ignored` counts the entries that asked. Servers
// from here are usable and carded; to vouch for one, name it in `lmPipe.mcpServers`.
//
// Total, like parse_mcp_servers: no file, unreadable, malformed, or the wrong shape all
// yield an empty list. A project without one is the common case, not a failure.
[[nodiscard]] std::vector<tools::McpServerConfig> parse_mcp_json_file(
    const std::string& workspace_root, std::string& signature, std::size_t& trusted_ignored);

// Settings entries win on name; file entries fill in the rest. A name configured in both
// places resolves to the settings one -- that is the operator's own machine overriding
// what a checkout brought with it, which is the only direction that can be right.
[[nodiscard]] std::vector<tools::McpServerConfig> merge_mcp_servers(
    std::vector<tools::McpServerConfig> from_settings,
    std::vector<tools::McpServerConfig> from_file);

// A repeated STRING field, for the same reason parse_mcp_servers exists just above: the
// surface::string_field extractors are a substring search for a scalar and cannot walk an
// array. Used for `image_paths` on lmp/start and lmp/message.
[[nodiscard]] std::vector<std::string> parse_string_array(const std::string& message,
                                                          std::string_view key);

} // namespace lmp::surface
