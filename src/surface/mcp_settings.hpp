#pragma once
//
// The MCP half of a run's settings: reading the server list off the wire, and connecting
// it into the tool registry (M1).
//
// Split out of sidecar.cpp because it pushed that file over the 800-line ratchet, which
// was the right signal -- this is a distinct responsibility with untrusted input, and it
// wants tests of its own. `trusted` in particular decides whether a tool that runs
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

} // namespace lmp::surface
