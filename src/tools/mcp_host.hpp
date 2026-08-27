#pragma once
//
// McpHost -- MCP servers as ordinary tools in the Registry (M1).
//
// src/mcp is a complete MCP client and server, interop-verified in both directions. This
// is the thing that makes it the way LM_Pipe actually uses MCP rather than a library
// sitting beside the agent: at run start each configured server is spawned over stdio,
// initialized, asked for its tools, and every tool it reports is registered into the same
// `Registry` the native tools live in.
//
// Registering them there rather than beside them is what makes them first class. The
// remote tool's `inputSchema` becomes a `parsephony::ToolSpec`, so `ToolCallGuard`
// constrains generation of a remote call byte by byte EXACTLY as it does a native one --
// an unregistered remote tool and a missing required parameter are both unrepresentable,
// not errors caught after the fact.
//
// THREE THINGS THIS FILE TREATS AS HOSTILE, because a server is someone else's process:
//
//   1. NAMES. A remote tool keeps the name the server declared when that name is
//      free in the registry, so `godot_guide` is callable as `godot_guide`. Collision
//      (a native `read_file`, a second server's `echo`) is the only case that
//      namespaces to `mcp__<server>__<tool>`. Always-prefixing made the grammar
//      advertise a name the conventions and the model never write; measured on
//      r-18cf2a6c7f16da98-14a63603 the model asked about Blender instead of calling
//      the Godoer tool sitting in the registry under `mcp__godoer__godot_model`.
//   2. CONTAINMENT. A remote tool runs in the SERVER's process, outside Seatbelt. We do
//      not sandbox it and cannot. So an untrusted server's tools are all declared
//      `irreversible`, which routes every call through the approval card. The operator
//      opts a server into `trusted` explicitly; the server's own MCP annotations never
//      decide this, because the MCP spec says they must not be trusted for security.
//      On a trusted server, `readOnlyHint` / `destructiveHint` DO decide mutates vs
//      read -- that is a statement about what the tool does, used for mode policy,
//      not a bypass of Seatbelt. Untrusted servers keep the assume-worst flags.
//   3. AVAILABILITY. A server that fails to spawn, refuses the handshake, or hangs leaves
//      its tools ABSENT and the run continues. A broken server must never be able to
//      stall a turn, so every call carries a timeout.
//
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <parsephony/toolcall.hpp>

#include "src/tools/registry.hpp"

namespace lmp::mcp {
class Client;
}

namespace lmp::tools {

// One MCP server to connect. Mirrors protocol::McpServerSettings, kept separate so
// src/tools does not depend on the wire types (S3: layers depend upward).
struct McpServerConfig {
    std::string name;
    std::string command;
    std::vector<std::string> args;
    std::vector<std::string> env;  // KEY=VALUE
    bool trusted = false;
    // Which config named this server: "settings" (empty, the default) or ".mcp.json".
    // The host does not read it -- it exists so the event log can say where a server came
    // from, which matters because only one of the two sources may set `trusted`.
    std::string source;
};

// What happened to one configured server. Reported rather than thrown: "that server is
// absent" is a normal outcome the run has to survive.
struct McpServerStatus {
    std::string name;
    bool connected = false;
    std::size_t registered = 0;
    std::vector<std::string> rejected;  // tool name + why, for the ones we would not take
    std::string error;                  // why the server itself is absent
    // From initialize().instructions, only stored when the operator trusted the server.
    // Empty if the server sent none or was untrusted (untrusted prose does not enter
    // the system prompt).
    std::string instructions;
};

// Builds the generation-constraining spec for one remote tool from its JSON Schema.
//
// Free and pure so it can be tested without a server: this is where untrusted input meets
// the grammar, and it is the part worth asserting hardest. `registered_name` is the
// registered name (the server's name, or namespaced on collision). Returns false
// and sets `why` when the tool cannot be
// represented, which is a rejection of that tool and never of the whole server.
[[nodiscard]] bool tool_spec_from_schema(const std::string& registered_name,
                                         const nlohmann::json& input_schema,
                                         parsephony::ToolSpec& out, std::string& why);

// The namespaced registry name for a server's tool, used when the declared name
// would collide with one already registered.
[[nodiscard]] std::string namespaced_tool_name(const std::string& server,
                                               const std::string& tool);

// The name the tool will actually register under: the server's name if it is
// usable and free, otherwise the namespaced form.
[[nodiscard]] std::string registered_mcp_tool_name(const Registry& registry,
                                                   const std::string& server,
                                                   const std::string& tool);

class McpHost {
  public:
    struct Options {
        // A server that does not complete the handshake in this long is absent.
        std::chrono::milliseconds connect_timeout{10000};
        // A call that does not answer in this long is a Timeout ToolResult, not a hang.
        std::chrono::milliseconds call_timeout{60000};
        // parsephony's own ceiling; a tool wanting more parameters is rejected.
        std::size_t max_params = 64;
    };

    // Two declarations rather than `Options options = {}`: a default argument naming a
    // nested type whose members have initializers is not usable inside the enclosing
    // class definition.
    McpHost();
    explicit McpHost(Options options);
    ~McpHost();

    McpHost(const McpHost&) = delete;
    McpHost& operator=(const McpHost&) = delete;

    // Connects every server and registers what it offers. Never throws.
    std::vector<McpServerStatus> connect_and_register(
        const std::vector<McpServerConfig>& servers, Registry& registry);

    // Trusted servers' initialize() instructions, concatenated, for the system prompt.
    // Godoer's BRIEF lives here: "prefer these tools over shelling out to blender".
    // Empty when no trusted server sent any.
    [[nodiscard]] std::string prompt_instructions() const;

    void close();

  private:
    struct Connection;
    Options options_;
    std::vector<std::unique_ptr<Connection>> connections_;
};

} // namespace lmp::tools
