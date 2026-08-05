#include "src/tools/mcp_host.hpp"

#include <algorithm>
#include <set>
#include <utility>

#include "src/mcp/client.hpp"
#include "src/mcp/content.hpp"
#include "src/mcp/subprocess.hpp"

namespace lmp::tools {
namespace {

// parsephony's format delimiters. A name carrying one of these would not merely fail --
// it would corrupt the grammar for every OTHER tool in the same guard, so a server that
// reports one gets that tool dropped.
[[nodiscard]] bool name_is_usable(const std::string& n) {
    return !n.empty() && n.find('>') == std::string::npos &&
           n.find('=') == std::string::npos && n.find('\n') == std::string::npos;
}

[[nodiscard]] parsephony::ParamType param_type_of(const nlohmann::json& prop) {
    if (!prop.is_object() || !prop.contains("type")) {
        return parsephony::ParamType::Json;
    }
    const nlohmann::json& t = prop.at("type");
    // JSON Schema allows a list of types ("type": ["string", "null"]). Anything we cannot
    // reduce to one shape travels as Json, which accepts any value rather than
    // constraining generation to a shape the server did not actually promise.
    if (!t.is_string()) {
        return parsephony::ParamType::Json;
    }
    const std::string s = t.get<std::string>();
    if (s == "string") {
        return parsephony::ParamType::Text;
    }
    if (s == "number" || s == "integer") {
        return parsephony::ParamType::Number;
    }
    if (s == "boolean") {
        return parsephony::ParamType::Boolean;
    }
    if (s == "object") {
        return parsephony::ParamType::Object;
    }
    if (s == "array") {
        return parsephony::ParamType::Array;
    }
    return parsephony::ParamType::Json;
}

// Flattens the MCP tool result's content blocks into the model-facing summary.
[[nodiscard]] std::string summarize(const mcp::ToolResult& r) {
    std::string out;
    if (r.content.is_array()) {
        for (const nlohmann::json& block : r.content) {
            if (!block.is_object()) {
                continue;
            }
            const auto type = block.value("type", std::string{});
            if (type == "text") {
                if (!out.empty()) {
                    out += "\n";
                }
                out += block.value("text", std::string{});
            } else if (!type.empty()) {
                // Images, audio and embedded resources are not text and must not be
                // silently dropped -- the model needs to know something came back.
                if (!out.empty()) {
                    out += "\n";
                }
                out += "[" + type + " content omitted]";
            }
        }
    }
    if (out.empty() && r.structured.has_value()) {
        out = r.structured->dump();
    }
    return out;
}

} // namespace

std::string namespaced_tool_name(const std::string& server, const std::string& tool) {
    return "mcp__" + server + "__" + tool;
}

bool tool_spec_from_schema(const std::string& registered_name,
                           const nlohmann::json& input_schema, parsephony::ToolSpec& out,
                           std::string& why) {
    if (!name_is_usable(registered_name)) {
        why = "tool name is empty or contains one of '>', '=', newline";
        return false;
    }
    out = parsephony::ToolSpec{};
    out.name = registered_name;

    // A tool with no properties is legal and useful (many take no arguments), so an
    // absent or non-object `properties` is an empty parameter list and not a rejection.
    if (!input_schema.is_object() || !input_schema.contains("properties") ||
        !input_schema.at("properties").is_object()) {
        return true;
    }

    std::set<std::string> required;
    if (input_schema.contains("required") && input_schema.at("required").is_array()) {
        for (const nlohmann::json& r : input_schema.at("required")) {
            if (r.is_string()) {
                required.insert(r.get<std::string>());
            }
        }
    }

    std::set<std::string> seen;
    for (const auto& [key, prop] : input_schema.at("properties").items()) {
        if (!name_is_usable(key)) {
            why = "parameter '" + key + "' is empty or contains one of '>', '=', newline";
            return false;
        }
        if (!seen.insert(key).second) {
            why = "duplicate parameter '" + key + "'";
            return false;
        }
        parsephony::ParamSpec p;
        p.name = key;
        p.type = param_type_of(prop);
        p.required = required.count(key) != 0;
        out.params.push_back(std::move(p));
    }
    return true;
}

// One live server: the client, and the config it was built from.
//
// shared_ptr, not unique_ptr, and this is load-bearing. The handlers installed into the
// Registry have to reach the client, and a raw pointer made close() -- or simply
// destroying the host before the registry -- a use-after-free on the next tool call.
// ASan caught exactly that (Client::next_id on freed memory) once these tests reached
// gate-asan. Shared ownership makes the lifetime correct by construction rather than by
// everyone remembering a destruction order: close() disconnects the client, and the
// object itself survives as long as some handler can still be called.
struct McpHost::Connection {
    McpServerConfig config;
    std::shared_ptr<mcp::Client> client;
};

McpHost::McpHost() : McpHost(Options{}) {}

McpHost::McpHost(Options options) : options_(options) {}

McpHost::~McpHost() { close(); }

void McpHost::close() {
    for (auto& conn : connections_) {
        if (conn && conn->client) {
            conn->client->close();
        }
    }
    connections_.clear();
}

std::vector<McpServerStatus> McpHost::connect_and_register(
    const std::vector<McpServerConfig>& servers, Registry& registry) {
    std::vector<McpServerStatus> report;
    std::set<std::string> server_names;

    for (const McpServerConfig& cfg : servers) {
        McpServerStatus status;
        status.name = cfg.name;

        if (!name_is_usable(cfg.name)) {
            status.error = "server name is empty or contains one of '>', '=', newline";
            report.push_back(std::move(status));
            continue;
        }
        if (!server_names.insert(cfg.name).second) {
            status.error = "duplicate server name; the second is ignored so its tools "
                           "cannot displace the first's";
            report.push_back(std::move(status));
            continue;
        }
        if (cfg.command.empty()) {
            status.error = "no command configured";
            report.push_back(std::move(status));
            continue;
        }

        auto conn = std::make_unique<Connection>();
        conn->config = cfg;

        // Everything from here is someone else's process. A throw is "that server is
        // absent", never a failed run.
        std::vector<mcp::Tool> tools;
        try {
            mcp::Client::Info info;
            info.name = "lmp";
            mcp::Client::Options copts;
            copts.default_timeout = options_.connect_timeout;
            conn->client = std::make_shared<mcp::Client>(info, copts);

            mcp::Subprocess::Options sub;
            sub.program = cfg.command;
            sub.args = cfg.args;
            // The sidecar's own stdio is a protocol channel, and a chatty server would
            // interleave its banner with ours. kDiscard rather than kCapture because
            // nothing here drains a stderr pipe, and an undrained one eventually blocks
            // the server mid-call -- the stall this class exists to prevent.
            sub.stderr_mode = mcp::StderrMode::kDiscard;
            for (const std::string& kv : cfg.env) {
                const std::size_t eq = kv.find('=');
                if (eq != std::string::npos) {
                    sub.env.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
                }
            }
            conn->client->connect_stdio(std::move(sub));
            (void)conn->client->initialize();
            tools = conn->client->list_tools();
        } catch (const std::exception& e) {
            status.error = e.what();
            report.push_back(std::move(status));
            continue;
        }

        status.connected = true;
        const std::shared_ptr<mcp::Client> client = conn->client;
        const bool trusted = cfg.trusted;
        const auto call_timeout = options_.call_timeout;

        for (const mcp::Tool& tool : tools) {
            const std::string registered = namespaced_tool_name(cfg.name, tool.name);
            parsephony::ToolSpec spec;
            std::string why;
            if (!tool_spec_from_schema(registered, tool.input_schema, spec, why)) {
                status.rejected.push_back(tool.name + ": " + why);
                continue;
            }
            if (spec.params.size() > options_.max_params) {
                status.rejected.push_back(tool.name + ": " +
                                          std::to_string(spec.params.size()) +
                                          " parameters exceeds the guard's limit");
                continue;
            }

            ToolDecl decl;
            decl.name = registered;
            decl.description = tool.description.empty()
                                   ? ("MCP tool '" + tool.name + "' from server '" +
                                      cfg.name + "'")
                                   : tool.description;
            // Attribution belongs in the model-facing text: this tool's behaviour is a
            // claim by another process, not something this binary implements.
            decl.description += "\n(provided by MCP server '" + cfg.name + "'";
            decl.description += trusted ? ")" : "; runs outside the sandbox)";
            decl.spec = spec;
            // We cannot see what a remote tool touches, so the honest answer to both of
            // these is "assume it might".
            decl.mutates_workspace = !trusted;
            decl.executes_commands = !trusted;
            // THE containment decision. A remote tool runs in the server's process, which
            // Seatbelt does not cover, so an untrusted server's tools each raise a card.
            // The server's own readOnlyHint/destructiveHint annotations deliberately do
            // not participate: per the MCP spec they are hints from an untrusted peer and
            // must not drive a security decision.
            decl.irreversible = !trusted;
            // Regardless of trust. The three flags above answer "does this call need a
            // human", and trust is a real answer to that; this one answers "can a mode
            // that permits no writes permit this call", and the honest answer never
            // depends on how the operator feels about the server. Plan mode was letting
            // trusted MCP tools straight through the one gate it has.
            decl.remote = true;

            const std::string remote_name = tool.name;
            Registry::Handler handler =
                [client, remote_name, call_timeout](
                    const std::vector<ToolParamValue>& params, int) -> ToolResult {
                // A closed or crashed server is a typed failure, never a hang and never a
                // call into a half-dead client.
                if (!client->connected()) {
                    return ToolResult::error(ErrorClass::Transient, false,
                                             "the MCP server providing '" + remote_name +
                                                 "' is no longer connected");
                }
                nlohmann::json args = nlohmann::json::object();
                for (const ToolParamValue& p : params) {
                    // The guard validated shape, not JSON: a param typed Object or Json
                    // arrives as the text the model emitted. Parse when it parses, and
                    // otherwise send the string rather than failing the call.
                    nlohmann::json parsed = nlohmann::json::parse(p.value, nullptr, false);
                    args[p.name] = parsed.is_discarded() ? nlohmann::json(p.value)
                                                         : std::move(parsed);
                }
                try {
                    const mcp::ToolResult r =
                        client->call_tool(remote_name, args, {}, call_timeout);
                    std::string text = summarize(r);
                    if (r.is_error) {
                        // The tool ran and failed. That is evidence the model can act on,
                        // so it is a ToolError and not a Refused.
                        return ToolResult::error(ErrorClass::None, false,
                                                 text.empty() ? "the MCP tool reported an "
                                                                "error with no message"
                                                              : text);
                    }
                    return ToolResult::okay(std::move(text));
                } catch (const mcp::McpError& e) {
                    ToolResult out = ToolResult::error(ErrorClass::Transient, false,
                                                       std::string("MCP call failed: ") +
                                                           e.what());
                    return out;
                } catch (const std::exception& e) {
                    return ToolResult::error(ErrorClass::Transient, false,
                                             std::string("MCP transport failed: ") +
                                                 e.what());
                }
            };

            if (!registry.declare_remote(std::move(decl), std::move(handler))) {
                status.rejected.push_back(tool.name + ": name already registered");
                continue;
            }
            ++status.registered;
        }

        connections_.push_back(std::move(conn));
        report.push_back(std::move(status));
    }
    return report;
}

} // namespace lmp::tools
