#include "src/tools/mcp_host.hpp"
#include "src/platform/event_log.hpp"
#include <unistd.h>
#include <system_error>
#include <fstream>
#include <filesystem>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <set>
#include <string_view>
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

[[nodiscard]] parsephony::ParamType param_type_name(std::string_view s) {
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

// Reduces JSON Schema `type` to a ParamType. Nullable unions ("type": ["string","null"])
// keep the non-null arm rather than collapsing to unconstrained Json (P2 §12).
struct ReducedType {
    parsephony::ParamType type = parsephony::ParamType::Json;
    bool nullable = false;
    bool ok = false;
};

[[nodiscard]] ReducedType reduce_type_field(const nlohmann::json& t) {
    ReducedType out;
    if (t.is_string()) {
        const std::string s = t.get<std::string>();
        if (s == "string" || s == "number" || s == "integer" || s == "boolean" ||
            s == "object" || s == "array") {
            out.type = param_type_name(s);
            out.ok = true;
        }
        return out;
    }
    if (!t.is_array()) {
        return out;
    }
    std::string non_null;
    bool saw_null = false;
    for (const nlohmann::json& el : t) {
        if (!el.is_string()) {
            return out;
        }
        const std::string s = el.get<std::string>();
        if (s == "null") {
            saw_null = true;
            continue;
        }
        if (!non_null.empty() && non_null != s) {
            return out; // multi-type union we cannot reduce
        }
        non_null = s;
    }
    if (non_null.empty()) {
        return out;
    }
    out.type = param_type_name(non_null);
    if (out.type == parsephony::ParamType::Json) {
        return out;
    }
    out.nullable = saw_null;
    out.ok = true;
    return out;
}

void fill_param_constraints(const nlohmann::json& prop, parsephony::ParamSpec& p) {
    if (!prop.is_object()) {
        return;
    }
    if (prop.contains("enum") && prop.at("enum").is_array()) {
        for (const nlohmann::json& el : prop.at("enum")) {
            if (el.is_string()) {
                p.enum_values.push_back(el.get<std::string>());
            } else if (el.is_number_integer()) {
                p.enum_values.push_back(std::to_string(el.get<std::int64_t>()));
            } else if (el.is_number_float()) {
                p.enum_values.push_back(std::to_string(el.get<double>()));
            } else if (el.is_boolean()) {
                p.enum_values.push_back(el.get<bool>() ? "true" : "false");
            }
        }
    }
    if (p.type == parsephony::ParamType::Array && prop.contains("items") &&
        prop.at("items").is_object()) {
        const nlohmann::json& items = prop.at("items");
        if (items.contains("type")) {
            const ReducedType rt = reduce_type_field(items.at("type"));
            if (rt.ok) {
                p.has_items_type = true;
                p.items_type = rt.type;
            }
        }
        // Preserve nested item object/array schema in tools_json when present.
        nlohmann::json extras = nlohmann::json::object();
        if (items.contains("properties") && items.at("properties").is_object()) {
            extras["items"] = items;
            p.schema_extras_json = extras.dump();
        } else if (items.contains("enum") && items.at("enum").is_array()) {
            extras["items"] = items;
            p.schema_extras_json = extras.dump();
        }
    }
    if (p.type == parsephony::ParamType::Object && prop.contains("properties") &&
        prop.at("properties").is_object()) {
        nlohmann::json extras = nlohmann::json::object();
        extras["properties"] = prop.at("properties");
        if (prop.contains("required") && prop.at("required").is_array()) {
            extras["required"] = prop.at("required");
        }
        p.schema_extras_json = extras.dump();
    }
}

[[nodiscard]] parsephony::ParamType param_type_of(const nlohmann::json& prop,
                                                  bool* nullable) {
    if (nullable != nullptr) {
        *nullable = false;
    }
    if (!prop.is_object() || !prop.contains("type")) {
        return parsephony::ParamType::Json;
    }
    const ReducedType rt = reduce_type_field(prop.at("type"));
    if (!rt.ok) {
        return parsephony::ParamType::Json;
    }
    if (nullable != nullptr) {
        *nullable = rt.nullable;
    }
    return rt.type;
}

// The file extension for an image block's mime type. Narrow on purpose: the decoder
// sniffs the bytes, so this only has to be plausible enough that view_image's own
// extension check and any human reading the spool directory agree with the content.
[[nodiscard]] std::string extension_for_mime(std::string_view mime) {
    if (mime == "image/png") { return "png"; }
    if (mime == "image/jpeg" || mime == "image/jpg") { return "jpg"; }
    if (mime == "image/gif") { return "gif"; }
    if (mime == "image/webp") { return "webp"; }
    if (mime == "image/bmp") { return "bmp"; }
    if (mime == "image/tiff") { return "tiff"; }
    if (mime == "image/heic" || mime == "image/heif") { return "heic"; }
    return {};
}

// Flattens the MCP tool result's content blocks into the model-facing summary, SPOOLING
// any images to disk so the model can actually be shown them.
//
// An MCP image arrives as base64 in a JSON block, and the rest of this codebase moves
// images by PATH -- ToolResult::images, then a re-read at prompt time. Writing the bytes
// into the workspace's spool directory is what joins the two, and it means an MCP
// screenshot goes through exactly the same decode, smart-resize, patch and splice path as
// a file the model opened itself. Nothing downstream needs to know where it came from.
//
// Before this, every image block became the literal text "[image content omitted]" -- so
// a server whose whole purpose was returning a picture could describe one and never show
// it.
[[nodiscard]] std::string summarize(const mcp::ToolResult& r, const std::string& spool_dir,
                                    std::vector<std::string>& images) {
    std::string out;
    if (r.content.is_array()) {
        int index = 0;
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
                continue;
            }
            if (type.empty()) {
                continue;
            }
            std::string note = "[" + type + " content omitted]";
            if (type == "image" && !spool_dir.empty()) {
                const std::string ext =
                    extension_for_mime(block.value("mimeType", std::string{}));
                std::string bytes;
                if (!ext.empty() &&
                    platform::base64_decode(block.value("data", std::string{}), bytes) &&
                    !bytes.empty()) {
                    const std::string path = spool_dir + "/mcp_image_" +
                                             std::to_string(::getpid()) + "_" +
                                             std::to_string(index++) + "." + ext;
                    std::error_code ec;
                    std::filesystem::create_directories(spool_dir, ec);
                    std::ofstream f(path, std::ios::binary);
                    if (f && f.write(bytes.data(),
                                     static_cast<std::streamsize>(bytes.size()))) {
                        f.close();
                        images.push_back(path);
                        // The model is told the picture is THERE as well as shown it: a
                        // path it can name in a later view_image call is worth more than
                        // an image it can only remember.
                        note = "[image spooled to " + path + " and shown to you]";
                    }
                }
            }
            // Audio, embedded resources and any image that could not be spooled: named,
            // never silently dropped -- the model needs to know something came back.
            if (!out.empty()) {
                out += "\n";
            }
            out += note;
        }
    }
    if (out.empty() && r.structured.has_value()) {
        out = r.structured->dump();
    }
    return out;
}

std::optional<bool> annotation_bool(const std::optional<nlohmann::json>& ann,
                                    const char* key) {
    if (!ann.has_value() || !ann->is_object() || !ann->contains(key)) {
        return std::nullopt;
    }
    const nlohmann::json& v = ann->at(key);
    if (v.is_boolean()) {
        return v.get<bool>();
    }
    return std::nullopt;
}

// Trust answers containment (cards). Annotations on a trusted server answer
// mutates vs read. The previous `mutates = !trusted` made every trusted tool look
// like a read, and `remote` was then used as a second gate to take that back —
// which hid orientation tools from Plan mode.
void apply_mcp_decl_flags(ToolDecl& decl, bool trusted, const mcp::Tool& tool) {
    decl.remote = true;
    decl.executes_commands = false;
    if (!trusted) {
        decl.mutates_workspace = true;
        decl.needs_execution = true;
        decl.irreversible = true;
        return;
    }
    const bool read_only =
        annotation_bool(tool.annotations, "readOnlyHint").value_or(false);
    decl.mutates_workspace = !read_only;
    decl.needs_execution = !read_only;
    decl.irreversible =
        annotation_bool(tool.annotations, "destructiveHint").value_or(false);
}

constexpr std::size_t kMaxInstructionsBytes = 32U * 1024;

} // namespace

std::string namespaced_tool_name(const std::string& server, const std::string& tool) {
    return "mcp__" + server + "__" + tool;
}

std::string registered_mcp_tool_name(const Registry& registry, const std::string& server,
                                     const std::string& tool) {
    // Keep the name the server declared when it would not collide and would not
    // corrupt the grammar. Namespace only to avoid a shadow or a delimiter.
    if (name_is_usable(tool) && registry.find(tool) == nullptr) {
        return tool;
    }
    return namespaced_tool_name(server, tool);
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
        bool nullable = false;
        p.type = param_type_of(prop, &nullable);
        p.nullable = nullable;
        p.required = required.count(key) != 0;
        fill_param_constraints(prop, p);
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
    std::string instructions;
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
            // THE WORKSPACE, so a relative path means to this server what the prompt told
            // the model it means. See Subprocess::Options::working_dir for the fifteen
            // identical failures that motivated it.
            sub.working_dir = registry.workspace().root;
            for (const std::string& kv : cfg.env) {
                const std::size_t eq = kv.find('=');
                if (eq != std::string::npos) {
                    sub.env.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
                }
            }
            conn->client->connect_stdio(std::move(sub));
            const mcp::ServerInfo sinfo = conn->client->initialize();
            if (cfg.trusted && !sinfo.instructions.empty()) {
                conn->instructions = sinfo.instructions;
                if (conn->instructions.size() > kMaxInstructionsBytes) {
                    conn->instructions.resize(kMaxInstructionsBytes);
                }
                status.instructions = conn->instructions;
            }
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
            const std::string registered =
                registered_mcp_tool_name(registry, cfg.name, tool.name);
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
            apply_mcp_decl_flags(decl, trusted, tool);

            const std::string remote_name = tool.name;
            // Capture the Registry so each call can observe the run-scoped CancelToken
            // without threading it through every Handler signature. The registry owns
            // this handler, so the reference outlives every call into it.
            Registry* const host_registry = &registry;
            // WHICH PARAMETERS THE SERVER DECLARED AS STRINGS, so the call does not hand
            // it a parsed object where its schema asked for text. See the parse below.
            std::vector<std::string> string_params;
            for (const parsephony::ParamSpec& ps : decl.spec.params) {
                if (ps.type == parsephony::ParamType::Text) {
                    string_params.push_back(ps.name);
                }
            }
            Registry::Handler handler =
                [client, remote_name, call_timeout, host_registry,
                 string_params = std::move(string_params)](
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
                    // A STRING PARAMETER IS SENT AS A STRING, whatever its contents happen
                    // to parse as. This parsed every value unconditionally, so a parameter
                    // the server's own schema declares as `string` arrived as an object,
                    // a number or a bool whenever its text looked like one -- and the
                    // server rejected the call it had itself specified.
                    //
                    // Measured on r-18ced29746aa7728-2ea858f4: godoer's `godot_build_scene`
                    // takes `spec` as a STRING holding a scene spec. The model sent one,
                    // this parsed it into an object, and godoer answered "Input should be
                    // a valid string [type=string_type, input_value={'path': ...},
                    // input_type=dict]". The same hazard turns a version string "1.20"
                    // into the number 1.2 and an id "007" into 7.
                    //
                    // The parse is still right for every other type: the guard validates
                    // SHAPE, not JSON, so a param typed Object, Array, Number or Boolean
                    // arrives as the text the model emitted and has to be parsed. When it
                    // does not parse, the string goes rather than the call failing.
                    const bool declared_string =
                        std::find(string_params.begin(), string_params.end(), p.name) !=
                        string_params.end();
                    if (declared_string) {
                        args[p.name] = p.value;
                        continue;
                    }
                    nlohmann::json parsed = nlohmann::json::parse(p.value, nullptr, false);
                    args[p.name] = parsed.is_discarded() ? nlohmann::json(p.value)
                                                         : std::move(parsed);
                }
                const model::CancelToken* cancel = host_registry->cancel_token();
                mcp::Client::CancelFn cancelled;
                if (cancel != nullptr) {
                    cancelled = [cancel]() { return cancel->cancelled(); };
                }
                try {
                    const mcp::ToolResult r = client->call_tool(
                        remote_name, args, {}, call_timeout, std::move(cancelled));
                    std::vector<std::string> images;
                    std::string text =
                        summarize(r, host_registry->workspace().spool_dir, images);
                    if (r.is_error) {
                        // The tool ran and failed. That is evidence the model can act on,
                        // so it is a ToolError and not a Refused.
                        return ToolResult::error(ErrorClass::None, false,
                                                 text.empty() ? "the MCP tool reported an "
                                                                "error with no message"
                                                              : text);
                    }
                    ToolResult ok = ToolResult::okay(std::move(text));
                    ok.images = std::move(images);
                    return ok;
                } catch (const mcp::McpError& e) {
                    if (e.code() == mcp::to_int(mcp::ErrorCode::kRequestCancelled)) {
                        return ToolResult::cancelled(std::string("MCP call cancelled: ") +
                                                     e.what());
                    }
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

std::string McpHost::prompt_instructions() const {
    std::string out;
    for (const auto& c : connections_) {
        if (c->instructions.empty()) {
            continue;
        }
        out += "\n\n# MCP server '";
        out += c->config.name;
        out += "'\n\n";
        out += c->instructions;
    }
    return out;
}

} // namespace lmp::tools
