#include "src/surface/mcp_settings.hpp"

#include "src/platform/fs.hpp"

#include <nlohmann/json.hpp>

namespace lmp::surface {
namespace {

[[nodiscard]] std::string exec_fingerprint(const tools::McpServerConfig& cfg) {
    std::string material = cfg.command;
    material.push_back('\0');
    for (const std::string& a : cfg.args) {
        material += a;
        material.push_back('\0');
    }
    return platform::content_sha256_hex(material);
}

// The string members of a JSON array field, skipping anything that is not a string.
std::vector<std::string> json_string_array(const nlohmann::json& obj, const char* key) {
    std::vector<std::string> out;
    if (!obj.contains(key) || !obj.at(key).is_array()) {
        return out;
    }
    for (const nlohmann::json& v : obj.at(key)) {
        if (v.is_string()) {
            out.push_back(v.get<std::string>());
        }
    }
    return out;
}

} // namespace

std::vector<tools::McpServerConfig> parse_mcp_servers(const std::string& message,
                                                     std::string& signature) {
    // nlohmann rather than the surface::string_field extractors, and this is the one
    // place in the sidecar that needs a real parser: those are a substring search for a
    // scalar and cannot walk an array of objects. Settings are read once per run, so the
    // parse costs nothing that matters -- and the alternative, a hand-encoded string the
    // schema does not describe, is exactly the drift the generated protocol prevents.
    std::vector<tools::McpServerConfig> out;
    signature.clear();
    const nlohmann::json root = nlohmann::json::parse(message, nullptr, false);
    if (root.is_discarded() || !root.is_object()) {
        return out;
    }
    const nlohmann::json& params =
        root.contains("params") && root.at("params").is_object() ? root.at("params") : root;
    if (!params.contains("mcp_servers") || !params.at("mcp_servers").is_array()) {
        return out;
    }
    const nlohmann::json& list = params.at("mcp_servers");
    for (const nlohmann::json& s : list) {
        if (!s.is_object()) {
            continue;
        }
        tools::McpServerConfig cfg;
        cfg.name = s.value("name", std::string{});
        cfg.command = s.value("command", std::string{});
        cfg.args = json_string_array(s, "args");
        cfg.env = json_string_array(s, "env");
        // Required to be a literal boolean true, never coerced. This is the operator
        // saying a server may run OUTSIDE the sandbox without a card for every call, so
        // anything we did not understand -- "true", 1, null, absent -- must read as no.
        // S7.5: no security input gets a permissive default.
        cfg.trusted = s.contains("trusted") && s.at("trusted").is_boolean() &&
                      s.at("trusted").get<bool>();
        out.push_back(std::move(cfg));
    }
    signature = list.dump();
    return out;
}

void connect_mcp_servers(tools::McpHost& host,
                         const std::vector<tools::McpServerConfig>& servers,
                         tools::Registry& registry, platform::EventLogWriter& log,
                         const platform::Clock& clock) {
    // Fingerprints before connect so a spawn failure still records what was asked for.
    std::vector<std::string> hashes;
    hashes.reserve(servers.size());
    for (const tools::McpServerConfig& cfg : servers) {
        hashes.push_back(exec_fingerprint(cfg));
        if (cfg.trusted) {
            platform::Event trust;
            trust.kind = "mcp_trust";
            trust.fields.push_back({"name", cfg.name});
            trust.fields.push_back({"command", cfg.command});
            trust.fields.push_back({"exec_hash", hashes.back()});
            trust.fields.push_back(
                {"why", "operator vouched: tools run outside Seatbelt without per-call cards"});
            log.append(trust, clock);
        }
    }
    const auto report = host.connect_and_register(servers, registry);
    for (std::size_t i = 0; i < report.size(); ++i) {
        const tools::McpServerStatus& st = report[i];
        platform::Event ev;
        ev.kind = "mcp_server";
        ev.fields.push_back({"name", st.name});
        ev.fields.push_back({"connected", st.connected ? "1" : "0"});
        ev.fields.push_back({"tools", std::to_string(st.registered)});
        if (i < servers.size()) {
            ev.fields.push_back({"trusted", servers[i].trusted ? "1" : "0"});
            ev.fields.push_back({"exec_hash", i < hashes.size() ? hashes[i] : ""});
        }
        if (!st.error.empty()) {
            // Absent, not fatal: the run continues without that server's tools.
            ev.fields.push_back({"error", st.error});
        }
        for (const std::string& why : st.rejected) {
            ev.fields.push_back({"rejected", why});
        }
        log.append(ev, clock);
    }
}

} // namespace lmp::surface
