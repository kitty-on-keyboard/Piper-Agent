#include "src/mcp/message.hpp"

#include <functional>

namespace lmp::mcp {

namespace {

// A JSON-RPC id must be a string or a number. The spec discourages fractional numbers,
// and we reject them outright rather than round-trip them lossily through int64.
bool is_integral_number(const nlohmann::json& j) {
    return j.is_number_integer() || j.is_number_unsigned();
}

} // namespace

nlohmann::json Id::to_json() const {
    if (is_number()) {
        return as_number();
    }
    if (is_string()) {
        return as_string();
    }
    return nullptr;
}

Id Id::from_json(const nlohmann::json& j) {
    if (is_integral_number(j)) {
        return Id::number(j.get<std::int64_t>());
    }
    if (j.is_string()) {
        return Id::string(j.get<std::string>());
    }
    return Id::none();
}

std::string Id::debug() const {
    if (is_number()) {
        return std::to_string(as_number());
    }
    if (is_string()) {
        return "\"" + as_string() + "\"";
    }
    return "<none>";
}

std::size_t Id::Hash::operator()(const Id& id) const noexcept {
    if (id.is_number()) {
        // Mixed with a constant so that number 1 and string "1" do not collide.
        return std::hash<std::int64_t>{}(id.as_number()) ^ 0x9e3779b97f4a7c15ULL;
    }
    if (id.is_string()) {
        return std::hash<std::string>{}(id.as_string());
    }
    return 0;
}

nlohmann::json Error::to_json() const {
    nlohmann::json j{{"code", code}, {"message", message}};
    if (data.has_value()) {
        j["data"] = *data;
    }
    return j;
}

Message classify(const nlohmann::json& j) {
    Message m;

    if (!j.is_object()) {
        m.invalid_reason = "message is not a JSON object";
        return m;
    }

    // We accept a missing "jsonrpc" but not a wrong one. Being strict about the version
    // string buys nothing and breaks against servers that omit it; being lax about a
    // value of "1.0" would hide a genuinely mismatched peer.
    if (j.contains("jsonrpc") && j["jsonrpc"] != "2.0") {
        m.invalid_reason = "jsonrpc field is not \"2.0\"";
        return m;
    }

    // Presence of the key, not its value. `"id": null` is present-but-invalid, which is
    // a different thing from a notification, and the two get different treatment below.
    const bool has_id_key = j.contains("id");
    const bool has_method = j.contains("method") && j["method"].is_string();
    const bool has_result = j.contains("result");
    const bool has_error  = j.contains("error");

    if (has_method) {
        m.method = j["method"].get<std::string>();
        if (j.contains("params")) {
            m.params = j["params"];
        }

        if (!has_id_key) {
            m.kind = MessageKind::kNotification;
            return m;
        }

        m.id = Id::from_json(j["id"]);
        if (m.id.is_none()) {
            m.invalid_reason = "request id must be a string or an integer";
            return m;
        }
        m.kind = MessageKind::kRequest;
        return m;
    }

    if (has_result || has_error) {
        if (has_result && has_error) {
            m.invalid_reason = "response carries both result and error";
            return m;
        }
        // A response to a request we could not parse legitimately carries a null id, so
        // an absent-or-null id is tolerated here where it would be fatal above.
        if (has_id_key) {
            m.id = Id::from_json(j["id"]);
        }
        if (has_result) {
            m.result = j["result"];
        } else {
            Error e;
            const auto& ej = j["error"];
            if (ej.is_object()) {
                if (ej.contains("code") && is_integral_number(ej["code"])) {
                    e.code = ej["code"].get<int>();
                }
                if (ej.contains("message") && ej["message"].is_string()) {
                    e.message = ej["message"].get<std::string>();
                }
                if (ej.contains("data")) {
                    e.data = ej["data"];
                }
            } else {
                e.message = "malformed error object";
            }
            m.error = std::move(e);
        }
        m.kind = MessageKind::kResponse;
        return m;
    }

    m.invalid_reason = "message has neither method nor result nor error";
    return m;
}

nlohmann::json make_request(const Id& id, std::string_view method, const nlohmann::json& params) {
    nlohmann::json j{
        {"jsonrpc", "2.0"},
        {"id", id.to_json()},
        {"method", std::string(method)},
    };
    // An absent params is not the same as an empty one to every server, so send the key
    // only when there is something to say.
    if (!params.is_null()) {
        j["params"] = params;
    }
    return j;
}

nlohmann::json make_notification(std::string_view method, const nlohmann::json& params) {
    nlohmann::json j{
        {"jsonrpc", "2.0"},
        {"method", std::string(method)},
    };
    if (!params.is_null()) {
        j["params"] = params;
    }
    return j;
}

nlohmann::json make_response(const Id& id, const nlohmann::json& result) {
    return nlohmann::json{
        {"jsonrpc", "2.0"},
        {"id", id.to_json()},
        {"result", result.is_null() ? nlohmann::json::object() : result},
    };
}

nlohmann::json make_error(const Id& id, ErrorCode code, std::string_view message,
                          const std::optional<nlohmann::json>& data) {
    Error e;
    e.code = to_int(code);
    e.message = std::string(message);
    e.data = data;
    return make_error(id, e);
}

nlohmann::json make_error(const Id& id, const Error& err) {
    return nlohmann::json{
        {"jsonrpc", "2.0"},
        {"id", id.to_json()},
        {"error", err.to_json()},
    };
}

std::string_view to_string(LogLevel level) noexcept {
    switch (level) {
    case LogLevel::kDebug:     return "debug";
    case LogLevel::kInfo:      return "info";
    case LogLevel::kNotice:    return "notice";
    case LogLevel::kWarning:   return "warning";
    case LogLevel::kError:     return "error";
    case LogLevel::kCritical:  return "critical";
    case LogLevel::kAlert:     return "alert";
    case LogLevel::kEmergency: return "emergency";
    }
    return "info";
}

bool parse_log_level(std::string_view name, LogLevel& out) noexcept {
    struct Entry {
        std::string_view name;
        LogLevel level;
    };
    static constexpr Entry kTable[]{
        {"debug", LogLevel::kDebug},         {"info", LogLevel::kInfo},
        {"notice", LogLevel::kNotice},       {"warning", LogLevel::kWarning},
        {"error", LogLevel::kError},         {"critical", LogLevel::kCritical},
        {"alert", LogLevel::kAlert},         {"emergency", LogLevel::kEmergency},
    };
    for (const auto& e : kTable) {
        if (e.name == name) {
            out = e.level;
            return true;
        }
    }
    return false;
}

} // namespace lmp::mcp
