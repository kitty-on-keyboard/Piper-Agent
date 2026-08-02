#pragma once
//
// JSON-RPC 2.0 message model.
//
// The one type here that earns its keep is Id. JSON-RPC ids are `string | number`, and
// a correlation map keyed on uint64 silently coerces `"abc-1"` to 0 -- which is how a
// client that talks to two servers at once starts delivering one server's replies to
// the other's futures. Six of the seven cook-off clients keyed on uint64
// (docs/BAKEOFF_MCP.md); they get away with it only because they are also the only
// party ever generating ids. A server has no such luxury: it must echo back exactly
// what it was sent, with the type intact.
//
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

#include <nlohmann/json.hpp>

#include "src/mcp/protocol.hpp"

namespace lmp::mcp {

// ---------------------------------------------------------------------------
// Id -- string | number | absent.
//
// "absent" is a distinct third state, not a null: a message with no `id` is a
// notification and must never be replied to, whereas `"id": null` is a malformed
// request that must be. Collapsing those two is what makes a server answer its own
// peer's notifications (cook-off entrant S5).
// ---------------------------------------------------------------------------
class Id {
public:
    Id() = default;

    static Id number(std::int64_t n) {
        Id id;
        id.v_ = n;
        return id;
    }
    static Id string(std::string s) {
        Id id;
        id.v_ = std::move(s);
        return id;
    }
    static Id none() { return Id{}; }

    [[nodiscard]] bool is_none() const noexcept { return std::holds_alternative<std::monostate>(v_); }
    [[nodiscard]] bool is_number() const noexcept { return std::holds_alternative<std::int64_t>(v_); }
    [[nodiscard]] bool is_string() const noexcept { return std::holds_alternative<std::string>(v_); }

    [[nodiscard]] std::int64_t as_number() const { return std::get<std::int64_t>(v_); }
    [[nodiscard]] const std::string& as_string() const { return std::get<std::string>(v_); }

    // Absent id serialises as JSON null. Callers that must distinguish check is_none()
    // first -- serialisation of a notification omits the key entirely.
    [[nodiscard]] nlohmann::json to_json() const;

    // Accepts number or string. A null, an object, or a fractional number yields none().
    [[nodiscard]] static Id from_json(const nlohmann::json& j);

    [[nodiscard]] bool operator==(const Id& other) const noexcept { return v_ == other.v_; }

    // For diagnostics only -- never parsed back.
    [[nodiscard]] std::string debug() const;

    struct Hash {
        [[nodiscard]] std::size_t operator()(const Id& id) const noexcept;
    };

private:
    std::variant<std::monostate, std::int64_t, std::string> v_;
};

// ---------------------------------------------------------------------------
struct Error {
    int code = to_int(ErrorCode::kInternalError);
    std::string message;
    std::optional<nlohmann::json> data;

    [[nodiscard]] nlohmann::json to_json() const;
};

enum class MessageKind {
    kRequest,      // has id and method
    kNotification, // has method, no id
    kResponse,     // has id and (result xor error)
    kInvalid,      // did not parse, or violates the above
};

struct Message {
    MessageKind kind = MessageKind::kInvalid;
    Id id;
    std::string method;
    nlohmann::json params;          // null when absent
    nlohmann::json result;          // responses only
    std::optional<Error> error;     // responses only
    std::string invalid_reason;     // kInvalid only

    [[nodiscard]] bool is_request() const noexcept { return kind == MessageKind::kRequest; }
    [[nodiscard]] bool is_notification() const noexcept { return kind == MessageKind::kNotification; }
    [[nodiscard]] bool is_response() const noexcept { return kind == MessageKind::kResponse; }
    [[nodiscard]] bool is_invalid() const noexcept { return kind == MessageKind::kInvalid; }
};

// Classify an already-parsed JSON value. Separate from JSON parsing because the two
// failures need different error codes: bad JSON is -32700, well-formed JSON that is not
// a valid JSON-RPC message is -32600.
[[nodiscard]] Message classify(const nlohmann::json& j);

// ---------------------------------------------------------------------------
// Constructors for outbound messages. These exist so that `"jsonrpc": "2.0"` is written
// in exactly one place per message shape.
// ---------------------------------------------------------------------------
[[nodiscard]] nlohmann::json make_request(const Id& id, std::string_view method,
                                          const nlohmann::json& params);
[[nodiscard]] nlohmann::json make_notification(std::string_view method,
                                               const nlohmann::json& params);
[[nodiscard]] nlohmann::json make_response(const Id& id, const nlohmann::json& result);
[[nodiscard]] nlohmann::json make_error(const Id& id, ErrorCode code, std::string_view message,
                                        const std::optional<nlohmann::json>& data = std::nullopt);
[[nodiscard]] nlohmann::json make_error(const Id& id, const Error& err);

} // namespace lmp::mcp
