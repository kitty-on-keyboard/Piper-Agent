#pragma once
//
// Model Context Protocol -- wire constants.
//
// Everything in this file is a fact about the protocol, not a choice we made, so it
// lives apart from the code that acts on it. The cook-off's most uniform defect was
// burying "2024-11-05" as a literal in fourteen separate call sites (docs/BAKEOFF_MCP.md);
// a version that appears once can be negotiated, and one that appears eleven times
// cannot.
//
#include <array>
#include <string_view>

namespace lmp::mcp {

// ---------------------------------------------------------------------------
// Protocol revisions, newest first. Order is load-bearing: negotiation walks this
// array and takes the first entry the peer also supports.
// ---------------------------------------------------------------------------
inline constexpr std::string_view kProtocolVersion = "2025-06-18";

inline constexpr std::array<std::string_view, 3> kSupportedVersions{
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
};

[[nodiscard]] constexpr bool is_supported_version(std::string_view v) noexcept {
    for (const auto& s : kSupportedVersions) {
        if (s == v) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Error codes.
//
// -32768..-32000 is the JSON-RPC reserved range. MCP adds its own above it; the two
// that matter in practice are kNotInitialized (a request arrived before the handshake
// finished) and kResourceNotFound.
// ---------------------------------------------------------------------------
enum class ErrorCode : int {
    kParseError     = -32700,
    kInvalidRequest = -32600,
    kMethodNotFound = -32601,
    kInvalidParams  = -32602,
    kInternalError  = -32603,

    // MCP-defined.
    kNotInitialized   = -32002,
    kResourceNotFound = -32002,
    kRequestCancelled = -32800,
};

[[nodiscard]] constexpr int to_int(ErrorCode c) noexcept {
    return static_cast<int>(c);
}

// ---------------------------------------------------------------------------
// Method names. Spelled once each.
// ---------------------------------------------------------------------------
namespace method {

inline constexpr std::string_view kInitialize = "initialize";
inline constexpr std::string_view kPing       = "ping";

inline constexpr std::string_view kToolsList = "tools/list";
inline constexpr std::string_view kToolsCall = "tools/call";

inline constexpr std::string_view kResourcesList          = "resources/list";
inline constexpr std::string_view kResourcesRead          = "resources/read";
inline constexpr std::string_view kResourcesTemplatesList = "resources/templates/list";
inline constexpr std::string_view kResourcesSubscribe     = "resources/subscribe";
inline constexpr std::string_view kResourcesUnsubscribe   = "resources/unsubscribe";

inline constexpr std::string_view kPromptsList = "prompts/list";
inline constexpr std::string_view kPromptsGet  = "prompts/get";

inline constexpr std::string_view kCompletionComplete = "completion/complete";
inline constexpr std::string_view kLoggingSetLevel    = "logging/setLevel";

// Client-implemented; the server calls these back the other way.
inline constexpr std::string_view kRootsList        = "roots/list";
inline constexpr std::string_view kSamplingCreate   = "sampling/createMessage";
inline constexpr std::string_view kElicitationCreate = "elicitation/create";

} // namespace method

namespace notification {

inline constexpr std::string_view kInitialized = "notifications/initialized";
inline constexpr std::string_view kCancelled   = "notifications/cancelled";
inline constexpr std::string_view kProgress    = "notifications/progress";
inline constexpr std::string_view kMessage     = "notifications/message";

inline constexpr std::string_view kToolsListChanged     = "notifications/tools/list_changed";
inline constexpr std::string_view kResourcesListChanged = "notifications/resources/list_changed";
inline constexpr std::string_view kResourcesUpdated     = "notifications/resources/updated";
inline constexpr std::string_view kPromptsListChanged   = "notifications/prompts/list_changed";
inline constexpr std::string_view kRootsListChanged     = "notifications/roots/list_changed";

} // namespace notification

// ---------------------------------------------------------------------------
// Logging levels, RFC 5424 order (most verbose first). Comparable: a message is
// emitted when its level is at least as severe as the configured minimum.
// ---------------------------------------------------------------------------
enum class LogLevel : int {
    kDebug = 0,
    kInfo,
    kNotice,
    kWarning,
    kError,
    kCritical,
    kAlert,
    kEmergency,
};

[[nodiscard]] std::string_view to_string(LogLevel level) noexcept;
[[nodiscard]] bool parse_log_level(std::string_view name, LogLevel& out) noexcept;

} // namespace lmp::mcp
