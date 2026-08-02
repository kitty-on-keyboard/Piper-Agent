#pragma once
//
// Content blocks and the feature payloads built from them.
//
// These are thin builders over nlohmann::json rather than a closed type hierarchy. The
// content union grows every protocol revision -- audio arrived in 2025-03-26, resource
// links in 2025-06-18 -- and a sealed variant would have to be reopened each time,
// while a server that wants to emit a block this file has not heard of can still do it.
//
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace lmp::mcp {

namespace content {

[[nodiscard]] nlohmann::json text(std::string_view s);
[[nodiscard]] nlohmann::json image(std::string_view base64_data, std::string_view mime_type);
[[nodiscard]] nlohmann::json audio(std::string_view base64_data, std::string_view mime_type);

// A pointer to a resource the client may then read, as opposed to embedding it.
[[nodiscard]] nlohmann::json resource_link(std::string_view uri, std::string_view name,
                                           std::string_view description = {},
                                           std::string_view mime_type = {});

// The resource itself, inline.
[[nodiscard]] nlohmann::json embedded_text_resource(std::string_view uri, std::string_view text,
                                                    std::string_view mime_type = "text/plain");

} // namespace content

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------
struct Tool {
    std::string name;
    std::string title;        // human-facing; optional
    std::string description;
    nlohmann::json input_schema = nlohmann::json{{"type", "object"}};
    std::optional<nlohmann::json> output_schema;

    [[nodiscard]] nlohmann::json to_json() const;
};

// The result of a tool call.
//
// The `is_error` flag is not the same as a JSON-RPC error, and the distinction is the
// point: a protocol error means the call could not be made, whereas is_error means the
// tool ran and failed. The model needs to see the second kind -- that is how it learns
// the file was not found and tries another path -- so it travels as a normal result.
struct ToolResult {
    nlohmann::json content = nlohmann::json::array();
    bool is_error = false;
    std::optional<nlohmann::json> structured;

    [[nodiscard]] static ToolResult text(std::string_view s);
    [[nodiscard]] static ToolResult failure(std::string_view message);

    ToolResult& add(nlohmann::json block) {
        content.push_back(std::move(block));
        return *this;
    }

    [[nodiscard]] nlohmann::json to_json() const;
};

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------
struct Resource {
    std::string uri;
    std::string name;
    std::string title;
    std::string description;
    std::string mime_type;

    [[nodiscard]] nlohmann::json to_json() const;
};

// An RFC 6570 URI template, for resources whose set is not enumerable.
struct ResourceTemplate {
    std::string uri_template;
    std::string name;
    std::string title;
    std::string description;
    std::string mime_type;

    [[nodiscard]] nlohmann::json to_json() const;
};

struct ResourceContents {
    std::string uri;
    std::string mime_type;
    std::optional<std::string> text;  // exactly one of text / blob
    std::optional<std::string> blob;  // base64

    [[nodiscard]] static ResourceContents from_text(std::string_view uri, std::string_view body,
                                                    std::string_view mime = "text/plain");
    [[nodiscard]] nlohmann::json to_json() const;
};

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------
struct PromptArgument {
    std::string name;
    std::string description;
    bool required = false;

    [[nodiscard]] nlohmann::json to_json() const;
};

struct Prompt {
    std::string name;
    std::string title;
    std::string description;
    std::vector<PromptArgument> arguments;

    [[nodiscard]] nlohmann::json to_json() const;
};

struct PromptMessage {
    std::string role = "user"; // "user" | "assistant"
    nlohmann::json content;

    [[nodiscard]] static PromptMessage user_text(std::string_view s);
    [[nodiscard]] static PromptMessage assistant_text(std::string_view s);
    [[nodiscard]] nlohmann::json to_json() const;
};

} // namespace lmp::mcp
