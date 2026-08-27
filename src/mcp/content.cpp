#include "src/mcp/content.hpp"

namespace lmp::mcp {

namespace {

// Optional string fields are omitted rather than sent empty. A client that renders
// `title` unconditionally shows a blank line for `"title": ""`, but skips the field
// when it is absent.
void set_if_nonempty(nlohmann::json& j, const char* key, const std::string& value) {
    if (!value.empty()) {
        j[key] = value;
    }
}

} // namespace

namespace content {

nlohmann::json text(std::string_view s) {
    return nlohmann::json{{"type", "text"}, {"text", std::string(s)}};
}

nlohmann::json image(std::string_view base64_data, std::string_view mime_type) {
    return nlohmann::json{{"type", "image"},
                          {"data", std::string(base64_data)},
                          {"mimeType", std::string(mime_type)}};
}

nlohmann::json audio(std::string_view base64_data, std::string_view mime_type) {
    return nlohmann::json{{"type", "audio"},
                          {"data", std::string(base64_data)},
                          {"mimeType", std::string(mime_type)}};
}

nlohmann::json resource_link(std::string_view uri, std::string_view name,
                             std::string_view description, std::string_view mime_type) {
    nlohmann::json j{{"type", "resource_link"},
                     {"uri", std::string(uri)},
                     {"name", std::string(name)}};
    if (!description.empty()) {
        j["description"] = std::string(description);
    }
    if (!mime_type.empty()) {
        j["mimeType"] = std::string(mime_type);
    }
    return j;
}

nlohmann::json embedded_text_resource(std::string_view uri, std::string_view body,
                                      std::string_view mime_type) {
    return nlohmann::json{
        {"type", "resource"},
        {"resource",
         {{"uri", std::string(uri)}, {"text", std::string(body)}, {"mimeType", std::string(mime_type)}}}};
}

} // namespace content

nlohmann::json Tool::to_json() const {
    nlohmann::json j{{"name", name}, {"inputSchema", input_schema}};
    set_if_nonempty(j, "title", title);
    set_if_nonempty(j, "description", description);
    if (output_schema.has_value()) {
        j["outputSchema"] = *output_schema;
    }
    if (annotations.has_value() && annotations->is_object() && !annotations->empty()) {
        j["annotations"] = *annotations;
    }
    return j;
}

ToolResult ToolResult::text(std::string_view s) {
    ToolResult r;
    r.content.push_back(content::text(s));
    return r;
}

ToolResult ToolResult::failure(std::string_view message) {
    ToolResult r;
    r.content.push_back(content::text(message));
    r.is_error = true;
    return r;
}

nlohmann::json ToolResult::to_json() const {
    nlohmann::json j{{"content", content}, {"isError", is_error}};
    if (structured.has_value()) {
        j["structuredContent"] = *structured;
    }
    return j;
}

nlohmann::json Resource::to_json() const {
    nlohmann::json j{{"uri", uri}, {"name", name}};
    set_if_nonempty(j, "title", title);
    set_if_nonempty(j, "description", description);
    set_if_nonempty(j, "mimeType", mime_type);
    return j;
}

nlohmann::json ResourceTemplate::to_json() const {
    nlohmann::json j{{"uriTemplate", uri_template}, {"name", name}};
    set_if_nonempty(j, "title", title);
    set_if_nonempty(j, "description", description);
    set_if_nonempty(j, "mimeType", mime_type);
    return j;
}

ResourceContents ResourceContents::from_text(std::string_view uri, std::string_view body,
                                             std::string_view mime) {
    ResourceContents c;
    c.uri = std::string(uri);
    c.mime_type = std::string(mime);
    c.text = std::string(body);
    return c;
}

nlohmann::json ResourceContents::to_json() const {
    nlohmann::json j{{"uri", uri}};
    set_if_nonempty(j, "mimeType", mime_type);
    if (text.has_value()) {
        j["text"] = *text;
    }
    if (blob.has_value()) {
        j["blob"] = *blob;
    }
    return j;
}

nlohmann::json PromptArgument::to_json() const {
    nlohmann::json j{{"name", name}, {"required", required}};
    set_if_nonempty(j, "description", description);
    return j;
}

nlohmann::json Prompt::to_json() const {
    nlohmann::json j{{"name", name}};
    set_if_nonempty(j, "title", title);
    set_if_nonempty(j, "description", description);
    if (!arguments.empty()) {
        nlohmann::json args = nlohmann::json::array();
        for (const auto& a : arguments) {
            args.push_back(a.to_json());
        }
        j["arguments"] = std::move(args);
    }
    return j;
}

PromptMessage PromptMessage::user_text(std::string_view s) {
    return PromptMessage{"user", content::text(s)};
}

PromptMessage PromptMessage::assistant_text(std::string_view s) {
    return PromptMessage{"assistant", content::text(s)};
}

nlohmann::json PromptMessage::to_json() const {
    return nlohmann::json{{"role", role}, {"content", content}};
}

} // namespace lmp::mcp
