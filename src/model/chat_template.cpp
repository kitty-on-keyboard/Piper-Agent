#include "src/model/chat_template.hpp"

namespace lmp::model {
namespace {

// Qwen 3.6's tool preamble and format reminder, matching the model's own
// chat_template.jinja (carried from v1, which verified it against the shipped
// template). Parameter values are raw multi-line text with NO JSON escaping -- this is
// what makes multi-line write_file content robust: the model never escapes newlines
// or quotes inside a JSON string.
constexpr std::string_view kToolsPreamble =
    "\n\n# Tools\n\n"
    "You have access to the following functions:\n\n"
    "<tools>";

constexpr std::string_view kToolsSuffix =
    "</tools>\n\n"
    "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> "
    "block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- Put exactly ONE tool call per turn\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE "
    "the function call, but NOT after\n"
    "</IMPORTANT>";

void append(std::vector<TokenId>& out, const std::vector<TokenId>& ids) {
    out.insert(out.end(), ids.begin(), ids.end());
}

} // namespace

void ChatTemplate::append_message(const Message& m, std::string_view tools_json,
                                  std::vector<TokenId>& out) const {
    const SpecialIds& s = tok_.specials();
    out.push_back(s.im_start);
    switch (m.role) {
        case Role::System: {
            append(out, tok_.encode_template("system\n"));
            append(out, tok_.encode_content(m.content));
            if (!tools_json.empty()) {
                // The tools block is structure: preamble and suffix go through the
                // template path, the schema JSON itself through the content path.
                append(out, tok_.encode_template(kToolsPreamble));
                append(out, tok_.encode_content(tools_json));
                append(out, tok_.encode_template(kToolsSuffix));
            }
            break;
        }
        case Role::User: {
            append(out, tok_.encode_template("user\n"));
            append(out, tok_.encode_content(m.content));
            break;
        }
        case Role::Assistant: {
            append(out, tok_.encode_template("assistant\n"));
            append(out, tok_.encode_content(m.content));
            break;
        }
        case Role::ToolResponse: {
            // Qwen's template wraps tool results in <tool_response> INSIDE a user turn.
            append(out, tok_.encode_template("user\n"));
            out.push_back(s.tool_response_open);
            append(out, tok_.encode_template("\n"));
            append(out, tok_.encode_content(m.content));
            append(out, tok_.encode_template("\n"));
            out.push_back(s.tool_response_close);
            break;
        }
    }
    out.push_back(s.im_end);
    append(out, tok_.encode_template("\n"));
}

void ChatTemplate::append_generation_prompt(std::vector<TokenId>& out) const {
    const SpecialIds& s = tok_.specials();
    out.push_back(s.im_start);
    append(out, tok_.encode_template("assistant\n"));
    out.push_back(s.think_open);
    append(out, tok_.encode_template("\n"));
}

std::vector<TokenId> ChatTemplate::render(const std::vector<Message>& messages,
                                          std::string_view tools_json) const {
    std::vector<TokenId> out;
    bool first = true;
    for (const Message& m : messages) {
        // The tools block rides on the first message iff it is the system message;
        // it is part of the KV prefix and therefore run-constant (S6.4).
        const bool with_tools = first && m.role == Role::System;
        append_message(m, with_tools ? tools_json : std::string_view{}, out);
        first = false;
    }
    append_generation_prompt(out);
    return out;
}

} // namespace lmp::model
