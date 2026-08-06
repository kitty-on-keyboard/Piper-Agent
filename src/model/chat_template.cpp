#include "src/model/chat_template.hpp"
#include "src/model/family_traits.hpp"

namespace lmp::model {
namespace {

// Qwen XML tool-format reminder (parameter values are raw multi-line text, no JSON
// escaping). Framing open/close tags come from FamilyTraits so the next checkpoint is
// load + traits, not a loop edit. XML remains the Qwen3 default (P2 §12).
constexpr std::string_view kQwenXmlToolReminder =
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
    "- Independent read-only calls (reading several files, say) may be batched into one "
    "turn, each in its own <tool_call> block; anything that writes or runs goes one call "
    "per turn so you see its result before the next move\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE "
    "the function call, but NOT after\n"
    "- Answering in plain text with no tool call ends the run as your final answer, so do "
    "that only when the work is finished (or, in plan mode, when it is the operator's turn)\n"
    "</IMPORTANT>";

void append(std::vector<TokenId>& out, const std::vector<TokenId>& ids) {
    out.insert(out.end(), ids.begin(), ids.end());
}

} // namespace

void ChatTemplate::append_message(const Message& m, std::string_view tools_json,
                                  std::vector<TokenId>& out) const {
    const SpecialIds& s = tok_.specials();
    const FamilyTraits traits = traits_for(tok_.family());
    out.push_back(s.im_start);
    switch (m.role) {
        case Role::System: {
            append(out, tok_.encode_template("system\n"));
            append(out, tok_.encode_content(m.content));
            if (!tools_json.empty()) {
                // The tools block is structure: preamble and suffix go through the
                // template path, the schema JSON itself through the content path.
                append(out, tok_.encode_template(traits.tools_preamble));
                append(out, tok_.encode_content(tools_json));
                std::string suffix(traits.tools_suffix_head);
                if (traits.tool_syntax == ToolCallSyntax::QwenXml) {
                    suffix.append(kQwenXmlToolReminder);
                }
                append(out, tok_.encode_template(suffix));
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
    const FamilyTraits traits = traits_for(tok_.family());
    out.push_back(s.im_start);
    append(out, tok_.encode_template("assistant\n"));
    if (traits.prime_think) {
        out.push_back(s.think_open);
        append(out, tok_.encode_template("\n"));
    }
}

std::vector<TokenId> ChatTemplate::render_with_offsets(
    const std::vector<Message>& messages, std::string_view tools_json,
    std::vector<std::size_t>& offsets) const {
    std::vector<TokenId> out;
    offsets.clear();
    offsets.reserve(messages.size() + 1);
    bool first = true;
    for (const Message& m : messages) {
        offsets.push_back(out.size());
        // The tools block rides on the first message iff it is the system message;
        // it is part of the KV prefix and therefore run-constant (S6.4).
        const bool with_tools = first && m.role == Role::System;
        append_message(m, with_tools ? tools_json : std::string_view{}, out);
        first = false;
    }
    offsets.push_back(out.size()); // where the generation prompt starts
    append_generation_prompt(out);
    return out;
}

// One line, so the two cannot drift: a second copy of the loop is how the offsets would
// eventually stop describing the ids.
std::vector<TokenId> ChatTemplate::render(const std::vector<Message>& messages,
                                          std::string_view tools_json) const {
    std::vector<std::size_t> ignored;
    return render_with_offsets(messages, tools_json, ignored);
}

} // namespace lmp::model
