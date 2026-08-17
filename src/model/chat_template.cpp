#include "src/model/chat_template.hpp"
#include "src/model/family_traits.hpp"

#include <stdexcept>

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

bool ChatTemplate::supports_images() const { return tok_.specials().has_vision(); }

// Each image in its own <|vision_start|> ... <|vision_end|>, before the message's text --
// the order the checkpoint's chat_template.jinja emits for a content list, and the one a
// question about a picture wants.
//
// The pads go in as the SPECIAL id, not as the literal characters: encode_content would
// turn "<|image_pad|>" into ordinary bytes, which the splice would then fail to find, and
// the model would read TEXT ABOUT an image rather than an image. That is the same split
// the tool-call framing makes, for the same reason.
void ChatTemplate::append_images(const Message& m, std::vector<TokenId>& out,
                                 std::vector<ImagePlacement>* placements,
                                 std::size_t message_index) const {
    const SpecialIds& s = tok_.specials();
    for (std::size_t i = 0; i < m.images.size(); ++i) {
        const MessageImage& img = m.images[i];
        if (img.tokens <= 0) {
            // Reached only if a caller rendered before preprocessing the pixels, which
            // would otherwise reserve a zero-length run and leave the splice with nowhere
            // to write.
            throw std::runtime_error(
                "chat template: image has no token count -- preprocess the pixels before "
                "rendering, so the pad run can be reserved");
        }
        out.push_back(s.vision_start);
        if (placements != nullptr) {
            placements->push_back({message_index, i, out.size(), img.tokens});
        }
        out.insert(out.end(), static_cast<std::size_t>(img.tokens), s.image_pad);
        out.push_back(s.vision_end);
    }
}

void ChatTemplate::append_message(const Message& m, std::string_view tools_json,
                                  std::vector<TokenId>& out,
                                  std::vector<ImagePlacement>* placements,
                                  std::size_t message_index) const {
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
            // Images first, each in its own <|vision_start|> ... <|vision_end|>, then the
            // text -- the order the checkpoint's chat_template.jinja emits for a content
            // list, and the one a question about a picture wants.
            //
            append_images(m, out, placements, message_index);
            append(out, tok_.encode_content(m.content));
            break;
        }
        case Role::Assistant: {
            append(out, tok_.encode_template("assistant\n"));
            append(out, tok_.encode_content(m.content));
            // The call, framed by the same SPECIAL ids the grammar accepts when the model
            // emits one -- not the literal characters "<tool_call>", which encode_content
            // would turn into ordinary bytes and which would teach the model a shape it is
            // not allowed to produce.
            if (!m.tool_call.empty()) {
                out.push_back(s.tool_call_open);
                append(out, tok_.encode_template("\n"));
                append(out, tok_.encode_content(m.tool_call));
                append(out, tok_.encode_template("\n"));
                out.push_back(s.tool_call_close);
            }
            break;
        }
        case Role::ToolResponse: {
            // Qwen's template wraps tool results in <tool_response> INSIDE a user turn.
            append(out, tok_.encode_template("user\n"));
            out.push_back(s.tool_response_open);
            append(out, tok_.encode_template("\n"));
            // A tool that returns an image puts it HERE, inside the response block, ahead
            // of whatever the tool said about it. This is still a user turn as far as the
            // model is concerned, which is the role the reference template allows an
            // image on.
            append_images(m, out, placements, message_index);
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
    std::vector<std::size_t>& offsets, std::vector<ImagePlacement>* placements) const {
    std::vector<TokenId> out;
    offsets.clear();
    offsets.reserve(messages.size() + 1);
    if (placements != nullptr) {
        placements->clear();
    }
    // Refused up front rather than per-message, so "this checkpoint cannot take an image"
    // is one clear failure instead of an kInvalidToken spliced into a prompt.
    for (const Message& m : messages) {
        if (m.images.empty()) {
            continue;
        }
        if (!supports_images()) {
            throw std::runtime_error(
                "chat template: a message carries an image but this checkpoint has no "
                "vision tokens");
        }
        // A tool response IS a user turn in Qwen's template (it wraps the result in
        // <tool_response> inside one), so a tool that returns an image is allowed. A
        // system message is not: the reference template raises on that, and so does this.
        if (m.role != Role::User && m.role != Role::ToolResponse) {
            throw std::runtime_error(
                "chat template: images are only legal on a user message or a tool "
                "response (the reference template raises on one in a system message)");
        }
    }
    bool first = true;
    std::size_t index = 0;
    for (const Message& m : messages) {
        offsets.push_back(out.size());
        // The tools block rides on the first message iff it is the system message;
        // it is part of the KV prefix and therefore run-constant (S6.4).
        const bool with_tools = first && m.role == Role::System;
        append_message(m, with_tools ? tools_json : std::string_view{}, out, placements,
                       index);
        first = false;
        ++index;
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
