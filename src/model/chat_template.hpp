#pragma once
//
// Chat template -- renders a conversation to TOKEN IDS, never to a string that someone
// tokenizes later (spec S5.7, S5.8).
//
// The format is Qwen 3.6's own by default (FamilyTraits / ToolCallSyntax::QwenXml),
// taken from the model's chat_template.jinja: ChatML delimiters, a <tools> schema
// block in the system message, XML tool calls, and tool results wrapped in
// <tool_response> inside a user turn. Family-specific framing lives in family_traits.hpp
// so the next Qwen checkpoint is load + config rather than a loop change.
//
// The golden tests assert exact id sequences. Not strings: a template that renders
// plausible text but tokenizes differently is invisible in a string diff and
// catastrophic at inference.
//
// Message CONTENT is encoded with encode_content(), so user text containing
// "<|im_end|>" cannot mint a control token; the structure is emitted through
// encode_template() and the specials table. That split is the token layer's entire
// prompt-injection defence (S5.4).
//
#include <string>
#include <string_view>
#include <vector>

#include "src/model/qwen_tokenizer.hpp"

namespace lmp::model {

enum class Role : std::uint8_t { System, User, Assistant, ToolResponse };

// An image attached to a message, as the TEMPLATE sees it: a count of `<|image_pad|>`
// ids to reserve. The pixels are the backend's business (InferenceTask::PromptImage);
// what the template owns is the framing around them and, critically, WHERE the run lands
// in the rendered id vector -- because the splice addresses it by absolute offset and
// nothing downstream can check that offset is right. A wrong one overwrites real text and
// the model reads a sentence with a hole in it.
struct MessageImage {
    // (grid_h * grid_w) / merge_unit, from PreprocessedImage::token_count(). Filled in by
    // whoever preprocessed the pixels; rendering with 0 throws rather than reserving an
    // empty run.
    int tokens = 0;
    // Where the pixels came from, carried through for the CALLER's benefit -- the
    // template neither reads it nor cares. It exists so that the ImagePlacement coming
    // back out of render_with_offsets can be paired with the image it describes without
    // the caller keeping a parallel list in render order and hoping it stays in step.
    std::string source = {};
};

struct Message {
    Role role = Role::User;
    std::string content;
    // THE CALL THE ASSISTANT MADE, as the `<function=...>` body without its framing
    // specials, which append_message pushes as real tokens. Assistant role only; empty on
    // a text-only turn.
    //
    // History used to carry the assistant's PROSE and the tool's RESPONSE and nothing in
    // between, so a run's own transcript showed tool results arriving after assistant
    // messages that contained no call -- the model never saw itself calling anything. See
    // ContextStore::render, and the measurement in agent.cpp's call_surface_form.
    // Defaulted so the many two-field brace inits stay valid: a message that is not an
    // assistant tool call should not have to say so.
    std::string tool_call = {};

    // Images, emitted BEFORE the text of this message and in this order -- which is what
    // the checkpoint's own chat_template.jinja does for a content list, and what a caller
    // asking "what is in this picture?" wants: the picture, then the question.
    //
    // USER MESSAGES ONLY. The reference template raises on an image in a system message,
    // and render() refuses the same way rather than emitting framing the model was never
    // trained to see there.
    std::vector<MessageImage> images = {};
};

// Where a message's image landed in the rendered ids. The caller pairs this with the
// pixels it preprocessed, in the same order, to build InferenceTask::PromptImage.
struct ImagePlacement {
    std::size_t message_index = 0;
    std::size_t image_index = 0; // within that message
    std::size_t pad_offset = 0;  // into the rendered id vector
    int tokens = 0;
};

class ChatTemplate {
  public:
    // `tools_json` is the serialized tool schemas placed inside <tools>...</tools> in
    // the system message; empty means no tools block. It is part of the KV prefix and
    // therefore run-constant (S6.4).
    explicit ChatTemplate(const QwenTokenizer& tok) : tok_(tok) {}

    [[nodiscard]] std::vector<TokenId> render(const std::vector<Message>& messages,
                                              std::string_view tools_json) const;

    // render(), plus where each message begins. `offsets` comes back with
    // messages.size() + 1 entries; the last is where the generation prompt starts.
    //
    // This exists because render() APPENDS THE GENERATION PROMPT, so rendering the first k
    // messages does NOT produce a token prefix of rendering all of them -- it ends in
    // "<|im_start|>assistant\n<think>\n". Anything that needs "how many tokens do the
    // first k messages occupy" (the KV checkpoint boundary, S5.10) must ask here rather
    // than re-render a sub-list. Getting that wrong does not crash; it reuses a cache
    // against the wrong prefix and the text stays fluent.
    // `placements`, when non-null, receives one entry per image across all messages, in
    // render order. Callers that send images MUST use this rather than computing offsets
    // themselves: the framing, the think primer and the tools block all move the run.
    [[nodiscard]] std::vector<TokenId> render_with_offsets(
        const std::vector<Message>& messages, std::string_view tools_json,
        std::vector<std::size_t>& offsets,
        std::vector<ImagePlacement>* placements = nullptr) const;

    void append_message(const Message& m, std::string_view tools_json,
                        std::vector<TokenId>& out,
                        std::vector<ImagePlacement>* placements = nullptr,
                        std::size_t message_index = 0) const;

    // Whether this tokenizer can frame an image at all. A checkpoint without the vision
    // specials cannot, and render() throws rather than emitting kInvalidToken.
    [[nodiscard]] bool supports_images() const;

  private:
    void append_images(const Message& m, std::vector<TokenId>& out,
                       std::vector<ImagePlacement>* placements,
                       std::size_t message_index) const;

  public:

    // "<|im_start|>assistant\n<think>\n" -- Qwen3 derails without the think scaffold
    // primed (S5.7).
    void append_generation_prompt(std::vector<TokenId>& out) const;

  private:
    const QwenTokenizer& tok_;
};

} // namespace lmp::model
