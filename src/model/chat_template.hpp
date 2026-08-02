#pragma once
//
// Chat template -- renders a conversation to TOKEN IDS, never to a string that someone
// tokenizes later (spec S5.7, S5.8).
//
// The format is Qwen 3.6's own, taken from the model's chat_template.jinja (v1 verified
// it against the shipped template; parsephone re-verified the tool-call framing):
// ChatML delimiters, a <tools> schema block in the system message, XML tool calls, and
// tool results wrapped in <tool_response> inside a user turn.
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

struct Message {
    Role role = Role::User;
    std::string content;
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
    [[nodiscard]] std::vector<TokenId> render_with_offsets(
        const std::vector<Message>& messages, std::string_view tools_json,
        std::vector<std::size_t>& offsets) const;

    void append_message(const Message& m, std::string_view tools_json,
                        std::vector<TokenId>& out) const;

    // "<|im_start|>assistant\n<think>\n" -- Qwen3 derails without the think scaffold
    // primed (S5.7).
    void append_generation_prompt(std::vector<TokenId>& out) const;

  private:
    const QwenTokenizer& tok_;
};

} // namespace lmp::model
