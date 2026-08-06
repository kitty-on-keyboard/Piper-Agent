#pragma once
//
// Per-family chat / tool-format defaults (P2 §12).
//
// Qwen XML tool calls remain the default. The next Qwen checkpoint should be load +
// FamilyTraits (and SpecialIds from the vocab) rather than loop changes. Assumptions
// about chat-template framing and tool-call syntax live here, not scattered through
// the agent loop.
//
#include <cstdint>
#include <string_view>

#include "src/model/qwen_tokenizer.hpp"

namespace lmp::model {

enum class ToolCallSyntax : std::uint8_t {
    QwenXml, // <tool_call><function=...><parameter=...> — default for Qwen3
};

struct FamilyTraits {
    Family family = Family::Qwen3;
    ToolCallSyntax tool_syntax = ToolCallSyntax::QwenXml;
    std::string_view tools_preamble =
        "\n\n# Tools\n\n"
        "You have access to the following functions:\n\n"
        "<tools>";
    std::string_view tools_suffix_head = "</tools>\n\n";
    // Generation scaffold after <|im_start|>assistant\n
    bool prime_think = true;
};

[[nodiscard]] inline constexpr FamilyTraits traits_for(Family family) noexcept {
    FamilyTraits t;
    t.family = family;
    // Only Qwen3 is loadable today; other families refuse at tokenizer load. When a
    // later Qwen checkpoint needs different specials or framing, add a Family enumerator
    // and a branch here — do not edit the ReAct loop.
    if (family == Family::Qwen3) {
        t.tool_syntax = ToolCallSyntax::QwenXml;
        t.prime_think = true;
    }
    return t;
}

} // namespace lmp::model
