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
#include <optional>
#include <string_view>

#include "src/model/qwen_tokenizer.hpp"

namespace lmp::model {

// How hard the model is asked to think. THREE LEVELS, not four: the reference template
// validates against exactly ('xhigh', 'medium', 'low') and raises on anything else, and
// `high` -- which reads like it belongs and is the one every summary of this feature
// claims exists -- is precisely what it raises on.
enum class ReasoningEffort : std::uint8_t { Default, Low, Medium, XHigh };

// Empty means "leave the checkpoint on its own default" and is the only non-level this
// accepts. Everything else is either one of the three words or nullopt, so an unknown
// spelling is refused where it arrives rather than silently rendering as no instruction.
[[nodiscard]] inline std::optional<ReasoningEffort> parse_reasoning_effort(
    std::string_view s) noexcept {
    if (s.empty()) {
        return ReasoningEffort::Default;
    }
    if (s == "low") {
        return ReasoningEffort::Low;
    }
    if (s == "medium") {
        return ReasoningEffort::Medium;
    }
    if (s == "xhigh") {
        return ReasoningEffort::XHigh;
    }
    return std::nullopt;
}

// The sentences are the checkpoint's OWN, copied verbatim from its chat_template.jinja.
// They are what the model was tuned against, so they are quoted rather than paraphrased --
// a better-worded instruction is a different instruction.
//
// MEDIUM IS DELIBERATELY EMPTY, and that is the template's behaviour, not an omission
// here: it sets reasoning_instructions only for xhigh and low, so medium is the level that
// says nothing at all. Note that this makes medium CHEAPER than the checkpoint's own
// default -- the template defaults to xhigh, so the fullest instruction is what an
// unconfigured run already gets.
[[nodiscard]] inline constexpr std::string_view reasoning_instructions_for(
    ReasoningEffort e) noexcept {
    switch (e) {
    case ReasoningEffort::XHigh:
        return "Reasoning effort is set to xhigh. Please think carefully through the task, "
               "validate key assumptions, consider plausible alternatives, and prioritize "
               "correctness, consistency, and clarity in the final answer.";
    case ReasoningEffort::Low:
        return "Reasoning effort is set to low. Keep your thinking brief and focused, "
               "moving directly to the conclusion without unnecessary elaboration.";
    case ReasoningEffort::Medium:
    case ReasoningEffort::Default:
        break;
    }
    return {};
}

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
