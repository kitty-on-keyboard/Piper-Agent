#pragma once
//
// Pulse -- schema-only Choice gate over option-token logprobs (T1 degenerate).
//
// Iteration A: neutral cue + A/B/C/D letter codes + structured features; option order
// shuffled per item; hindsight stripped from any probe summary. Default OFF
// (`LMP_PULSE=0`). See docs/PULSE.md and research/pulse/ITERATION_A_PROMPT.md.
//
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/model/qwen_tokenizer.hpp"
#include "src/model/token_mask.hpp"

namespace lmp::loop {

// T1 questionnaire options. Pulse only selects among harness actions.
enum class PulseChoice : std::uint8_t {
    ForceTool = 0,
    Nudge = 1,
    Stall = 2,
    Compact = 3,
};

inline constexpr std::size_t kPulseT1OptionCount = 4;

// Decode mask uses single-letter codes A/B/C/D (vocab-stable first-token ids).
// Journal still records the enum name (force_tool|nudge|stall|compact).
inline constexpr char kPulseT1Letters[kPulseT1OptionCount] = {'A', 'B', 'C', 'D'};

[[nodiscard]] constexpr std::string_view pulse_choice_name(PulseChoice c) noexcept {
    switch (c) {
        case PulseChoice::ForceTool:
            return "force_tool";
        case PulseChoice::Nudge:
            return "nudge";
        case PulseChoice::Stall:
            return "stall";
        case PulseChoice::Compact:
            return "compact";
    }
    return "nudge";
}

inline constexpr const char* kPulseT1OptionNames[kPulseT1OptionCount] = {
    "force_tool", "nudge", "stall", "compact"};

[[nodiscard]] constexpr const char* const* pulse_t1_option_names() noexcept {
    return kPulseT1OptionNames;
}

// One-line defs for the Iteration A forced prefix (letter assigned after shuffle).
[[nodiscard]] constexpr std::string_view pulse_choice_def(PulseChoice c) noexcept {
    switch (c) {
        case PulseChoice::ForceTool:
            return "force_tool — task clearly needs a tool next (read/edit/test/shell); "
                   "stop pure think/text.";
        case PulseChoice::Nudge:
            return "nudge — one more recovery chance; mild stuck, not exhausted.";
        case PulseChoice::Stall:
            return "stall — further nudges futile; end cleanly.";
        case PulseChoice::Compact:
            return "compact — context bloated or re-read storm; compact before another "
                   "nudge.";
    }
    return "nudge — one more recovery chance; mild stuck, not exhausted.";
}

// Structured features only — no free prose essay (Iteration A).
struct PulseFeatures {
    int consec = 0;
    int streak = 0;
    int prompt_tok = 0;
    int reread_max = 0;
    int think = 0;
    int text = 0;
    int tool_tok = 0;
    // loop_cut | no_progress | degenerate | none (and harness why_detail aliases).
    std::string why = "none";
};

[[nodiscard]] std::string format_pulse_features(const PulseFeatures& f);

// Remove hindsight / post-decision trails from probe context before decode.
// Strips `next=`, `outcome=`, and similar post-decision fields.
[[nodiscard]] std::string strip_pulse_hindsight(std::string_view text);

// Presentation order of the four choices (letters A..D map to these slots).
using PulseChoiceOrder = std::array<PulseChoice, kPulseT1OptionCount>;

// Seeded shuffle so force_tool is not always letter A / first slot.
[[nodiscard]] PulseChoiceOrder shuffle_t1_choices(std::uint64_t seed) noexcept;

// Neutral forced prefix + letter defs in shuffled order + Features block.
// Optional mission stub is omitted when empty; callers must keep it ≤20 tokens.
[[nodiscard]] std::string pulse_t1_forced_prefix(const PulseChoiceOrder& order,
                                                 const PulseFeatures& features,
                                                 std::string_view mission = {});

struct PulseOptionIds {
    // Token id per presented letter (A..D order), parallel to `choices`.
    std::vector<model::TokenId> ids;
    // Enum for each presented letter slot (same order as ids).
    std::vector<PulseChoice> choices;
    // "letter" when masking A/B/C/D single-token ids (Iteration A default).
    std::string encoding = "letter";
    std::string error;
    [[nodiscard]] bool ok() const noexcept {
        return error.empty() && ids.size() == kPulseT1OptionCount &&
               choices.size() == kPulseT1OptionCount;
    }
};

// Resolve single-letter token ids for A/B/C/D in presentation order.
// Prefers encode_content("A") etc.; requires unique first tokens.
[[nodiscard]] PulseOptionIds resolve_t1_letter_ids(const model::QwenTokenizer& tok,
                                                   const PulseChoiceOrder& order);

// Enum mask: ONLY the given option token ids are legal. Everything else is denied.
[[nodiscard]] model::TokenMask pulse_option_mask(std::size_t vocab_size,
                                                 const std::vector<model::TokenId>& option_ids);

// True when `id` is outside the option set (mask escape / free-text attempt).
[[nodiscard]] bool pulse_mask_escape(const model::TokenMask& mask, model::TokenId id) noexcept;

struct PulseMicroResult {
    bool ok = false;
    std::string error;
    PulseChoice choice = PulseChoice::Nudge;
    std::string choice_name = "nudge";
    char letter = '?'; // A..D as presented
    // Normalised mass over the presented letter set (A..D order). Empty on failure.
    std::vector<float> p_vec;
    float p = 0.0F; // P[choice]
    double latency_ms = 0.0;
    std::size_t prefill_reused_tokens = 0;
    std::string kv_reuse; // Extend|Restore|Reset|unknown|n/a
    std::string encoding; // "letter"
    std::string order;    // e.g. "stall,force_tool,nudge,compact"
};

// Softmax over option token ids only. `choices` is parallel to `option_ids` (presentation
// order after shuffle). Maps argmax letter slot → enum for journal.
[[nodiscard]] PulseMicroResult pulse_decode_from_logits(
    const std::vector<float>& logits, const std::vector<model::TokenId>& option_ids,
    const std::vector<PulseChoice>& choices, double latency_ms = 0.0);

// Argmax if P[choice] >= p_min; otherwise Fallback (caller keeps #150 / stall heuristics).
enum class PulsePolicy : std::uint8_t {
    ForceTool,
    Nudge,
    Stall,
    Compact,
    Fallback,
};

inline constexpr float kPulseDefaultPMin = 0.55F;

[[nodiscard]] PulsePolicy apply_pulse_policy(const PulseMicroResult& result,
                                             float p_min = kPulseDefaultPMin) noexcept;

[[nodiscard]] constexpr std::string_view pulse_policy_name(PulsePolicy p) noexcept {
    switch (p) {
        case PulsePolicy::ForceTool:
            return "force_tool";
        case PulsePolicy::Nudge:
            return "nudge";
        case PulsePolicy::Stall:
            return "stall";
        case PulsePolicy::Compact:
            return "compact";
        case PulsePolicy::Fallback:
            return "fallback";
    }
    return "fallback";
}

// Fixed MaskSource over a precomputed option TokenMask (block-stable; one-token Pulse).
class PulseEnumMask final : public model::MaskSource {
  public:
    explicit PulseEnumMask(model::TokenMask mask) : mask_(std::move(mask)) {}

    [[nodiscard]] const model::TokenMask& mask() const override { return mask_; }
    [[nodiscard]] bool mask_is_block_stable() const override { return true; }
    [[nodiscard]] bool is_block_boundary(model::TokenId /*id*/) const override {
        return true;
    }

  private:
    model::TokenMask mask_;
};

using PulseProbe = std::function<std::optional<PulseMicroResult>()>;

} // namespace lmp::loop
