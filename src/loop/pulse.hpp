#pragma once
//
// Pulse -- schema-only Choice gate over option-token logprobs (T1 degenerate).
//
// Between turns, the already-loaded model answers a tiny closed questionnaire under an
// enum mask; harness branches on (choice, P). No free text. Default OFF (`LMP_PULSE=0`).
// See docs/PULSE.md and research notes PULSE_IMPLEMENTABLE_NOTE.
//
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

[[nodiscard]] constexpr const char* const* pulse_t1_option_names() noexcept {
    static constexpr const char* kNames[kPulseT1OptionCount] = {
        "force_tool", "nudge", "stall", "compact"};
    return kNames;
}

// Forced prefix closed over the four T1 options. Appended as a Pulse suffix; prefer
// Extend on the live KV when plan_turn_reuse allows.
[[nodiscard]] inline std::string pulse_t1_forced_prefix() {
    return "[Pulse] Degenerate / text-instead-of-tool pressure. "
           "Answer with exactly one of these tokens and nothing else:\n"
           "force_tool\n"
           "nudge\n"
           "stall\n"
           "compact\n"
           "Answer:";
}

struct PulseOptionIds {
    // First-token id per option, parallel to pulse_t1_option_names().
    std::vector<model::TokenId> ids;
    std::string error; // non-empty when resolution failed (empty encode, collision, ...)
    [[nodiscard]] bool ok() const noexcept {
        return error.empty() && ids.size() == kPulseT1OptionCount;
    }
};

// Resolve first-token ids for T1 options. Colliding first tokens are a shape failure.
[[nodiscard]] PulseOptionIds resolve_t1_option_ids(const model::QwenTokenizer& tok);

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
    // Normalised mass over the option set, parallel to option order. Empty on failure.
    std::vector<float> p_vec;
    float p = 0.0F; // P[choice]
    double latency_ms = 0.0;
    // Filled by the live decode path when known; unit tests leave empty/"unknown".
    std::size_t prefill_reused_tokens = 0;
    std::string kv_reuse; // Extend|Restore|Reset|unknown|n/a
};

// Softmax over option token ids only (raw last-step logits). No free text: if the option
// set is empty or every option id is out of range, returns ok=false.
[[nodiscard]] PulseMicroResult pulse_decode_from_logits(
    const std::vector<float>& logits, const std::vector<model::TokenId>& option_ids,
    double latency_ms = 0.0);

// Argmax if P[choice] >= p_min; otherwise Fallback (caller keeps #150 / stall heuristics).
enum class PulsePolicy : std::uint8_t {
    ForceTool,
    Nudge,
    Stall,
    Compact,
    Fallback, // P below floor -- do not hard-fail open
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

// Optional test / Mac-probe seam. When set on AgentConfig, T1 uses this instead of the
// backend. Returning nullopt means "no Pulse answer" → policy Fallback.
using PulseProbe = std::function<std::optional<PulseMicroResult>()>;

} // namespace lmp::loop
