#pragma once
//
// Pulse -- schema-only Choice gate over option-token logprobs (T1 degenerate).
//
// KEEP-seed (force-only binary): Stage-0 decides stall/compact from features (no LLM);
// residual is a 2-way letter mask (force_tool vs nudge). Default OFF (`LMP_PULSE=0`).
// Not a 4-way reopen. See docs/PULSE.md.
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

// Binary questionnaire options only. Stall/compact are Stage-0 (never asked of the model).
enum class PulseChoice : std::uint8_t {
    ForceTool = 0,
    Nudge = 1,
};

inline constexpr std::size_t kPulseT1OptionCount = 2;

// Decode mask uses single-letter codes A/B (vocab-stable first-token ids).
// Journal still records the enum name (force_tool|nudge).
inline constexpr char kPulseT1Letters[kPulseT1OptionCount] = {'A', 'B'};

[[nodiscard]] constexpr std::string_view pulse_choice_name(PulseChoice c) noexcept {
    switch (c) {
        case PulseChoice::ForceTool:
            return "force_tool";
        case PulseChoice::Nudge:
            return "nudge";
    }
    return "nudge";
}

inline constexpr const char* kPulseT1OptionNames[kPulseT1OptionCount] = {
    "force_tool", "nudge"};

[[nodiscard]] constexpr const char* const* pulse_t1_option_names() noexcept {
    return kPulseT1OptionNames;
}

// One-line defs for the forced prefix (letter assigned after shuffle).
[[nodiscard]] constexpr std::string_view pulse_choice_def(PulseChoice c) noexcept {
    switch (c) {
        case PulseChoice::ForceTool:
            return "force_tool — task clearly needs a tool next (read/edit/test/shell); "
                   "stop pure think/text.";
        case PulseChoice::Nudge:
            return "nudge — one more recovery chance; mild stuck, not exhausted.";
    }
    return "nudge — one more recovery chance; mild stuck, not exhausted.";
}

// Structured features only — no free prose essay.
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

// Presentation order of the two choices (letters A..B map to these slots).
using PulseChoiceOrder = std::array<PulseChoice, kPulseT1OptionCount>;

// Seeded shuffle so force_tool is not always letter A / first slot.
[[nodiscard]] PulseChoiceOrder shuffle_t1_choices(std::uint64_t seed) noexcept;

// Neutral forced prefix + letter defs in shuffled order + Features block.
// Optional mission stub is omitted when empty; callers must keep it ≤20 tokens.
[[nodiscard]] std::string pulse_t1_forced_prefix(const PulseChoiceOrder& order,
                                                 const PulseFeatures& features,
                                                 std::string_view mission = {});

struct PulseOptionIds {
    // Token id per presented letter (A..B order), parallel to `choices`.
    std::vector<model::TokenId> ids;
    // Enum for each presented letter slot (same order as ids).
    std::vector<PulseChoice> choices;
    // "letter" when masking A/B single-token ids.
    std::string encoding = "letter";
    std::string error;
    [[nodiscard]] bool ok() const noexcept {
        return error.empty() && ids.size() == kPulseT1OptionCount &&
               choices.size() == kPulseT1OptionCount;
    }
};

// Resolve single-letter token ids for A/B in presentation order.
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
    char letter = '?'; // A..B as presented
    // Normalised mass over the presented letter set (A..B order). Empty on failure.
    std::vector<float> p_vec;
    float p = 0.0F; // P[choice]
    // Mass on force_tool after letter→enum map (0 when force not in the set / failure).
    float p_force = 0.0F;
    double latency_ms = 0.0;
    std::size_t prefill_reused_tokens = 0;
    std::string kv_reuse; // Extend|Restore|Reset|unknown|n/a|stage0
    std::string encoding; // "letter" | "stage0"
    std::string order;    // e.g. "nudge,force_tool"
};

// Softmax over option token ids only. `choices` is parallel to `option_ids` (presentation
// order after shuffle). Maps argmax letter slot → enum for journal; also fills p_force.
[[nodiscard]] PulseMicroResult pulse_decode_from_logits(
    const std::vector<float>& logits, const std::vector<model::TokenId>& option_ids,
    const std::vector<PulseChoice>& choices, double latency_ms = 0.0);

// Stage-0 + binary residual policy outcomes (stall/compact never come from the LLM).
enum class PulsePolicy : std::uint8_t {
    ForceTool,
    Nudge,
    Stall,
    Compact,
    Fallback,
};

// Per-model seed defaults (Research amend 2026-09-21):
//   27B dense (primary product path): 0.55 — best joint acc/ff/forceR on N=34
//   A3B / MoE transfer path:          0.50 — 0.55 fails force R 6/10; 0.50 passes 9/10
// Env `LMP_PULSE_P_MIN` still overrides either default.
inline constexpr float kPulsePMin27B = 0.55F;
inline constexpr float kPulsePMinA3B = 0.50F;
inline constexpr float kPulseDefaultPMin = kPulsePMin27B;

// True when `model_id` (checkpoint path or name) looks like an A3B / MoE transfer target.
[[nodiscard]] bool pulse_model_is_a3b_moe(std::string_view model_id) noexcept;

// Family-keyed seed default; 27B/unknown → 0.55, A3B/MoE → 0.50.
[[nodiscard]] float pulse_default_p_min(std::string_view model_id) noexcept;

// Stage-0 thresholds (deterministic; no LLM):
//   stall   if consec >= 3 OR streak >= 4
//   else compact if prompt_tok >= 16000 OR reread_max >= 3
//   else nullopt → run binary Pulse
[[nodiscard]] std::optional<PulsePolicy> stage0_pulse_policy(const PulseFeatures& f) noexcept;

// Binary residual: force_tool iff p_force >= p_min, else nudge.
// Probe failure → Fallback (caller keeps #150 / stall heuristics).
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
