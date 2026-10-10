#pragma once
//
// Tiny gate — second-process typed Choice over a small Qwen3 MLX model (T1).
//
// Hypothesis (non-binding): a tiny Qwen may be sharper than same-weights 27B Pulse
// answering its own stuckness. Default OFF (`LMP_TINY_GATE` unset/0). Never a second
// full load inside the main sidecar (S5.11 one-load); the gate is a sibling process.
//
// Policy when on: Stage-0 owns stall/compact; residual is binary force vs nudge via
// letter-mask Choice over HTTP/stdio JSON. Not MTP. Not 4-way Pulse. See docs/TINY_GATE.md.
//
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lmp::loop {

// Binary questionnaire options only. Stall/compact are Stage-0 (never asked of the model).
enum class GateChoice : std::uint8_t {
    ForceTool = 0,
    Nudge = 1,
};

inline constexpr std::size_t kGateT1OptionCount = 2;

// Decode mask uses single-letter codes A/B (Qwen3 vocab-stable first-token ids).
inline constexpr char kGateT1Letters[kGateT1OptionCount] = {'A', 'B'};

[[nodiscard]] constexpr std::string_view gate_choice_name(GateChoice c) noexcept {
    switch (c) {
        case GateChoice::ForceTool:
            return "force_tool";
        case GateChoice::Nudge:
            return "nudge";
    }
    return "nudge";
}

[[nodiscard]] constexpr std::string_view gate_choice_def(GateChoice c) noexcept {
    switch (c) {
        case GateChoice::ForceTool:
            return "force_tool — task clearly needs a tool next (read/edit/test/shell); "
                   "stop pure think/text.";
        case GateChoice::Nudge:
            return "nudge — one more recovery chance; mild stuck, not exhausted.";
    }
    return "nudge — one more recovery chance; mild stuck, not exhausted.";
}

// Structured features only — no free prose; keep gate context ~100–200 tok.
struct GateFeatures {
    int consec = 0;
    int streak = 0;
    int prompt_tok = 0;
    int reread_max = 0;
    int think = 0;
    int text = 0;
    int tool_tok = 0;
    // loop_cut | no_progress | degenerate | none
    std::string why = "none";
};

[[nodiscard]] std::string format_gate_features(const GateFeatures& f);

// Strip hindsight next=/outcome=/… before any gate prompt material.
[[nodiscard]] std::string strip_gate_hindsight(std::string_view text);

using GateChoiceOrder = std::array<GateChoice, kGateT1OptionCount>;

// Seeded shuffle so force_tool is not always letter A.
[[nodiscard]] GateChoiceOrder shuffle_gate_t1_choices(std::uint64_t seed) noexcept;

// Neutral forced prefix + letter defs + Features block (tiny context).
[[nodiscard]] std::string gate_t1_forced_prefix(const GateChoiceOrder& order,
                                                const GateFeatures& features,
                                                std::string_view mission = {});

struct GateMicroResult {
    bool ok = false;
    std::string error;
    GateChoice choice = GateChoice::Nudge;
    std::string choice_name = "nudge";
    char letter = '?';
    std::vector<float> p_vec;
    float p = 0.0F;
    float p_force = 0.0F;
    double latency_ms = 0.0;
    std::string encoding; // "letter" | "stage0"
    std::string order;    // e.g. "nudge,force_tool"
    std::string model;    // gate model path when known
};

// Stage-0 + binary residual policy outcomes.
enum class GatePolicy : std::uint8_t {
    ForceTool,
    Nudge,
    Stall,
    Compact,
    Fallback,
};

// Default p_min for the tiny-gate residual (research seed: 0.55). Env override:
// LMP_TINY_GATE_P_MIN.
inline constexpr float kGateDefaultPMin = 0.55F;

// Stage-0 (deterministic; no LLM):
//   stall   if consec >= 3 OR streak >= 4
//   else compact if prompt_tok >= 16000 OR reread_max >= 3
//   else nullopt → call tiny gate process
[[nodiscard]] std::optional<GatePolicy> stage0_gate_policy(const GateFeatures& f) noexcept;

// Binary residual: force_tool iff p_force >= p_min, else nudge. Failure → Fallback.
[[nodiscard]] GatePolicy apply_gate_policy(const GateMicroResult& result,
                                           float p_min = kGateDefaultPMin) noexcept;

[[nodiscard]] constexpr std::string_view gate_policy_name(GatePolicy p) noexcept {
    switch (p) {
        case GatePolicy::ForceTool:
            return "force_tool";
        case GatePolicy::Nudge:
            return "nudge";
        case GatePolicy::Stall:
            return "stall";
        case GatePolicy::Compact:
            return "compact";
        case GatePolicy::Fallback:
            return "fallback";
    }
    return "fallback";
}

// Parse a gate-helper JSON response body (no GPU). Used by the HTTP/stdio client and
// unit tests. Accepts the helper's response schema (ok, letter, choice, p, p_force,
// p_vec, latency_ms, model, encoding, order, error).
[[nodiscard]] GateMicroResult parse_gate_response_json(std::string_view body);

// Build the JSON request body the helper expects.
[[nodiscard]] std::string build_gate_request_json(std::string_view prompt,
                                                  const GateChoiceOrder& order);

// HTTP client: POST JSON to a gate helper URL (e.g. http://127.0.0.1:18765/v1/choose).
// Timeout is best-effort via SO_RCVTIMEO. Returns Fallback-shaped failure on transport
// errors (ok=false). Does not spawn the process — Benchbot/sidecar starts it separately.
[[nodiscard]] GateMicroResult query_gate_http(std::string_view base_url,
                                              std::string_view request_json,
                                              int timeout_ms = 5000);

// Test / probe seam: when set and tiny_gate is on, T1 uses this instead of HTTP.
using GateProbe = std::function<std::optional<GateMicroResult>()>;

} // namespace lmp::loop
