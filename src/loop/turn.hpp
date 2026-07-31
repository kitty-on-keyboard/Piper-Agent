#pragma once
//
// The turn machine (spec S9, S10).
//
// ONE TURN, ONE OUTCOME (S9.1). TurnClassifier maps a completed generation to exactly
// one Outcome -- not two, not zero. It asks "did this call actually EXECUTE?" before
// anything else, because in v1 a second observation silently erased the first.
//
// STEER WITH MECHANISM, NEVER PROSE (S9.2). Every corrective must change state,
// synthesize a tool call, or alter control flow. A corrective that only composes a
// sentence asking the model to behave is forbidden, and scripts/run_ratchets.py counts
// prose-only sites -- the required count is 0. v1 had 11 of 34, including one named
// ForceWrite that forced nothing while logging that it was forcing something.
//
// At most ONE instruction per turn, chosen by rank. Exactly ONE repeat detector, ONE
// stall breaker, ONE budget module.
//
#include <cstdint>
#include <string>
#include <vector>

#include "src/context/context.hpp"
#include "src/model/backend.hpp"
#include "src/model/grammar.hpp"
#include "src/tools/registry.hpp"

namespace lmp::loop {

enum class Outcome : std::uint8_t {
    ToolCallExecuted,   // the call ran; the result is the observation
    ToolCallRefused,    // policy or HITL said no; the tool NEVER RAN
    TextOnly,           // the model answered without acting
    LengthCapped,       // generation hit the cap -- NOT completion
    Cancelled,
    BackendError,
};

[[nodiscard]] std::string_view to_string(Outcome o) noexcept;

struct TurnResult {
    Outcome outcome = Outcome::BackendError;
    std::string assistant_text;
    std::string reasoning;      // peeled off, surfaced separately, never in the answer
    std::string tool_name;
    std::vector<tools::ToolParamValue> tool_params;
    tools::ToolResult tool_result;
    model::GenResult generation;
};

// Mode policy is applied in ONE place (S9.3); the loop does not apply it itself, so a
// config that skips the policy cannot run with writes enabled anyway.
enum class Mode : std::uint8_t { Plan, Debug, Agent };

struct ModePolicy {
    int sandbox_tier = 0;
    bool allow_workspace_writes = false;

    [[nodiscard]] static ModePolicy for_mode(Mode m) noexcept;
};

// --- mechanism steering -----------------------------------------------------
//
// Every Corrective below CHANGES SOMETHING. None of them is "tell the model to try
// harder". The ratchet asserts that.

enum class Corrective : std::uint8_t {
    None,
    // Mechanism: drop the repeated call's result from context and force a different
    // tool by narrowing the grammar's registry for the next turn.
    BreakRepeat,
    // Mechanism: synthesize the verification tool call the model keeps describing but
    // not making.
    SynthesizeVerification,
    // Mechanism: end the run. Not a request -- a control-flow change.
    HaltOnBudget,
};

// Exactly one repeat detector. A call is a repeat when the same tool is invoked with
// the same arguments as a previous turn AND that turn's observation was not an error
// (repeating after an error is legitimate retry).
class RepeatDetector {
  public:
    // Returns the number of times this exact call has already been made successfully.
    [[nodiscard]] std::size_t seen_count(const std::string& tool,
                                         const std::vector<tools::ToolParamValue>& params) const;
    void record(const std::string& tool, const std::vector<tools::ToolParamValue>& params);
    void clear() noexcept { seen_.clear(); }

  private:
    [[nodiscard]] static std::string key(const std::string& tool,
                                         const std::vector<tools::ToolParamValue>& params);
    std::vector<std::pair<std::string, std::size_t>> seen_;
};

struct Budget {
    int max_iterations = 40;
    int wall_clock_seconds = 900;
};

// Ranks the applicable correctives and returns the single highest. At most one per
// turn (S9.2).
[[nodiscard]] Corrective choose_corrective(const TurnResult& turn,
                                           const RepeatDetector& repeats,
                                           int iterations_used, const Budget& budget,
                                           bool wall_clock_exhausted);

// --- classification ---------------------------------------------------------

// `executed` answers "did this call actually EXECUTE?" -- asked first, and the reason
// the classifier takes it as an input rather than inferring it.
[[nodiscard]] Outcome classify_turn(const model::GenResult& gen,
                                    const model::TurnGrammar& grammar, bool executed,
                                    bool refused);

// --- completion -------------------------------------------------------------

// Driven by the checklist and the deliverable ledger, never by prose guessing whether
// the model sounded finished (S10.4).
struct CompletionVerdict {
    bool complete = false;
    std::string reason;
};

[[nodiscard]] CompletionVerdict evaluate_completion(const context::ContextStore& ctx);

} // namespace lmp::loop
