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
    // What applying a `plan` call did. Reported back to the model as the tool result, so
    // a malformed checklist is a correctable observation rather than a silent no-op.
    struct PlanOutcome {
        bool ok = false;
        std::string detail;
    };

    // A call batched into the same turn behind the first one. Each still gets its own
    // history record and its own UI row -- batching changes how many prefills a turn
    // costs, not how honestly its calls are reported.
    struct ExtraCall {
        std::string tool_name;
        std::vector<tools::ToolParamValue> params;
        tools::ToolResult result;
    };

    Outcome outcome = Outcome::BackendError;
    std::string assistant_text;
    std::string reasoning;      // peeled off, surfaced separately, never in the answer
    std::string tool_name;
    std::vector<tools::ToolParamValue> tool_params;
    tools::ToolResult tool_result;
    std::vector<ExtraCall> extra_calls;
    model::GenResult generation;
};

// Value of a named param, or empty. The grammar guarantees required params are present,
// so an empty return means "not supplied" rather than "lost".
[[nodiscard]] std::string param_value(const std::vector<tools::ToolParamValue>& params,
                                      std::string_view name);

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
    // Mechanism: make a tool the operator has refused twice UNSAMPLABLE for the rest of
    // the run, by dropping it from the grammar's spec list. Asking again is then not
    // discouraged, it is impossible.
    BlockRefusedTool,
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

// Refusals, counted by TOOL rather than by (tool, params).
//
// RepeatDetector deliberately ignores refusals -- S9.1 says a refused tool NEVER RAN, so
// it is not an execution and repeating it is not a repeat. That is right for the ledgers,
// and it left re-asking free: the destructive fixture kept its files but burned its
// entire budget re-attempting `delete_file` after each refusal, because a refusal is not
// an error either and so nothing fired.
//
// Counted by tool, not by exact call, on purpose. The operator refused a CAPABILITY; a
// model that varies the path by one character has not been told something different, and
// a per-(tool, params) counter would never reach two.
class RefusalLedger {
  public:
    void record(const std::string& tool);
    [[nodiscard]] std::size_t refused_count(const std::string& tool) const;

    // Set by the BlockRefusedTool corrective; read when the next turn's grammar is built.
    void block(std::string tool);
    [[nodiscard]] bool is_blocked(const std::string& tool) const;
    [[nodiscard]] const std::vector<std::string>& blocked() const noexcept {
        return blocked_;
    }

  private:
    std::vector<std::pair<std::string, std::size_t>> refusals_;
    std::vector<std::string> blocked_;
};

// Drops every tool BlockRefusedTool has blocked from a turn's samplable spec list.
//
// Applied to the grammar rather than to the registry, so the block is scoped to
// SAMPLING: a corrective or a verification may still drive the tool directly, and only
// the model is stopped from choosing it again.
//
// Never returns an empty list. With nothing samplable a turn can emit no call at all,
// which is exactly the deadlock the deleted `must_reconcile` restriction caused; `plan`
// is always a legal move and always clears its own precondition, so it is the floor.
[[nodiscard]] std::vector<parsephony::ToolSpec> without_blocked(
    const std::vector<parsephony::ToolSpec>& specs, const RefusalLedger& refusals,
    const std::vector<parsephony::ToolSpec>& all_specs);

struct Budget {
    int max_iterations = 40;
    int wall_clock_seconds = 900;
};

// Ranks the applicable correctives and returns the single highest. At most one per
// turn (S9.2).
// `have_verify_contract` gates SynthesizeVerification. With no contract declared there is
// nothing to synthesize, and the corrective used to run a hardcoded `cmake --build build`
// regardless -- which on the eight Python fixtures in evals/agent is a command that cannot
// work. Passed as a bool rather than the store so this stays a pure function (S11.2).
[[nodiscard]] Corrective choose_corrective(const TurnResult& turn,
                                           const RepeatDetector& repeats,
                                           const RefusalLedger& refusals,
                                           int iterations_used, const Budget& budget,
                                           bool wall_clock_exhausted,
                                           bool have_verify_contract);

// --- classification ---------------------------------------------------------

// `executed` answers "did this call actually EXECUTE?" -- asked first, and the reason
// the classifier takes it as an input rather than inferring it.
[[nodiscard]] Outcome classify_turn(const model::GenResult& gen,
                                    const model::TurnGrammar& grammar, bool executed,
                                    bool refused);

// --- completion -------------------------------------------------------------

// Driven by the deliverable and verification ledgers, never by prose guessing whether the
// model sounded finished -- and, since the seventh pass, never by the model's checklist
// ticks either, which are a self-report wearing a checkbox (S10.4).
//
// `open_items` is REPORTED, not enforced. completing with open_items > 0 means the
// evidence says the mission is met while the model's own list says otherwise. That
// disagreement is worth a human's attention and worth none of the harness's authority.
struct CompletionVerdict {
    bool complete = false;
    std::string reason;
    std::size_t open_items = 0;
    // WHO CHOSE THE PROOF. A complete verdict against a model-chosen contract means "the
    // model's own criterion is satisfied", which is a strictly weaker claim than "the
    // operator's criterion is satisfied" -- and until this field existed the two were
    // reported with the same word. See ContextStore::ContractSource.
    context::ContextStore::ContractSource contract_source =
        context::ContextStore::ContractSource::Model;
    // True when the verdict rests on a contract nobody but the model has vouched for.
    // Reported, never enforced: refusing to complete without an operator contract would
    // break every run that does not have one, which is most of them.
    [[nodiscard]] bool self_declared() const noexcept {
        return complete &&
               contract_source == context::ContextStore::ContractSource::Model;
    }
};

[[nodiscard]] CompletionVerdict evaluate_completion(const context::ContextStore& ctx);

} // namespace lmp::loop
