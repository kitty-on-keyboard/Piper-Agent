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
    // Mechanism: make `plan` the only samplable tool for one turn, so the run must restate
    // its verification contract, and hand it the failure that has stopped being about the
    // code.
    //
    // The one corrective aimed at the CRITERION rather than the work. A contract whose red
    // no amount of work moves cannot pass, so completion is unreachable and every
    // remaining turn is spent on the wrong problem -- and none of the other correctives
    // can see it, because from their side the run looks busy and productive.
    //
    // MEASURED: 45 turns and 2508 seconds against `xcodebuild build -scheme ResMon` in a
    // project whose only scheme was `Untitled Project`. The run found the real scheme by
    // itself on turn ~30 and rebuilt with it; the harness kept requiring the contract it
    // was given. See unmoved_contract() for the full trace.
    //
    // Uses the SAME mechanism as a stale plan -- narrowing the grammar to `plan` -- because
    // that mechanism already exists and already terminates: restating the checklist is what
    // clears it. Prose would not do it. This repo's `prose_correctives` ratchet exists
    // because a corrective that composes a sentence and changes nothing was tried first,
    // eleven times, and none of them worked.
    RederiveContract,
    // Mechanism: pin the next turn's grammar to `plan`, so the run must restate its
    // checklist at the moment its evidence says the mission is met and its own list says
    // otherwise.
    //
    // The disagreement this resolves is the one the completion gate used to just PRINT.
    // Evidence green, three of eleven items ticked, run reported "Complete" -- and the two
    // readings of that are opposite. Either the work is done and the list is stale, or the
    // list is right and the contract the model chose is weaker than the mission. Nothing in
    // the ledgers can tell them apart, because the list is the only place the remaining
    // scope is written down.
    //
    // MEASURED: a five-task macOS mission ended `completed` at 3/11 and again at 3/11 of a
    // different eleven. Both times the work was in fact further along than the list said,
    // and both times the ending read as "gave up two thirds of the way in".
    //
    // So ASK, once, with the mechanism that already exists for making a run restate its
    // plan. One `plan` call answers it either way: tick what is done, or leave items open
    // and declare a verify_with that covers them. This is not the deleted `must_reconcile`
    // restriction -- that one latched, re-entered its own precondition every turn, and
    // deadlocked at fourteen consecutive plan turns. This fires ONCE per run (see
    // Agent::reconcile_asked_), spends one turn like every other narrowing, and the run
    // proceeds whatever the answer is.
    ReconcileChecklist,
    // Mechanism: tell the run how many turns remain, once, while it still has enough of
    // them to land safely.
    //
    // This exists because the completion rule has a sharp edge. A green check does not
    // finish a run until it has been shown capable of RED (S10.2), so a run whose contract
    // was already green when declared has to prove it: break the behaviour, watch it fail,
    // restore, watch it pass. That is the right procedure and the model performs it -- but
    // it is the one procedure with a state in the middle where the workspace is
    // DELIBERATELY BROKEN, and HaltOnBudget does not care.
    //
    // MEASURED, twice: a run that had written a complete, passing implementation started
    // the proof around turn 38 and the budget ended at 40, mid-proof. Both times the
    // workspace was left with the injected bug still in it -- once with 9 failing tests,
    // once not even importable. The harness's own evidence rule corrupted the deliverable
    // it was there to protect.
    BudgetNearlyGone,
    // Mechanism: end the run. Not a request -- a control-flow change.
    HaltOnBudget,
};

// The corrective's name, for the trace. The log used to print these as hand-written
// strings at each emit site, which is how a name and a mechanism drift apart.
[[nodiscard]] std::string_view to_string(Corrective c) noexcept;

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
// Drops ONE tool for ONE turn -- the narrowing half of BreakRepeat, which was declared as
// this corrective's mechanism from the start and was never implemented. Until it was, the
// corrective appended a note asking the model to stop repeating itself and left the
// identical call fully samplable on the next turn.
//
// MEASURED: a real editor run spent all 80 turns alternating `list_dir ResMon` and
// `list_dir ResMon/`, and on the turns BreakRepeat did fire, nothing changed.
//
// Separate from without_blocked() because the lifetime is different -- a refused tool is
// gone for the rest of the run, this one is gone for a few turns -- and because a
// repeated tool is usually the RIGHT tool with wrong arguments. Same floor: never returns
// an empty list, since a turn with nothing samplable can emit no call at all.
//
// A SET WITH DEADLINES, not one tool for one turn. One tool for one turn was the first
// fix, and it turned a one-cycle into a two-cycle: a model with two ways to ask the same
// question just asks the other way and comes straight back. The run that prompted this
// alternated `list_dir ResMon` and `shell find . -name '*.swift'` for twenty-seven turns
// -- a third of its budget -- with BreakRepeat firing on nearly every one of them and
// suppressing a tool that was already not the one about to be called.
//
// So suppressions accumulate and their windows grow with the repeat count (see
// Agent::kMaxSuppressTurns). Once both halves of a ping-pong are held down at once, the
// only samplable moves left are the ones that make progress.
[[nodiscard]] std::vector<parsephony::ToolSpec> without_suppressed(
    const std::vector<parsephony::ToolSpec>& specs,
    const std::vector<std::pair<std::string, int>>& suppressed);

[[nodiscard]] std::vector<parsephony::ToolSpec> without_blocked(
    const std::vector<parsephony::ToolSpec>& specs, const RefusalLedger& refusals,
    const std::vector<parsephony::ToolSpec>& all_specs);

// What a run is allowed to spend. These are ceilings on a RUNAWAY, not a target for a
// healthy run, and they were set too low to finish real work.
//
// MEASURED on the KV-store-with-TTL-and-transactions mission (write an implementation,
// write a suite, iterate until green, prove the check falsifiable): the run that got all
// the way through declared completion on turn 41. At the old ceiling of 40 it was cut off
// one turn short -- and the two before it were cut off mid-falsifiability-proof, each
// leaving a deliberately-injected bug in the workspace. Coding and debugging take turns;
// a ceiling tight enough to end ordinary work is measuring the ceiling, not the agent.
//
// 80 was then measured too low IN TURN, on a bigger mission than the KV store: a five-task
// macOS app (a Mach telemetry actor, a ring buffer, an @Observable view model, Canvas
// visualisers, a MenuBarExtra shell). It ended `turn_budget_exhausted` at exactly 80 with
// the build still red -- not thrashing, not looping, just still writing the sixth of ten
// files it needed. A multi-task mission spends turns on ORDINARY work: each file is a
// write, each build is a shell call, each compile error is a read and an edit. Eighty
// turns is roughly two files' worth of that once a Swift build starts talking back.
//
// 200 turns is the ceiling for a mission of that size with room to iterate, and it is
// still a ceiling: the run that hangs or ping-pongs is caught by the correctives above
// long before it gets here, and by the clock if they miss.
//
// The two must move TOGETHER. Turns measured 12.1-12.7 s on the KV mission, so 200 turns
// is ~2500 s of wall clock; the old 1800 s would have ended every long run on time before
// it ended on turns -- swapping one arbitrary cutoff for another while looking like a fix,
// and reading afterwards as if the turn limit were the thing that fired (see
// Agent::halt_reason_ for why that mattered enough to name the two separately). 4800 s is
// ~1.9x the measured rate, which absorbs the slow turns (a 3000-token write is ~40 s)
// without letting a genuinely hung run sit forever.
//
// Both are operator-settable -- lmPipe.maxIterations and lmPipe.wallClockSeconds, live in
// the sidebar drawer -- and the drawer says which of the two will stop the run first,
// because a raised turn limit under an unraised clock does nothing at all.
struct Budget {
    int max_iterations = 200;
    int wall_clock_seconds = 4800;
};

// How many iterations before the limit the run is warned. See Corrective::BudgetNearlyGone
// for why a warning exists at all: it is the margin in which a half-finished
// falsifiability proof can be undone.
//
// Eight rather than five: five was tried and measured too short. The run it fired on was
// part-way through restoring a multi-method edit, acted on the warning immediately, and
// still ran out -- so the margin has to cover a restore that is several edits long, not
// just one file write and a re-run.
inline constexpr int kBudgetWarningTurns = 8;

// Ranks the applicable correctives and returns the single highest. At most one per
// turn (S9.2).
// `have_verify_contract` gates SynthesizeVerification. With no contract declared there is
// nothing to synthesize, and the corrective used to run a hardcoded `cmake --build build`
// regardless -- which on the eight Python fixtures in evals/agent is a command that cannot
// work. Passed as a bool rather than the store so this stays a pure function (S11.2).
// `contract_unmoved` gates RederiveContract. Passed as a bool for the same reason
// `have_verify_contract` is -- this stays a pure function (S11.2) -- and it carries the
// Agent's once-per-contract policy as well as the ledger's verdict: a run that has already
// been told about this exact contract and re-declared it unchanged must not be pinned to
// `plan` every turn thereafter. That is the deadlock the deleted `must_reconcile`
// restriction caused, and it is worth not building a second time.
// `checklist_unreconciled` gates ReconcileChecklist, and carries the once-per-run policy
// for the same reason: the Agent knows whether it has already asked, and this function has
// no memory to know it with.
[[nodiscard]] Corrective choose_corrective(const TurnResult& turn,
                                           const RepeatDetector& repeats,
                                           const RefusalLedger& refusals,
                                           int iterations_used, const Budget& budget,
                                           bool wall_clock_exhausted,
                                           bool have_verify_contract,
                                           bool contract_unmoved,
                                           bool checklist_unreconciled);

// --- classification ---------------------------------------------------------

// `executed` answers "did this call actually EXECUTE?" -- asked first, and the reason
// the classifier takes it as an input rather than inferring it.
[[nodiscard]] Outcome classify_turn(const model::GenResult& gen,
                                    const model::TurnGrammar& grammar, bool executed,
                                    bool refused);

// --- completion -------------------------------------------------------------

// Driven by the deliverable and verification ledgers, never by prose guessing whether the
// model sounded finished (S10.4).
//
// THE CHECKLIST IS NOT A VOTE, IT IS A SCOPE STATEMENT. The seventh pass dropped it from
// the gate entirely, reasoning that a tick is a self-report and a run that had proved its
// fix should not need the model to agree in checkbox form. Half right. A tick is indeed no
// evidence that work HAPPENED -- but an untick is the model's own statement that work
// REMAINS, and the ledgers cannot contradict it, because nothing in them knows what the
// mission's remaining scope is. Only the list does.
//
// So the disagreement gets resolved instead of printed. `evidence_complete` says every
// evidential gate passed; `complete` additionally requires the run's own list to agree, or
// `checklist_waived` -- which the Agent sets only after asking once, with a mechanism, and
// getting no answer (Corrective::ReconcileChecklist). A waived completion still reports
// `open_items`, because "the evidence says done and the run never said so" is a real
// ending and worth a human's attention.
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
    // Every EVIDENTIAL gate passed: a deliverable was written, the declared contract ran,
    // it is green, that green has been proven capable of red, and the reading postdates
    // the latest instruction. Separate from `complete` because the checklist gate sits on
    // top of it, and the corrective that clears that gate has to be able to tell "the
    // evidence is not there yet" from "the evidence is there and the list disagrees".
    bool evidence_complete = false;
};

// `checklist_waived` lets a run finish over its own open items. It is the Agent's answer
// to a question it has already asked once and been given no answer to -- never a default,
// and never something the model can set for itself.
[[nodiscard]] CompletionVerdict evaluate_completion(const context::ContextStore& ctx,
                                                    bool checklist_waived = false);

} // namespace lmp::loop
