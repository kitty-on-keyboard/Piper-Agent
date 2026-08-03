#include "src/loop/turn.hpp"

#include <algorithm>

// For canonicalize_check: the completion gate has to look a contract up in the same form
// the Verifier files it under.
#include "src/loop/verification.hpp"
// For lexically_normal: a repeat is the same call, not the same bytes. See
// RepeatDetector::key.
#include "src/platform/fs.hpp"

namespace lmp::loop {

std::string_view to_string(Outcome o) noexcept {
    switch (o) {
        case Outcome::ToolCallExecuted:
            return "ToolCallExecuted";
        case Outcome::ToolCallRefused:
            return "ToolCallRefused";
        case Outcome::TextOnly:
            return "TextOnly";
        case Outcome::LengthCapped:
            return "LengthCapped";
        case Outcome::Cancelled:
            return "Cancelled";
        case Outcome::BackendError:
            return "BackendError";
    }
    return "BackendError";
}

ModePolicy ModePolicy::for_mode(Mode m) noexcept {
    switch (m) {
        case Mode::Plan:
            return {0, false}; // T0: no execution, no writes
        case Mode::Debug:
            return {1, false}; // T1 sandbox, read and run, but no workspace mutation
        case Mode::Agent:
            return {1, true};
    }
    return {0, false}; // an unknown mode is the most restrictive, never the least
}

Outcome classify_turn(const model::GenResult& gen, const model::TurnGrammar& grammar,
                      bool executed, bool refused) {
    // "Did this call actually EXECUTE?" is asked FIRST (S9.1). Everything else is a
    // property of a turn that did not run a tool, so nothing downstream can overwrite
    // an execution that happened.
    if (executed) {
        return Outcome::ToolCallExecuted;
    }
    if (refused) {
        return Outcome::ToolCallRefused;
    }
    switch (gen.status) {
        case model::GenStatus::Cancelled:
            return Outcome::Cancelled;
        case model::GenStatus::BackendError:
            return Outcome::BackendError;
        case model::GenStatus::LengthCapped:
            // NOT completion. v1 blurred these and reported a truncated turn as a
            // finished one.
            return Outcome::LengthCapped;
        case model::GenStatus::Complete:
            break;
    }
    // Accepted by the grammar with no tool call: a text answer.
    return grammar.has_tool_call() ? Outcome::ToolCallRefused : Outcome::TextOnly;
}

// A repeat is the same CALL, not the same bytes. `ResMon` and `ResMon/` name one
// directory, and keying on the raw value made them two different calls -- so a run could
// alternate the trailing slash and repeat itself forever without the detector ever
// counting past one.
//
// MEASURED: a real run in the editor spent its entire 80-turn budget alternating
// `list_dir ResMon` and `list_dir ResMon/`, learning nothing, and BreakRepeat fired on
// almost none of it. Normalising the path arguments is what makes the two the same key.
//
// Only path-shaped parameters are normalised. A `command` or a `content` argument is raw
// text where a trailing slash is a real difference.
std::string RepeatDetector::key(const std::string& tool,
                                const std::vector<tools::ToolParamValue>& params) {
    std::string k = tool;
    for (const tools::ToolParamValue& p : params) {
        k += '\x1f';
        k += p.name;
        k += '\x1e';
        k += p.name == "path" ? platform::lexically_normal(p.value) : p.value;
    }
    return k;
}

std::size_t RepeatDetector::seen_count(
    const std::string& tool, const std::vector<tools::ToolParamValue>& params) const {
    const std::string k = key(tool, params);
    for (const auto& [seen_key, count] : seen_) {
        if (seen_key == k) {
            return count;
        }
    }
    return 0;
}

void RepeatDetector::record(const std::string& tool,
                            const std::vector<tools::ToolParamValue>& params) {
    const std::string k = key(tool, params);
    for (auto& [seen_key, count] : seen_) {
        if (seen_key == k) {
            ++count;
            return;
        }
    }
    seen_.emplace_back(k, 1);
}

void RefusalLedger::record(const std::string& tool) {
    for (auto& [name, count] : refusals_) {
        if (name == tool) {
            ++count;
            return;
        }
    }
    refusals_.emplace_back(tool, 1);
}

std::size_t RefusalLedger::refused_count(const std::string& tool) const {
    for (const auto& [name, count] : refusals_) {
        if (name == tool) {
            return count;
        }
    }
    return 0;
}

void RefusalLedger::block(std::string tool) {
    if (!is_blocked(tool)) {
        blocked_.push_back(std::move(tool));
    }
}

bool RefusalLedger::is_blocked(const std::string& tool) const {
    for (const std::string& name : blocked_) {
        if (name == tool) {
            return true;
        }
    }
    return false;
}

std::vector<parsephony::ToolSpec> without_suppressed(
    const std::vector<parsephony::ToolSpec>& specs, const std::string& tool) {
    if (tool.empty()) {
        return specs;
    }
    std::vector<parsephony::ToolSpec> allowed;
    for (const parsephony::ToolSpec& s : specs) {
        if (s.name != tool) {
            allowed.push_back(s);
        }
    }
    // Suppressing the only samplable tool would leave the grammar unsatisfiable, and the
    // turn would come back TextOnly for a reason the model cannot see. Better to allow the
    // repeat than to hand it a turn it cannot spend.
    return allowed.empty() ? specs : allowed;
}

std::vector<parsephony::ToolSpec> without_blocked(
    const std::vector<parsephony::ToolSpec>& specs, const RefusalLedger& refusals,
    const std::vector<parsephony::ToolSpec>& all_specs) {
    if (refusals.blocked().empty()) {
        return specs;
    }
    std::vector<parsephony::ToolSpec> allowed;
    for (const parsephony::ToolSpec& s : specs) {
        if (!refusals.is_blocked(s.name)) {
            allowed.push_back(s);
        }
    }
    if (allowed.empty()) {
        for (const parsephony::ToolSpec& s : all_specs) {
            if (s.name == "plan") {
                allowed.push_back(s);
            }
        }
    }
    return allowed;
}

Corrective choose_corrective(const TurnResult& turn, const RepeatDetector& repeats,
                             const RefusalLedger& refusals, int iterations_used,
                             const Budget& budget, bool wall_clock_exhausted,
                             bool have_verify_contract) {
    // Ranked; the highest applicable one wins, and only one is returned (S9.2).
    if (wall_clock_exhausted || iterations_used >= budget.max_iterations) {
        return Corrective::HaltOnBudget;
    }
    // Immediately below the halt, and above everything else: a run about to be cut off
    // needs to hear that before it needs anything else. Fires on ONE exact iteration, so
    // it cannot repeat -- and the loop's own emit records that it was delivered.
    //
    // Five turns is enough to restore a broken file and re-run the check (the proof it is
    // most likely to be in the middle of), and small enough that a healthy run never sees
    // it as anything but a footnote.
    if (iterations_used == budget.max_iterations - kBudgetWarningTurns) {
        return Corrective::BudgetNearlyGone;
    }
    // Above BreakRepeat: this one is the difference between a run that ends and a run
    // that spends its whole budget asking a question already answered. The second
    // refusal is the trigger -- the first is legitimate (the model could not have known),
    // and blocking on it would take the tool away over a single "no".
    if (turn.outcome == Outcome::ToolCallRefused &&
        refusals.refused_count(turn.tool_name) >= 2 && !refusals.is_blocked(turn.tool_name)) {
        return Corrective::BlockRefusedTool;
    }
    // A repeat is a repeat whether it SUCCEEDED or failed unrecoverably.
    //
    // This used to require ok(), on the reasoning that "repeating after an error is
    // legitimate retry". That is true of a TRANSIENT error and false of anything else: an
    // ambiguous edit, a path that does not exist, a malformed argument. Those are pure
    // functions of the bytes sent, so re-sending the same bytes gets the same answer, and
    // the run just buys the same failure again.
    //
    // MEASURED. failing_test_median sent a byte-identical replace_in_file five times --
    // "old_text matches more than one site (lines 7, 8)" each time -- and nothing fired,
    // because a ToolError is not ok(). It is the same hole RefusalLedger was added to
    // close, one class over: refusals were not errors, and unretryable errors were not
    // repeats.
    const bool unrecoverable_repeat =
        !turn.tool_result.ok() && !turn.tool_result.retryable;
    if (turn.outcome == Outcome::ToolCallExecuted &&
        (turn.tool_result.ok() || unrecoverable_repeat) &&
        repeats.seen_count(turn.tool_name, turn.tool_params) > 1) {
        return Corrective::BreakRepeat;
    }
    // The model described a verification but did not make one. Synthesizing the call is
    // a mechanism; asking it to please run the build would be prose.
    //
    // Only when a contract exists to synthesize. Without one the corrective has nothing to
    // run, and running something else -- as the hardcoded `cmake --build build` did -- files
    // a guaranteed failure against a contract nobody declared.
    if (have_verify_contract && turn.outcome == Outcome::TextOnly &&
        !turn.assistant_text.empty()) {
        const std::string& t = turn.assistant_text;
        const bool claims_verification =
            t.find("should pass") != std::string::npos ||
            t.find("should now build") != std::string::npos ||
            t.find("should work") != std::string::npos;
        if (claims_verification) {
            return Corrective::SynthesizeVerification;
        }
    }
    return Corrective::None;
}

std::string param_value(const std::vector<tools::ToolParamValue>& params,
                        std::string_view name) {
    for (const tools::ToolParamValue& p : params) {
        if (p.name == name) {
            return p.value;
        }
    }
    return {};
}

CompletionVerdict evaluate_completion(const context::ContextStore& ctx) {
    // COMPLETION IS EVIDENTIAL (S10.4). Every gate below is an observed fact: a file the
    // harness watched get written, and a command the harness watched go red and then
    // green. None of them is the model's own account of how it went.
    //
    // The model used to have a vote here, in the form of "every checklist item ticked".
    // It no longer does, for two reasons.
    //
    // First, a tick is a SELF-REPORT -- the same prose-trust this design refuses
    // everywhere else. A run that fixed the bug, proved the fix, and then narrated its
    // success instead of restating the list ended `text_only_no_progress` on work that
    // was demonstrably finished; the evidence was complete and the gate was waiting on
    // the model to agree with it.
    //
    // Second, the industry line is drawn elsewhere: the agent decides when to STOP, the
    // harness decides whether it SUCCEEDED, and benchmarks score the transition of the
    // tests rather than the agent's claim. One boolean was doing both jobs.
    //
    // So an unticked list no longer blocks -- it is reported (`open_items`, and
    // `unfinished_items` on the wire) so a human can see the disagreement.
    const std::size_t open = ctx.open_checklist_items();

    if (ctx.checklist().empty()) {
        return {false, "no checklist: the run has not stated what it must produce", open};
    }
    // An instruction that arrived after the current plan is unfinished business by
    // definition, whatever the ledgers already hold.
    if (ctx.plan_is_stale()) {
        return {false, "an instruction has arrived that the checklist predates", open};
    }
    if (ctx.deliverables().empty()) {
        return {false, "no deliverable was recorded", open};
    }
    const auto& vs = ctx.verifications();
    if (vs.empty()) {
        return {false, "no verification has been run", open};
    }

    // The LATEST reading for the declared contract, not every reading ever taken.
    //
    // Requiring all of them to be green made completion unreachable by construction: the
    // baseline check records a deliberate red at declaration time -- that red IS the
    // proof of falsifiability -- so the ledger of a healthy run always contains a
    // failure. Scanning the whole ledger read the evidence of rigour as evidence of
    // breakage.
    // CANONICALIZED, because that is the form every record is filed under -- the store
    // keeps the contract exactly as it was declared, and both writers (baseline_check and
    // the shell path in dispatch_call) key their records by canonicalize_check() of it.
    // Comparing the raw string here means a contract that is not already in canonical
    // form matches NOTHING: `latest` stays null, and a run whose check is green and proven
    // reports "the declared contract has not run" until the budget kills it.
    //
    // Any wrapper is enough to trigger it -- doubled whitespace, a trailing `2>&1`, a
    // `| tail -20` -- and the failure is silent, because every individual reading in the
    // ledger looks correct.
    const std::string declared = canonicalize_check(ctx.verify_contract());
    const context::VerificationRecord* latest = nullptr;
    for (const context::VerificationRecord& v : vs) {
        if (!v.ran) {
            continue; // a refusal never ran, so it is not evidence either way (S6.2)
        }
        if (!declared.empty() && v.contract != declared) {
            continue;
        }
        latest = &v; // the ledger is append-ordered, so the last match is the current one
    }
    if (latest == nullptr) {
        return {false,
                declared.empty() ? "no verification has actually run"
                                 : "the declared contract '" + declared + "' has not run",
                open};
    }
    if (!latest->passed) {
        return {false, "verification still failing: " + latest->contract, open};
    }
    // A green counts only if that exact check has been proven capable of red (S10.2).
    // An unproven green does not complete a run.
    if (!latest->falsifiable) {
        return {false,
                "verification '" + latest->contract +
                    "' passed but has never been shown capable of failing; "
                    "it does not count as evidence yet",
                open};
    }
    // Evidence has to POSTDATE the instruction it is offered against. Without this a
    // follow-up would complete on its predecessor's green before doing any of the new
    // work -- the ledger was already full, and nothing in it knew it was stale.
    if (latest->seq <= ctx.last_directive_seq()) {
        return {false,
                "'" + latest->contract +
                    "' last passed before the latest instruction; it has not been re-run "
                    "since",
                open};
    }
    const auto source = ctx.verify_contract_source();
    std::string why = "deliverables recorded and the ";
    // The word that was missing. "The declared contract passes" reads identically whether
    // the operator set the criterion or the model picked one it could satisfy, and those
    // are not the same claim.
    why += source == context::ContextStore::ContractSource::Operator
               ? "operator's contract passes provably"
               : "contract the MODEL chose passes provably (nobody else vouched for it "
                 "as the mission's criterion)";
    if (open != 0) {
        why += ", though " + std::to_string(open) + " checklist item(s) are still unticked";
    }
    return {true, why, open, source};
}

} // namespace lmp::loop
