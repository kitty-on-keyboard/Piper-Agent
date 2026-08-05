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

std::string_view to_string(Corrective c) noexcept {
    switch (c) {
        case Corrective::None:
            return "none";
        case Corrective::BreakRepeat:
            return "break_repeat";
        case Corrective::SynthesizeVerification:
            return "synthesize_verification";
        case Corrective::ForceVerification:
            return "force_verification";
        case Corrective::BlockRefusedTool:
            return "block_refused_tool";
        case Corrective::RederiveContract:
            return "rederive_contract";
        case Corrective::ReconcileChecklist:
            return "reconcile_checklist";
        case Corrective::BudgetNearlyGone:
            return "budget_nearly_gone";
        case Corrective::HaltOnBudget:
            return "halt_on_budget";
    }
    return "none";
}

ModePolicy ModePolicy::for_mode(Mode m) noexcept {
    switch (m) {
        // T0: no execution, no writes, and it TALKS. The first three fields were the
        // whole of plan mode for a long time, and they are the half that never mattered:
        // a mode the model is not told it is in, whose write tools are still advertised
        // to it, spends its turns discovering the refusals one at a time.
        case Mode::Plan:
            return {0, false, false, false, true};
        // Debug WRITES. It could not, which made it useless for the one thing it is named
        // after -- you cannot add a log line, cannot save a reproduction, cannot apply the
        // fix you just proved. What it still cannot do is destroy: instrumenting a bug
        // never requires deleting a file, so the power is not granted.
        case Mode::Debug:
            return {1, true, false, true, false};
        case Mode::Agent:
            return {1, true, true, true, false};
    }
    // An unknown mode is the most restrictive, never the least -- and note that the most
    // restrictive is NOT conversational: a mode nobody declared has no operator waiting on
    // it, and yielding to a human who is not there is a hang, not a safety property.
    return {0, false, false, false, false};
}

// Written as what the mode IS and what it is FOR, not as a list of prohibitions. The
// prohibitions are already enforced twice -- the tool is not advertised and the gate would
// refuse it -- so spending prompt on them would be telling the model not to do something
// it has no way to do. What it cannot get anywhere else is the purpose.
std::string_view mode_brief(Mode m) noexcept {
    switch (m) {
        case Mode::Plan:
            return "# Plan mode\n"
                   "\n"
                   "You are planning, not building. Nothing you do here changes a file or "
                   "runs a command -- the tools for that are not loaded, so do not reach "
                   "for them and do not write as though you had used them. Say what you "
                   "would do, never what you did.\n"
                   "\n"
                   "- Read first. Go and look at the code the request touches; a plan "
                   "written from assumptions is worth less than no plan, because it reads "
                   "as though someone checked.\n"
                   "- Name what you found, including anything that makes the request "
                   "harder or different than it sounds. That is the part the human cannot "
                   "get anywhere else.\n"
                   "- When something is genuinely undecided and the answer would change "
                   "the design, ask with `ask_user` -- one question, the load-bearing one, "
                   "and say what each answer would change. Do not ask to confirm what you "
                   "already believe, and do not ask for permission to continue.\n"
                   "- When the approach is settled, call `exit_plan_mode` with the whole "
                   "plan. It becomes the mission of the run that implements it, so "
                   "anything you leave out is something that run will not know.\n"
                   "\n"
                   "The persona above tells you to run your tests. You cannot, here. Say "
                   "which command WOULD prove the work correct and leave it for the run "
                   "that can execute it.\n";
        case Mode::Debug:
            return "# Debug mode\n"
                   "\n"
                   "You are finding out why something is wrong, and the answer has to be "
                   "observed rather than argued. You can read, run and edit; you cannot "
                   "delete.\n"
                   "\n"
                   "- Reproduce it first. A failure you have watched happen is worth more "
                   "than any amount of reading, and until you have one you are guessing "
                   "about which of several stories is true.\n"
                   "- Instrument rather than theorise. Add the log line, print the value, "
                   "run the command -- and then READ what came back. A hypothesis you did "
                   "not test is not evidence, however well it fits.\n"
                   "- Narrow before you fix. Get to the smallest thing that still fails; "
                   "a fix applied to the whole area is a fix you cannot prove.\n"
                   "- Then fix it, and run the same reproduction again. Ending on 'that "
                   "should do it' is how a bug survives being fixed.\n";
        case Mode::Agent:
            return "";
    }
    return "";
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
//
// And one parameter is dropped from the key entirely: `replace_in_file`'s `new_text`.
//
// What decides whether that call can do anything is the path and `old_text` -- the text
// being searched for. If old_text is not in the file the call fails for EVERY new_text, and
// if it is, the first call consumed it. So (path, old_text) is the whole identity of the
// call, and including new_text meant a model could vary the replacement by one character
// and mint a fresh key for a call that cannot behave any differently.
//
// This is a claim about the tool's contract, not a similarity threshold. `write_file`'s
// `content` stays in the key, because two different contents genuinely are two different
// calls -- the identical-content case is caught upstream now, by the write door refusing
// to write bytes the file already holds (tools::CommitOutcome::unchanged), which both
// costs less and tells the model something a repeat count cannot.
std::string RepeatDetector::key(const std::string& tool,
                                const std::vector<tools::ToolParamValue>& params) {
    std::string k = tool;
    for (const tools::ToolParamValue& p : params) {
        if (tool == "replace_in_file" && p.name == "new_text") {
            continue;
        }
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
    const std::vector<parsephony::ToolSpec>& specs,
    const std::vector<std::pair<std::string, int>>& suppressed) {
    if (suppressed.empty()) {
        return specs;
    }
    std::vector<parsephony::ToolSpec> allowed;
    for (const parsephony::ToolSpec& s : specs) {
        bool held = false;
        for (const auto& [name, turns_left] : suppressed) {
            if (turns_left > 0 && name == s.name) {
                held = true;
                break;
            }
        }
        if (!held) {
            allowed.push_back(s);
        }
    }
    // Suppressing every samplable tool would leave the grammar unsatisfiable, and the
    // turn would come back TextOnly for a reason the model cannot see. Better to allow the
    // repeat than to hand it a turn it cannot spend. This floor matters more now that
    // suppressions accumulate: holding down two tools is the point, holding down all of
    // them would be a turn the run cannot spend at all.
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
                             bool have_verify_contract, bool contract_unmoved,
                             bool checklist_unreconciled, bool writes_unverified) {
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
    // Above every corrective aimed at the WORK, because this run is one turn from stopping
    // and none of them apply to a run that is about to stop. A repeat, an unmoved contract,
    // a described-but-unmade verification are all diagnoses of a run still grinding; this
    // one is the last question asked of a run whose grinding is over.
    //
    // Below the budget arms and below BlockRefusedTool for the same reasons everything else
    // is: a run about to be cut off, and an operator who has said no twice, both outrank a
    // bookkeeping disagreement.
    if (checklist_unreconciled) {
        return Corrective::ReconcileChecklist;
    }
    // ABOVE BreakRepeat, and above everything else a working run can trigger.
    //
    // A run with a broken criterion IS repeating itself, and it is re-reading, and it will
    // narrate -- so the lower correctives all have something to say and every one of them
    // aims at the work. None of them can end the run, because the run is not failing at the
    // work. Fixing the criterion is the only move that changes the outcome, so it outranks
    // the symptoms it causes.
    //
    // Below BlockRefusedTool because that one is the operator's own "no", and below the
    // budget arms because a run about to be cut off needs to land before it needs a better
    // criterion.
    if (contract_unmoved) {
        return Corrective::RederiveContract;
    }
    // ABOVE BreakRepeat, because a run editing without checking is the DISEASE and the
    // repeated call is the symptom. Suppressing the tool first answers "you sent that
    // twice" when the run's actual problem is that it has no idea whether the first one
    // worked -- and a model with no new evidence, denied its editor for a turn, comes back
    // and sends the same guess with the tool it has left.
    //
    // BELOW RederiveContract, because forcing a run of a contract that no work can move
    // buys another copy of a failure the run has already been shown twice.
    //
    // Only when there is a contract to run. Without one this has no mechanism, and a
    // corrective with no mechanism is the thing this enum exists to forbid.
    if (have_verify_contract && writes_unverified) {
        return Corrective::ForceVerification;
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
    // ANY CALL IN THE TURN, not only the one at the front of it.
    //
    // A turn may batch several calls, and this looked at the primary alone -- so a turn
    // that re-read four files was judged entirely on the first of them. Vary that one and
    // the other three repeat forever, unexamined.
    //
    // MEASURED: turns 34, 35 and 37 of a cancelled 38-turn run each re-read the same four
    // unchanged files. BreakRepeat could only ever see one quarter of what those turns did.
    //
    // The primary keeps its own test above because it carries the turn's outcome and its
    // error class; a batched call has a result but no outcome, so `ok()` is the whole
    // condition available for it, and a failed batched call is left to the retry it may
    // legitimately be.
    if (turn.outcome == Outcome::ToolCallExecuted) {
        for (const TurnResult::ExtraCall& extra : turn.extra_calls) {
            if (extra.result.ok() && repeats.seen_count(extra.tool_name, extra.params) > 1) {
                return Corrective::BreakRepeat;
            }
        }
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
        // BOTH TENSES. These were all forward-looking -- "should pass", "should work" --
        // which catches a model predicting success and misses a model ASSERTING it. The
        // second is the more dangerous claim and the more common ending.
        //
        // MEASURED: a run implemented both modules, made the suite green, beat a stale
        // Makefile by symlinking the suite into the path the Makefile expected, then ran
        // pytest directly and finished with "Confirmed: `make test` passes with 7/7 tests
        // green." It never re-ran its DECLARED contract, so the ledger held two reds and no
        // green, and the run ended text_only_no_progress with the mission complete and the
        // workspace correct. The corrective that exists for exactly this did not fire,
        // because the model happened to state it in the past tense.
        //
        // Still only on a TextOnly turn with a contract declared, so the cost of a false
        // positive is one run of a command the run already said proves it -- and the
        // Verifier is the only thing that may write the ledger, so a synthesized run is
        // recorded exactly as an honest one is.
        static constexpr std::string_view kClaims[] = {
            "should pass", "should now build", "should work",
            "tests pass",  "tests now pass",   "all tests pass",
            "suite passes", "passes with",     "is now green",
            "are now green", "builds cleanly",
        };
        bool claims_verification = false;
        for (const std::string_view claim : kClaims) {
            if (t.find(claim) != std::string::npos) {
                claims_verification = true;
                break;
            }
        }
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

CompletionVerdict evaluate_completion(const context::ContextStore& ctx,
                                      bool checklist_waived) {
    // COMPLETION IS EVIDENTIAL (S10.4). Every gate below is an observed fact: a file the
    // harness watched get written, and a command the harness watched go red and then
    // green. None of them is the model's own account of how it went.
    //
    // Those gates decide `evidence_complete`, and they are the whole of it. The checklist
    // is applied afterwards, once, and it decides something different -- see below and see
    // CompletionVerdict.
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
    // EVERY ATOMIC CHECK, each judged on its own latest reading.
    //
    // A contract is a set of criteria, not one string (see contract_checks). This used to
    // look up the whole declared string as a single ledger key, so `swift test && swift
    // build` matched only a command containing that exact text -- which is not what running
    // a two-part contract looks like. The run below satisfied both halves repeatedly and
    // the gate reported "has not run" the entire time.
    //
    // ALL of them, because `&&` means all of them. One green half does not finish a run,
    // and each half carries its own falsifiability proof.
    const std::vector<std::string> checks = contract_checks(ctx.verify_contract());
    const context::VerificationRecord* latest = nullptr;
    for (const std::string& check : checks) {
        const context::VerificationRecord* newest = nullptr;
        for (const context::VerificationRecord& v : vs) {
            if (!v.ran) {
                continue; // a refusal never ran, so it is not evidence either way (S6.2)
            }
            if (!check.empty() && v.contract != check) {
                continue;
            }
            newest = &v; // append-ordered, so the last match is the current one
        }
        if (newest == nullptr) {
            return {false,
                    check.empty() ? "no verification has actually run"
                                  : "the declared contract '" + check + "' has not run",
                    open};
        }
        if (!newest->passed) {
            return {false, "verification still failing: " + newest->contract, open};
        }
        // The one reported by the gates below is the WEAKEST link: an unproven check is
        // what stops the run, so naming a proven one instead would send it to fix the
        // wrong criterion. Falsifiability is the only test left, so the first unproven
        // check wins and an all-proven set falls through on the last.
        if (latest == nullptr || !newest->falsifiable) {
            latest = newest;
        }
    }
    if (latest == nullptr) {
        return {false,
                declared.empty() ? "no verification has actually run"
                                 : "the declared contract '" + declared + "' has not run",
                open};
    }
    // A green counts only if that exact check has been proven capable of red (S10.2).
    // An unproven green does not complete a run.
    if (!latest->falsifiable) {
        // Names the ACTION that clears this, because the completion reason is shown to the
        // model and a reason with no exit is read as a demand to keep grinding. The action
        // is never "break your code": a check that passed before the work began is the
        // wrong check, and re-declaring it costs one call.
        return {false,
                "verification '" + latest->contract +
                    "' has passed but has never been seen to fail, so it is not evidence "
                    "yet -- declare a verify_with that was red before this work and is "
                    "green now",
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

    // Every evidential gate has passed. From here on the question is no longer "did this
    // work happen" -- the harness watched it happen -- but "was this work the whole
    // mission", and the ledgers cannot answer that. A verification proves one command
    // green. Whether that command covers what was asked for is written in exactly one
    // place: the run's own checklist.
    //
    // An open item is therefore not a missing tick. It is the run stating that scope
    // remains, and completing over it publishes a claim the run itself contradicts. The
    // seventh pass reported that contradiction and finished anyway; two consecutive real
    // runs finished at 3 of 11 items and read as having given up (see
    // Corrective::ReconcileChecklist for the trace).
    //
    // It is still not enforced FOREVER -- that was the deleted `must_reconcile` deadlock,
    // and a stale list must never be able to trap a run that is genuinely done. It is
    // enforced until the run has been ASKED, once, by a mechanism. `checklist_waived` is
    // the Agent reporting that it asked and got nothing back.
    if (open != 0 && !checklist_waived) {
        CompletionVerdict v;
        v.evidence_complete = true;
        v.open_items = open;
        v.contract_source = source;
        v.reason = "the evidence is green, but " + std::to_string(open) +
                   " item(s) on the run's own checklist are still open -- restate the "
                   "checklist to tick what is done, or declare a verify_with that covers "
                   "what is not";
        return v;
    }

    std::string why = "deliverables recorded and the ";
    // The word that was missing. "The declared contract passes" reads identically whether
    // the operator set the criterion or the model picked one it could satisfy, and those
    // are not the same claim.
    why += source == context::ContextStore::ContractSource::Operator
               ? "operator's contract passes provably"
               : "contract the MODEL chose passes provably (nobody else vouched for it "
                 "as the mission's criterion)";
    if (open != 0) {
        // Only reachable through the waiver, so this is not "the model forgot to tick" --
        // it is "the model was asked and did not answer". Reported, because a human
        // reading `completed` deserves to know it was not unanimous.
        why += ", though " + std::to_string(open) +
               " checklist item(s) were left open after the run was asked to reconcile them";
    }
    CompletionVerdict v;
    v.complete = true;
    v.evidence_complete = true;
    v.reason = std::move(why);
    v.open_items = open;
    v.contract_source = source;
    return v;
}

} // namespace lmp::loop
