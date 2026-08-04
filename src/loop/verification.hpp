#pragma once
//
// Verification -- one choke point, falsifiability by intervention (spec S10).
//
// 1. ONE CHOKE POINT records every verification result. There is no other way to
//    assign one; the context store's ledger is written only from here.
// 2. FALSIFIABILITY: a green counts only if that exact check has been PROVEN capable of
//    red. The proof is by intervention (S19.3), not by matching text: break the
//    workspace, re-run, confirm red, restore, confirm green again.
// 3. CANONICALISATION so the proof is paid for once. `cmake --build build` and
//    `cmake --build build ; echo $?` are the same check -- a reporting wrapper must not
//    mint a second identity that needs its own proof.
//
#include <string>
#include <vector>

#include "src/context/context.hpp"
#include "src/tools/registry.hpp"

namespace lmp::loop {

// The command with its status-swallowing wrappers removed, and NOTHING else changed --
// this is the form that is actually executed.
//
// A shell pipeline exits with the status of its last element, so `pytest tests/ | tail -20`
// reports tail's success as the check's: it cannot fail, and a run whose contract cannot
// fail has no feedback loop at all. Running this form instead is what makes the exit code
// mean what the ledger says it means.
[[nodiscard]] std::string executable_form(std::string_view command);

// Strips reporting wrappers and normalises whitespace so one contract has one identity.
//
// The IDENTITY, never something to run: the whitespace collapse would rewrite a quoted
// argument (`pytest -k "a  or  b"`). Use executable_form() to execute.
[[nodiscard]] std::string canonicalize_check(std::string_view command);

// A contract, decomposed into the ATOMIC CHECKS it is made of -- one per top-level `&&`,
// each canonicalized, in declaration order. A contract with no top-level `&&` yields itself
// and everything downstream behaves exactly as it did before this existed.
//
// WHY A CONTRACT IS A SET, NOT A STRING. `swift test && swift build` is two criteria
// written on one line, and a run satisfies it by running two commands -- which is what
// models do, and what a person does. Matching by containment against the whole string then
// recognises NEITHER: `cd /path && swift test` does not contain `swift test && swift build`.
//
// MEASURED: a run declared exactly that, ran both halves repeatedly over 18 turns, and the
// Verifier saw none of it. The ledger kept a single red from turn 20 with
// `workspace_writes=0`, `not_complete: verification still failing` never changed, and the
// run could not have completed no matter what it wrote.
//
// This is NOT a widening of the containment match, which would let a WEAKER command be
// recorded as the whole contract passing. Each atomic check keeps its own identity, its own
// ledger history and its own falsifiability proof, and completion requires EVERY one of
// them to be green -- so running only `swift build` records only `swift build`, and the
// contract stays unsatisfied until `swift test` has its own green.
//
// Splits on `&&` ALONE. `||` means either-of, `;` means regardless-of, and a pipeline is
// one command -- decomposing any of those would change what the operator asked for. Segments
// that assert nothing (`cd somewhere`, `VAR=value`) are dropped rather than kept as vacuous
// always-green checks that could never be proven falsifiable and so would deadlock the gate.
[[nodiscard]] std::vector<std::string> contract_checks(std::string_view contract);

// Why this command's exit status cannot mean what a criterion needs it to mean, or empty
// when there is nothing wrong with it. Takes the EXECUTABLE form -- the swallowers
// executable_form() can simply remove are removed, and this catches what survives that.
//
// The point is to tell a run its CRITERION is broken instead of letting it conclude its
// WORK is. A check that cannot go red never becomes evidence, so the run is told its green
// is unproven every turn forever -- and the honest reading of that, if nobody names the
// real cause, is "my code must still be wrong".
[[nodiscard]] std::string_view unfalsifiable_reason(std::string_view command);

// The part of a failure that is ABOUT THE WORKSPACE, with the volatile parts removed.
//
// Two readings of one contract get compared to answer "did any of the work move this
// failure?", and a raw string compare answers "no" for every tool that stamps its output.
//
// MEASURED on the run this exists for: two `xcodebuild` failures nineteen turns and eleven
// file writes apart, both reporting the same missing scheme, differed in exactly three
// places -- a wall-clock timestamp, a `pid:tid` pair, and a result-bundle filename with
// the time embedded in it. "Byte-identical across attempts" is the right idea and the
// wrong comparison; it would never once have fired.
//
// Digit runs collapse to `#` and whitespace collapses. One rule covers timestamps, pids,
// temp-directory names, uuids and elapsed times, rather than a pattern list that rots. It
// deliberately also collapses LINE NUMBERS and error COUNTS, so "3 errors" and "10 errors"
// share a signature -- safe only because the rest of a compiler's output (the file names,
// the messages) differs whenever the errors themselves differ. A run making progress never
// produces two identical signatures; a run whose entire output is the same modulo numbers
// has not moved.
[[nodiscard]] std::string failure_signature(std::string_view detail);

// Whether ledger[index] is a red that says nothing about the code.
//
// TRUE when another reading of the SAME contract failed with the SAME signature at a
// DIFFERENT number of workspace writes. That pair is the whole argument: the workspace
// changed between them and the failure did not, so whatever this check is looking at, it
// is not the work.
//
// SYMMETRIC on purpose -- both readings in such a pair are disqualified, not just the
// later one. The asymmetric version (only a red with a matching red after it) leaves the
// most recent red standing, and the most recent red is exactly the one is_proven() reaches
// first. The run that motivated this certified `falsifiable: 1` off a "scheme not found"
// that could never have been about the code, and would have gone on doing so.
//
// A repeat at the SAME write count is not unmoved: re-running a check without touching
// anything in between is expected to report the same thing, and says nothing either way.
[[nodiscard]] bool failure_is_unmoved(
    const std::vector<context::VerificationRecord>& ledger, std::size_t index);

// The declared contract, when its latest reading is a red that no work has moved.
//
// This is a finding about the CRITERION, and the distinction is the whole point. A run
// whose contract cannot pass is not failing -- it is unable to succeed, at any budget, and
// every turn it spends on the code is spent on the wrong thing.
//
// MEASURED: a 45-turn, 2508-second run declared `xcodebuild build -scheme ResMon` against
// a project whose only scheme is `Untitled Project`. It ran the contract twice, nineteen
// turns apart, and got the same "does not contain a scheme named ResMon" both times. It
// then DIAGNOSED the problem itself, found the real scheme, and rebuilt with it -- and
// because that command no longer contains the declared contract it never reached the
// Verifier at all. The last 31 minutes of the run produced no verification of any kind.
// Completion was unreachable from turn one and nothing in the harness said so.
struct UnmovedContract {
    bool unmoved = false;
    std::string contract;
    std::string failure; // the latest reading's detail, for the observation
};
[[nodiscard]] UnmovedContract unmoved_contract(const context::ContextStore& ctx);

// The program a check invokes: the first word that is not a directory change, an
// environment assignment or a shell connective.
//
// `cd /abs/path && xcodebuild -scheme X` is a check whose program is `xcodebuild`, and
// every model writes it that way -- the contract is declared bare and executed with a `cd`
// in front. Taking the literal first word would make the program `cd` for nearly every
// real invocation, which is the same as having no rule at all.
[[nodiscard]] std::string_view check_program(std::string_view command);

// Whether `command` runs the same program as `contract` WITHOUT being that check.
//
// The declared contract is matched by containment (Agent::dispatch_call), so a command
// that runs the same tool a different way is not the check and never reaches the ledger.
// That is the correct default -- widening the match would let a WEAKER command be recorded
// as the contract passing, which is the `rename_across_files` failure this whole gate
// exists to stop -- but it is silent, and silence is what cost the run.
//
// MEASURED: a 45-turn run declared `xcodebuild -scheme ResMon`, discovered mid-run that the
// only scheme was `Untitled Project`, rebuilt correctly with it, and recorded ZERO
// verifications for its last 31 minutes. It did the right thing and the harness simply
// stopped watching. Naming the near miss turns a silent gap into a choice the run can act
// on, without changing what counts as evidence.
[[nodiscard]] bool is_near_miss(std::string_view command, std::string_view contract);

class Verifier {
  public:
    Verifier(tools::Registry& registry, context::ContextStore& ctx)
        : registry_(registry), ctx_(ctx) {}

    // THE choke point. Runs `command`, records the result in the ledger, and returns
    // whether it passed. Nothing else in the codebase writes a VerificationRecord.
    bool run_and_record(const std::string& command, int approved_tier);

    // Same, but filed under `contract_id` rather than under the command's own canonical
    // form. The loop uses this so that every way the model spells its check --
    // `pytest x`, `cd d && python3 -m pytest x`, the same with a `tail` on the end --
    // lands on the ONE contract the run declared. Filing them separately gave each
    // variation a fresh identity with no history, so none could ever be falsifiable.
    bool run_and_record_as(const std::string& command, int approved_tier,
                           const std::string& contract_id);

    // ONE EXECUTION, recorded against every atomic check it covers.
    //
    // A decomposed contract (`swift test && swift build`) can be satisfied by a single
    // command that runs both halves, and that command must produce one reading per half.
    // Calling run_and_record_as() in a loop would instead EXECUTE the operator's
    // verification once per half -- two builds for one request -- and two records of two
    // different executions are not two readings of the same event.
    //
    // Returns whether the execution passed. Empty ids records nothing and returns false.
    bool run_and_record_as(const std::string& command, int approved_tier,
                           const std::vector<std::string>& contract_ids);

    // Proves the check can fail, by intervention: `breaker` mutates the workspace so
    // the check MUST fail, the check is re-run and required to be red, then `restore`
    // puts it back and it is required to be green again. Only after that does a pass
    // from this contract count as evidence.
    //
    // Returns true when the proof succeeded. A failed proof is a finding about the
    // CHECK, not about the workspace: it means the check does not test what it claims.
    bool prove_falsifiable(const std::string& command, int approved_tier,
                           const std::function<bool()>& breaker,
                           const std::function<bool()>& restore);

    [[nodiscard]] bool is_proven(const std::string& command) const;

  private:
    // is_proven() asked against the ledger WITH `pending` appended, but with `pending`
    // itself barred from being the proof. See the call site: a record has to be visible to
    // disqualify an earlier identical red, and must still never certify itself.
    [[nodiscard]] bool proven_by(const std::string& contract,
                                 const context::VerificationRecord& pending) const;

    tools::Registry& registry_;
    context::ContextStore& ctx_;
    std::vector<std::string> proven_; // canonical forms
};

} // namespace lmp::loop
