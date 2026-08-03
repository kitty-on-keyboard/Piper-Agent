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
    tools::Registry& registry_;
    context::ContextStore& ctx_;
    std::vector<std::string> proven_; // canonical forms
};

} // namespace lmp::loop
